"""Submission notebook: test DICOMs -> submission.csv. Runs on Kaggle, no internet, <=9h.

The inference path the whole project exists to feed. It reproduces the feature cache live --
there is no cache for the test set -- then runs the fusion head trained on the M5.

    /kaggle/input/<competition>          the test DICOMs
    /kaggle/input/rsna-knee-code         this repo, so pipeline/preprocess.py is importable
    /kaggle/input/dinov2-weights         DINOv2 .safetensors/.bin  <- NOT downloaded, see below
    /kaggle/input/rsna-knee-fusion       fold*.pt from fusion/train.py

FOUR THINGS THAT KILL A SUBMISSION, ALL HANDLED HERE:

 1. NO INTERNET. `timm.create_model(..., pretrained=True)` reaches out to HuggingFace and dies.
    The weights must be attached as a Dataset and loaded from disk. This is the single most
    common submission-day failure and it does not reproduce in an interactive session, where
    the weights are already cached.
 2. PREPROCESSING DRIFT. If this file preprocessed test data even slightly differently from
    kaggle_02, the model would score badly with no error anywhere. Both import the same
    pipeline/preprocess.py, and assert_matches() compares the cache manifest's fingerprint
    against this file's before a single study is read.
 3. MISSING SERIES. 87.2% of studies lack at least one of the six types (FINDINGS.md 3.2) and
    some test studies will lack a plane entirely. Every per-study call is wrapped: a study that
    fails for any reason emits 0.5 rather than taking the submission down with it.
 4. THE HEADER. `Baker's` contains an apostrophe and several columns contain spaces, so the
    column order and quoting are written explicitly rather than left to a dict's ordering.

NO horizontal-flip TTA. hflip swaps Medial and Lateral, so it is only valid with the output
pairs swapped too (PLAN.md 3.2). Handedness is canonicalised in preprocessing instead.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

for _p in ("/kaggle/input/rsna-knee-code", str(Path(__file__).resolve().parents[1])):
    if Path(_p).exists():
        sys.path.insert(0, _p + "/pipeline")
        sys.path.insert(0, _p + "/fusion")
        break
from preprocess import (BATCH_HINT, MODEL, PLANE_ID, assert_matches,   # noqa: E402
                        imagenet_normalise, load_series, to_25d)
from dataset import series_type_id                                     # noqa: E402
from model import FusionHead                                           # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

COMP = None          # resolved below
WEIGHTS_DIR = Path("/kaggle/input/dinov2-weights")
FUSION_DIR = Path("/kaggle/input/rsna-knee-fusion")
CACHE_MANIFEST = FUSION_DIR / "manifest.json"
OUT = Path("/kaggle/working/submission.csv")


def find_competition() -> Path:
    for p in sorted(Path("/kaggle/input").iterdir()):
        if p.is_dir() and "knee" in p.name.lower() and not any(
                x in p.name.lower() for x in ("code", "fusion", "weights")):
            return p
    sys.exit("competition data not found under /kaggle/input")


def load_backbone(dev: str):
    """DINOv2 from an attached Dataset. pretrained=False so timm never touches the network."""
    import timm
    model = timm.create_model(MODEL, pretrained=False, num_classes=0)
    files = [p for p in WEIGHTS_DIR.rglob("*") if p.suffix in (".safetensors", ".bin", ".pt")]
    if not files:
        sys.exit(f"no DINOv2 weights under {WEIGHTS_DIR}. Attach them as a Dataset -- with no "
                 f"internet, pretrained=True cannot work.")
    w = files[0]
    if w.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(w))
    else:
        sd = torch.load(w, map_location="cpu")
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"backbone {w.name}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if len(missing) > 10:
        sys.exit(f"backbone weights do not fit {MODEL} -- {len(missing)} missing keys. Wrong "
                 f"checkpoint would silently produce garbage features.")
    return model.eval().to(dev)


def load_heads(dev: str) -> list:
    ckpts = sorted(FUSION_DIR.glob("fold*.pt"))
    if not ckpts:
        sys.exit(f"no fold*.pt under {FUSION_DIR}")
    heads = []
    for c in ckpts:
        obj = torch.load(c, map_location="cpu")
        m = FusionHead(d=obj.get("d", 256))
        m.load_state_dict(obj["state_dict"])
        heads.append(m.eval().to(dev))
    print(f"fusion heads: {len(heads)} folds from {FUSION_DIR.name}")
    return heads


@torch.no_grad()
def embed_series(backbone, vol, dev):
    x = to_25d(vol)
    out = []
    for i in range(0, len(x), BATCH_HINT):
        b = imagenet_normalise(x[i:i + BATCH_HINT].to(dev))
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            tok = backbone.forward_features(b)
        cls, patches = tok[:, 0], tok[:, backbone.num_prefix_tokens:].mean(1)
        out.append(torch.cat([cls, patches], -1).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def predict_study(backbone, heads, sdir: Path, meta, dev, n_slices: int) -> np.ndarray:
    feats, types = [], []
    for ser in sorted(p for p in sdir.iterdir() if p.is_dir()):
        files = sorted(ser.glob("*.dcm"))
        if not files:
            continue
        row = meta.loc[ser.name] if ser.name in meta.index else None
        plane_name = getattr(row, "Anatomical_Plane", None)
        vol, _ = load_series(files, plane_name)
        if vol is None:
            continue
        e = embed_series(backbone, vol, dev)
        idx = np.linspace(0, len(e) - 1, min(n_slices, len(e))).round().astype(int)
        feats.append(e[idx])
        types.append(series_type_id(PLANE_ID.get(plane_name, -1),
                                    int(getattr(row, "Fluid_Sensitive", -1))))
    if not feats:
        raise RuntimeError("no decodable series")

    K, S = len(feats), n_slices
    x = torch.zeros(1, K, S, feats[0].shape[-1])
    smask = torch.zeros(1, K, S, dtype=torch.bool)
    for k, f in enumerate(feats):
        x[0, k, :len(f)] = f
        smask[0, k, :len(f)] = True
    sermask = torch.ones(1, K, dtype=torch.bool)
    stype = torch.tensor(types, dtype=torch.long).view(1, K)

    x, smask, sermask, stype = (t.to(dev) for t in (x, smask, sermask, stype))
    p = [torch.sigmoid(h(x, smask, sermask, stype)).float().cpu().numpy()[0] for h in heads]
    return np.mean(p, axis=0)


def main() -> None:
    global COMP
    COMP = find_competition()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if CACHE_MANIFEST.exists():
        assert_matches(json.loads(CACHE_MANIFEST.read_text()))
        print("preprocessing fingerprint matches the cache the heads were trained on")
    else:
        print(f"WARNING: no {CACHE_MANIFEST.name} attached -- cannot verify that this file "
              f"preprocesses exactly as the training cache did (see docstring point 2)")

    test = pd.read_csv(COMP / "test.csv")
    series = pd.read_csv(COMP / "test_series.csv")
    meta = series.set_index("SeriesInstanceUID")
    n_slices = int(json.loads(CACHE_MANIFEST.read_text()).get("slices_per_series_train", 24)
                   if CACHE_MANIFEST.exists() else 24)

    backbone = load_backbone(dev)
    heads = load_heads(dev)

    rows, failures, t0 = [], 0, time.time()
    for n, uid in enumerate(test.StudyInstanceUID, 1):
        try:
            sdir = next((d for d in COMP.rglob(uid) if d.is_dir()), None)
            if sdir is None:
                raise FileNotFoundError("study directory not found")
            p = predict_study(backbone, heads, sdir, meta, dev, n_slices)
            p = np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)
        except Exception:
            # One unreadable DICOM must not zero the submission (PLAN.md 5).
            failures += 1
            if failures <= 3:
                traceback.print_exc()
            p = np.full(len(LABELS), 0.5)
        rows.append([uid, *p.tolist()])
        if n % 100 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(test)}  {el:.0f}s  eta {el / n * (len(test) - n):.0f}s")

    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
    sub.to_csv(OUT, index=False)

    el = time.time() - t0
    print(f"\nwrote {OUT}  {len(sub)} rows, {failures} fallbacks in {el:.0f}s")
    # PLAN.md 6.1: efficiency = AUC/(Benchmark-maxAUC) + RuntimeSeconds/32400, and the exchange
    # rate is 0.001 macro-AUC ~ 93 s. Printed so the runtime term is a measured number rather
    # than the estimate in 6.2, which predates the DINOv2 decision in 7.1.
    print(f"runtime term {el / 32400:.4f}  (= {el / 93:.1f} milli-AUC at the 6.2 exchange rate)")
    assert list(sub.columns) == ["StudyInstanceUID"] + LABELS
    assert sub[LABELS].notna().all().all() and sub[LABELS].between(0, 1).all().all()
    print("header and value checks passed")


if __name__ == "__main__":
    main()
