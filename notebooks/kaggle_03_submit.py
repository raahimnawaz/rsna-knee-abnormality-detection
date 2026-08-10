"""Submission notebook: test DICOMs -> submission.csv. Runs on Kaggle, no internet, <=9h.

The inference path the whole project exists to feed. It reproduces the feature cache live --
there is no cache for the test set -- then runs the fusion head trained on the M5.

    /kaggle/input/<competition>          the test DICOMs
    /kaggle/input/rsna-knee-code         this repo, so pipeline/preprocess.py is importable
    /kaggle/input/dinov2-weights         DINOv2 .safetensors/.bin  <- NOT downloaded, see below
    /kaggle/input/rsna-knee-fusion       fold*.pt AND manifest.json from fusion/train.py

FOUR THINGS THAT KILL A SUBMISSION, ALL HANDLED HERE:

 1. NO INTERNET. `timm.create_model(..., pretrained=True)` reaches out to HuggingFace and dies.
    The weights must be attached as a Dataset and loaded from disk. This is the single most
    common submission-day failure and it does not reproduce in an interactive session, where
    the weights are already cached.
 2. PREPROCESSING DRIFT. If this file preprocessed test data even slightly differently from
    kaggle_02, the model would score badly with no error anywhere. Both import the same
    pipeline/preprocess.py, and assert_matches() compares the cache manifest's fingerprint
    against this file's before a single study is read. A MISSING manifest is fatal, not a
    warning: an unverifiable run is the exact failure this point exists to prevent.
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

# Locate pipeline/preprocess.py wherever the code Dataset landed. Globbing beats hardcoding:
# the mount name depends on the Dataset slug, and a Dataset version still being created when the
# kernel starts can leave the path briefly absent -- which surfaced as a bare ModuleNotFoundError
# and cost a GPU session. Fail with a listing instead.
def _repo_preprocess() -> str | None:
    """This repo's own copy, for a checkout rather than a Kaggle mount.

    Lazy and guarded on purpose. __main__ has no __file__ in a notebook cell or under exec(),
    and evaluating this eagerly while building the pattern list turned "no code Dataset
    attached" into a bare NameError raised before the helpful SystemExit below could run.
    """
    f = globals().get("__file__")
    return str(Path(f).resolve().parents[1] / "pipeline" / "preprocess.py") if f else None


def _bootstrap_preprocess() -> None:
    import glob
    # BOUNDED DEPTH FIRST, recursion only as a fallback. Kaggle nests sources under
    # competitions/ and datasets/ when a kernel has more than one, so the depth is not fixed
    # (measured 2026-08-07 -- /kaggle/input held exactly those two and a one-level glob found
    # nothing) -- but it is shallow, and every depth below 3 is image data. A `**` pattern
    # descends into ~29k series directories on a mount measured at ~19 ms per open, and here
    # that walk is charged straight to the runtime term of the efficiency score.
    pats = [f"/kaggle/input/{'*/' * d}pipeline/preprocess.py" for d in range(4)]
    pats += ["/kaggle/usr/lib/*/preprocess.py", "/kaggle/usr/lib/**/preprocess.py",
             "/kaggle/input/**/pipeline/preprocess.py", _repo_preprocess()]
    for pat in pats:
        hits = sorted(glob.glob(pat, recursive=True)) if pat else []
        if hits:
            root = Path(hits[0]).parents[1]
            sys.path.insert(0, str(root / "pipeline"))
            sys.path.insert(0, str(root / "fusion"))
            return
    listing = sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*"))
    raise SystemExit(
        "cannot find pipeline/preprocess.py. Attach the rsna-knee-code Dataset to this "
        f"notebook. /kaggle/input currently holds: {listing}")


_bootstrap_preprocess()
from preprocess import (BATCH_HINT, MODEL, PLANE_ID, SLICES_PER_SERIES_TRAIN,   # noqa: E402
                        assert_matches, build_study_index, embed,
                        find_competition_root, load_series, pick_device, to_25d)
from dataset import series_type_id                                     # noqa: E402
from model import FusionHead                                           # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

COMP = None          # resolved below
WEIGHTS_DIR = Path("/kaggle/input/dinov2-weights")
FUSION_DIR = Path("/kaggle/input/rsna-knee-fusion")
OUT = Path("/kaggle/working/submission.csv")


def load_cache_manifest() -> dict | None:
    """The manifest of the feature cache these heads were trained on, or None.

    fusion/train.py copies it next to the fold checkpoints precisely so this lookup succeeds --
    an earlier version of this file expected a bare manifest.json that NOTHING in the repo
    wrote, so exists() was always False, assert_matches() never ran, and the fingerprint check
    the docstring calls a structural defence was in practice a print statement. _shard*.json is
    accepted too: that is what kaggle_02 writes, so a features Dataset attached directly also
    works.
    """
    for cand in [FUSION_DIR / "manifest.json", *sorted(FUSION_DIR.glob("_shard*.json"))]:
        if cand.exists():
            try:
                m = json.loads(cand.read_text())
            except Exception as e:
                sys.exit(f"{cand} is not readable JSON ({e}). It records which preprocessing "
                         f"built the features the heads were trained on and cannot be skipped.")
            print(f"cache manifest: {cand.name}")
            return m
    return None


def load_backbone(dev: str):
    """DINOv2 from an attached Dataset. pretrained=False so timm never touches the network."""
    import timm
    # dynamic_img_size must match kaggle_02 exactly -- see the note there. It is also what makes
    # the load below work at all: the checkpoint is 518-native, strict=False forgives missing and
    # unexpected keys but NOT a shape mismatch, so a model built at IMG_SIZE=224 would raise on
    # pos_embed instead of falling through to the missing-key guard.
    model = timm.create_model(MODEL, pretrained=False, num_classes=0, dynamic_img_size=True)
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


def load_heads(dev: str) -> tuple[list, int | None]:
    """-> (heads, n_slices_train). The second value is what the heads were actually FED.

    It cannot come from the cache manifest, which is what this file used to do. The manifest
    describes the CACHE, and the slice count is the one parameter that does not change a single
    cached feature value -- the cache always stores 32 and the head samples a subset -- so it is
    deliberately outside PREPROCESS_VERSION and `assert_matches()` is blind to it (IMPROVEMENTS
    6.2). Read it off the checkpoints, which is the only artefact that knows.
    """
    ckpts = sorted(FUSION_DIR.glob("fold*.pt"))
    if not ckpts:
        sys.exit(f"no fold*.pt under {FUSION_DIR}")
    heads, counts = [], set()
    for c in ckpts:
        obj = torch.load(c, map_location="cpu")
        m = FusionHead(d=obj.get("d", 256))
        m.load_state_dict(obj["state_dict"])
        heads.append(m.eval().to(dev))
        counts.add(obj.get("n_slices_train"))
    print(f"fusion heads: {len(heads)} folds from {FUSION_DIR.name}")

    if len(counts) > 1:
        sys.exit(f"fold checkpoints disagree on n_slices_train: {sorted(map(str, counts))}. They "
                 f"were not produced by one training run and must not be ensembled.")
    n = counts.pop()
    if n is None:
        print(f"WARNING: checkpoints carry no n_slices_train -- written by a fusion/train.py "
              f"older than 2026-08-09. Falling back to the manifest, then to "
              f"SLICES_PER_SERIES_TRAIN={SLICES_PER_SERIES_TRAIN}. Re-train to remove the guess.")
    return heads, n


def embed_series(backbone, vol, dev):
    """One series -> [S, EMBED_DIM] float32, via the SHARED embed().

    This used to be a fourth hand-written copy of the embedding loop. It was the LAST one, and
    the most dangerous: preprocess.embed's own docstring claims three copies were consolidated
    so train and test cannot drift, and this file -- the test side -- was not migrated. Any
    future change to pooling or autocast would have applied to training features and not to
    these. That is K12's shape exactly, under a comment asserting it could not happen.

    embed() already returns fp16, which is the round-trip that matters: kaggle_02 stores the
    cache as fp16 and fusion/dataset.py upcasts per batch, so the head has only ever seen
    fp16-quantised vectors. Handing it full fp32 here would be a train/serve mismatch the
    fingerprint cannot catch, because it hashes constants and not dtypes.
    """
    return torch.from_numpy(embed(backbone, to_25d(vol), dev, BATCH_HINT)).float()


@torch.no_grad()
def predict_study(backbone, heads, sdir: Path, meta, dev, n_slices: int) -> np.ndarray:
    feats, types = [], []
    for ser in sorted(p for p in sdir.iterdir() if p.is_dir()):
        files = sorted(ser.glob("*.dcm"))
        if not files:
            continue
        row = meta.loc[ser.name] if ser.name in meta.index else None
        plane_name = getattr(row, "Anatomical_Plane", None)
        vol, _, _ = load_series(files, plane_name)
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
    COMP = find_competition_root()
    # At submission time a CPU fallback is still better than no submission at all -- but only
    # because of the placeholder written below. Without it the fallback produced NOTHING: at
    # 518px on Kaggle CPU the loop cannot finish inside 9 h, and the single to_csv sat after it.
    dev = pick_device()
    if dev == "unusable":
        print("continuing on CPU. This will almost certainly hit the 9 h cap before the last "
              "study; the partial submission below is flushed as it goes, but RE-RUN for a T4.")
        dev = "cpu"

    cache_manifest = load_cache_manifest()
    if cache_manifest is None:
        sys.exit(
            f"no cache manifest under {FUSION_DIR}. It records the preprocessing that built the "
            f"features these heads were trained on, and without it a silent train/test "
            f"preprocessing mismatch would score badly with no error anywhere (docstring point "
            f"2). fusion/train.py writes manifest.json beside fold*.pt -- re-run it and "
            f"re-upload the Dataset, or attach the feature cache, which carries _shard*.json.")
    assert_matches(cache_manifest)
    print("preprocessing fingerprint matches the cache the heads were trained on")
    n_slices = int(cache_manifest.get("slices_per_series_train", SLICES_PER_SERIES_TRAIN))

    test = pd.read_csv(COMP / "test.csv")
    series = pd.read_csv(COMP / "test_series.csv")
    meta = series.set_index("SeriesInstanceUID")

    # Every study starts at 0.5 and the file is written BEFORE the expensive part, then
    # rewritten as results land. A kernel killed at the 9 h cap, or dying on the backbone load,
    # then still leaves a valid submission holding whatever finished.
    preds = {uid: np.full(len(LABELS), 0.5) for uid in test.StudyInstanceUID}

    def write_submission() -> pd.DataFrame:
        sub = pd.DataFrame([[uid, *preds[uid].tolist()] for uid in test.StudyInstanceUID],
                           columns=["StudyInstanceUID"] + LABELS)
        sub.to_csv(OUT, index=False)
        return sub

    write_submission()
    print(f"placeholder {OUT} written for {len(preds)} studies")

    backbone = load_backbone(dev)
    heads, head_n_slices = load_heads(dev)
    if head_n_slices is not None and head_n_slices != n_slices:
        # The manifest and the checkpoints disagree about how many slices the head saw. Feeding
        # the wrong count is a silent scorer: slice_pos is a learned per-index embedding, so the
        # positions simply mean something else, and nothing raises.
        sys.exit(f"slice-count mismatch: the heads were trained on {head_n_slices} slices per "
                 f"series, the manifest says {n_slices}. The head's learned slice positions "
                 f"would be read at the wrong indices and this would score badly with no error. "
                 f"Re-upload the fusion Dataset and its manifest from one training run.")
    if head_n_slices is not None:
        n_slices = head_n_slices
    print(f"slices per series at inference: {n_slices} "
          f"({'from checkpoints' if head_n_slices is not None else 'from manifest'})")

    index = build_study_index(COMP)     # one pass; per-study rglob is O(n^2) over 570 GB
    failures, t0 = 0, time.time()
    for n, uid in enumerate(test.StudyInstanceUID, 1):
        try:
            sdir = index.get(uid)
            if sdir is None:
                raise FileNotFoundError("study directory not found")
            p = predict_study(backbone, heads, sdir, meta, dev, n_slices)
            preds[uid] = np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)
        except Exception:
            # One unreadable DICOM must not zero the submission (PLAN.md 5).
            failures += 1
            if failures <= 3:
                traceback.print_exc()
        if n % 100 == 0:
            el = time.time() - t0
            write_submission()
            print(f"  {n}/{len(test)}  {el:.0f}s  eta {el / n * (len(test) - n):.0f}s")

    sub = write_submission()

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
