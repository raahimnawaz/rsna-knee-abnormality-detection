"""Kaggle-side: frozen DINOv2 embeddings for every series. The unlock for local iteration.

Everyone has the same DINOv2 weights, so the backbone cannot differentiate us (PLAN.md 7.1).
What it can do is make our differentiator cheap to iterate: freeze it, run it once, cache the
per-slice features, and the fusion head from 3.3 -- slice transformer, attention pool,
series-type embedding, series attention -- then trains on the Mac Studio in minutes instead of
needing a GPU session per experiment.

Output is ~2.4 GB for the whole corpus: 24,371 series x 32 slices x 1536 dims fp16. (Earlier
drafts of this file said ~800 MB, computed at 768 dims -- but embed() concatenates CLS with the
patch mean, so it is 1536. Corrected 2026-08-07.) Publish /kaggle/working/features as a Kaggle
Dataset and pull it down.

ALL preprocessing lives in pipeline/preprocess.py, imported here rather than defined here. The
submission notebook imports the same file, and if the two ever disagree the model is fed a
distribution it never trained on with nothing raising an error. The manifest written beside the
shards carries PREPROCESS_VERSION so that mismatch is detectable instead of silent.

RUNS ACROSS MULTIPLE SESSIONS BY DESIGN. Decode is the bottleneck, not the GPU. Set
SHARD/N_SHARDS and run N sessions; finished studies are skipped on restart, so an interrupted
session loses at most one study.

  IMG_SIZE  518 gives 37x37 patches and is what "DINOv2 at meniscus resolution" is about -- a
            meniscal tear is small and 16x16 patches at 224 lose it. It costs roughly 5x. Do a
            224 pass first to get the pipeline honest, then re-run at 518.
  SLICES    32 cached, 24 used at train time. The gap is slice jitter, the only pixel-space
            augmentation that survives a frozen backbone (see fusion/dataset.py).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Attach the repo as a Kaggle Dataset (or add pipeline/preprocess.py as a Utility Script).
# Copy-pasting the file instead works right up until someone edits one copy.
for _p in ("/kaggle/input/rsna-knee-code/pipeline", "/kaggle/usr/lib/preprocess",
           str(Path(__file__).resolve().parents[1] / "pipeline")):
    if Path(_p).exists():
        sys.path.insert(0, _p)
        break
from preprocess import (BATCH_HINT, MODEL, PLANE_ID, PREPROCESS_VERSION,  # noqa: E402
                        imagenet_normalise, load_series, manifest, to_25d)

BATCH = BATCH_HINT
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
OUT = Path("/kaggle/working/features")


def find_root() -> Path:
    base = Path("/kaggle/input")
    if not base.exists():
        sys.exit("no /kaggle/input -- this runs on Kaggle, not locally")
    for p in base.iterdir():
        if p.is_dir() and "knee" in p.name.lower() and "code" not in p.name.lower():
            return p
    sys.exit(f"competition data not found under {base}")


@torch.no_grad()
def embed(model, x: torch.Tensor, dev: str) -> np.ndarray:
    """[S,3,H,W] -> [S, 2D] fp16: CLS concatenated with the patch mean.

    Both halves earn their place: CLS carries the global impression, the patch mean retains
    localised signal that a single token averages away -- and the findings here are small.
    """
    out = []
    for i in range(0, len(x), BATCH):
        b = imagenet_normalise(x[i:i + BATCH].to(dev))
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            tok = model.forward_features(b)
        cls, patches = tok[:, 0], tok[:, model.num_prefix_tokens:].mean(1)
        out.append(torch.cat([cls, patches], -1).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main() -> None:
    import timm
    root = find_root()
    OUT.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(root / "train_series.csv")
    studies = sorted(series.StudyInstanceUID.unique())
    mine = [s for i, s in enumerate(studies) if i % N_SHARDS == SHARD]
    print(f"shard {SHARD}/{N_SHARDS}: {len(mine):,} of {len(studies):,} studies")
    print(f"preprocess version {PREPROCESS_VERSION}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(dev)
    print(f"{MODEL} on {dev}, prefix_tokens={model.num_prefix_tokens}")

    meta = series.set_index("SeriesInstanceUID")
    done = skipped = 0
    lat_seen = {"L": 0, "R": 0, "unknown": 0}
    t0 = time.time()

    for study in mine:
        dst = OUT / f"{study}.npz"
        if dst.exists():                       # resume: an interrupted session loses one study
            skipped += 1
            continue
        sdir = next((d for d in root.rglob(study) if d.is_dir()), None)
        if sdir is None:
            continue

        feats, sid, plane, fs, lat = [], [], [], [], []
        for k, ser in enumerate(sorted(p for p in sdir.iterdir() if p.is_dir())):
            files = sorted(ser.glob("*.dcm"))
            if not files:
                continue
            row = meta.loc[ser.name] if ser.name in meta.index else None
            plane_name = getattr(row, "Anatomical_Plane", None)

            vol, side = load_series(files, plane_name)
            lat_seen[side if side in ("L", "R") else "unknown"] += 1
            if vol is None:
                continue

            e = embed(model, to_25d(vol), dev)
            feats.append(e)
            sid += [k] * len(e)
            plane += [PLANE_ID.get(plane_name, -1)] * len(e)
            # Fluid_Sensitive and Fat_Suppression are perfectly redundant (FINDINGS.md 3.1),
            # so one flag is the whole story -> 6 series types, not 12.
            fs += [int(getattr(row, "Fluid_Sensitive", -1))] * len(e)
            # Recorded per series so laterality coverage can be audited after the fact, and so
            # a study whose handedness was unknown can be found again without a full rebuild.
            lat += [{"L": 0, "R": 1}.get(side, -1)] * len(e)

        if not feats:
            continue
        np.savez_compressed(dst, feats=np.concatenate(feats),
                            series_idx=np.array(sid, np.int16),
                            plane=np.array(plane, np.int8),
                            fluid_sensitive=np.array(fs, np.int8),
                            laterality=np.array(lat, np.int8))
        done += 1
        if done % 20 == 0:
            rate = done / (time.time() - t0)
            left = (len(mine) - skipped - done) / max(rate, 1e-9) / 3600
            print(f"  {done:>5} done  {rate * 3600:>6.0f} studies/h  ~{left:.1f} h left")

    total_lat = sum(lat_seen.values()) or 1
    print(f"\nshard {SHARD}: {done:,} written, {skipped:,} already present -> {OUT}")
    print(f"laterality: L {lat_seen['L']}, R {lat_seen['R']}, "
          f"unknown {lat_seen['unknown']} ({100 * lat_seen['unknown'] / total_lat:.1f}%)")
    if lat_seen["unknown"] > total_lat * 0.05:
        print("  WARNING: >5% of series have no laterality tag. Medial/Lateral labels are only "
              "as good as this -- see PLAN.md 3.2 and the kaggle_01 audit before trusting them.")

    (OUT / f"_shard{SHARD}.json").write_text(json.dumps(
        manifest(written=done, shard=SHARD, n_shards=N_SHARDS, laterality=lat_seen), indent=2))
    print("Publish /kaggle/working/features as a Kaggle Dataset, then train the fusion head "
          "on the Studio.")


if __name__ == "__main__":
    main()
