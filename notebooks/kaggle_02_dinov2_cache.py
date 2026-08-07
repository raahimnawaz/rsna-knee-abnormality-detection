"""Kaggle-side: frozen DINOv2 embeddings for every series. The unlock for local iteration.

Everyone has the same DINOv2 weights, so the backbone cannot differentiate us (PLAN.md 7.1).
What it can do is make our differentiator cheap to iterate: freeze it, run it once, cache the
per-slice features, and the fusion head from 3.3 -- slice transformer, attention pool,
series-type embedding, series attention -- then trains on a laptop in minutes instead of
needing a GPU session per experiment.

Output is ~800 MB for the whole corpus at ViT-B/14 (4,407 studies x ~5 series x 24 slices x
768 dims, fp16). Publish /kaggle/working/features as a Kaggle Dataset and pull it down.

RUNS ACROSS MULTIPLE SESSIONS BY DESIGN. Decode is the bottleneck, not the GPU: ~530k slices
at the ms/slice the audit script measured, on Kaggle's ~4 usable cores. Set SHARD/N_SHARDS and
run N sessions; finished studies are skipped on restart, so an interrupted session loses at
most one study.

  IMG_SIZE  518 gives 37x37 patches and is what "DINOv2 at meniscus resolution" is about --
            a meniscal tear is small and 16x16 patches at 224 lose it. It costs roughly 5x.
            Do a 224 pass first to get the pipeline honest, then re-run at 518.

Preprocessing follows PLAN.md 3.1 exactly, including sorting slices by ImagePositionPatient
projected onto the slice normal rather than InstanceNumber -- InstanceNumber is not reliably
spatial and a mis-ordered volume silently destroys the slice transformer's positional signal.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- config
IMG_SIZE = 518            # 518 (meniscus resolution) or 224 (fast first pass)
SLICES_PER_SERIES = 24
TARGET_MM = 0.35          # in-plane resample target, mm/px
FOV_MM = 160.0            # centre crop/pad field of view
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"   # registers variant; cleaner attention maps
BATCH = 16
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
OUT = Path("/kaggle/working/features")

PLANE_ID = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find_root() -> Path:
    base = Path("/kaggle/input")
    if not base.exists():
        sys.exit("no /kaggle/input -- this runs on Kaggle, not locally")
    for p in base.iterdir():
        if p.is_dir() and "knee" in p.name.lower():
            return p
    sys.exit(f"competition data not found under {base}")


def slice_order(headers: list) -> list[int]:
    """Sort indices along the true through-plane axis (PLAN.md 3.1 step 1).

    The slice normal is the cross product of the row and column direction cosines from
    ImageOrientationPatient; projecting ImagePositionPatient onto it gives a real spatial
    coordinate. Falls back to InstanceNumber only when the geometry is missing.
    """
    keys = []
    for i, ds in enumerate(headers):
        try:
            iop = np.asarray(ds.ImageOrientationPatient, float)
            ipp = np.asarray(ds.ImagePositionPatient, float)
            normal = np.cross(iop[:3], iop[3:])
            keys.append((float(np.dot(ipp, normal)), i))
        except Exception:
            keys.append((float(getattr(ds, "InstanceNumber", i) or i), i))
    return [i for _, i in sorted(keys)]


def load_series(paths: list[Path]) -> np.ndarray | None:
    """-> [S, H, W] float32 in [0,1], resampled to TARGET_MM and cropped to FOV_MM."""
    headers, keep = [], []
    for p in paths:
        try:
            headers.append(pydicom.dcmread(p, stop_before_pixels=True, force=True))
            keep.append(p)
        except Exception:
            pass
    if len(keep) < 3:
        return None

    order = slice_order(headers)
    # Even spread across the volume; a knee series is protocol-centred so the middle is signal.
    pick = np.unique(np.linspace(0, len(order) - 1, SLICES_PER_SERIES).round().astype(int))
    idx = [order[i] for i in pick]

    spacing = None
    try:
        spacing = float(np.asarray(headers[idx[0]].PixelSpacing, float)[0])
    except Exception:
        pass

    vol = []
    for i in idx:
        try:
            arr = pydicom.dcmread(keep[i], force=True).pixel_array.astype(np.float32)
        except Exception:
            continue
        vol.append(arr)
    if len(vol) < 3:
        return None

    shape = max({a.shape for a in vol}, key=lambda s: s[0] * s[1])
    vol = [a for a in vol if a.shape == shape]
    if len(vol) < 3:
        return None
    v = torch.from_numpy(np.stack(vol))[None]                      # [1,S,H,W]

    # MRI has no HU standard -> per-volume robust percentile normalisation (3.1 step 3).
    lo, hi = torch.quantile(v.flatten(), torch.tensor([0.005, 0.995]))
    v = ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1)

    if spacing and spacing > 0:                                    # 3.1 step 4
        scale = spacing / TARGET_MM
        if abs(scale - 1) > 0.02:
            v = F.interpolate(v, scale_factor=scale, mode="bilinear", align_corners=False)

    side = int(round(FOV_MM / TARGET_MM))
    v = center_fit(v, side)
    return v[0].numpy()


def center_fit(v: torch.Tensor, side: int) -> torch.Tensor:
    """Centre crop or zero-pad [1,S,H,W] to side x side. The knee is protocol-centred."""
    _, _, h, w = v.shape
    if h > side:
        t = (h - side) // 2; v = v[:, :, t:t + side, :]
    if w > side:
        l = (w - side) // 2; v = v[:, :, :, l:l + side]
    _, _, h, w = v.shape
    if h < side or w < side:
        v = F.pad(v, (0, max(0, side - w), 0, max(0, side - h)))
    return v


def to_25d(vol: np.ndarray) -> torch.Tensor:
    """[S,H,W] -> [S,3,H,W]: three adjacent slices as RGB, matching DINOv2's 3-channel stem."""
    t = torch.from_numpy(vol)
    prev = torch.cat([t[:1], t[:-1]])
    nxt = torch.cat([t[1:], t[-1:]])
    return torch.stack([prev, t, nxt], dim=1)


@torch.no_grad()
def embed(model, x: torch.Tensor, dev: str) -> np.ndarray:
    """-> [S, 2D] fp16: CLS concatenated with the patch mean."""
    out = []
    for i in range(0, len(x), BATCH):
        b = x[i:i + BATCH].to(dev)
        b = F.interpolate(b, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        b = (b - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
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

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(dev)
    print(f"{MODEL} on {dev}, img={IMG_SIZE}, prefix_tokens={model.num_prefix_tokens}")

    meta = series.set_index("SeriesInstanceUID")
    done = skipped = 0
    t0 = time.time()

    for n, study in enumerate(mine, 1):
        dst = OUT / f"{study}.npz"
        if dst.exists():                       # resume: an interrupted session loses one study
            skipped += 1
            continue
        sdir = next((d for d in root.rglob(study) if d.is_dir()), None)
        if sdir is None:
            continue

        feats, sid, plane, fs = [], [], [], []
        for k, ser in enumerate(sorted(p for p in sdir.iterdir() if p.is_dir())):
            files = sorted(ser.glob("*.dcm"))
            if not files:
                continue
            vol = load_series(files)
            if vol is None:
                continue
            e = embed(model, to_25d(vol), dev)
            row = meta.loc[ser.name] if ser.name in meta.index else None
            feats.append(e)
            sid += [k] * len(e)
            plane += [PLANE_ID.get(getattr(row, "Anatomical_Plane", None), -1)] * len(e)
            # Fluid_Sensitive and Fat_Suppression are perfectly redundant (FINDINGS.md 3.1),
            # so one flag is the whole story -> 6 series types, not 12.
            fs += [int(getattr(row, "Fluid_Sensitive", -1))] * len(e)

        if not feats:
            continue
        np.savez_compressed(dst, feats=np.concatenate(feats),
                            series_idx=np.array(sid, np.int16),
                            plane=np.array(plane, np.int8),
                            fluid_sensitive=np.array(fs, np.int8))
        done += 1
        if done % 20 == 0:
            rate = done / (time.time() - t0)
            left = (len(mine) - skipped - done) / max(rate, 1e-9) / 3600
            print(f"  {done:>5} done  {rate*3600:>6.0f} studies/h  ~{left:.1f} h left")

    print(f"\nshard {SHARD}: {done:,} written, {skipped:,} already present -> {OUT}")
    (OUT / f"_shard{SHARD}.json").write_text(json.dumps(
        {"model": MODEL, "img_size": IMG_SIZE, "slices_per_series": SLICES_PER_SERIES,
         "target_mm": TARGET_MM, "fov_mm": FOV_MM, "written": done}, indent=2))
    print("Publish /kaggle/working/features as a Kaggle Dataset, then train the fusion head "
          "locally.")


if __name__ == "__main__":
    main()
