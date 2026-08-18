"""Pixels for the public RadImageNet arm: 3 fat-suppressed planes x 8 slices at 224 px, full frame.

§C-3 / §4b. `fusion/rad_model.py` is the model half and is structurally verified against
`rad_heads_manifest.json` (all six SHA-256s, strict load, 3,174,924 params). This is the pixel half,
and per `ARCHITECTURES.md` it is the half that fails SILENTLY when a convention is crossed.

WHY THIS REUSES `pilkwang_pixels.read_slot_nifti` RATHER THAN TRANSCRIBING A NEW READER.
That looks like a shortcut and is the opposite. Their kernel's RadImageNet path calls the notebook's
**shared** `build_cache` -- the same function its DINOv2 arm uses -- with the rad constants
rebound (`N_SLOT` 3, `CACHE_SLICES` 8, `IMG` 224, no crop, band 0.12-0.88). That shared reader is
`pilkwang`'s `read_slot` lineage, which `pilkwang_pixels` already transcribes and which the §3i gate
validated at 20/20 fingerprints and a clean fold partition. **Using a different reader here would be
the divergence, not using this one.**

THE CONTRACT, from `rad_heads_manifest.json`, every field of which differs from pilkwang's:

    image_size          224          (pilkwang 336)
    crop                full-frame   (pilkwang 130 mm centre crop)   <- `CROP_FULL` below
    slices_per_plane    8            (pilkwang 12)
    slice_band          0.12-0.88    (pilkwang 0.20-0.80)
    planes              Sagittal, Coronal, Axial -- FAT-SUPPRESSED ONLY   (pilkwang 6 slots)
    window              per-series percentile 1/99      (same as pilkwang's)
    normalization       grayscale repeated to RGB, x/127.5 - 1   (NOT ImageNet mean/std)

⚠️ THE NORMALISATION IS APPLIED IN `rad_features`, NOT HERE. This module emits **uint8** exactly as
`pilkwang_pixels` does; `_rad_encode`'s `.float().div_(127.5).sub_(1.0)` then `.expand(-1, 3, ...)`
is reproduced at the encoder. Keeping the cache uint8 is what makes it the same object the other
arms cache, and it is a quarter of the memory.

⚠️ SLOT ORDER IS THE `plane` EMBEDDING INDEX AND MUST NOT BE SORTED. The manifest lists planes as
[Sagittal, Coronal, Axial]; `pilkwang_pixels.SLOT_NAMES[:3]` is
[SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS] -- **the same order**, which is why the first three
pilkwang slots can be reused directly. `rad_model.FoundationQueryHead.plane` is indexed by this
position, and §9h is what a permuted conditioning table costs: it runs perfectly and scores wrongly.

⚠️ FULL-FRAME IS EXPRESSED AS A CROP LARGER THAN THE MATRIX, NOT AS A SPECIAL CASE.
`read_slot_nifti` applies its crop only `if 16 < want < min(h, w)`, so any `crop_mm` beyond the
field of view leaves the volume untouched and records `cropped=False`. `CROP_FULL = 10_000.0`
therefore reproduces "full-frame" through the *same* code path, with no branch to get wrong.

    python fusion/rad_pixels.py --limit 8 --report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
D = PROJ / "data"

from fusion.pilkwang_pixels import (NII, SLOT_NAMES, SLOT_PLANE,  # noqa: E402
                                    direction_map, read_slot_nifti)
from pipeline.preprocess import study_laterality  # noqa: E402

# The three fat-suppressed planes, in the manifest's order. NOT sorted, NOT a set.
RAD_SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS"]
assert RAD_SLOTS == SLOT_NAMES[:3], "pilkwang slot order changed -- the plane embedding would shift"

IMG = 224
N_SLICE = 8
BAND = (0.12, 0.88)
CROP_FULL = 10_000.0        # any value past the FOV => read_slot_nifti leaves the frame alone


def build_rad_cache(studies, slots: pd.DataFrame, *, out_size: int = IMG,
                    n_slice: int = N_SLICE, band: tuple[float, float] = BAND,
                    use_direction: bool = True, verbose: bool = True):
    """(study, plane) -> uint8 [n_study, 3, n_slice, out, out] plus the [n_study, 3] presence mask.

    Mirrors `pilkwang_pixels.build_cache`, restricted to the three fat-suppressed planes and
    carrying this arm's constants. An absent plane is a zero row and a False in the mask, which
    `FoundationQueryHead` turns into `key_padding` -- the same contract `SlotHead` has.
    """
    lat_map = study_laterality(D / "study_meta.csv")
    rev = direction_map() if use_direction else {}
    slots = slots[slots["StudyInstanceUID"].isin(set(studies)) & slots["slot"].isin(RAD_SLOTS)]
    by_study = {s: g for s, g in slots.groupby("StudyInstanceUID")}

    n = len(studies)
    cache = np.zeros((n, len(RAD_SLOTS), n_slice, out_size, out_size), np.uint8)
    mask = np.zeros((n, len(RAD_SLOTS)), np.float32)
    stats = {"filled": 0, "no_file": 0, "unreadable": 0, "reversed": 0, "no_lat": 0}

    for i, st in enumerate(studies):
        g = by_study.get(st)
        if g is None:
            continue
        lat = (lat_map.get(st) or (None,))[0]
        if lat is None:
            stats["no_lat"] += 1
        for _, r in g.iterrows():
            j = RAD_SLOTS.index(r["slot"])
            f = NII / f"{st}_{r['SeriesInstanceUID']}.nii"
            if not f.exists():
                f = f.with_suffix(".nii.gz")
            if not f.exists():
                stats["no_file"] += 1
                continue
            rv = bool(rev.get(r["SeriesInstanceUID"], False))
            img, info = read_slot_nifti(f, SLOT_PLANE[r["slot"]], lat, crop_mm=CROP_FULL,
                                        out_size=out_size, n_slice=n_slice, band=band,
                                        reverse=rv, px_hdr=r.get("px"))
            if img is None:
                stats["unreadable"] += 1
                continue
            cache[i, j] = img.numpy()
            mask[i, j] = 1.0
            stats["filled"] += 1
            stats["reversed"] += int(info["reversed"])
            if info["cropped"]:
                raise RuntimeError("read_slot_nifti CROPPED a full-frame read -- CROP_FULL too "
                                   "small for this matrix; the arm expects the whole field")
        if verbose and (i + 1) % 100 == 0:
            print(f"    [{i + 1:>5}/{n}] {stats['filled']:,} planes", flush=True)
    return cache, mask, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    slots = pd.read_csv(D / "slots_pilkwang.csv")
    have = {p.name.split("_")[1].replace(".nii", "") for p in NII.glob("*.nii*")}
    slots = slots[slots["SeriesInstanceUID"].isin(have)]
    studies = sorted(slots["StudyInstanceUID"].unique())[:a.limit]

    print(f"{len(studies)} studies · {IMG}px · {N_SLICE} slices · band {BAND} · FULL FRAME")
    cache, mask, stats = build_rad_cache(studies, slots)
    print(f"\ncache {cache.shape} = {cache.nbytes / 1e9:.3f} GB")
    print(f"mask: {int(mask.sum())}/{mask.size} planes ({mask.sum() / mask.size:.1%}), "
          f"per study {mask.sum(1).mean():.2f}")
    for k, v in stats.items():
        print(f"  {k:<12} {v}")
    if a.report:
        nz = cache[mask.astype(bool)]
        print(f"\nuint8 over filled planes: mean {nz.mean():.1f} std {nz.std():.1f} "
              f"min {nz.min()} max {nz.max()}")
        x = nz.astype(np.float32) / 127.5 - 1.0
        print(f"after x/127.5-1 (what the encoder sees): mean {x.mean():+.3f} "
              f"range [{x.min():+.2f}, {x.max():+.2f}] -- must sit in [-1, +1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
