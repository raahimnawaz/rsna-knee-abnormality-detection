"""The fork's `read_slot`, rebuilt against our NIfTI corpus, with the crop as a parameter.

WHAT THIS IS FOR. `IMPROVEMENTS.md` §3g established that the members' fingerprint checks the
WEIGHTS and not the pixels, so `CROP_MM` and `SLICE_BAND` can be moved without tripping any guard.
That opens F2-cheap -- tighter or off-centre crops as ADDITIONAL TTA windows on the twenty frozen
members, with no training run. This module is the pixel half of it. `fusion/pilkwang_model.py` is
the model half and is already verified exact: 20/20 fingerprints at 7e-06.

They read DICOM off the Kaggle mount; we hold the same corpus as per-series NIfTI. Everything else
is transcribed: the band sample over `[0.2, 0.8]`, the centre crop of `CROP_MM / px` pixels, the
1st-99th percentile normalisation, the bilinear resize, uint8, and `normalise_laterality`.

    python fusion/pilkwang_pixels.py --limit 8 --report     # what a rebuild looks like

THE ONE PLACE THE TRANSCRIPTION CANNOT BE LITERAL, AND IT IS THE RISK IN THIS FILE.
`order_slices` sorts DICOM files by the signed through-plane projection `k = p . (r_x x r_y)`.
Our NIfTI conversion **dropped the patient coordinate system** (`PLAN.md` 9.1), so that projection
is not recoverable from the file and the stored slice order is a convention rather than a
measurement. What we hold instead is K16: `data/slice_direction_resolved.csv`, **8,048 sagittal
series measured and cross-validated 21/21**. Axial and coronal have no per-series bit; the
`validate_nifti` stratified sample read Axial 12/12 forward and Coronal 14/18, so they ride on that
and it is a known weakness rather than a hidden one.

**Why order matters here at all**, since the band is symmetric and a reversal maps the sampled SET
to itself: it does not map the sampled SEQUENCE to itself. The twelve slices become ten
three-channel windows, so a reversal reverses every window's channels and reverses the window
order. And on sagittal it matters twice, because `normalise_laterality` normalises a right knee by
reversing the slice axis rather than mirroring in plane -- so a wrong direction bit and a wrong
laterality cancel into a silently plausible stack.

**Do not fix a disagreement here by tuning until the gate passes.** The gate compares against the
fork's own OOF, and a convention chosen to make that number good is a convention fitted to the
thing it is supposed to test. Directions come from K16 or they are declared unknown.

CROP GEOMETRY, and this is what F2 will move. Their crop is a pure centre crop in pixel space --
`cy, cx = h // 2, w // 2`, `half = want // 2` -- with no anatomy in it whatsoever. `centre_mm`
here is our addition: a (row, col) offset in millimetres, defaulting to (0, 0), which reproduces
them exactly. It is the two-line change that lets §2l's canonical axes aim a box at the posterior
horn of the lateral meniscus while the members stay frozen. Offsets are applied in the
CANONICALISED frame, i.e. after the laterality decision, so a left and a right knee take the same
sign -- see `normalise_laterality` below for why sagittal is the awkward one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
D = PROJ / "data"
NII = D / "nifti" / "nifti_train"

from pipeline.preprocess import nifti_geometry, study_laterality  # noqa: E402

# --- the fork's constants, from the member config -------------------------------------------- #
SLOT_NAMES = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
              "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
SLOT_PLANE = {"SAG_FLUID_FS": "Sagittal", "COR_FLUID_FS": "Coronal", "AX_FLUID_FS": "Axial",
              "SAG_FLUID_NOFS": "Sagittal", "COR_T1": "Coronal", "SAG_T1": "Sagittal"}
CROP_MM = 130.0
SLICE_BAND = (0.20, 0.80)
IMG = 336
N_SLICE = 12          # cfg["slices"]; with group 3 this is ten overlapping TTA windows
GROUP = 3


def band_indices(n: int, n_slice: int = N_SLICE) -> np.ndarray:
    """Their sampler, verbatim: linspace across the central band, deduplicated, then padded."""
    lo, hi = int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    return idx[:n_slice]


def normalise_laterality(img: torch.Tensor, plane: str, lat: str | None) -> torch.Tensor:
    """Map every knee onto a left-knee convention. Theirs, unchanged.

    Coronal and axial mirror under a horizontal flip. Sagittal stacks are not mirror images of
    each other -- the slice order runs medial-to-lateral in opposite directions -- so the channel
    order is reversed instead. `lat is None` leaves the volume alone and the caller records the
    study as uncanonicalised, which is what `canonicalise` does on the NIfTI path and is honest
    rather than safe.
    """
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


def read_slot_nifti(path: Path, plane: str, lat: str | None, *, crop_mm: float = CROP_MM,
                    centre_mm: tuple[float, float] = (0.0, 0.0), out_size: int = IMG,
                    n_slice: int = N_SLICE, reverse: bool = False,
                    px_hdr: float | None = None) -> tuple[torch.Tensor | None, dict]:
    """One slot image: uint8 [n_slice, out, out], plus what was decided while making it."""
    info: dict = {"reversed": bool(reverse), "px": None, "px_hdr": px_hdr,
                  "cropped": False, "n_src": 0}
    g = nifti_geometry(path)
    if g is None:
        return None, info
    import nibabel as nib
    try:
        arr = np.asanyarray(nib.load(str(path)).dataobj)
    except Exception:
        return None, info

    vol = np.transpose(arr, (g["slice_axis"], g["row_axis"], g["col_axis"])).astype(np.float32)
    if reverse:
        vol = vol[::-1]
    n = len(vol)
    info["n_src"] = n
    if n < 3:
        return None, info

    px = float(g["in_plane_mm"])
    info["px"] = px
    vol = np.ascontiguousarray(vol[band_indices(n, n_slice)])

    # constant physical extent, then resize -- their comment: PixelSpacing varies 3.4x
    if px and np.isfinite(px) and px > 0:
        want = int(round(crop_mm / px))
        h, w = vol.shape[1:]
        if 16 < want < min(h, w):
            dy, dx = (int(round(centre_mm[0] / px)), int(round(centre_mm[1] / px)))
            cy, cx = h // 2 + dy, w // 2 + dx
            half = want // 2
            # Clamp so an offset box stays inside the matrix. Their centre crop can never
            # leave it, ours can, and a numpy slice with a negative start silently wraps.
            cy = int(np.clip(cy, half, h - half)) if h >= 2 * half else h // 2
            cx = int(np.clip(cx, half, w - half)) if w >= 2 * half else w // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            info["cropped"] = True

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)

    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    t = (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)
    return normalise_laterality(t, plane, lat), info


def direction_map() -> dict[str, bool]:
    """SeriesInstanceUID -> True if the stored NIfTI stack runs BACKWARDS (K16).

    Sagittal only, 8,048 series, `rule == 'measured'`. Anything absent is treated as forward and
    counted, never guessed: `PLAN.md` 9b's whole point is that a header rule reads 56.9-60.8%
    against ~50% chance, so an inferred bit is worse than a declared unknown.
    """
    p = D / "slice_direction_resolved.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d[d["rule"] == "measured"] if "rule" in d.columns else d
    return dict(zip(d["SeriesInstanceUID"], d["direction"].eq("reversed")))


def build_cache(studies, slots: pd.DataFrame, *, crop_mm: float = CROP_MM,
                centre_mm: tuple[float, float] = (0.0, 0.0), out_size: int = IMG,
                n_slice: int = N_SLICE, use_direction: bool = True,
                verbose: bool = True):
    """Decode (study, slot) -> uint8 [n_study, 6, n_slice, out, out] plus the presence mask.

    An absent slot is a False in the mask and zeros in the cache, which is exactly what the
    fork's `SlotHead` masks out of its softmax. A slot with no NIfTI on disk is absent in the
    same sense as one never acquired -- the model cannot tell, and neither can we.
    """
    lat_map = study_laterality(D / "study_meta.csv")
    # `use_direction=False` is the K16 ABLATION, not a convenience. K16 was measured on the 01c
    # thumbnails and cross-validated 21/21 against a genuinely independent instrument, but it has
    # never been tested against anything that cares -- the fold-0 gate sat at depth 0.5 where a
    # reversal maps the middle slice to itself. Running the gate both ways gives K16 its first
    # test with a predicted sign: if the measured bit is right AND slice order is what the
    # residual is made of, applying it must LOWER the residual. Either outcome is information,
    # which is what separates this from tuning the convention until the number looks good.
    rev = direction_map() if use_direction else {}
    slots = slots[slots["StudyInstanceUID"].isin(set(studies))]
    by_study = {s: g for s, g in slots.groupby("StudyInstanceUID")}

    n = len(studies)
    cache = np.zeros((n, len(SLOT_NAMES), n_slice, out_size, out_size), np.uint8)
    mask = np.zeros((n, len(SLOT_NAMES)), np.float32)
    stats = {"filled": 0, "no_file": 0, "unreadable": 0, "no_crop": 0,
             "reversed": 0, "px_mismatch": 0, "no_lat": 0}

    for i, st in enumerate(studies):
        g = by_study.get(st)
        if g is None:
            continue
        lat = (lat_map.get(st) or (None,))[0]
        if lat is None:
            stats["no_lat"] += 1
        for _, r in g.iterrows():
            j = SLOT_NAMES.index(r["slot"])
            f = NII / f"{st}_{r['SeriesInstanceUID']}.nii"
            if not f.exists():
                f = f.with_suffix(".nii.gz")
            if not f.exists():
                stats["no_file"] += 1
                continue
            rv = bool(rev.get(r["SeriesInstanceUID"], False))
            img, info = read_slot_nifti(f, SLOT_PLANE[r["slot"]], lat, crop_mm=crop_mm,
                                        centre_mm=centre_mm, out_size=out_size,
                                        n_slice=n_slice, reverse=rv,
                                        px_hdr=r.get("px"))
            if img is None:
                stats["unreadable"] += 1
                continue
            cache[i, j] = img.numpy()
            mask[i, j] = 1.0
            stats["filled"] += 1
            stats["reversed"] += int(info["reversed"])
            stats["no_crop"] += int(not info["cropped"])
            if info["px"] and info["px_hdr"] and np.isfinite(info["px_hdr"]) \
                    and abs(info["px"] - float(info["px_hdr"])) > 1e-3:
                stats["px_mismatch"] += 1
        if verbose and (i + 1) % 50 == 0:
            print(f"    [{i + 1:>5}/{n}] {stats['filled']:,} slots", flush=True)
    return cache, mask, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--crop-mm", type=float, default=CROP_MM)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    slots = pd.read_csv(D / "slots_pilkwang.csv")
    have = {p.name.split("_")[1].replace(".nii", "") for p in NII.glob("*.nii*")}
    slots = slots[slots["SeriesInstanceUID"].isin(have)]
    studies = sorted(slots["StudyInstanceUID"].unique())[:a.limit]

    print(f"rebuilding {len(studies)} studies at crop {a.crop_mm:g} mm, "
          f"{N_SLICE} slices, {IMG}px")
    cache, mask, stats = build_cache(studies, slots, crop_mm=a.crop_mm)
    print(f"\ncache {cache.shape} = {cache.nbytes / 1e9:.2f} GB")
    print(f"mask: {int(mask.sum())} of {mask.size} slots filled "
          f"({mask.sum() / mask.size:.1%}), per study mean {mask.sum(1).mean():.2f}")
    for k, v in stats.items():
        print(f"  {k:<14} {v}")
    if a.report:
        nz = cache[mask.astype(bool)]
        print(f"\npixel stats over filled slots: mean {nz.mean():.1f} "
              f"std {nz.std():.1f} min {nz.min()} max {nz.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
