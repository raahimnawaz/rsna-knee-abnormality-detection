"""`rsna-ft-b`'s `load_series`, rebuilt against our NIfTI corpus, and the gate that judges it.

    python fusion/ft_b_pixels.py --gate        # the pre-registered reproduction gate
    python fusion/ft_b_pixels.py --limit 4 --report

WHAT IS TRANSCRIBED LITERALLY from their inference kernel: K=32 slices taken by `linspace` over
the ordered stack; per-SERIES percentile 0.5/99.5 (not per-slice, which would destroy inter-slice
contrast); background trim at `max>8` with a 4 px margin; resize to 336; grey repeated to 3
channels; ImageNet mean/std. Canonicalisation is to a **right** knee — note this is the opposite
convention to `pilkwang_pixels.normalise_laterality`, which maps onto a left knee, and mixing the
two is a silent error rather than a loud one.

TWO DEVIATIONS, both forced, both declared:

1. **`cv2.resize(INTER_AREA)` → `F.interpolate(mode='area')`.** Mathematically the same operation
   for downscaling, which is the only direction used here. `cv2` is not installed in `.venv`.
2. **Plane comes from `train_series.csv:Anatomical_Plane`, not from `plane_from_iop`.** §2l
   measured the nearest signed LPS axis as unanimous **132/132** per plane, so these agree; and
   the competition ships the column free.

THE PART THAT CANNOT BE LITERAL, AND IT IS THE RISK IN THIS FILE — same as `pilkwang_pixels`.
Their `load_series` orders slices by projecting `ImagePositionPatient` onto the slice normal, and
orders sagittal by each slice's `center_x`. **Measured 2026-08-13: 0 of 300 sampled NIfTI files
retain a patient coordinate system** — `has_position` and `has_orientation` are both false in
100% of the sample, the affine is a bare diagonal of voxel spacings. So neither projection is
recoverable and the stored order is a convention, not a measurement. What we have instead is K16
(`data/slice_direction_resolved.csv`, 8,048 sagittal series, measured and cross-validated 21/21).

**The medial-first bit is resolved from the repo's own convention, NOT fitted to this gate.**
`pipeline/slot_cache.py` line 53 records, as a load-bearing fact used by `sag_med`/`sag_lat`:
*"the normal is −x, so ascending spatial order runs medial → lateral for a canonical right
knee."* That is exactly their `slice 0 := most medial`. So: K16-correct to DICOM-forward, then
reverse sagittal for **left** knees only. Coronal and axial mirror in plane for left knees.

**Do not resolve any remaining ambiguity by trying both and keeping the better score.** That is
`pilkwang_pixels`'s standing rule and §3b's whole lesson — a convention chosen to make a gate
pass is fitted to the thing it is meant to test.

===============================================================================================
THE GATE. PRE-REGISTERED 2026-08-13 pm, BEFORE ANY ft_b PREDICTION EXISTED.
===============================================================================================

`ft_b_model.py --check` proved the architecture (5/5 strict loads). It cannot prove our pixels are
their pixels, and this arm ships **no fingerprint and no fold split**, so §3h's 7e-06 trick and
§3i's fold recovery are both unavailable. What IS available is §3l-2's measured offset.

**The test.** Run all five folds on the gold studies with NIfTI coverage, mean of sigmoids exactly
as their `predict_study` does, and score on `fusion/score_gold.py`.

**The prediction.** Their claim is **LB 0.883**. §3l-2 measured **gold + 0.046 ≈ LB** over four
independent systems, so an honest read of this arm should sit near **gold 0.837**.

**Why the comparison is one-directional, which is what makes it usable at n=47.** Gold studies are
in their training corpus (everyone trains on report labels over all 4,407), and we average all
five folds rather than the one that held each study out. So **our number is biased UPWARD** while
0.837 is derived from their honest hidden-test score.

  * **gold ≥ 0.837** → consistent with a working pixel path. Not proof — the bias could be masking
    a small defect — but the route is open and F6 proceeds.
  * **gold materially < 0.837** → the pixel path is BROKEN, because the bias runs the other way.
    First suspect is **§3i's slice-order residual** (a third of series arrive permuted, not merely
    reversed, and this arm's K=32 `linspace` over physical order has more surface area to that
    than pilkwang's symmetric band did). Second suspect is the medial-first bit above. **Neither
    is to be fixed by tuning against this number.**

**What this gate may NOT do:** choose a convention, choose K, choose a normalisation, or be re-run
with a variant kept because it scored better. One shot, stated in advance.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "extractor"))
D = PROJ / "data"
NII = D / "nifti" / "nifti_train"

from pipeline.preprocess import nifti_geometry, study_laterality  # noqa: E402
from fusion.pilkwang_pixels import direction_map  # noqa: E402
from fusion.ft_b_model import IMG, K, LABELS, embed, load_fold, pick_device  # noqa: E402

PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
IM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], np.float32)
#: their fallback when a study cannot be read at all -- prevalence, so it scores at chance
PREV = dict(zip(LABELS, [0.196, 0.103, 0.429, 0.210, 0.228, 0.157, 0.307,
                         0.410, 0.407, 0.242, 0.216, 0.223]))


def crop_background(vol: np.ndarray) -> np.ndarray:
    """Trim the empty border so the knee fills more of the frame. Theirs, on uint8."""
    m = vol.max(axis=0) > 8
    ys, xs = np.where(m)
    if len(ys) == 0:
        return vol
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    y0, x0 = max(0, y0 - 4), max(0, x0 - 4)
    y1, x1 = min(vol.shape[1], y1 + 4), min(vol.shape[2], x1 + 4)
    return vol[:, y0:y1, x0:x1]


def load_series_nifti(path: Path, plane: str, lat: str | None,
                      reverse: bool) -> torch.Tensor | None:
    """One series -> (K, 3, IMG, IMG) float32, canonicalised to a RIGHT knee."""
    g = nifti_geometry(path)
    if g is None:
        return None
    import nibabel as nib
    try:
        arr = np.asanyarray(nib.load(str(path)).dataobj)
    except Exception:
        return None
    vol = np.transpose(arr, (g["slice_axis"], g["row_axis"], g["col_axis"])).astype(np.float32)
    if reverse:                       # K16: the stored stack runs backwards
        vol = vol[::-1]
    n = len(vol)
    if n < 3:
        return None

    # slice 0 := most medial. Ascending order is medial->lateral for a canonical RIGHT knee
    # (slot_cache.py:53), so only left knees are reversed.
    if plane == "Sagittal" and lat == "L":
        vol = vol[::-1]

    sel = np.linspace(0, n - 1, K).round().astype(int)
    vol = np.ascontiguousarray(vol[sel])

    lo, hi = np.percentile(vol, [0.5, 99.5])       # over the SERIES, theirs
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    vol = (vol * 255).astype(np.uint8)
    vol = crop_background(vol)

    if plane in ("Coronal", "Axial") and lat == "L":
        vol = vol[:, :, ::-1]                       # mirror onto a right knee

    t = torch.from_numpy(np.ascontiguousarray(vol)).float().unsqueeze(0)
    t = F.interpolate(t, size=(IMG, IMG), mode="area")   # == cv2.INTER_AREA downscaling
    t = (t.squeeze(0) / 255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    t = (t - torch.from_numpy(IM_MEAN)[None, :, None, None]) / \
        torch.from_numpy(IM_STD)[None, :, None, None]
    return t


def series_table() -> pd.DataFrame:
    """Every series with a NIfTI on disk, with its plane. `<study>_<series>.nii` is the layout."""
    have = {p.name[:-4].split("_")[1]: p for p in NII.glob("*.nii")}
    s = pd.read_csv(D / "train_series.csv")
    s = s[s["SeriesInstanceUID"].isin(have)].copy()
    s["path"] = s["SeriesInstanceUID"].map(have)
    s["plane"] = s["Anatomical_Plane"].str.title()
    return s[s["plane"].isin(PLANES)]


@torch.no_grad()
def predict_studies(studies, tab, models, dev, verbose=True):
    """(n_studies, 5 folds, 12) sigmoid probabilities. Pixels decoded once, reused per fold."""
    lat_map = study_laterality(D / "study_meta.csv")
    rev = direction_map()
    by_study = {k: v for k, v in tab.groupby("StudyInstanceUID")}
    out = np.full((len(studies), len(models), 12), np.nan, np.float32)
    t0 = time.time()
    for i, st in enumerate(studies):
        rows = by_study.get(st)
        if rows is None:
            continue
        lat = (lat_map.get(st) or (None,))[0]
        xs, pls = [], []
        for _, r in rows.iterrows():
            x = load_series_nifti(r["path"], r["plane"],
                                  lat, bool(rev.get(r["SeriesInstanceUID"], False)))
            if x is not None:
                xs.append(x.to(dev))
                pls.append(PLANES[r["plane"]])
        if not xs:
            continue
        pl = torch.tensor(pls, device=dev)[None]
        m = torch.ones_like(pl, dtype=torch.bool)
        for k, (bb, hd, _) in enumerate(models):
            e = torch.stack([embed(bb, x) for x in xs])[None]
            out[i, k] = torch.sigmoid(hd(e, pl, m))[0].cpu().numpy()
        del xs
        if verbose and (i % 5 == 0 or i == len(studies) - 1):
            el = time.time() - t0
            print(f"  [{i + 1:>3}/{len(studies)}] {el / 60:5.1f} min  "
                  f"ETA {el / max(i + 1, 1) * len(studies) / 60:5.1f} min", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", action="store_true", help="the pre-registered reproduction gate")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default="data/_ft_b_gold.npz")
    a = ap.parse_args()

    tab = series_table()
    print(f"{len(tab):,} series on disk over {tab.StudyInstanceUID.nunique():,} studies")

    if a.report or a.limit:
        lat_map = study_laterality(D / "study_meta.csv")
        rev = direction_map()
        for _, r in tab.head(a.limit or 4).iterrows():
            lat = (lat_map.get(r["StudyInstanceUID"]) or (None,))[0]
            x = load_series_nifti(r["path"], r["plane"], lat,
                                  bool(rev.get(r["SeriesInstanceUID"], False)))
            print(f"  {r['plane']:9s} lat={lat}  -> "
                  f"{tuple(x.shape) if x is not None else None}"
                  f"{'' if x is None else f'  mean {x.mean():+.3f} sd {x.std():.3f}'}")
        if not a.gate:
            return 0

    if not a.gate:
        ap.print_help()
        return 0

    ref = np.load(D / "external" / "pilkwang_weights" / "oof.npz", allow_pickle=True)
    gold = set(np.array(ref["ids"])[ref["gold_mask"]].tolist())
    studies = sorted(g for g in gold if g in set(tab.StudyInstanceUID))
    print(f"\nGATE: {len(studies)} of {len(gold)} gold studies have NIfTI coverage")

    dev = pick_device()
    models = [load_fold(f, dev) for f in range(5)]
    print(f"5 folds loaded on {dev}\n")
    P = predict_studies(studies, tab, models, dev)

    np.savez_compressed(a.out, pred=P, ids=np.array(studies), labels=np.array(LABELS))
    print(f"\nwrote {a.out}")

    from metrics import auc
    y = (pd.read_csv(D / "train.csv").set_index("StudyInstanceUID")
         .reindex(studies)[LABELS].to_numpy())
    p = np.nanmean(P, axis=1)
    keep = ~np.isnan(y).any(1) & ~np.isnan(p).any(1)
    y, p = y[keep].astype(int), p[keep]
    per = [auc(y[:, j], p[:, j]) for j in range(12)]
    macro = float(np.nanmean(per))
    print("\n" + "=" * 66)
    print(f"  ft_b all-5-fold on {len(y)} gold studies:  macro {macro:.4f}")
    print(f"  gate                                    >= 0.837  "
          f"(= their LB 0.883 - §3l-2's 0.046)")
    print(f"  VERDICT: {'PASS' if macro >= 0.837 else 'FAIL'} "
          f"({macro - 0.837:+.4f} vs the bar, and our read is biased UP)")
    print("=" * 66)
    for j, t in enumerate(LABELS):
        print(f"  {t:18s} {per[j]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
