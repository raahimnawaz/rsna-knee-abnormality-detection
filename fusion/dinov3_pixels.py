"""The DINOv3 arm's pixel path, rebuilt against our NIfTI corpus — and the probe that decides
whether it can be built at fidelity at all.

    python fusion/dinov3_pixels.py --probe            # the slice-direction probe (start here)
    python fusion/dinov3_pixels.py --probe --limit 12 # a fast pass while iterating

===============================================================================================
WHY THERE IS A PROBE BEFORE THERE IS A GATE
===============================================================================================

`ft_b`'s kernel orders slices **geometrically** (project `ImagePositionPatient` onto the slice
normal), and geometry is something we can recover: K16 measured it for 8,048 sagittal series and
cross-validated 21/21 (§2n/§3i). **This arm does not order geometrically. It orders by
`int(InstanceNumber)`** — see `ordered_files` in its kernel — and InstanceNumber is **not on
disk in any form**:

* the NIfTI files carry no patient coordinate system at all (`has_position` and `has_orientation`
  false in 100% of a 300-file sample, §3n) and no DICOM tags;
* `data/external/dicom_headers_zhukovoleksiy.parquet` is **one row per SERIES** — 24,371 rows,
  `n_slices`/`PixelSpacing`/`Rows` and 40 other series-level fields, **nothing per slice**.

§2n already priced what that costs: sorting by InstanceNumber agrees with the true geometric
direction **56.9%** of the time, and since `inst` and `loc` sit at **|rho| = 1.000**, the
disagreement is a **direction bit, not a scramble**. So we can place the slices in the right
*order* and cannot know the right *sign* for roughly 43% of series.

**Why that bites harder here than anywhere else in this repo.** `stem: 'native'` means the 16
slices enter as **input CHANNELS** (`patch_embed.proj.weight` is (384, **16**, 16, 16)). Channels
are not exchangeable — each has its own filter — so a reversed stack is a genuine input change,
not a re-ordering the model averages over.

**But their training data had exactly the same coin-flip in it**, because they trained on
InstanceNumber order too. A model fed an arbitrary per-series direction for 4,407 studies may
simply have learned to be direction-robust. **That is measurable, it is cheap, and it decides
everything downstream:**

* **flip Δ << member Δ** → the model is direction-robust, the ordering problem is moot, and the
  full path can be built and gated normally.
* **flip Δ ≈ member Δ** → reversal is as damaging as swapping in a different model (§3i's scale
  for a random permutation was 0.0501, "as much as swapping in a different member"). The path
  then needs **direction TTA** (run both, average) and that has to be declared *before* any gold
  number exists, not chosen after.

The comparison is deliberately made against an **in-model** reference — the spread between two
different folds on the identical input — so it reads in the units that matter rather than in
abstract sigmoid points.

===============================================================================================
WHAT IS TRANSCRIBED, AND WHERE IT DIFFERS FROM `ft_b_pixels.py`
===============================================================================================

All four differences are from `PLAN.md` §9h's corrections, and every one of them is silent if got
wrong — they change pixel values, not shapes:

| | this arm | `ft_b` |
|---|---|---|
| slices | **16, as input channels** | 32, as a batch of 3-channel images |
| band | **`(0.12, 0.88)` of the stack** | full stack |
| window | **per-SLICE 1/99 percentile** | per-SERIES 0.5/99.5 |
| normalisation | **none — `uint8/255` and stop** | ImageNet mean/std |
| crop | **fixed 130 mm centred, via PixelSpacing** | background trim at `max>8` |
| laterality | **non-sagittal only; sagittal untouched** | sagittal reversed too |

**On laterality.** Their rule is `flip = plane != 'Sagittal' and ImagePositionPatient[0] < 0`. In
LPS, −x is the patient's **right**, so this mirrors right knees onto a **left**-knee convention —
the same direction as `pilkwang_pixels.normalise_laterality` and the *opposite* of
`ft_b_pixels`. Sagittal is left alone here, where pilkwang reverses its channel order. We take
the side from `study_meta.csv` (the DICOM answer, §2's `study_laterality`) rather than from the
NIfTI, which cannot supply it.

**Nothing in this file may be tuned against a score.** Same standing rule as `ft_b_pixels` and
§3b: a convention picked because it made a number look better is fitted to the thing it is meant
to test.
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
D = PROJ / "data"
NII = D / "nifti" / "nifti_train"

from pipeline.preprocess import nifti_geometry, study_laterality  # noqa: E402
from fusion.dinov3_model import (CROP_MM, LABELS, N_SLICE, SIZE, SLICE_BAND,  # noqa: E402
                                 SLOTS, load_fold)


def load_volume(path: Path) -> tuple[np.ndarray, float] | None:
    """`<study>_<series>.nii` -> (n_slices, rows, cols) float32 + in-plane mm. No selection yet:
    the probe needs the raw stack so it can reverse it *before* the band is applied."""
    g = nifti_geometry(path)
    if g is None:
        return None
    import nibabel as nib
    try:
        arr = np.asanyarray(nib.load(str(path)).dataobj)
    except Exception:
        return None
    vol = np.transpose(arr, (g["slice_axis"], g["row_axis"], g["col_axis"])).astype(np.float32)
    return (vol, g["in_plane_mm"]) if len(vol) >= 3 else None


def centre_crop_mm(sl: np.ndarray, in_plane_mm: float) -> np.ndarray:
    """Their `read_crop`: a fixed CROP_MM box about the array centre, in millimetres."""
    ps = in_plane_mm if in_plane_mm > 1e-6 else CROP_MM / max(sl.shape)
    half = int(round(CROP_MM / ps / 2))
    cy, cx = sl.shape[0] // 2, sl.shape[1] // 2
    y0, y1 = max(0, cy - half), min(sl.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(sl.shape[1], cx + half)
    return sl[y0:y1, x0:x1]


def render_slot(vol: np.ndarray, in_plane_mm: float, plane: str, lat: str | None,
                reverse: bool) -> torch.Tensor:
    """One series -> (N_SLICE, SIZE, SIZE) float32 in [0,1]. `reverse` flips the stack BEFORE the
    band is taken, which is what a wrong InstanceNumber sign would actually do."""
    if reverse:
        vol = vol[::-1]
    n = len(vol)
    lo_f, hi_f = SLICE_BAND
    i0, i1 = int(round(lo_f * (n - 1))), int(round(hi_f * (n - 1)))
    avail = list(range(i0, i1 + 1))
    if len(avail) >= N_SLICE:
        picks = [avail[int(round(t))] for t in np.linspace(0, len(avail) - 1, N_SLICE)]
        off = 0
    else:
        picks, off = avail, (N_SLICE - len(avail)) // 2

    out = np.zeros((N_SLICE, SIZE, SIZE), np.float32)
    for c, p in enumerate(picks):
        crop = centre_crop_mm(vol[p], in_plane_mm)
        if crop.size == 0:
            continue
        lo, hi = np.percentile(crop[::4, ::4], [1, 99])        # per-SLICE, theirs
        img = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1)
        t = torch.from_numpy(np.ascontiguousarray(img))[None, None]
        t = F.interpolate(t, size=(SIZE, SIZE), mode="area")   # == cv2.INTER_AREA downscaling
        out[off + c] = t[0, 0].numpy()

    t = torch.from_numpy(out)
    if plane in ("Coronal", "Axial") and lat == "R":
        t = torch.flip(t, dims=[-1])       # mirror onto a LEFT knee; sagittal untouched
    return t


def slot_table() -> pd.DataFrame:
    """Every series with a NIfTI on disk, tagged with its DINOv3 slot index (or -1)."""
    have = {p.name[:-4].split("_")[1]: p for p in NII.glob("*.nii")}
    s = pd.read_csv(D / "train_series.csv")
    s = s[s["SeriesInstanceUID"].isin(have)].copy()
    s["path"] = s["SeriesInstanceUID"].map(have)
    slot = {(pl, fs): i for i, (pl, fs) in enumerate(SLOTS)}
    s["slot"] = [slot.get((p, f), -1) for p, f in zip(s["Anatomical_Plane"], s["Fat_Suppression"])]
    return s[s["slot"] >= 0]


def build_study(rows: pd.DataFrame, lat: str | None, reverse: bool,
                cache: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
    """-> (n_present, N_SLICE, SIZE, SIZE) and the 1-based slot ids, their `build_study` order."""
    ims, slots = [], []
    for si in range(len(SLOTS)):
        sub = rows[rows["slot"] == si]
        if sub.empty:
            continue
        r = sub.iloc[0]                                        # first match, theirs
        key = r["SeriesInstanceUID"]
        if key not in cache:
            got = load_volume(r["path"])
            if got is None:
                continue
            cache[key] = got
        vol, mm = cache[key]
        ims.append(render_slot(vol, mm, SLOTS[si][0], lat, reverse))
        slots.append(si + 1)                                   # 1-based; 0 is the padding index
    if not ims:
        return None
    return torch.stack(ims), torch.tensor(slots, dtype=torch.long)


@torch.no_grad()
def probe(limit: int | None, dev: torch.device) -> int:
    """Reversal Δ against the between-fold Δ on identical input. See the module docstring."""
    z = np.load(D / "_ft_b_gold.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    if limit:
        ids = ids[:limit]
    tab = slot_table()
    by = {s: g for s, g in tab.groupby("StudyInstanceUID") if s in set(ids)}
    ids = [s for s in ids if s in by]
    lat_map = study_laterality(D / "study_meta.csv")

    print(f"loading 5 folds onto {dev} ...")
    models = [load_fold(f, dev)[0] for f in range(5)]
    print(f"probing {len(ids)} gold studies\n")

    fwd = np.zeros((len(models), len(ids), len(LABELS)), np.float32)
    rev = np.zeros_like(fwd)
    t0 = time.time()
    for i, sid in enumerate(ids):
        rows, cache = by[sid], {}
        lat = (lat_map.get(sid) or (None,))[0]
        for tag, store in (("fwd", fwd), ("rev", rev)):
            got = build_study(rows, lat, reverse=(tag == "rev"), cache=cache)
            if got is None:
                store[:, i] = np.nan
                continue
            im, sl = got
            im, sl = im.to(dev), sl.to(dev)
            sidx = torch.zeros(len(sl), dtype=torch.long, device=dev)
            sm = torch.zeros(len(sl), 0, device=dev)
            for mi, m in enumerate(models):
                store[mi, i] = torch.sigmoid(m(im, sl, sm, sidx, 1).float())[0].cpu().numpy()
        if (i + 1) % 10 == 0 or i + 1 == len(ids):
            print(f"  {i+1}/{len(ids)}  {time.time()-t0:.0f}s")

    ok = ~np.isnan(fwd[0, :, 0])
    f, r = fwd[:, ok], rev[:, ok]
    flip = np.abs(f - r).mean()
    member = np.mean([np.abs(f[a] - f[b]).mean()
                      for a in range(5) for b in range(a + 1, 5)])
    corr = np.corrcoef(f.ravel(), r.ravel())[0, 1]

    print(f"\n{'='*76}\nSLICE-DIRECTION PROBE — n={int(ok.sum())} gold studies, 5 folds")
    print(f"{'='*76}")
    print(f"  reversal Δ (mean |Δsigmoid|)          : {flip:.4f}")
    print(f"  between-fold Δ, identical input       : {member:.4f}   <- the reference scale")
    print(f"  ratio  reversal / between-fold        : {flip/member:.3f}")
    print(f"  corr(forward, reversed) over all preds: {corr:.4f}")
    print("\n  per-label reversal Δ:")
    for li, lab in enumerate(LABELS):
        print(f"    {lab:18s} {np.abs(f[:, :, li] - r[:, :, li]).mean():.4f}")

    print(f"\n{'-'*76}")
    if flip < 0.35 * member:
        print("  VERDICT: DIRECTION-ROBUST. Reversal moves predictions well under the spread")
        print("  between two folds, so the unrecoverable InstanceNumber sign is not a blocker.")
        print("  Build the full path; no direction TTA needed.")
    elif flip < 0.8 * member:
        print("  VERDICT: PARTIALLY SENSITIVE. Declare direction TTA (run both, average) BEFORE")
        print("  any gold number exists -- 3b forbids choosing it afterwards.")
    else:
        print("  VERDICT: DIRECTION-SENSITIVE. Reversal costs about as much as swapping the")
        print("  model. 43% of series would be fed backwards, so an honest local score is not")
        print("  reachable without TTA, and even with it the input distribution is off-train.")
    print(f"{'-'*76}")
    print("\nNOTE: these thresholds were written before the numbers were seen. They are a")
    print("reading rule, not a gate -- nothing here selects, so 3b is not engaged.")
    return 0


@torch.no_grad()
def predict(dev: torch.device, limit: int | None = None) -> int:
    """Score the gold studies with DIRECTION TTA and write `data/_dinov3_gold.npz`.

    TTA was pre-registered in `PLAN.md` §9h **before this function produced a number**, on the
    reasoning that guessing the InstanceNumber sign is fully wrong on ~43% of series while the
    mean of both renders is half-right on all of them. Both directions are saved alongside the
    mean so the choice stays auditable and nobody has to re-run to check it.

    Output shape is `(n_studies, 5, 12)` to match `_ft_b_gold.npz`, so `fold_recover.py` can read
    it without a special case.
    """
    ids = [str(x) for x in np.load(D / "_ft_b_gold.npz", allow_pickle=True)["ids"]]
    if limit:
        ids = ids[:limit]
    tab = slot_table()
    by = {s: g for s, g in tab.groupby("StudyInstanceUID") if s in set(ids)}
    ids = [s for s in ids if s in by]
    lat_map = study_laterality(D / "study_meta.csv")

    print(f"loading 5 folds onto {dev} ...")
    models = [load_fold(f, dev)[0] for f in range(5)]
    print(f"scoring {len(ids)} gold studies, both directions\n")

    out = {k: np.full((len(ids), 5, len(LABELS)), np.nan, np.float32) for k in ("fwd", "rev")}
    t0 = time.time()
    for i, sid in enumerate(ids):
        rows, cache = by[sid], {}
        lat = (lat_map.get(sid) or (None,))[0]
        for tag in ("fwd", "rev"):
            got = build_study(rows, lat, reverse=(tag == "rev"), cache=cache)
            if got is None:
                continue
            im, sl = got[0].to(dev), got[1].to(dev)
            sidx = torch.zeros(len(sl), dtype=torch.long, device=dev)
            sm = torch.zeros(len(sl), 0, device=dev)
            for mi, m in enumerate(models):
                out[tag][i, mi] = torch.sigmoid(m(im, sl, sm, sidx, 1).float())[0].cpu().numpy()
        if (i + 1) % 10 == 0 or i + 1 == len(ids):
            print(f"  {i+1}/{len(ids)}  {time.time()-t0:.0f}s")

    tta = np.nanmean([out["fwd"], out["rev"]], axis=0)
    np.savez(D / "_dinov3_gold.npz", pred=tta, fwd=out["fwd"], rev=out["rev"],
             ids=np.array(ids), labels=np.array(LABELS))
    print(f"\nwrote data/_dinov3_gold.npz  pred{tta.shape} (direction-TTA mean)")
    print("  `pred` is the pre-registered arm. `fwd`/`rev` are kept for audit.")
    print("\nNEXT: python fusion/fold_recover.py --arm dinov3")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if a.probe:
        return probe(a.limit, dev)
    if a.predict:
        return predict(dev, a.limit)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
