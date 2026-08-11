"""Measure the K16 direction bit per series instead of inferring it. CPU only, sagittal only.

WHY THIS REPLACES kaggle_01d. 01d exported three header rules that could have been the NIfTI
converter's sort key -- InstanceNumber, sorted-filename, SliceLocation -- on the theory that one
of them would reproduce the 51 series whose direction the thumbnails already settle. Measured
2026-08-10, none does:

    inst 56.9%    file 60.8%    loc 56.9%        (n=51, chance is ~50%)

and filtering to the most confident half of the ground truth does not move any of them. The
sampling was not the limit either: `inst` and `loc` both come back at |rho| = 1.000, i.e.
perfectly monotone in projection, so reading every header rather than six would export the same
numbers at ten times the cost. **The converter's order is simply not a function of any header
this corpus carries.** That is a real answer, and it retires the "2-3 header reads per series"
route in `PLAN.md` 9 Phase 0 step 2 along with its 3.7 h fallback.

WHAT THIS DOES INSTEAD. It stops asking what rule the converter followed and measures the answer
directly, the same way `validate_nifti` check 4b measures it for 51 series -- just for every
series that needs it. Per series: read the headers, sort spatially, and ship three 32x32
thumbnails (spatially first, middle, last). Locally, `resolve_slice_direction.py --measured`
correlates NIfTI slice k=0 against the first and the last and reads the bit off. No rule, no
extrapolation, no adoption bar to clear.

WHY SAGITTAL ONLY, AND WHY THAT IS NOT A SHORTCUT. The bit exists to serve K18: medial/lateral
is the SLICE axis for sagittal and the in-plane x-axis for axial and coronal, so those two are
already handled by `canonicalise`'s existing mirror and do not need it. Sagittal is 9,864 of
24,371 series. That is what makes this affordable -- a full-header pass over the whole corpus is
~95 min at 01d's measured 122 opens/s, and over sagittal alone it is ~20.

    ~9,864 series x ~30 slices = ~296k header opens, plus 29,592 pixel decodes
    output ~30 MB

Read ALL slice headers within a series, not a sample: the spatial sort is the thing being
established and a partial series cannot establish it. That is the same reason kaggle_01c reads
them all.

Writes /kaggle/working/direction_thumbs.npz + direction_index.csv.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


def _bootstrap_preprocess() -> None:
    """Same glob-not-hardcode bootstrap as kaggle_01b/01c/01d -- see its comment; it cost a session."""
    import glob
    for pat in ("/kaggle/input/**/pipeline/preprocess.py", "/kaggle/usr/lib/**/preprocess.py",
                str(Path(__file__).resolve().parents[1] / "pipeline" / "preprocess.py")):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            sys.path.insert(0, str(Path(hits[0]).parent))
            return
    listing = sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*"))
    raise SystemExit(
        "cannot find pipeline/preprocess.py. Attach the rsna-knee-code Dataset to this "
        f"notebook. /kaggle/input currently holds: {listing}")


_bootstrap_preprocess()

from preprocess import build_study_index, find_competition_root, read_pixels  # noqa: E402

OUT = Path("/kaggle/working")
PLANES = ("Sagittal",)
THUMB = 32            # enough to correlate two ends of a knee apart; 1 KB per slice
WORKERS = 8


def thumb_of(a: np.ndarray) -> np.ndarray:
    """[H,W] -> [THUMB,THUMB] uint8. Nearest-neighbour and percentile-normalised.

    Nearest-neighbour on purpose: the local side reproduces this exact operation on the NIfTI
    slice, so any interpolation would have to match to be comparable. Same 0.5/99.5 percentiles
    as normalise_and_resample, so the two are on one intensity scale. uint8 rather than
    kaggle_01c's float16 because this ships 30k of them and a correlation does not need 11 bits.
    """
    r = np.linspace(0, a.shape[0] - 1, THUMB).round().astype(int)
    c = np.linspace(0, a.shape[1] - 1, THUMB).round().astype(int)
    t = a[np.ix_(r, c)].astype(np.float32)
    lo, hi = np.percentile(t, [0.5, 99.5])
    return (np.clip((t - lo) / (hi - lo + 1e-6), 0, 1) * 255).round().astype(np.uint8)


def probe(args):
    """One series -> (key, {first,mid,last} thumbnails, row). Header pass, then 3 decodes."""
    study, series, sdir = args
    sd = Path(sdir) / series
    try:
        files = sorted(sd.glob("*.dcm"))
    except OSError:
        return None
    if len(files) < 3:
        return None

    heads = []
    for f in files:
        try:
            heads.append((f, pydicom.dcmread(f, stop_before_pixels=True, force=True)))
        except Exception:                                          # noqa: BLE001
            pass
    if len(heads) < 3:
        return None
    try:
        iop = np.asarray(heads[0][1].ImageOrientationPatient, float)
        normal = np.cross(iop[:3], iop[3:])
    except Exception:                                              # noqa: BLE001
        return None

    keyed = []
    for f, ds in heads:
        try:
            p = float(np.dot(np.asarray(ds.ImagePositionPatient, float), normal))
        except Exception:                                          # noqa: BLE001
            continue
        if np.isfinite(p):
            keyed.append((p, f))
    if len(keyed) < 3:
        return None
    keyed.sort(key=lambda t: t[0])

    picks = {"first": keyed[0][1], "mid": keyed[len(keyed) // 2][1], "last": keyed[-1][1]}
    thumbs = {}
    for tag, f in picks.items():
        a = read_pixels(f)
        if a is None or a.ndim != 2:
            return None
        thumbs[tag] = thumb_of(a)

    key = f"{study}_{series}"
    return key, thumbs, {"StudyInstanceUID": study, "SeriesInstanceUID": series,
                         "n_slices": len(keyed), "mid_index": len(keyed) // 2,
                         "proj_first": keyed[0][0], "proj_last": keyed[-1][0]}


def main(root=None, out=None) -> None:
    root = Path(root) if root else find_competition_root()
    out = Path(out) if out else OUT
    ser = pd.read_csv(root / "train_series.csv")
    ser = ser[ser.Anatomical_Plane.isin(PLANES)]
    print(f"{len(ser):,} {'/'.join(PLANES)} series / {ser.StudyInstanceUID.nunique():,} studies")

    index = build_study_index(root)
    print(f"indexed {len(index):,} study directories")

    work = []
    for r in ser.itertuples():
        sdir = index.get(r.StudyInstanceUID)
        if sdir is not None:
            work.append((r.StudyInstanceUID, r.SeriesInstanceUID, str(sdir)))
    print(f"{len(work):,} resolve to a directory; full header pass + 3 decodes each, "
          f"{WORKERS} workers\n")

    thumbs, rows = {}, []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for n, res in enumerate(ex.map(probe, work), 1):
            if res is not None:
                key, t, row = res
                for tag, a in t.items():
                    thumbs[f"{key}|{tag}"] = a
                rows.append(row)
            if n % 1000 == 0:
                print(f"  {n:,}/{len(work):,}  ({len(rows):,} usable)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "direction_index.csv", index=False)
    np.savez_compressed(out / "direction_thumbs.npz", **thumbs)
    print(f"\n{'=' * 66}\nwrote direction_index.csv ({len(df):,} series) + "
          f"direction_thumbs.npz ({len(thumbs):,} thumbnails)\n{'=' * 66}")
    print("Next: download both into data/, then")
    print("  python pipeline/resolve_slice_direction.py --measured")


if __name__ == "__main__":
    main()
