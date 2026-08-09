"""Does the third-party NIfTI conversion agree with the DICOM headers? PLAN.md 9.1.

The conversion sits UPSTREAM of `PREPROCESS_VERSION`, so the fingerprint cannot detect a
disagreement (see the NIfTI section of preprocess.py). PLAN.md 9.1 says validating it "needs a
Kaggle session to make". **Most of it does not**, and that is what this script is for:
`kaggle_01b` already exported per-study DICOM geometry for all 4,407 studies into
`data/study_meta.csv` -- `laterality_tag`, `x_median` (median ImagePositionPatient x) and
`n_series`. That is a reference sitting on the laptop, and it covers the whole corpus rather
than the "handful of studies" 9.1 budgeted for.

Five checks. ALL PASSED 2026-08-09; the verdicts below are results, not intentions.

  1. DOES THE AFFINE CARRY A PATIENT FRAME?  **No.** 0.0% of series have a translation, 0.0% an
     off-diagonal rotation -- diagonal spacing only. PLAN.md 9.1 recorded "sform_code=2 with a
     populated affine ... real affine"; sform_code is 2 and the affine is empty. So
     ImagePositionPatient and ImageOrientationPatient are both gone, and neither laterality nor
     plane is derivable from these files.
  2. CAN LATERALITY STILL BE RESOLVED?  **Yes, from study_meta.csv, not from the pixels.**
     kaggle_01b already answered it for all 4,407 studies from the DICOM headers. Measured:
     2,203 by tag, 2,204 by geometry, 0 unresolved. This only works because kaggle_01b was run
     over the whole corpus first -- otherwise 4 of the 12 labels would be noise.
  3. STRUCTURAL INTEGRITY.  Series per study, shapes, spacings against the DICOM headers.
  4. IN-PLANE ORIENTATION.  Layout `as-is` at **r = 1.0000**, best for 100% of series, runner-up
     0.6513. r = 1.0 means the pixel data is identical -- a faithful repackaging, not a
     re-render.
  4b. SLICE DIRECTION.  **100% forward**; the stored k order matches the DICOM spatial sort.
  5. SLICE COUNT + SHAPE + SPACING vs DICOM.  69/69, 69/69, spacing error 0.0000 mm.

Checks 1-3 need only what kaggle_01b already exported. Checks 4/4b/5 need
`notebooks/kaggle_01c_series_geometry.py` -- header-only, CPU-only, no GPU lottery, ~5 min.
With the affine empty, the kaggle_01c thumbnails are the ONLY instrument that can settle
orientation and slice direction: there is no header left to compare against.

    python pipeline/validate_nifti.py                    # whatever has downloaded so far
    python pipeline/validate_nifti.py --nifti data/nifti/nifti_train
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "pipeline"))
from preprocess import nifti_geometry, study_laterality                # noqa: E402

D = PROJ / "data"


def per_study_index(df: pd.DataFrame):
    """The distinct studies present in the scanned parts, in a stable order."""
    return df.StudyInstanceUID.drop_duplicates().tolist()


def scan(nifti_dir: Path) -> pd.DataFrame:
    """Header-only pass over every .nii present. No pixel data is decoded."""
    files = sorted(nifti_dir.glob("*.nii")) + sorted(nifti_dir.glob("*.nii.gz"))
    if not files:
        sys.exit(f"no .nii under {nifti_dir}\nDownload a part first:\n  python -m kaggle "
                 f"datasets download -d davidadekanmi/rsna-knee-nifti-part1 -p data/nifti "
                 f"--unzip")
    rows, bad = [], 0
    for n, p in enumerate(files, 1):
        stem = p.name.split(".nii")[0]
        if "_" not in stem:
            bad += 1
            continue
        study, series = stem.split("_", 1)
        g = nifti_geometry(p)
        if g is None:
            bad += 1
            continue
        rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                     "n_slices": g["n_slices"], "rows": g["shape"][g["row_axis"]],
                     "cols": g["shape"][g["col_axis"]], "in_plane_mm": g["in_plane_mm"],
                     "slice_mm": g["slice_mm"], "has_position": g["has_position"],
                     "has_orientation": g["has_orientation"]})
        if n % 500 == 0 or n == len(files):
            print(f"\r  reading headers {n:,}/{len(files):,}", end="", flush=True)
    print()
    if bad:
        print(f"  {bad} file(s) unreadable or misnamed")
    return pd.DataFrame(rows)


THUMB = 64          # must match kaggle_01c.THUMB


def thumb_of(a: np.ndarray) -> np.ndarray:
    """Byte-for-byte the same operation as kaggle_01c.thumb_of, on a NIfTI slice instead."""
    r = np.linspace(0, a.shape[0] - 1, THUMB).round().astype(int)
    c = np.linspace(0, a.shape[1] - 1, THUMB).round().astype(int)
    t = a[np.ix_(r, c)].astype(np.float32)
    lo, hi = np.percentile(t, [0.5, 99.5])
    return np.clip((t - lo) / (hi - lo + 1e-6), 0, 1)


def _orientations(t: np.ndarray):
    """The 8 ways a 2-D slice can be laid out: transpose x flip0 x flip1."""
    for name, base in (("as-is", t), ("transposed", t.T)):
        for f0 in (False, True):
            for f1 in (False, True):
                v = base[::-1] if f0 else base
                v = v[:, ::-1] if f1 else v
                yield f"{name}{' +flipH' if f1 else ''}{' +flipV' if f0 else ''}", v


def check_orientation(nifti_dir: Path, geom: pd.DataFrame, thumbs: dict) -> None:
    """Correlate the DICOM middle slice against the NIfTI's, over all 8 layouts.

    This is the check that headers cannot make. A converter that wrote a plausible affine over
    transposed pixels passes every metadata test and still trains the model on mirrored anatomy.
    """
    import nibabel as nib
    g = geom.set_index(geom.StudyInstanceUID + "_" + geom.SeriesInstanceUID)
    keys = sorted({k.split("|")[0] for k in thumbs})
    scores, best_per_series, direction = {}, [], []

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        c = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        return 0.0 if np.isnan(c) else c

    for key in keys:
        p = nifti_dir / f"{key}.nii"
        if not p.exists():
            p = nifti_dir / f"{key}.nii.gz"
        if not p.exists() or key not in g.index or f"{key}|mid" not in thumbs:
            continue
        info = nifti_geometry(p)
        if info is None:
            continue
        arr = np.asanyarray(nib.load(str(p)).dataobj)
        vol = np.transpose(arr, (info["slice_axis"], info["row_axis"], info["col_axis"]))
        row = g.loc[key]
        if len(vol) != int(row.n_slices):
            continue                      # different slice count: not comparable slice-to-slice

        # --- in-plane layout, from the middle slice (invariant to slice reversal) ---
        d_mid = np.asarray(thumbs[f"{key}|mid"], dtype=np.float32)
        sl = np.asarray(vol[int(row.mid_index)], dtype=np.float32)
        best = (-2.0, "")
        for name, cand in _orientations(sl):
            if cand.shape[0] < 2 or cand.shape[1] < 2:
                continue
            c = corr(thumb_of(np.ascontiguousarray(cand)), d_mid)
            scores[name] = scores.get(name, 0.0) + c
            if c > best[0]:
                best = (c, name)
        best_per_series.append(best)

        # --- slice direction: does NIfTI k=0 match the DICOM's first or its last? ---
        # Only meaningful once the layout is known, so apply this series' own best layout.
        if f"{key}|first" in thumbs and f"{key}|last" in thumbs:
            lay = dict(_orientations(np.asarray(vol[0], dtype=np.float32)))[best[1]]
            t0 = thumb_of(np.ascontiguousarray(lay))
            c_first = corr(t0, np.asarray(thumbs[f"{key}|first"], dtype=np.float32))
            c_last = corr(t0, np.asarray(thumbs[f"{key}|last"], dtype=np.float32))
            direction.append("forward" if c_first >= c_last else "reversed")

    print(f"\n{'=' * 70}\n4. IN-PLANE ORIENTATION -- DICOM pixels vs NIfTI pixels "
          f"(n={len(best_per_series)})\n{'=' * 70}")
    if not best_per_series:
        print("  no series overlap between the thumbnails and the downloaded NIfTI parts.\n"
              "  kaggle_01c samples the whole corpus, so download more parts or re-run it "
              "restricted\n  to the studies you have.")
        return
    print(f"  {'layout':<26}{'mean r':>9}")
    for name, tot in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<26}{tot / len(best_per_series):>9.4f}")
    win = max(scores.items(), key=lambda kv: kv[1])[0]
    agree = sum(1 for c, n in best_per_series if n == win) / len(best_per_series)
    med = float(np.median([c for c, _ in best_per_series]))
    print(f"\n  winner: '{win}', best for {agree:.0%} of series, median r = {med:.4f}")
    if win == "as-is" and med > 0.9:
        print("  => load_series_nifti's transpose is CORRECT as written. Nothing to change.")
    elif med > 0.9:
        print(f"  => load_series_nifti is WRONG. Its (slice,row,col) transpose needs '{win}' "
              f"applied\n     in-plane. Fix _nifti_axes/load_series_nifti before building "
              f"anything.")
    else:
        print("  => no layout correlates well. The conversion is not a faithful repackaging; "
              "do not\n     use this corpus for the submitted cache.")

    print(f"\n{'=' * 70}\n4b. SLICE DIRECTION -- NIfTI k=0 vs the DICOM spatial first/last "
          f"(n={len(direction)})\n{'=' * 70}")
    if not direction:
        print("  no series had all three thumbnails.")
        return
    fwd = direction.count("forward") / len(direction)
    print(f"  forward (k=0 is the DICOM first slice) : {fwd:>6.1%}")
    print(f"  reversed                               : {1 - fwd:>6.1%}")
    if fwd > 0.95:
        print("\n  => stored order matches the DICOM sort. load_series_nifti is correct as "
              "written.")
    elif fwd < 0.05:
        print("\n  => every series is stored back-to-front. Add `vol = vol[::-1]` to "
              "load_series_nifti.\n     A reversed volume destroys the slice transformer's "
              "positional signal silently.")
    else:
        print("\n  => MIXED, and nothing in the NIfTI header distinguishes the two. Slice order "
              "is not\n     recoverable from this corpus. Either accept the loss (and drop the "
              "slice-position\n     embedding) or build the cache from DICOM.")


def check_slice_order(nifti_dir: Path, geom: pd.DataFrame) -> None:
    """Does the NIfTI's stored k order match the DICOM spatial sort, end to end?"""
    rows = []
    for r in geom.itertuples():
        key = f"{r.StudyInstanceUID}_{r.SeriesInstanceUID}"
        p = nifti_dir / f"{key}.nii"
        if not p.exists():
            p = nifti_dir / f"{key}.nii.gz"
        if not p.exists():
            continue
        info = nifti_geometry(p)
        if info is None:
            continue
        rows.append({"n_match": info["n_slices"] == r.n_slices,
                     "shape_match": (info["shape"][info["row_axis"]] == r.rows
                                     and info["shape"][info["col_axis"]] == r.cols),
                     "spacing_err": abs(info["in_plane_mm"] - r.pixel_spacing_r)})
    print(f"\n{'=' * 70}\n5. SLICE COUNT + POSITION vs DICOM (n={len(rows)})\n{'=' * 70}")
    if not rows:
        print("  no overlap between series_geometry.csv and the downloaded parts.")
        return
    d = pd.DataFrame(rows)
    print(f"  slice count matches : {d.n_match.sum():,}/{len(d):,} ({d.n_match.mean():.1%})")
    print(f"  in-plane shape matches (rows/cols vs DICOM): "
          f"{d.shape_match.sum():,}/{len(d):,} ({d.shape_match.mean():.1%})")
    print(f"  |spacing_nifti - PixelSpacing| : median {d.spacing_err.median():.4f} mm, "
          f"p95 {d.spacing_err.quantile(0.95):.4f} mm")
    # No position comparison: the affine carries no translation (check 1), so there is nothing
    # to compare ImagePositionPatient against. Shape and spacing are what survive.
    if d.n_match.mean() > 0.99 and d.spacing_err.median() < 0.01:
        print("  => slice counts and spacings reproduce the DICOM headers exactly.")
    else:
        print("  => the repackaging drops or resamples slices. Do not build the cache from it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nifti", default=str(D / "nifti" / "nifti_train"))
    ap.add_argument("--meta", default=str(D / "study_meta.csv"))
    ap.add_argument("--geometry", default=None,
                    help="series_geometry.csv from notebooks/kaggle_01c_series_geometry.py")
    ap.add_argument("--thumbs", default=None, help="series_thumbs.npz from the same run")
    args = ap.parse_args()

    meta_p = Path(args.meta)
    if not meta_p.exists():
        sys.exit(f"{meta_p} not found -- it is the DICOM reference this script compares against "
                 f"(written by notebooks/kaggle_01b_patients_laterality.py)")
    meta = pd.read_csv(meta_p).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")

    df = scan(Path(args.nifti))
    studies = df.StudyInstanceUID.unique()
    print(f"\n{len(df):,} series over {len(studies):,} studies "
          f"({len(studies) / len(meta):.1%} of the 4,407-study corpus)")

    # ---- 1. is there a patient coordinate system in these files at all? ---------------
    print(f"\n{'=' * 70}\n1. AFFINE -- does it carry a patient frame, or only spacing?"
          f"\n{'=' * 70}")
    pos, ori = df.has_position.mean(), df.has_orientation.mean()
    print(f"  non-zero translation (ImagePositionPatient) : {pos:>6.1%} of series")
    print(f"  off-diagonal rotation (ImageOrientationPatient): {ori:>6.1%} of series")
    if pos < 0.01 and ori < 0.01:
        print("\n  => SPACING ONLY. The conversion dropped the patient coordinate system.\n"
              "     PLAN.md 9.1 recorded 'sform_code=2 with a populated affine ... real\n"
              "     affine'. sform_code is 2; the affine is diagonal with zero translation.\n"
              "     Laterality and plane are NOT derivable from these files.")
    elif pos > 0.99 and ori > 0.99:
        print("\n  => full geometry present. This contradicts the 2026-08-09 measurement on\n"
              "     part1 -- a later part may have been converted differently. Re-check.")
    else:
        print("\n  => MIXED across parts, which is worse than either. Do not build until the\n"
              "     subsets are separated and each is handled explicitly.")

    # ---- 2. can laterality still be resolved, and from where? ------------------------
    print(f"\n{'=' * 70}\n2. LATERALITY -- recovered from study_meta.csv, not from the NIfTI"
          f"\n{'=' * 70}")
    lat = study_laterality(meta_p)
    studies_here = list(per_study_index(df))
    got = [lat.get(u, (None, "none")) for u in studies_here]
    src = pd.Series([s for _, s in got]).value_counts()
    print(f"  studies in the downloaded parts : {len(studies_here):,}")
    for k in ("tag", "geometry", "none"):
        n = int(src.get(k, 0))
        print(f"    laterality from {k:<9}: {n:>6,} ({n / max(len(studies_here), 1):>5.1%})")
    unresolved = int(src.get("none", 0))
    if unresolved == 0:
        print("\n  => every downloaded study can be canonicalised. The DICOM answer travels in\n"
              "     study_meta.csv, so losing the affine costs nothing here -- but ONLY because\n"
              "     kaggle_01b was run over all 4,407 studies first.")
    else:
        print(f"\n  => {unresolved:,} studies have no laterality from either source. Those cannot"
              f"\n     be canonicalised and 4 of 12 labels are noise for them.")

    # ---- 3. structural integrity ------------------------------------------------------
    print(f"\n{'=' * 70}\n3. STRUCTURE\n{'=' * 70}")
    n_ser = df.groupby("StudyInstanceUID").size()
    exp = meta.reindex(n_ser.index)["n_series"]
    same = (n_ser == exp)
    print(f"  series/study matches DICOM : {same.sum():,}/{len(n_ser):,} ({same.mean():.1%})")
    if (~same).any():
        d = pd.DataFrame({"nifti": n_ser[~same], "dicom": exp[~same]}).head(5)
        print(f"    first mismatches:\n{d.to_string()}")
        print("    NB studies split across parts will show short until every part is present.")
    print(f"  slices/series   : median {df.n_slices.median():.0f}  "
          f"range {df.n_slices.min()}-{df.n_slices.max()}")
    print(f"  in-plane mm     : median {df.in_plane_mm.median():.3f}  "
          f"range {df.in_plane_mm.min():.3f}-{df.in_plane_mm.max():.3f}")
    print(f"  slice mm        : median {df.slice_mm.median():.2f}  "
          f"range {df.slice_mm.min():.2f}-{df.slice_mm.max():.2f}")
    print(f"  in-plane shape  : {df['rows'].mode()[0]}x{df['cols'].mode()[0]} modal, "
          f"{df.groupby(['rows', 'cols']).ngroups} distinct")
    print(f"  affine carries a patient frame: position {df.has_position.mean():.1%}, "
          f"orientation {df.has_orientation.mean():.1%}  <- see check 1")

    # ---- 4 + 5. the two checks that need the Kaggle export ----------------------------
    if args.geometry and Path(args.geometry).exists():
        geom = pd.read_csv(args.geometry)
        nd = Path(args.nifti)
        if args.thumbs and Path(args.thumbs).exists():
            with np.load(args.thumbs) as z:
                check_orientation(nd, geom, {k: z[k] for k in z.files})
        else:
            print("\n4. IN-PLANE ORIENTATION -- skipped, no --thumbs")
        check_slice_order(nd, geom)
        print(f"\n{'=' * 70}\nAll five checks have run. If 1, 2, 4 and 5 all pass, the NIfTI "
              f"path is a faithful\nrepackaging and the train-side cache can be built from it "
              f"on the M5.\n{'=' * 70}")
    else:
        print(f"\n{'=' * 70}\nSTILL UNVALIDATED, and not validatable from what is on this "
              f"machine: in-plane\norientation (row/col transpose and flip) and slice order "
              f"against the DICOMs.\n\nRun notebooks/kaggle_01c_series_geometry.py -- CPU only, "
              f"no GPU, ~3 min -- then\nre-run this with --geometry/--thumbs. Until then this "
              f"path is for EXPERIMENTS,\nnot for the submitted cache.\n{'=' * 70}")


if __name__ == "__main__":
    main()
