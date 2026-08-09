"""Per-series DICOM geometry + thumbnails, so the NIfTI conversion can be validated locally.

**CPU ONLY. No GPU, therefore no lottery.** Minutes, not a session. This is the last thing the
train-side cache needs from Kaggle (PLAN.md 9.1).

WHY THIS EXISTS. The corpus repackaged as one NIfTI per series
(`davidadekanmi/rsna-knee-nifti-part1..12`) removes the ~19 ms/open latency wall that killed
five cache attempts. But that conversion happened UPSTREAM of `PREPROCESS_VERSION`, so the
fingerprint structurally cannot detect a disagreement with `load_series()`. Train features would
flow through the NIfTI reader and test features through the DICOM reader, and if the two
disagree on slice order or in-plane orientation the model scores badly with no other symptom.

`pipeline/validate_nifti.py` settles what it can locally against `study_meta.csv` from
`kaggle_01b`: whether the affine carries a patient frame at all (it does not -- diagonal spacing,
zero translation), and whether laterality can still be recovered (it can, from study_meta rather
than from the pixels). Two things it cannot settle from any header, precisely because that frame
is missing:

  IN-PLANE ORIENTATION  the row/col transpose, and either axis possibly flipped. 8 combinations.
  SLICE ORDER           whether the NIfTI's stored k order matches the DICOM spatial sort.

This exports what resolves both.

WHY IT ALSO SHIPS THUMBNAILS. `ImageOrientationPatient` resolves the orientation *if the
converter wrote a faithful affine*. A converter that wrote a wrong affine over correct pixels --
or the reverse -- would pass a header-only check and still be wrong, which is precisely the
class of failure this whole exercise is guarding against. Measured 2026-08-09, the affine in
this corpus is diagonal spacing with ZERO translation and no rotation, so header-only checking
is not merely weak here, it is impossible. So it writes three 64x64 slices per sampled series --
first, middle and last in spatial order. Correlating the middle against the NIfTI over all 8
transpose/flip layouts settles orientation; matching NIfTI slice 0 against first-vs-last settles
slice direction. ~1.5 MB.

WHY IT SAMPLES. Reading every header is ~700k opens at ~19 ms -- 3.7 h, the exact wall this is
meant to route around. Orientation conventions are systematic, not per-study: they either match
or they do not, and 400 series stratified over all six types is decisive for that. It reads ALL
slice headers within a sampled series, because the spatial sort is the thing being checked and a
partial series cannot establish it.

    SERIES = 400        # ~8,800 header opens, ~3 min at the measured latency
    THUMBS = 60         # pixel decode, the only part that touches pixel data

Writes /kaggle/working/series_geometry.csv + series_thumbs.npz. Download both into data/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


def _bootstrap_preprocess() -> None:
    """Same glob-not-hardcode bootstrap as kaggle_01b -- see its comment; it cost a session."""
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
SERIES = 400          # sampled series for geometry
THUMBS = 60           # of those, how many also get a pixel thumbnail
THUMB = 64            # thumbnail side
SEED = 20260809


def thumb_of(a: np.ndarray) -> np.ndarray:
    """[H,W] -> [THUMB,THUMB] float16 in [0,1], nearest-neighbour and percentile-normalised.

    Nearest-neighbour on purpose: the local side reproduces this exact operation on the NIfTI
    slice, so any interpolation would have to match bit-for-bit to be comparable. Intensity uses
    the same 0.5/99.5 percentiles as normalise_and_resample, so the two are on one scale.
    """
    r = np.linspace(0, a.shape[0] - 1, THUMB).round().astype(int)
    c = np.linspace(0, a.shape[1] - 1, THUMB).round().astype(int)
    t = a[np.ix_(r, c)].astype(np.float32)
    lo, hi = np.percentile(t, [0.5, 99.5])
    return np.clip((t - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float16)


def main(root=None, out=None) -> None:
    root = Path(root) if root else find_competition_root()
    out = Path(out) if out else OUT
    ser = pd.read_csv(root / "train_series.csv")
    print(f"{len(ser):,} series / {ser.StudyInstanceUID.nunique():,} studies")

    # Stratify over the six series types (3 planes x fluid-sensitive) so no type is unchecked --
    # Axial nonFS is only 19.4% of studies (FINDINGS.md 3.2) and would be thin under a blind
    # sample, yet it is a type the fusion head has an embedding for.
    # NB the column must not start with an underscore: DataFrame.itertuples() renames any
    # column that is not a valid Python identifier to a positional `_1`, `_2`, ... and a
    # leading underscore qualifies. `r._type` then raises AttributeError deep in the loop,
    # ~5 minutes into the run, after the study index has already been built. It did.
    ser["series_type"] = ser.Anatomical_Plane.astype(str) + "_" + ser.Fluid_Sensitive.astype(str)
    per = max(SERIES // max(ser.series_type.nunique(), 1), 1)
    # Explicit index sampling rather than groupby.apply: apply on the grouping column is
    # deprecated, and include_groups=False would drop the very column being selected on.
    idx = []
    for _t, grp in ser.groupby("series_type"):
        idx += list(grp.sample(min(len(grp), per), random_state=SEED).index)
    pick = ser.loc[idx].reset_index(drop=True)
    # SHUFFLE. `idx` is accumulated per group, so without this the first THUMBS usable series are
    # all from the first group -- v2 of this script thumbnailed 60 series and every one was
    # Axial_0, leaving Coronal and Sagittal (where ACL, menisci and MCL live) completely
    # unvalidated while the report claimed "100% of series". The thumbnails are the ONLY
    # instrument for orientation, so their coverage is the validation's coverage.
    pick = pick.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"sampled {len(pick):,} series over {pick.series_type.nunique()} types:")
    print(pick.series_type.value_counts().to_string())

    index = build_study_index(root)
    print(f"indexed {len(index):,} study directories")

    rows, thumbs = [], {}
    for n, r in enumerate(pick.itertuples(), 1):
        sdir = index.get(r.StudyInstanceUID)
        if sdir is None:
            continue
        sd = sdir / r.SeriesInstanceUID
        if not sd.is_dir():
            continue
        files = sorted(sd.glob("*.dcm"))
        if len(files) < 3:
            continue

        heads = []
        for f in files:
            try:
                heads.append((f, pydicom.dcmread(f, stop_before_pixels=True, force=True)))
            except Exception:
                pass
        if len(heads) < 3:
            continue

        try:
            iop = np.asarray(heads[0][1].ImageOrientationPatient, float)
            normal = np.cross(iop[:3], iop[3:])
        except Exception:
            continue

        # The spatial sort -- the same one slice_order() does, and the thing being validated.
        keyed = []
        for f, ds in heads:
            try:
                ipp = np.asarray(ds.ImagePositionPatient, float)
                keyed.append((float(np.dot(ipp, normal)), f, ds, ipp))
            except Exception:
                pass
        if len(keyed) < 3:
            continue
        keyed.sort(key=lambda t: t[0])

        first, last = keyed[0], keyed[-1]
        mid = keyed[len(keyed) // 2]
        try:
            ps = np.asarray(heads[0][1].PixelSpacing, float)
        except Exception:
            ps = np.array([np.nan, np.nan])

        rows.append({
            "StudyInstanceUID": r.StudyInstanceUID, "SeriesInstanceUID": r.SeriesInstanceUID,
            "series_type": r.series_type, "n_slices": len(keyed),
            "rows": int(getattr(heads[0][1], "Rows", 0)),
            "cols": int(getattr(heads[0][1], "Columns", 0)),
            "pixel_spacing_r": float(ps[0]), "pixel_spacing_c": float(ps[-1]),
            "slice_thickness": float(getattr(heads[0][1], "SliceThickness", np.nan) or np.nan),
            # ImageOrientationPatient: first triple is the ROW direction cosine (increasing
            # column index), second is the COLUMN direction (increasing row index). Both LPS.
            **{f"iop{i}": float(v) for i, v in enumerate(iop)},
            **{f"ipp_first{i}": float(v) for i, v in enumerate(first[3])},
            **{f"ipp_last{i}": float(v) for i, v in enumerate(last[3])},
            "proj_first": first[0], "proj_last": last[0],
            "x_median": float(np.median([k[3][0] for k in keyed])),
            "mid_index": len(keyed) // 2,
            "laterality_tag": str(getattr(heads[0][1], "Laterality", "") or "").strip().upper()[:1],
        })

        # THREE thumbnails per series, in spatial order: first, middle, last.
        #
        # The middle one alone settles in-plane orientation. It cannot settle slice DIRECTION,
        # because the middle of a reversed sequence is still the middle -- and with the NIfTI
        # affine carrying no rotation (measured 2026-08-09) direction is otherwise unknowable.
        # Matching NIfTI slice 0 against DICOM-first vs DICOM-last is what resolves it, and a
        # back-to-front volume destroys the slice transformer's positional signal silently.
        if len([k for k in thumbs if k.endswith("|mid")]) < THUMBS:
            key = f"{r.StudyInstanceUID}_{r.SeriesInstanceUID}"
            for tag, sl in (("first", first), ("mid", mid), ("last", last)):
                a = read_pixels(sl[1])
                if a is not None and a.ndim == 2:
                    thumbs[f"{key}|{tag}"] = thumb_of(a)

        if n % 50 == 0:
            print(f"  {n:,}/{len(pick):,}  ({len(rows):,} usable, {len(thumbs)} thumbs)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "series_geometry.csv", index=False)
    np.savez_compressed(out / "series_thumbs.npz", **thumbs)

    n_ser_thumbed = len({k.split('|')[0] for k in thumbs})
    print(f"\n{'=' * 66}\nwrote series_geometry.csv ({len(df):,} series) + "
          f"series_thumbs.npz ({len(thumbs)} slices over {n_ser_thumbed} series)\n{'=' * 66}")
    if len(df):
        asc = (df.proj_last > df.proj_first).mean()
        print(f"  slices/series : median {df.n_slices.median():.0f}  "
              f"range {df.n_slices.min()}-{df.n_slices.max()}")
        print(f"  in-plane mm   : median {df.pixel_spacing_r.median():.3f}")
        print(f"  sorted order ascends in projection: {asc:.1%}  (must be 100% -- it is the sort)")
        print(f"  distinct IOP rows: {df[[f'iop{i}' for i in range(6)]].round(3).drop_duplicates().shape[0]}"
              f"  <- if this is ~3 the protocol is clean per plane")
    print("\nNext: download both files into data/, then\n  python pipeline/validate_nifti.py "
          "--geometry data/series_geometry.csv --thumbs data/series_thumbs.npz")


def self_test() -> None:
    """Drive the WHOLE loop on synthetic DICOMs. Seconds locally, and it guards a session.

    Version 1 of this script died on Kaggle after ~5 minutes -- the study index had already
    been walked -- with `AttributeError: 'Pandas' object has no attribute '_type'`, because
    DataFrame.itertuples() renames any column whose name is not a valid Python identifier and
    a leading underscore disqualifies it. A single pass over three fake studies would have
    caught it instantly. kaggle_02 has --self-test for exactly this reason; this did not.

    The fixture writes real DICOMs with the tags the loop actually reads -- orientation,
    position, spacing, pixel data -- including a series whose files are deliberately named out
    of spatial order, so the projection sort is exercised rather than assumed.
    """
    import shutil
    import tempfile
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    tmp = Path(tempfile.mkdtemp(prefix="k01c_selftest_"))
    try:
        rows, planes = [], ["Axial", "Coronal", "Sagittal"]
        for si in range(3):
            study = f"1.2.3.{si}"
            for ci in range(2):
                series = f"1.2.3.{si}.{ci}"
                plane = planes[si % 3]
                rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                             "Fluid_Sensitive": ci, "Fat_Suppression": ci,
                             "Anatomical_Plane": plane})
                d = tmp / "train_series" / study / series
                d.mkdir(parents=True, exist_ok=True)
                n = 5
                for k in range(n):
                    ds = Dataset()
                    ds.file_meta = FileMetaDataset()
                    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
                    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
                    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
                    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                    # z advances with k; x is constant per study so laterality is well defined
                    ds.ImagePositionPatient = [-150.0 + si, 10.0, 3.0 * k]
                    ds.PixelSpacing = [0.33, 0.33]
                    ds.SliceThickness = 3.0
                    ds.Laterality = "R"
                    ds.Rows, ds.Columns = 16, 16
                    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
                    ds.SamplesPerPixel, ds.PixelRepresentation = 1, 0
                    ds.PhotometricInterpretation = "MONOCHROME2"
                    a = np.zeros((16, 16), np.uint16)
                    a[k % 16, :] = 1000 + k          # slice-dependent, so thumbs differ
                    ds.PixelData = a.tobytes()
                    # Reverse the FILENAME order against the spatial order, so a loop that
                    # trusted sorted(glob) instead of the projection sort would be caught.
                    pydicom.dcmwrite(d / f"{n - 1 - k:03d}.dcm", ds, enforce_file_format=False)
        pd.DataFrame(rows).to_csv(tmp / "train_series.csv", index=False)

        main(root=tmp, out=tmp)

        g = pd.read_csv(tmp / "series_geometry.csv")
        assert len(g) == 6, f"expected 6 series, got {len(g)}"
        assert (g.n_slices == 5).all(), g.n_slices.tolist()
        assert (g.proj_last > g.proj_first).all(), "projection sort did not order the slices"
        # The stratification is the reason this script samples at all, so assert it landed --
        # and assert on series_type specifically, which is the attribute itertuples mangled.
        assert set(g.series_type) == {f"{p}_{c}" for p in planes for c in (0, 1)}, \
            f"stratification lost a type: {sorted(set(g.series_type))}"
        assert g.laterality_tag.eq("R").all(), "Laterality tag not read"
        with np.load(tmp / "series_thumbs.npz") as z:
            keys = list(z.files)
            assert len(keys) == 18, f"expected 3 thumbs x 6 series, got {len(keys)}"
            for tag in ("first", "mid", "last"):
                assert sum(k.endswith(f"|{tag}") for k in keys) == 6, tag
            first = z[[k for k in keys if k.endswith("|first")][0]]
            last = z[[k for k in keys if k.endswith("|last")][0]]
            assert first.shape == (THUMB, THUMB), first.shape
            assert not np.allclose(first, last), \
                "first and last thumbnails are identical -- slice direction is untestable"
        print("\nself-test PASSED -- 6 series, 18 thumbnails, projection sort verified")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
