"""Turn the K16 header export into the per-series direction bit -- or refuse to.

`notebooks/kaggle_01d_slice_direction.py` exports three CANDIDATE predictors of the NIfTI
converter's slice order, for all 24,371 series: whether InstanceNumber, sorted-filename order or
SliceLocation ascends in spatial projection. None of them is the answer on its own. The converter
is a third party's and is not observable, so the only thing that can promote a candidate to
"the bit" is agreement with series whose direction is already known.

Those exist, and they are free: `validate_nifti.py` check 4b settles direction for the 51
stratified series that have DICOM thumbnails, by correlating NIfTI slice k=0 against the DICOM's
spatially-first and spatially-last slice. That set is small but it is ground truth, and the
question being asked of it is not "what is the average direction" -- which 51 samples could not
answer for 24,371 series -- but "which of three deterministic header rules reproduces it".
A rule that is genuinely the converter's sort key reproduces it 51/51; anything materially short
of that is a coincidence and must not be shipped.

    ADOPT_MIN  = 1.00 agreement, i.e. every single one
    n = 51 -> a rule with a true 90% accuracy passes this by chance with p = 0.9**51 = 0.5%

So this either writes `data/slice_direction_resolved.csv` with a `direction` column for every
series, or it writes nothing and says which rule got closest. Nothing downstream reads a
partially-trusted bit: `pipeline/slot_cache.py` refuses the sagittal anatomical slabs when the
file is absent, which is the honest failure, and the fallback stated in `PLAN.md` 9 Phase 0 is to
build the cache from DICOM instead.

    !! "Nothing downstream reads a partially-trusted bit" IS FALSE -- code review 2026-08-11,
    !! IMPROVEMENTS.md 2r-A1. slot_cache refuses only when the FILE is absent. A file that
    !! covers some series and not others -- which is what --measured produces, since it writes
    !! only series with NIfTI on disk, thumbnails present and a matching slice count -- passes
    !! the gate, and the uncovered series are built with slice_direction=None. Note also that
    !! this script's own `unknown` rows are dropped by load_direction(), making them
    !! indistinguishable from series it never measured.

    python pipeline/resolve_slice_direction.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "pipeline"))

from preprocess import nifti_geometry                      # noqa: E402
from validate_nifti import _orientations, thumb_of         # noqa: E402

D = PROJ / "data"
ADOPT_MIN = 1.00
RULES = ["inst", "file", "loc"]
THUMB = 32                     # must match kaggle_01e.THUMB -- the grids are correlated directly


def thumb32(a: np.ndarray) -> np.ndarray:
    """Line-for-line with kaggle_01e.thumb_of. Nearest-neighbour, 0.5/99.5 percentiles, uint8.

    Duplicated rather than imported because the Kaggle side is a single-file script and cannot
    import this one. The two must agree exactly: a different resample or a different intensity
    reference would put the DICOM and NIfTI thumbnails on different scales and the correlation
    would compare rendering rather than anatomy.
    """
    r = np.linspace(0, a.shape[0] - 1, THUMB).round().astype(int)
    c = np.linspace(0, a.shape[1] - 1, THUMB).round().astype(int)
    t = a[np.ix_(r, c)].astype(np.float32)
    lo, hi = np.percentile(t, [0.5, 99.5])
    return (np.clip((t - lo) / (hi - lo + 1e-6), 0, 1) * 255).round().astype(np.uint8)


def measure(nifti_dir: Path, idx: pd.DataFrame, thumbs) -> pd.DataFrame:
    """Read the bit off directly, per series: is NIfTI k=0 the DICOM's spatial first or its last?

    No rule and no extrapolation -- this is check 4b's arithmetic applied to every series that
    kaggle_01e shipped thumbnails for. `margin` is |r_first - r_last|, i.e. how far apart the two
    hypotheses actually landed, so a series where the two ends look alike reports its own
    weakness instead of contributing a coin flip as though it were a measurement.
    """
    import nibabel as nib
    out = []

    def corr(a, b) -> float:
        c = float(np.corrcoef(np.asarray(a, np.float32).ravel(),
                              np.asarray(b, np.float32).ravel())[0, 1])
        return 0.0 if np.isnan(c) else c

    for r in idx.itertuples():
        key = f"{r.StudyInstanceUID}_{r.SeriesInstanceUID}"
        if any(f"{key}|{t}" not in thumbs for t in ("first", "mid", "last")):
            continue
        p = nifti_dir / f"{key}.nii"
        if not p.exists():
            continue
        info = nifti_geometry(p)
        if info is None:
            continue
        arr = np.asanyarray(nib.load(str(p)).dataobj)
        vol = np.transpose(arr, (info["slice_axis"], info["row_axis"], info["col_axis"]))
        if len(vol) != int(r.n_slices):
            continue                      # different slice count: not comparable slice-to-slice

        # Layout first, direction second -- the middle slice is invariant to slice reversal, so
        # it identifies the in-plane layout without presupposing the answer. check 4 measured
        # 'as-is' winning for 98% of series across all six types, but it is resolved per series
        # here rather than assumed, at no cost.
        d_mid = thumbs[f"{key}|mid"]
        best = (-2.0, "")
        for name, cand in _orientations(np.asarray(vol[int(r.mid_index)], np.float32)):
            if cand.shape[0] < 2 or cand.shape[1] < 2:
                continue
            c = corr(thumb32(np.ascontiguousarray(cand)), d_mid)
            if c > best[0]:
                best = (c, name)
        if not best[1]:
            continue
        lay = dict(_orientations(np.asarray(vol[0], np.float32)))[best[1]]
        t0 = thumb32(np.ascontiguousarray(lay))
        c_first = corr(t0, thumbs[f"{key}|first"])
        c_last = corr(t0, thumbs[f"{key}|last"])
        out.append({"StudyInstanceUID": r.StudyInstanceUID,
                    "SeriesInstanceUID": r.SeriesInstanceUID,
                    "direction": "forward" if c_first >= c_last else "reversed",
                    "rule": "measured", "layout": best[1], "layout_r": best[0],
                    "rho": abs(c_first - c_last)})
    return pd.DataFrame(out)


def ground_truth(nifti_dir: Path, geom: pd.DataFrame, thumbs) -> pd.DataFrame:
    """The 51 series whose direction the thumbnails settle. Same arithmetic as check 4b.

    Duplicated deliberately rather than imported: check_orientation() prints a report and
    returns None, and refactoring it to return data would change the script that produced every
    direction number this project has quoted. The two must agree, so the port is line-for-line.
    """
    import nibabel as nib
    g = geom.set_index(geom.StudyInstanceUID + "_" + geom.SeriesInstanceUID)
    out = []

    def corr(a, b) -> float:
        c = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        return 0.0 if np.isnan(c) else c

    for key in sorted({k.split("|")[0] for k in thumbs}):
        p = nifti_dir / f"{key}.nii"
        if not p.exists() or key not in g.index:
            continue
        if any(f"{key}|{t}" not in thumbs for t in ("first", "mid", "last")):
            continue
        info = nifti_geometry(p)
        if info is None:
            continue
        arr = np.asanyarray(nib.load(str(p)).dataobj)
        vol = np.transpose(arr, (info["slice_axis"], info["row_axis"], info["col_axis"]))
        row = g.loc[key]
        if len(vol) != int(row.n_slices):
            continue

        d_mid = np.asarray(thumbs[f"{key}|mid"], dtype=np.float32)
        best = (-2.0, "")
        for name, cand in _orientations(np.asarray(vol[int(row.mid_index)], dtype=np.float32)):
            if cand.shape[0] < 2 or cand.shape[1] < 2:
                continue
            c = corr(thumb_of(np.ascontiguousarray(cand)), d_mid)
            if c > best[0]:
                best = (c, name)
        if not best[1]:
            continue
        lay = dict(_orientations(np.asarray(vol[0], dtype=np.float32)))[best[1]]
        t0 = thumb_of(np.ascontiguousarray(lay))
        c_first = corr(t0, np.asarray(thumbs[f"{key}|first"], dtype=np.float32))
        c_last = corr(t0, np.asarray(thumbs[f"{key}|last"], dtype=np.float32))
        out.append({"SeriesInstanceUID": row.SeriesInstanceUID,
                    "series_type": row.series_type,
                    "truth": "forward" if c_first >= c_last else "reversed",
                    "margin": abs(c_first - c_last)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", default=str(D / "slice_direction.csv"))
    ap.add_argument("--nifti", default=str(D / "nifti" / "nifti_train"))
    ap.add_argument("--geometry", default=str(D / "series_geometry.csv"))
    ap.add_argument("--thumbs", default=str(D / "series_thumbs.npz"))
    ap.add_argument("--out", default=str(D / "slice_direction_resolved.csv"))
    ap.add_argument("--measured", action="store_true",
                    help="read the bit off kaggle_01e's thumbnails directly, no rule to adopt")
    ap.add_argument("--measured-index", default=str(D / "direction_index.csv"))
    ap.add_argument("--measured-thumbs", default=str(D / "direction_thumbs.npz"))
    a = ap.parse_args()

    if a.measured:
        ip, tp = Path(a.measured_index), Path(a.measured_thumbs)
        if not (ip.exists() and tp.exists()):
            raise SystemExit(
                f"need {ip.name} and {tp.name}. Run notebooks/kaggle_01e_direction_measure.py "
                "on Kaggle (CPU), then\n  kaggle kernels output "
                "raahimnawaz/rsna-knee-direction-measure -p data/")
        idx = pd.read_csv(ip)
        res = measure(Path(a.nifti), idx, np.load(tp))
        print(__doc__.splitlines()[0])
        print(f"shipped {len(idx):,} series; {len(res):,} have NIfTI on disk and were measured")
        if res.empty:
            raise SystemExit("no overlap between the shipped series and the downloaded NIfTI")
        rev = (res.direction == "reversed").mean()
        print(f"  reversed {rev:.1%}   forward {1 - rev:.1%}")
        print(f"  in-plane layout 'as-is' for {(res.layout == 'as-is').mean():.1%} "
              f"(median r {res.layout_r.median():.4f})")
        # A near-zero margin means the two ends of that series look alike, so the call is weak.
        # Reported, and kept: 'unknown' would drop the series from the sagittal slabs entirely,
        # which is a worse trade than a 55/45 call on a series whose ends are near-symmetric.
        weak = (res.rho < 0.05).mean()
        print(f"  margin |r_first - r_last|: median {res.rho.median():.3f}, "
              f"{weak:.1%} under 0.05")
        res.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}: {len(res):,} series, measured not inferred")
        print("\nNext: SAGITTAL_LR=1 python pipeline/slot_cache.py --slots anatomical")
        return

    exp_path = Path(a.export)
    if not exp_path.exists():
        raise SystemExit(
            f"{exp_path} not found. Run notebooks/kaggle_01d_slice_direction.py on Kaggle "
            "(CPU, no GPU lottery) and\n  kaggle kernels output raahimnawaz/"
            "rsna-knee-slice-direction -p data/")
    exp = pd.read_csv(exp_path).drop_duplicates("SeriesInstanceUID").set_index(
        "SeriesInstanceUID")
    print(__doc__.splitlines()[0])
    print(f"export: {len(exp):,} series")
    for r in RULES:
        s = exp[f"{r}_sign"]
        print(f"  {r:<5} ascending {(s > 0).mean():>6.1%}  descending {(s < 0).mean():>6.1%}"
              f"  undetermined {(s == 0).mean():>6.1%}   |rho| median "
              f"{exp[f'{r}_rho'].abs().median():.3f}")

    truth = ground_truth(Path(a.nifti), pd.read_csv(a.geometry), np.load(a.thumbs))
    print(f"\nground truth from thumbnails: {len(truth)} series "
          f"({(truth.truth == 'forward').mean():.1%} forward)")
    if truth.empty:
        raise SystemExit("no thumbnailed series overlap the downloaded NIfTI; nothing to test")

    j = truth.join(exp, on="SeriesInstanceUID", how="inner")
    print(f"of those, {len(j)} appear in the export\n")

    print(f"{'rule':<8}{'agree':>10}{'n':>6}   (a rule that IS the sort key agrees 51/51)")
    best, best_acc = None, -1.0
    for r in RULES:
        # +1 sign == the ordering ascends in projection == NIfTI k=0 is the spatial first.
        pred = np.where(j[f"{r}_sign"] > 0, "forward",
                        np.where(j[f"{r}_sign"] < 0, "reversed", "unknown"))
        ok = pred == j.truth.to_numpy()
        acc = float(ok.mean())
        print(f"{r:<8}{acc:>9.1%}{len(j):>6}")
        if acc > best_acc:
            best, best_acc = r, acc

    if best_acc < ADOPT_MIN:
        print(f"\nNO RULE ADOPTED. Best is '{best}' at {best_acc:.1%}, under the {ADOPT_MIN:.0%} "
              "bar.\nA header rule that is not the converter's actual sort key is a coin flip "
              "dressed as a\nmeasurement, and it would land on the axis medial/lateral depends "
              "on. Nothing written.\nFallback (PLAN.md 9 Phase 0): build the cache from DICOM, "
              "or extend the export with more\ncandidate keys (AcquisitionNumber, "
              "SOPInstanceUID order, SeriesNumber).")
        raise SystemExit(1)

    print(f"\nADOPTED: '{best}' at {best_acc:.1%} over n={len(j)}.")
    sign = exp[f"{best}_sign"]
    res = pd.DataFrame({
        "SeriesInstanceUID": exp.index,
        "StudyInstanceUID": exp.StudyInstanceUID,
        "direction": np.where(sign > 0, "forward",
                              np.where(sign < 0, "reversed", "unknown")),
        "rule": best,
        "rho": exp[f"{best}_rho"].abs().to_numpy(),
    })
    res.to_csv(a.out, index=False)
    n_unk = int((res.direction == "unknown").sum())
    print(f"wrote {a.out}: {len(res):,} series "
          f"({(res.direction == 'reversed').mean():.1%} reversed, {n_unk} undetermined)")
    print("\nUndetermined series keep slice_direction=None, which leaves their slice axis "
          "alone --\nthe same honest default canonicalise() uses for unknown laterality.")
    print("\nNext: SAGITTAL_LR=1 python pipeline/slot_cache.py --slots anatomical")


if __name__ == "__main__":
    main()
