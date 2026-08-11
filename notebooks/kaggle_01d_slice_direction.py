"""Per-series slice-direction predictors for all 24,371 series -- the K16 export.

**CPU ONLY. No GPU, therefore no lottery.** Minutes, not a session.

WHY THIS EXISTS. `validate_nifti.py` check 4b measures, on a stratified 51-series sample with
thumbnails, that **33% of the NIfTI series are stored back-to-front**: Axial 12/12 forward,
Coronal 14/18, Sagittal 8/21. The affine carries no direction cosines (check 1: position 0.0%,
orientation 0.0%), so `load_series_nifti` cannot know which, while `load_series` -- the test-time
path -- always sorts ascending by ImagePositionPatient projection. Train and test therefore
disagree on a third of series, in the axis medial/lateral depends on, and `PREPROCESS_VERSION`
cannot see it because the conversion happened upstream of the fingerprint. K16/K18 in
`IMPROVEMENTS.md`; the sagittal handedness fix is an XOR against this bit and cannot ship
without it.

WHAT IT ACTUALLY EXPORTS, AND WHY IT IS NOT THE ANSWER BY ITSELF. The converter is a third
party's, it is not observable from here, and the NIfTI files are not mounted in this notebook.
So this cannot measure the direction bit directly. What it exports is the **three candidate
predictors** of the converter's slice order, per series:

    inst_sign   does InstanceNumber order ascend in spatial projection?
    file_sign   does sorted-filename order ascend in spatial projection?
    loc_sign    does SliceLocation ascend in spatial projection?

`pipeline/resolve_slice_direction.py` then scores each against the 51 series whose direction the
thumbnails already settle, locally and for free. The predictor that explains those 51 is the
converter's sort key, and its column becomes the bit for all 24,371 series. A predictor that
explains only some of them is not adopted -- the fallback is to rebuild the cache from DICOM,
which is a fact worth learning for the price of one CPU notebook rather than after a training run.

WHY IT SAMPLES SLICES RATHER THAN READING EVERY HEADER. `PLAN.md` 9 Phase 0 step 2 budgets
"2-3 header reads per series (~50k opens, ~20 min)" against a full pass of "~700k opens, 3.7 h",
and notes the ~19 ms/open figure underneath both is itself unmeasured. It is now measured, from
this kernel's own predecessor: `rsna-knee-series-geometry` read every header of 396 series plus
180 pixel thumbnails in **108 s**, or 0.27 s/series -- so a full pass is ~1.8 h, not 3.7 h, and
still not worth paying. Every predictor above is a MONOTONE relationship, and the sign of a
monotone relationship is settled by a handful of points: SAMPLE=6 slices spread over the series
gives ~146k opens and a rank correlation whose magnitude reports its own confidence.

The sampled slices are taken by position in the filename listing, which is an arbitrary but
deterministic handle -- it need not be spatially meaningful, and whether it is happens to be
exactly what `file_sign` measures.

Writes /kaggle/working/slice_direction.csv. Download it into data/.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


def _bootstrap_preprocess() -> None:
    """Same glob-not-hardcode bootstrap as kaggle_01b/01c -- see its comment; it cost a session."""
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

from preprocess import build_study_index, find_competition_root  # noqa: E402

OUT = Path("/kaggle/working")
SAMPLE = 6            # headers per series; the sign of a monotone relation needs very few
WORKERS = 8           # I/O bound on a mounted dataset, so oversubscribe the 4 vCPUs


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, so ties (repeated SliceLocation in a multi-echo series) do not fake a sign."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    # average over tied groups
    s = a[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return r


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman. Returns 0.0 when either side is constant -- an honest 'this says nothing'."""
    if len(x) < 3:
        return 0.0
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    c = float(np.corrcoef(rx, ry)[0, 1])
    return 0.0 if np.isnan(c) else c


def probe(args) -> dict | None:
    """One series -> the three predictors. Header-only; no pixel data is decoded."""
    study, series, sdir = args
    sd = Path(sdir) / series
    try:
        files = sorted(sd.glob("*.dcm"))
    except OSError:
        return None
    if len(files) < 3:
        return None

    # Even spread over the listing. np.unique guards a series shorter than SAMPLE.
    idx = np.unique(np.linspace(0, len(files) - 1, SAMPLE).round().astype(int))
    picked = [files[i] for i in idx]

    heads = []
    for rank_in_listing, f in zip(idx, picked):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:                                         # noqa: BLE001
            continue
        heads.append((int(rank_in_listing), ds))
    if len(heads) < 3:
        return None

    # The normal comes from ImageOrientationPatient, and the series is only interpretable if the
    # sampled slices agree on it. A localiser or a multi-plane series that does not is reported
    # rather than silently averaged -- iop_consistent is a column, not a filter.
    iops = []
    for _, ds in heads:
        try:
            iops.append(np.asarray(ds.ImageOrientationPatient, float))
        except Exception:                                         # noqa: BLE001
            pass
    if not iops:
        return None
    iop = iops[0]
    iop_consistent = all(np.allclose(v, iop, atol=1e-3) for v in iops)
    normal = np.cross(iop[:3], iop[3:])
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-6:
        return None

    file_rank, proj, inst, loc = [], [], [], []
    for rank_in_listing, ds in heads:
        try:
            ipp = np.asarray(ds.ImagePositionPatient, float)
            p = float(np.dot(ipp, normal))
        except Exception:                                         # noqa: BLE001
            continue
        if not np.isfinite(p):
            continue
        file_rank.append(rank_in_listing)
        proj.append(p)
        try:
            inst.append(float(ds.InstanceNumber))
        except Exception:                                         # noqa: BLE001
            inst.append(np.nan)
        try:
            loc.append(float(ds.SliceLocation))
        except Exception:                                         # noqa: BLE001
            loc.append(np.nan)
    if len(proj) < 3:
        return None

    proj = np.asarray(proj)
    inst = np.asarray(inst)
    loc = np.asarray(loc)
    ok_i = np.isfinite(inst)
    ok_l = np.isfinite(loc)

    r_inst = _rho(inst[ok_i], proj[ok_i]) if ok_i.sum() >= 3 else 0.0
    r_file = _rho(np.asarray(file_rank, float), proj)
    r_loc = _rho(loc[ok_l], proj[ok_l]) if ok_l.sum() >= 3 else 0.0

    return {
        "StudyInstanceUID": study, "SeriesInstanceUID": series,
        "n_files": len(files), "n_sampled": len(proj),
        "iop_consistent": bool(iop_consistent),
        # sign: +1 means that ordering ascends in projection, i.e. "forward" under that key.
        # 0 means the sample could not tell, which downstream must treat as unknown, not forward.
        "inst_rho": r_inst, "inst_sign": int(np.sign(r_inst)),
        "file_rho": r_file, "file_sign": int(np.sign(r_file)),
        "loc_rho": r_loc, "loc_sign": int(np.sign(r_loc)),
        "proj_span": float(proj.max() - proj.min()),
    }


def main(root=None, out=None) -> None:
    root = Path(root) if root else find_competition_root()
    out = Path(out) if out else OUT
    ser = pd.read_csv(root / "train_series.csv")
    print(f"{len(ser):,} series / {ser.StudyInstanceUID.nunique():,} studies")

    index = build_study_index(root)
    print(f"indexed {len(index):,} study directories")

    work = []
    for r in ser.itertuples():
        sdir = index.get(r.StudyInstanceUID)
        if sdir is not None:
            work.append((r.StudyInstanceUID, r.SeriesInstanceUID, str(sdir)))
    print(f"{len(work):,} series resolve to a directory; probing {SAMPLE} headers each "
          f"(~{len(work) * SAMPLE:,} opens) on {WORKERS} workers\n")

    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for n, res in enumerate(ex.map(probe, work), 1):
            if res is not None:
                rows.append(res)
            if n % 2000 == 0:
                print(f"  {n:,}/{len(work):,}  ({len(rows):,} usable)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "slice_direction.csv", index=False)

    print(f"\n{'=' * 66}\nwrote slice_direction.csv ({len(df):,} series)\n{'=' * 66}")
    if df.empty:
        return
    print(f"  IOP consistent across sampled slices : {df.iop_consistent.mean():>6.1%}")
    for k in ("inst", "file", "loc"):
        sign, rho = df[f"{k}_sign"], df[f"{k}_rho"].abs()
        print(f"  {k:<5} ascending {(sign > 0).mean():>6.1%}  descending {(sign < 0).mean():>6.1%}"
              f"  undetermined {(sign == 0).mean():>6.1%}   |rho| median {rho.median():.3f}")
    # If a key is monotone it will be ~1.0 here; a key that is not monotone is not the sort key.
    print("\n  A sort key the converter actually used should be near-perfectly monotone in")
    print("  projection (|rho| = 1.0) for nearly every series. Anything else is a red herring.")
    print("\nNext: download slice_direction.csv into data/, then")
    print("  python pipeline/resolve_slice_direction.py")


if __name__ == "__main__":
    main()
