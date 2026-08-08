"""Header-only pass over EVERY study. Two things the 200-study audit surfaced but left open.

Runs in minutes and touches no pixel data, so it is worth doing before spending GPU quota on
the feature cache -- both answers change what that cache should contain.

  Q1  PATIENT GROUPING. (0010,0020) PatientID is present in 200/200 sampled DICOMs, but it is
      NOT in train.csv. fusion/folds.py currently groups on a hash of the report text as a
      patient proxy, which catches only the 150 studies that literally share a report. Real
      PatientIDs let the folds group properly, and bilateral knees or follow-ups on one patient
      stop leaking across the train/val boundary and inflating CV.

  Q2  LATERALITY FALLBACK. The audit found (0020,0060) Laterality in 163/200 studies -- but 64
      of those are EMPTY, so only 99/200 (49.5%) carry a usable value. Half the corpus cannot
      be canonicalised from the tag, and PLAN.md 3.2 is blunt about what that costs: 'medial'
      sits on opposite sides of the image for a left vs a right knee, so Medial/Lateral
      Meniscus and Medial/Lateral OA become noise no backbone can recover.

      The audit also reported ImagePositionPatient usable (118 negative x, 79 positive, both
      signs well separated, median -56). In the DICOM patient coordinate system +x runs toward
      the patient's LEFT, so the sign of x should indicate which knee. That is a HYPOTHESIS.
      This script tests it the only way that means anything: on the studies that have BOTH a
      tag and geometry, does the sign predict the tag? If agreement is high the rule can
      canonicalise the other half; if not, we need a pixel-based L/R classifier and should know
      that before building a cache against it.

Writes /kaggle/working/study_meta.csv (one row per study) + laterality_check.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

# Locate pipeline/preprocess.py wherever the code Dataset landed. Globbing beats hardcoding:
# the mount name depends on the Dataset slug, and a Dataset version still being created when the
# kernel starts can leave the path briefly absent -- which surfaced as a bare ModuleNotFoundError
# and cost a GPU session. Fail with a listing instead.
def _bootstrap_preprocess() -> None:
    import glob
    # Recursive: Kaggle nests sources under competitions/ and datasets/ when a kernel has more
    # than one, so the depth of the mount is not fixed. Measured 2026-08-07 -- /kaggle/input held
    # exactly ['competitions', 'datasets'] and a one-level glob found nothing.
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
FILES_PER_SERIES = 3       # majority vote; guards against one odd slice


def norm_side(v) -> str | None:
    """'R' / 'RIGHT' / 'r' -> 'R'. Empty or missing -> None.

    The audit saw both single-letter and full-word forms ('RIGHT' x2, 'LEFT' x1) alongside 64
    empty values, so an empty string must NOT be read as a side.
    """
    if v is None:
        return None
    s = str(v).strip().upper()
    return s[0] if s[:1] in ("L", "R") else None


def main() -> None:
    root = find_competition_root()
    series = pd.read_csv(root / "train_series.csv")
    by_study = series.groupby("StudyInstanceUID").SeriesInstanceUID.apply(list).to_dict()
    print(f"{len(by_study):,} studies / {len(series):,} series")

    index = build_study_index(root)             # one pass, not one walk per study
    print(f"indexed {len(index):,} study directories")

    rows = []
    for n, (study, sers) in enumerate(sorted(by_study.items()), 1):
        sdir = index.get(study)
        if sdir is None:
            continue
        pids, sides, xs = Counter(), Counter(), []
        for ser in sdir.iterdir():
            if not ser.is_dir():
                continue
            files = sorted(ser.glob("*.dcm"))[:FILES_PER_SERIES]
            for f in files:
                try:
                    ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                except Exception:
                    continue
                pid = getattr(ds, "PatientID", None)
                if pid:
                    pids[str(pid)] += 1
                s = norm_side(getattr(ds, "Laterality", None) or
                              getattr(ds, "ImageLaterality", None))
                if s:
                    sides[s] += 1
                try:
                    xs.append(float(np.asarray(ds.ImagePositionPatient, float)[0]))
                except Exception:
                    pass
        rows.append({
            "StudyInstanceUID": study,
            "PatientID": pids.most_common(1)[0][0] if pids else "",
            "laterality_tag": sides.most_common(1)[0][0] if sides else "",
            "x_median": float(np.median(xs)) if xs else np.nan,
            "n_series": len(sers),
        })
        if n % 250 == 0:
            print(f"  {n:,}/{len(by_study):,}")

    df = pd.DataFrame(rows)
    # +x is toward the patient's LEFT, so a knee sitting at negative x is the RIGHT knee.
    df["x_side"] = np.where(df.x_median.isna(), "",
                            np.where(df.x_median < 0, "R", "L"))
    df.to_csv(OUT / "study_meta.csv", index=False)

    # ---- Q1 -------------------------------------------------------------------------
    have_pid = (df.PatientID != "").sum()
    n_pat = df[df.PatientID != ""].PatientID.nunique()
    multi = df[df.PatientID != ""].groupby("PatientID").size()
    print(f"\n{'=' * 66}\nQ1 PATIENT GROUPING\n{'=' * 66}")
    print(f"  studies with a PatientID : {have_pid:,}/{len(df):,}")
    print(f"  distinct patients        : {n_pat:,}")
    print(f"  patients with >1 study   : {(multi > 1).sum():,} "
          f"(max {multi.max() if len(multi) else 0} studies)")
    print(f"  studies sharing a patient: {int((multi[multi > 1]).sum()):,}   "
          f"<- these leak across folds today")

    # ---- Q2 -------------------------------------------------------------------------
    both = df[(df.laterality_tag != "") & (df.x_side != "")]
    agree = (both.laterality_tag == both.x_side).mean() if len(both) else float("nan")
    print(f"\n{'=' * 66}\nQ2 LATERALITY FALLBACK\n{'=' * 66}")
    print(f"  usable tag      : {(df.laterality_tag != '').sum():,}/{len(df):,}")
    print(f"  usable geometry : {(df.x_side != '').sum():,}/{len(df):,}")
    print(f"  both            : {len(both):,}")
    print(f"\n  x-sign vs tag agreement: {agree:.3f}")
    if len(both):
        print("\n  confusion (rows = tag, cols = x_side):")
        print(pd.crosstab(both.laterality_tag, both.x_side).to_string())

    covered = ((df.laterality_tag != "") | (df.x_side != "")).mean()
    print(f"\n  coverage if the fallback is adopted: {100 * covered:.1f}% of studies")
    if agree > 0.95:
        verdict = "ADOPT -- sign predicts the tag; canonicalise the untagged half by geometry"
    elif agree > 0.8:
        verdict = ("MARGINAL -- better than nothing but it will mislabel a few percent. "
                   "Weigh against a pixel-based L/R classifier")
    else:
        verdict = ("REJECT -- geometry does not track the tag. De-identification has probably "
                   "shifted the coordinates. Needs a pixel L/R classifier (PLAN.md 3.2)")
    print(f"\n  VERDICT: {verdict}")

    (OUT / "laterality_check.json").write_text(json.dumps({
        "n_studies": int(len(df)), "with_patient_id": int(have_pid), "n_patients": int(n_pat),
        "studies_sharing_patient": int((multi[multi > 1]).sum()) if len(multi) else 0,
        "usable_tag": int((df.laterality_tag != "").sum()),
        "usable_geometry": int((df.x_side != "").sum()),
        "agreement": None if np.isnan(agree) else float(agree),
        "coverage_with_fallback": float(covered), "verdict": verdict,
    }, indent=2))
    print(f"\nwrote study_meta.csv ({len(df):,} rows) + laterality_check.json")


if __name__ == "__main__":
    main()
