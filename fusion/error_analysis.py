"""Open the black box: where does the 0.891 ensemble actually fail, and on whom?

WHY THIS EXISTS. Until §2y the fork was a leaderboard number. It now has honest OOF over all
4,407 studies locally (`fusion/runs_pilkwang`), which makes it auditable for the first time. This
asks three things no leaderboard score can answer:

  1. **Which labels carry the loss?** A macro over 12 is dominated by its worst few, and rare
     labels have noisy AUCs. Prevalence is reported beside every number for that reason.
  2. **Is there a LATERALITY defect?** Every one of the 20 members runs `rules: {lat: 'centre'}`
     -- no mirroring of left knees onto right. But *medial and lateral swap sides* between a left
     and a right knee, so the four compartment labels (Medial/Lateral Meniscus, Medial/Lateral OA)
     have to be learned twice from a corpus that is ~57/43 R/L. The eight non-compartment labels
     are the built-in CONTROL: they should show no laterality gap. **A defect shows as a
     compartment-vs-control DIFFERENCE, not as a raw L/R gap** -- left knees could simply be
     rarer or sicker, and that would move every label equally.
  3. **Which acquisition subgroups are underserved?** Sex, field strength and manufacturer, from
     the published DICOM header dump.

    python fusion/error_analysis.py

Scored with the `score_oof.py` definition -- `lixin_gpt56`, non-gold studies, binarised at 0.5.
Both arms of every comparison are the same model, so the reference's bias cancels; this measures
*disparity*, which is exactly the case a non-neutral reference still supports.

CAUTION ON SUBGROUP AUCs. A subgroup AUC is only as stable as its positive count, and §3b is the
standing warning about acting on small-sample differences. Every row carries n and n_pos, gaps are
bootstrapped, and nothing here should be selected on without the leaderboard confirming it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "extractor"))
from metrics import auc  # noqa: E402

D = PROJ / "data"
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
COMPARTMENT = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]
NEUTRAL = D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56" / \
    "labels_llm_gpt56sol.csv"


def boot_gap(y, p, a, b, n=300, seed=0):
    """Bootstrap the AUC gap between two disjoint groups, resampling within each."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ia = rng.choice(a, len(a), replace=True)
        ib = rng.choice(b, len(b), replace=True)
        if len(np.unique(y[ia])) < 2 or len(np.unique(y[ib])) < 2:
            continue
        out.append(auc(y[ia], p[ia]) - auc(y[ib], p[ib]))
    return (float(np.mean(out)), float(np.std(out))) if out else (np.nan, np.nan)


def subgroup(name, keys, y, P, ids, min_n=150):
    vals = pd.Series(keys).value_counts()
    vals = [v for v, c in vals.items() if c >= min_n and str(v) not in ("nan", "")]
    if len(vals) < 2:
        print(f"\n### {name}: not enough groups\n")
        return
    vals = vals[:2]
    a = np.where(np.asarray(keys) == vals[0])[0]
    b = np.where(np.asarray(keys) == vals[1])[0]
    print(f"\n### {name}:  {vals[0]} (n={len(a)})  vs  {vals[1]} (n={len(b)})")
    print(f"  {'label':<18}{'n_pos':>7}{str(vals[0])[:9]:>11}{str(vals[1])[:9]:>11}"
          f"{'gap':>9}{'sigma':>7}")
    rows = {}
    for i, lab in enumerate(L):
        ya, yb = y[a, i], y[b, i]
        if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
            continue
        va, vb = auc(ya, P[a, i]), auc(yb, P[b, i])
        g, sd = boot_gap(y[:, i], P[:, i], a, b)
        rows[lab] = g
        sig = abs(g) / sd if sd else 0
        star = "  <<" if sig >= 2 else ""
        print(f"  {lab:<18}{int(y[:, i].sum()):>7}{va:>11.3f}{vb:>11.3f}"
              f"{g:>+9.3f}{sig:>7.1f}{star}")
    if rows:
        comp = [rows[c] for c in COMPARTMENT if c in rows]
        ctrl = [v for k, v in rows.items() if k not in COMPARTMENT]
        if comp and ctrl:
            print(f"  {'-'*60}")
            print(f"  mean gap  compartment {np.mean(comp):+.4f}   "
                  f"control {np.mean(ctrl):+.4f}   "
                  f"DIFFERENCE {np.mean(comp)-np.mean(ctrl):+.4f}")


def main() -> None:
    oof = pd.read_csv(PROJ / "fusion" / "runs_pilkwang" / "oof_all.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")
    ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    gold = set(pd.read_csv(D / "train.csv").dropna(subset=L).StudyInstanceUID)

    ids = [u for u in oof.index if u in ref.index and u not in gold]
    yv = ref.loc[ids, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~np.isnan(yv).any(axis=1)
    ids = [u for u, k in zip(ids, ok) if k]
    y = (yv[ok] > 0.5).astype(float)
    P = oof.loc[ids, L].to_numpy(float)
    print(f"pilkwang 20-member OOF · reference {NEUTRAL.name} · gold excluded · n={len(ids)}")

    print("\n### 1. Where the macro actually loses")
    print(f"  {'label':<18}{'n_pos':>7}{'prev':>8}{'AUC':>9}")
    per = {}
    for i, lab in enumerate(L):
        per[lab] = auc(y[:, i], P[:, i])
        print(f"  {lab:<18}{int(y[:, i].sum()):>7}{y[:, i].mean():>8.3f}{per[lab]:>9.3f}")
    print(f"  {'-'*42}\n  {'MACRO':<18}{'':>15}{np.mean(list(per.values())):>9.4f}")
    worst = sorted(per.items(), key=lambda x: x[1])[:3]
    print(f"  weakest three: " + ", ".join(f"{k} {v:.3f}" for k, v in worst))

    meta = pd.read_csv(D / "study_meta.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")
    subgroup("2. LATERALITY (members run lat='centre', no mirroring)",
             meta.reindex(ids)["x_side"].tolist(), y, P, ids)

    try:
        import pyarrow.parquet as pq
        h = pq.read_table(D / "external" / "dicom_headers_zhukovoleksiy.parquet",
                          columns=["StudyInstanceUID", "PatientSex", "MagneticFieldStrength",
                                   "Manufacturer"]).to_pandas()
        h = h.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(ids)
        subgroup("3. PATIENT SEX", h["PatientSex"].tolist(), y, P, ids)
        fs = pd.to_numeric(h["MagneticFieldStrength"], errors="coerce").round(1)
        subgroup("4. FIELD STRENGTH (T)", fs.tolist(), y, P, ids)
        subgroup("5. MANUFACTURER", h["Manufacturer"].tolist(), y, P, ids)
    except Exception as e:                                   # noqa: BLE001
        print(f"\n  header subgroups unavailable: {e}")


if __name__ == "__main__":
    main()
