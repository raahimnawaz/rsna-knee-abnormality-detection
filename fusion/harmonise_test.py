"""Does per-scanner rank harmonisation of the OUTPUT move macro-AUROC? A re-rank, not a calibration.

WHY THIS EXISTS, AND WHY IT IS NOT KILLED BY §3a's FILTER. §3a established that any per-label
*monotone* transform of finished predictions is worth exactly zero to AUC. A **group-conditional**
transform is not one: normalising scores within scanner group reorders studies **across** groups,
so it can change the metric. That is the loophole, and it is the only post-hoc lever we have found
that survives the filter.

The motivation is measured, not borrowed. `fusion/error_analysis.py` finds the model's largest
disparities are by MANUFACTURER -- Synovitis 0.848 Siemens vs 0.914 GE, Effusion 0.841 vs 0.892,
Fracture 0.897 vs 0.946 -- larger than by sex, field strength or laterality. The harmonisation
literature independently reports scanner manufacturer as *the* dominant site effect, with ComBat
as the standard remedy. We hold 265 scanner fingerprints (`data/site_fingerprint.csv`) plus
manufacturer and field strength from the published header dump, and now hold the fork's OOF, so
this is testable locally with no GPU and no submission.

    python fusion/harmonise_test.py

METHOD. Within each group, replace each study's score for a label by its within-group rank in
[0,1], then pool. This removes any group-level location/scale difference with **no fitted
parameters**, so unlike a ComBat fit it cannot overfit -- which matters because §3b showed
selection on small groups is actively harmful, and 265 fingerprints over ~4,300 studies is ~16
studies each.

THE TENSION THIS MEASURES. Within-group ranking assumes the group's score *distribution* is an
artefact. If prevalence genuinely differs by scanner -- a site that scans sicker knees -- then the
shift is signal and removing it must HURT. Groups are therefore reported with their prevalence, so
a gain or loss can be read against whether case mix actually differs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "extractor"))
from metrics import auc  # noqa: E402

D = PROJ / "data"
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
NEUTRAL = D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56" / \
    "labels_llm_gpt56sol.csv"


def macro(y, P):
    return float(np.nanmean([auc(y[:, i], P[:, i]) for i in range(len(L))]))


def harmonise(P, groups, min_n=30):
    """Within-group rank in [0,1] per label. Groups smaller than min_n are pooled as 'other'."""
    g = pd.Series(groups).astype(str).fillna("nan").to_numpy()
    counts = pd.Series(g).value_counts()
    small = {k for k, c in counts.items() if c < min_n}
    g = np.array(["__other__" if x in small else x for x in g])
    out = np.empty_like(P, dtype=float)
    for grp in np.unique(g):
        m = np.where(g == grp)[0]
        for i in range(P.shape[1]):
            out[m, i] = rankdata(P[m, i]) / len(m)
    return out, g


def boot_delta(y, Pa, Pb, n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        d.append(macro(y[k], Pb[k]) - macro(y[k], Pa[k]))
    return float(np.mean(d)), float(np.std(d))


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
    base = macro(y, P)
    print(f"reference {NEUTRAL.name} · gold excluded · n={len(ids)}")
    print(f"  baseline macro (pilkwang OOF as shipped): {base:.4f}\n")

    site = pd.read_csv(D / "site_fingerprint.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID").reindex(ids)
    keys = {"site_id (265 fingerprints)": site["site_id"].tolist()}
    try:
        import pyarrow.parquet as pq
        h = pq.read_table(D / "external" / "dicom_headers_zhukovoleksiy.parquet",
                          columns=["StudyInstanceUID", "Manufacturer",
                                   "MagneticFieldStrength", "ManufacturerModelName"]
                          ).to_pandas().drop_duplicates(
            "StudyInstanceUID").set_index("StudyInstanceUID").reindex(ids)
        keys["Manufacturer"] = h["Manufacturer"].tolist()
        keys["Manufacturer x FieldStrength"] = (
            h["Manufacturer"].astype(str) + "|"
            + pd.to_numeric(h["MagneticFieldStrength"], errors="coerce").round(1).astype(str)
        ).tolist()
        keys["ManufacturerModelName"] = h["ManufacturerModelName"].tolist()
    except Exception as e:                                     # noqa: BLE001
        print(f"  headers unavailable: {e}")

    print(f"  {'grouping':<32}{'groups':>8}{'macro':>9}{'delta':>9}{'sigma':>7}")
    for name, k in keys.items():
        Ph, g = harmonise(P, k)
        m = macro(y, Ph)
        d, sd = boot_delta(y, P, Ph)
        print(f"  {name:<32}{len(np.unique(g)):>8}{m:>9.4f}{d:>+9.4f}"
              f"{(abs(d)/sd if sd else 0):>7.1f}")

    # Does case mix actually differ by manufacturer? If it does, harmonising destroys signal.
    if "Manufacturer" in keys:
        print("\n  prevalence by manufacturer (is the shift artefact or case mix?)")
        gm = pd.Series(keys["Manufacturer"]).astype(str)
        top = [v for v, c in gm.value_counts().items() if c >= 300][:3]
        print(f"    {'label':<18}" + "".join(f"{str(t)[:11]:>13}" for t in top))
        for i, lab in enumerate(L):
            row = "".join(f"{y[np.where(gm == t)[0], i].mean():>13.3f}" for t in top)
            print(f"    {lab:<18}{row}")


if __name__ == "__main__":
    main()
