"""5-fold split: grouped by patient-proxy, stratified on the rarest labels.

PLAN.md 4 asks for MultilabelStratifiedGroupKFold "grouped by patient". **There is no patient
column.** train.csv is StudyInstanceUID, Report, and the twelve labels -- nothing else. So the
grouping key has to be inferred, and the only same-patient signal in the data is shared report
text: 131 studies repeat a report that appeared earlier, and IMPROVEMENTS.md 2b-i measured 177
studies involved in sharing overall. Those are bilateral knees or follow-ups on one patient.
Split them across folds and the model memorises a patient in train and is scored on them in
val, which inflates CV and shows up later as an unexplained CV/LB gap.

So: group on a hash of the normalised report text. It is a floor, not a solution -- two
studies on one patient with genuinely different reports still leak, and we cannot detect that.
Recorded as a known limitation rather than papered over.

Stratification is greedy rather than exact. Exact multilabel stratification is NP-hard and the
standard Kaggle approach (iterative-stratification) has no grouped variant, so we do what its
grouped forks do: order groups by how constrained they are (rarest positive first), then assign
each to whichever fold is currently furthest below its target count on that group's labels.

Gold studies are pinned as `is_gold` in the output. Per PLAN.md 4 they stay identifiable in
every fold and every metric is reported on them separately -- pseudo-labels inherit the
extractor's biases and scoring against them flatters the model.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
D = PROJ / "data"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
N_FOLDS = 5
SEED = 20260807


def group_key(report: str) -> str:
    """Patient proxy: hash of whitespace-normalised report text."""
    if not isinstance(report, str):
        return "nan"
    return hashlib.sha1(" ".join(report.split()).encode("utf-8")).hexdigest()[:16]


def assign(groups: list, y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Greedy label-balanced assignment of GROUPS (not rows) to folds.

    y is [n_groups, n_labels] of positive counts within each group. Study count is appended as
    an extra balancing column, otherwise the label terms swamp it and folds come out anywhere
    from 664 to 1,077 studies -- measured, on the first version of this function.

    Cost is the mean across columns of the standard deviation across folds of each column's
    share of its total. Normalising by the column total puts a 9-positive label and a
    4,407-study count column on the same scale, so no term can dominate by magnitude alone.

    Groups are placed largest-first: a 37-study group has no good home once the folds are
    nearly full, whereas a singleton always does.
    """
    rng = np.random.default_rng(seed)
    cols = np.concatenate([y, np.ones((len(y), 1))], axis=1)   # last column = study count
    total = np.maximum(cols.sum(0), 1.0)
    counts = np.zeros((n_folds, cols.shape[1]))

    rarity = 1.0 / np.maximum(y.sum(0), 1)
    priority = (cols[:, -1] * 1000.0                        # group size dominates
                + (y * rarity[None, :]).sum(1)              # then rarest-label content
                + rng.random(len(y)) * 1e-9)                # ties broken randomly, not by name
    order = np.argsort(-priority)

    out = np.full(len(groups), -1, dtype=int)
    for gi in order:
        best_f, best_cost = 0, np.inf
        for f in range(n_folds):
            counts[f] += cols[gi]
            cost = float((counts / total).std(axis=0).mean())
            counts[f] -= cols[gi]
            if cost < best_cost:
                best_f, best_cost = f, cost
        out[gi] = best_f
        counts[best_f] += cols[gi]
    return out


def build(soft_labels: bool = True) -> pd.DataFrame:
    tr = pd.read_csv(D / "train.csv")
    tr["group"] = tr.Report.map(group_key)
    tr["is_gold"] = tr[LABELS].notna().all(axis=1)

    # Stratify on the labels we will actually train against. Gold where it exists, otherwise
    # the extractor's soft targets thresholded -- stratification only needs the shape of the
    # positive distribution, not calibrated truth.
    y = tr[LABELS].to_numpy(dtype=float)
    if soft_labels:
        pl = D / "pseudo_labels.csv"
        if not pl.exists():
            sys.exit("data/pseudo_labels.csv missing -- run extractor/run_extract.py first")
        soft = (pd.read_csv(pl).set_index("StudyInstanceUID")[LABELS]
                  .reindex(tr.StudyInstanceUID).to_numpy(dtype=float))
        gold_rows = tr.is_gold.to_numpy()[:, None]
        y = np.where(gold_rows, y, soft >= 0.5)
    y = np.nan_to_num(y, nan=0.0)

    gdf = pd.DataFrame(y, columns=LABELS)
    gdf["group"] = tr.group.values
    agg = gdf.groupby("group", sort=True)[LABELS].sum()
    fold_of_group = assign(list(agg.index), agg.values, N_FOLDS, SEED)
    gmap = dict(zip(agg.index, fold_of_group))

    out = tr[["StudyInstanceUID", "is_gold", "group"]].copy()
    out["fold"] = out.group.map(gmap).astype(int)
    return out


def report(folds: pd.DataFrame) -> None:
    tr = pd.read_csv(D / "train.csv")
    m = folds.merge(tr, on="StudyInstanceUID")
    print(f"\n{len(folds):,} studies / {folds.group.nunique():,} groups "
          f"({len(folds) - folds.group.nunique()} studies share a group)")

    multi = folds.groupby("group").size()
    print(f"groups with >1 study: {(multi > 1).sum()}  (max {multi.max()} studies)")
    leaked = folds.groupby("group").fold.nunique()
    print(f"groups split across folds: {(leaked > 1).sum()}  <- must be 0")

    print(f"\n{'fold':<6}{'n':>7}{'gold':>7}   " + "".join(f"{l[:9]:>10}" for l in LABELS[:6]))
    for f in range(N_FOLDS):
        s = m[m.fold == f]
        rates = "".join(f"{100 * s[l].mean(skipna=True):>9.1f}%" for l in LABELS[:6])
        print(f"{f:<6}{len(s):>7}{int(s.is_gold.sum()):>7}   {rates}")
    print("  (gold-label positive rates, first 6 labels; blanks are the 4,349 unlabelled)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(D / "folds.csv"))
    args = ap.parse_args()

    folds = build()
    folds.to_csv(args.out, index=False)
    report(folds)

    bad = folds.groupby("group").fold.nunique().max()
    if bad > 1:
        sys.exit(f"FAIL: a group landed in {bad} folds -- grouping is broken")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
