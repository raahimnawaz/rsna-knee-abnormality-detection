"""5-fold split, stratified on the rarest labels. NOT grouped -- and that is a measured call.

PLAN.md 4 asks for MultilabelStratifiedGroupKFold "grouped by patient". Getting there took two
wrong turns, both worth recording because both are the obvious thing to do:

  1. train.csv has no patient column, so the first version grouped on a hash of the report
     text, reasoning that studies sharing a report are bilateral knees or follow-ups.
  2. The DICOM audit then found (0010,0020) PatientID in 200/200 headers, which looked like
     the real answer.

Both are wrong. kaggle_01b read PatientID for every study: **4,407 studies, 4,407 distinct
PatientIDs, zero patients with more than one study.** The IDs are de-identified per study, so
there is no patient linkage in this dataset at all.

And the report-hash groups are not patients either -- they are TEMPLATES. The largest group is
37 studies sharing one Turkish boilerplate normal report ('Diz eklemi içi sıvı miktarı normal.
Çapraz ve yan bağlar normal...'), i.e. 37 different people who got identical text. Forcing them
into one fold damaged fold balance (664-1,077 studies per fold) to prevent a leak that cannot
happen: the model consumes images, and the report is the target's source, never an input.

So there is nothing to group by, grouping on text made the folds worse, and plain multilabel
stratification is correct here. The GROUP_KEY hook stays as a single function so that if real
patient linkage ever appears it is a one-line change.

Stratification is greedy rather than exact. Exact multilabel stratification is NP-hard and the
standard Kaggle approach (iterative-stratification) has no grouped variant, so we do what its
grouped forks do: order groups by how constrained they are (rarest positive first), then assign
each to whichever fold is currently furthest below its target count on that group's labels.

Gold studies are pinned as `is_gold` in the output. Per PLAN.md 4 they stay identifiable in
every fold and every metric is reported on them separately -- pseudo-labels inherit the
extractor's biases and scoring against them flatters the model.
"""
import argparse
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


def group_key(row) -> str:
    """The default grouping key: the study itself -- i.e. no grouping.

    The module docstring explains why this is right for *patients* (there are none) and for
    *report templates* (the leak cannot occur for an image-only model). It is NOT right for
    scanners -- see `site_groups()`.
    """
    return row.StudyInstanceUID


def site_groups(uids: pd.Series) -> pd.Series:
    """Scanner-fingerprint grouping. `IMPROVEMENTS.md` §2i-a.

    This is the one grouping that survived scrutiny, and unlike the other two it is backed by a
    number rather than an argument: a public probe scores DICOM headers alone at 0.6516 under
    random folds and 0.5981 grouped on this fingerprint, so **0.053 of macro AUC is available
    from recognising the scanner instead of the knee**. A model given pixels can reach the same
    shortcut through image appearance, and ungrouped folds reward it.

    Not a replacement for the default. Report both: the ungrouped-minus-grouped gap is our own
    site-leakage number, and the grouped one is what the reproduction gate must use, because
    that gate compares our score against someone else's.
    """
    p = D / "site_fingerprint.csv"
    if not p.exists():
        sys.exit(f"{p} not found -- run: python pipeline/site_fingerprint.py")
    m = pd.read_csv(p).set_index("StudyInstanceUID").site_id
    g = uids.map(m)
    if g.isna().any():
        # A study with no header row cannot be grouped, and silently dropping it into a shared
        # "unknown" bucket would put every such study in one fold. Give each its own group,
        # which is the ungrouped behaviour for exactly those rows and nothing else.
        miss = g.isna()
        print(f"  {int(miss.sum())} studies have no header row; each becomes its own group")
        g = g.astype("object")
        g[miss] = ["nofp_" + u for u in uids[miss]]
    return g.astype(str)


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


def build(soft_labels: bool = True, group_by: str = "study",
          labels_csv: Path | None = None) -> pd.DataFrame:
    tr = pd.read_csv(D / "train.csv")
    if group_by == "site":
        tr["group"] = site_groups(tr.StudyInstanceUID).values
    else:
        tr["group"] = [group_key(r) for r in tr.itertuples()]
    tr["is_gold"] = tr[LABELS].notna().all(axis=1)

    # Stratify on the labels we will actually train against. Gold where it exists, otherwise
    # the extractor's soft targets thresholded -- stratification only needs the shape of the
    # positive distribution, not calibrated truth.
    y = tr[LABELS].to_numpy(dtype=float)
    if soft_labels:
        # Default moved off pseudo_labels.csv 2026-08-10: the rule extractor is retired as a
        # target source (IMPROVEMENTS §2f) and stratifying on it would shape the folds around
        # labels nothing trains on any more.
        pl = Path(labels_csv) if labels_csv else (D / "targets.csv")
        if not pl.exists() and not labels_csv:
            pl = D / "pseudo_labels.csv"
        if not pl.exists():
            sys.exit(f"{pl} missing -- see README Phase 0 step 1 (ship steven_v2 as targets.csv)")
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
    ap.add_argument("--group-by", choices=["study", "site"], default="study",
                    help="'site' groups on the scanner fingerprint (IMPROVEMENTS §2i-a). "
                         "Build BOTH: the gap between them is our site-leakage number, and "
                         "the grouped one is what the reproduction gate must use")
    ap.add_argument("--labels", default=None,
                    help="soft-target CSV for stratification; default data/targets.csv")
    args = ap.parse_args()

    folds = build(group_by=args.group_by, labels_csv=args.labels)
    folds.to_csv(args.out, index=False)
    report(folds)

    bad = folds.groupby("group").fold.nunique().max()
    if bad > 1:
        sys.exit(f"FAIL: a group landed in {bad} folds -- grouping is broken")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
