"""5-fold study split, stratified on the rarest labels, optionally grouped by scanner site."""
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
    """The default grouping key: the study itself -- i.e. no grouping."""
    return row.StudyInstanceUID


def site_groups(uids: pd.Series) -> pd.Series:
    """Scanner-fingerprint grouping."""
    p = D / "site_fingerprint.csv"
    if not p.exists():
        sys.exit(f"{p} not found -- run: python pipeline/site_fingerprint.py")
    m = pd.read_csv(p).set_index("StudyInstanceUID").site_id
    g = uids.map(m)
    if g.isna().any():
        miss = g.isna()
        print(f"  {int(miss.sum())} studies have no header row; each becomes its own group")
        g = g.astype("object")
        g[miss] = ["nofp_" + u for u in uids[miss]]
    return g.astype(str)


def assign(groups: list, y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Greedy label-balanced assignment of GROUPS (not rows) to folds."""
    rng = np.random.default_rng(seed)
    cols = np.concatenate([y, np.ones((len(y), 1))], axis=1)
    total = np.maximum(cols.sum(0), 1.0)
    counts = np.zeros((n_folds, cols.shape[1]))

    rarity = 1.0 / np.maximum(y.sum(0), 1)
    priority = (cols[:, -1] * 1000.0
                + (y * rarity[None, :]).sum(1)
                + rng.random(len(y)) * 1e-9)
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

    y = tr[LABELS].to_numpy(dtype=float)
    if soft_labels:
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
                    help="'site' groups on the scanner fingerprint. "
                         "Build BOTH: the gap between them is our site-leakage number, and "
                         "the grouped one is what the reproduction gate must use")
    ap.add_argument("--labels", default=None,
                    help="soft-target CSV for stratification; default data/targets.csv")
    ap.add_argument("--accept-new", action="store_true",
                    help="let a partition that differs from provenance/ replace the record")
    args = ap.parse_args()

    folds = build(group_by=args.group_by, labels_csv=args.labels)
    folds.to_csv(args.out, index=False)
    report(folds)

    bad = folds.groupby("group").fold.nunique().max()
    if bad > 1:
        sys.exit(f"FAIL: a group landed in {bad} folds -- grouping is broken")
    print(f"\nwrote {args.out}")

    sys.path.insert(0, str(PROJ / "pipeline"))
    from provenance import check
    check(Path(args.out), f"fusion/folds.py --group-by {args.group_by}",
          accept_new=args.accept_new, keep_copy=True)


if __name__ == "__main__":
    main()
