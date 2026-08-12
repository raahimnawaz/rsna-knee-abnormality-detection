"""How much of a gain selected on n studies is real? Quantifies the gold-58 selection trap.

WHY THIS EXISTS. `prvsiyan` -- the 0.906 notebook §2w decided to build on -- selects per-target
rank-blend weights on the **58** image-adjudicated studies, and reports its V34 PCA arm as worth
**+0.0273**. They are explicit that this "measures estimator stability rather than independent
clinical generalization". §2z showed low-dimensional selection is robust to a *bad fold scheme*;
it says nothing about a *tiny selection set*, which is the regime that actually overfits.

Their per-arm predictions are NOT published (their artifact datasets are the blank entries in
`kernel-metadata.json`), so their arm cannot be scored directly. **What can be measured exactly is
the procedure**: select a per-target blend weight on n studies, then see what that choice delivers
on studies it did not see.

    python fusion/selection_optimism.py --merge-gain <merge_gain.npz>

The two arms are `ours` / `imported` from `pilkwang/rsna-knee-weights::merge_gain.npz` -- the only
two aligned prediction matrices any public artifact exposes. The arms are a stand-in; **the
optimism curve is a property of the selection procedure and the sample size**, which is what
transfers to their number and to any future gold-58 decision of ours.

Reports, for each n:
  * CLAIMED  -- the gain the selector sees on its own selection set. This is the number a
    competitor reports, and the number we would report if we selected on gold-58.
  * REALIZED -- the same chosen weights, scored on the held-out remainder. This is the truth.
  * the gap between them is the optimism, and it is the whole point.
"""
from __future__ import annotations

import argparse
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
GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


def rank01(v):
    return rankdata(v) / len(v)


def macro_at(y, A, B, idx, w):
    """Macro AUC on `idx` with per-target weights `w`. Ranks computed within `idx`."""
    out = []
    for i in range(len(L)):
        ya = y[idx, i]
        if len(np.unique(ya)) < 2:
            continue
        r = (1 - w[i]) * rank01(A[idx, i]) + w[i] * rank01(B[idx, i])
        out.append(auc(ya, r))
    return float(np.nanmean(out)) if out else np.nan


def select(y, A, B, idx):
    w = np.zeros(len(L))
    for i in range(len(L)):
        ya = y[idx, i]
        if len(np.unique(ya)) < 2:
            continue           # label unusable in this subset -> keep baseline weight
        ra, rb = rank01(A[idx, i]), rank01(B[idx, i])
        w[i] = GRID[int(np.nanargmax([auc(ya, (1 - g) * ra + g * rb) for g in GRID]))]
    return w


def select_bagged(y, A, B, idx, rng, k=25):
    """`prvsiyan`'s actual shape: repeated resampled selection, averaged -- and the fix the
    low-data ensembling literature recommends. Resampling WITHIN the selection set, because that
    is what they do: "99.4% of 500 partitions" is 500 re-partitions of the same 58 subjects."""
    ws = []
    for _ in range(k):
        b = rng.choice(idx, len(idx), replace=True)
        ws.append(select(y, A, B, b))
    return np.mean(ws, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--merge-gain", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--sizes", type=int, nargs="+", default=[58, 150, 400, 1000, 2500])
    a = ap.parse_args()

    g = np.load(a.merge_gain, allow_pickle=True)
    ids = [str(u) for u in g["ids"]]
    A_all, B_all = np.asarray(g["ours"], float), np.asarray(g["imported"], float)

    ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    gold = set(pd.read_csv(D / "train.csv").dropna(subset=L).StudyInstanceUID)
    pos = {u: k for k, u in enumerate(ids)}
    keep = [u for u in ids if u in ref.index and u not in gold]
    yv = ref.loc[keep, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~np.isnan(yv).any(axis=1)
    keep = [u for u, k in zip(keep, ok) if k]
    y = (yv[ok] > 0.5).astype(float)
    rows = [pos[u] for u in keep]
    A, B = A_all[rows], B_all[rows]
    n = len(keep)
    print(f"reference: {NEUTRAL.name} · gold excluded · n={n} studies · "
          f"{a.trials} trials per size\n")
    print("  Two arms, per-target rank-blend weight chosen by argmax on the selection set.")
    print("  CLAIMED = gain the selector sees on its own set. "
          "REALIZED = same weights, held-out studies.\n")
    print(f"  {'n_select':>9} {'mode':>7} {'CLAIMED':>10} {'REALIZED':>10} {'OPTIMISM':>10} "
          f"{'P(realized<=0)':>15}")

    base = np.zeros(len(L))
    for m in a.sizes:
        if m >= n:
            continue
        for mode in ("argmax", "bagged"):
            rng = np.random.default_rng(0)          # same subsets for both modes
            cl, re_, neg = [], [], 0
            for _ in range(a.trials):
                sel = rng.choice(n, m, replace=False)
                held = np.setdiff1d(np.arange(n), sel, assume_unique=False)
                w = (select(y, A, B, sel) if mode == "argmax"
                     else select_bagged(y, A, B, sel, rng))
                c = macro_at(y, A, B, sel, w) - macro_at(y, A, B, sel, base)
                r = macro_at(y, A, B, held, w) - macro_at(y, A, B, held, base)
                if np.isnan(c) or np.isnan(r):
                    continue
                cl.append(c)
                re_.append(r)
                neg += (r <= 0)
            cl, re_ = np.array(cl), np.array(re_)
            print(f"  {m:>9} {mode:>7} {cl.mean():>+10.4f} {re_.mean():>+10.4f} "
                  f"{cl.mean()-re_.mean():>+10.4f} {neg/len(re_):>14.0%}")
            if m == 58:
                for thr in (0.0100, 0.0200, 0.0273):
                    print(f"            P(CLAIMED >= {thr:+.4f}) = {(cl >= thr).mean():.0%}"
                          + ("   <- prvsiyan's V34 PCA claim" if thr == 0.0273 else ""))

    print("\n  Read the n=58 rows against prvsiyan's claimed +0.0273 for the V34 PCA arm.")


if __name__ == "__main__":
    main()
