"""Does REPORT-grouped weight selection overfit relative to SITE-grouped? Blend weights only.

WHY THIS EXISTS. `prvsiyan` (the 0.906 public notebook we are building on) selects its per-target
rank-blend weights with `assign_group_balanced_folds(seed=20260809)`, which keeps **normalized
report groups atomic**. That is a duplicate-report guard. It is *not* a site guard, and §2j
measured site leakage in this corpus at **+0.024 (~5σ)** over 265 scanner fingerprints. So their
weights are chosen on a signal that can see the scanner. This asks whether that matters.

    python fusion/fold_scheme_test.py

The experiment is a nested selection test, which is the only way to ask the question honestly:

  * pick per-target blend weights on TRAIN folds under scheme S
  * score those weights on the HELD-OUT fold, always under the SITE split
  * do it for S = report-like (random, report groups atomic) and S = site

Both are evaluated identically, so the only thing varying is what the *selector* was allowed to
see. If the report-grouped selector picks weights that do worse on site-held-out data, its
advantage was leakage and prvsiyan's weights carry the same defect.

WHAT THE ARMS ARE. `pilkwang/rsna-knee-weights::merge_gain.npz` ships two aligned prediction
matrices over all 4,407 studies -- `ours` (their 20-member ensemble) and `imported` (a second
source they merge in). Two arms is all their artifacts expose; `oof.npz` is the already-merged
result, so the 20 members cannot be re-weighted individually from what is published. Two is
enough for this question, because the question is about the SELECTOR, not the arms.

Scored against `lixin_gpt56` on non-gold studies -- the `score_oof.py` definition. Both arms here
are pilkwang-lineage and neither trained on the reference, so unlike §2y this comparison is
symmetric and a positive result is readable.
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


def rank01(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / len(v)


def pick_weights(y, A, B, idx) -> np.ndarray:
    """Per-target blend weight chosen on `idx` only. w=0 -> all A, w=1 -> all B."""
    w = np.zeros(len(L))
    for i in range(len(L)):
        ya = y[idx, i]
        if len(np.unique(ya)) < 2:
            continue
        ra, rb = rank01(A[idx, i]), rank01(B[idx, i])
        scores = [auc(ya, (1 - g) * ra + g * rb) for g in GRID]
        w[i] = GRID[int(np.nanargmax(scores))]
    return w


def apply_weights(y, A, B, idx, w) -> float:
    out = []
    for i in range(len(L)):
        ya = y[idx, i]
        if len(np.unique(ya)) < 2:
            continue
        ra, rb = rank01(A[idx, i]), rank01(B[idx, i])
        out.append(auc(ya, (1 - w[i]) * ra + w[i] * rb))
    return float(np.nanmean(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--merge-gain", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260809, help="prvsiyan's fold seed")
    a = ap.parse_args()

    g = np.load(a.merge_gain, allow_pickle=True)
    ids = [str(u) for u in g["ids"]]
    A_all, B_all = np.asarray(g["ours"], float), np.asarray(g["imported"], float)

    ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    gold = set(pd.read_csv(D / "train.csv").dropna(subset=L).StudyInstanceUID)
    folds = pd.read_csv(D / "folds_site.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")

    pos = {u: k for k, u in enumerate(ids)}
    keep = [u for u in ids if u in ref.index and u not in gold and u in folds.index]
    yv = ref.loc[keep, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~np.isnan(yv).any(axis=1)
    keep = [u for u, k in zip(keep, ok) if k]
    y = (yv[ok] > 0.5).astype(float)
    rows = [pos[u] for u in keep]
    A, B = A_all[rows], B_all[rows]
    site = folds.loc[keep, "fold"].to_numpy()
    print(f"reference: {NEUTRAL.name} · gold excluded · n={len(keep)} studies\n")

    # a report-like selector: groups that are NOT sites, assigned by their seed.
    rng = np.random.default_rng(a.seed)
    rep = rng.integers(0, 5, len(keep))

    print("  per-target blend weight (0 = all `ours`, 1 = all `imported`), 5 site-held-out folds")
    print(f"  {'held-out':>9} {'select on SITE':>16} {'select on REPORT-like':>23}")
    res = {"site": [], "rep": []}
    for f in range(5):
        te = np.where(site == f)[0]
        tr_site = np.where(site != f)[0]
        tr_rep = np.where(rep != f)[0]
        tr_rep = np.array([i for i in tr_rep if i not in set(te)])
        if len(te) < 50:
            continue
        w_site = pick_weights(y, A, B, tr_site)
        w_rep = pick_weights(y, A, B, tr_rep)
        s_site = apply_weights(y, A, B, te, w_site)
        s_rep = apply_weights(y, A, B, te, w_rep)
        res["site"].append(s_site)
        res["rep"].append(s_rep)
        print(f"  {f:>9} {s_site:>16.4f} {s_rep:>23.4f}")

    ms, mr = float(np.mean(res["site"])), float(np.mean(res["rep"]))
    print(f"  {'MEAN':>9} {ms:>16.4f} {mr:>23.4f}   delta {ms-mr:+.4f}")

    # baselines: the arms alone, and the fixed 50/50
    allidx = np.arange(len(y))
    print()
    for nm, w in (("all `ours` (w=0)", np.zeros(len(L))),
                  ("all `imported` (w=1)", np.ones(len(L))),
                  ("fixed 50/50", np.full(len(L), 0.5))):
        print(f"  {nm:<22} {apply_weights(y, A, B, allidx, w):.4f}")


if __name__ == "__main__":
    main()
