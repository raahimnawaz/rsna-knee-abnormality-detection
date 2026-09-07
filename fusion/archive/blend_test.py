"""Does adding an arm to a rank blend HELP? The slot test, not the "is it better alone" test.

WHY THIS EXISTS. §2w step 4 says the port must EARN an ensemble slot, and `score_oof.py` cannot
answer that: a member can be much weaker on its own and still pay for its slot if it is wrong in
a *different direction* from the ensemble. That is the whole mechanism behind rank-mean gains.
So the question is not "does it beat the fork" but "does fork+port beat fork", swept over the
blend weight, on the studies they share.

    python fusion/blend_test.py fusion/runs_pilkwang fusion/runs_port

Rank blending, not probability averaging, because the metric reads order only (per §1 of the
public notebook and our own 2j). Ranks are computed per label over the scored studies.

READ THE ASYMMETRY. `score_oof.py`'s reference (`lixin_gpt56`) correlates 0.947 with `steven_v2`
-- what the port trained on -- and 0.866 with `pilkwang_v2`. Every number here therefore flatters
the port. A negative result is clean; a small positive one is not, and would need gold-58 to
confirm. Same rule as §2s.

Also reports the rank correlation between the arms, which is the thing that decides whether a
slot was ever plausible: §2f closed per-label reader fusion because the public readers ran at
mean |r| 0.87-0.95 and fusing lost to the best single. The same test, one level up.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "extractor"))
from metrics import auc  # noqa: E402

D = PROJ / "data"
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
NEUTRAL = D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56" / \
    "labels_llm_gpt56sol.csv"


def macro(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean([auc(y[:, i], p[:, i]) for i in range(len(L))]))


def boot_delta(y, pa, pb, n=400, seed=0) -> tuple[float, float]:
    """PAIRED bootstrap of macro(pb) - macro(pa). Paired because the arms share studies."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        out.append(macro(y[k], pb[k]) - macro(y[k], pa[k]))
    return float(np.mean(out)), float(np.std(out))


def to_rank(p: np.ndarray) -> np.ndarray:
    return np.column_stack([rankdata(p[:, i]) / len(p) for i in range(p.shape[1])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", help="run dir: the ensemble being added to")
    ap.add_argument("cand", help="run dir: the candidate member")
    a = ap.parse_args()

    ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    gold = set(pd.read_csv(D / "train.csv").dropna(subset=L).StudyInstanceUID)

    def load(r):
        return pd.read_csv(Path(r) / "oof_all.csv").drop_duplicates(
            "StudyInstanceUID").set_index("StudyInstanceUID")

    A, B = load(a.base), load(a.cand)
    ids = [u for u in A.index if u in B.index and u in ref.index and u not in gold]
    y = ref.loc[ids, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    keep = ~np.isnan(y).any(axis=1)
    ids = [u for u, k in zip(ids, keep) if k]
    y = (y[keep] > 0.5).astype(float)
    pa = A.loc[ids, L].to_numpy(float)
    pb = B.loc[ids, L].to_numpy(float)
    print(f"reference: {NEUTRAL.name} · gold excluded · n={len(ids)} shared studies\n")

    ra, rb = to_rank(pa), to_rank(pb)
    rho = [spearmanr(pa[:, i], pb[:, i]).statistic for i in range(len(L))]
    print(f"  rank correlation between arms: mean rho {np.mean(rho):.3f} "
          f"(min {np.min(rho):.3f}, max {np.max(rho):.3f})")
    print("  -- §2f closed reader fusion at |r| 0.87-0.95; a slot needs DIVERSITY, "
          "not just quality\n")

    m_base = macro(y, ra)
    print(f"  {'w':>6}  {'macro':>8}  {'vs base':>9}")
    print(f"  {0.0:>6.2f}  {m_base:>8.4f}  {'--':>9}   <- base alone")
    best = (0.0, m_base)
    for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        blend = (1 - w) * ra + w * rb
        m = macro(y, blend)
        print(f"  {w:>6.2f}  {m:>8.4f}  {m-m_base:>+9.4f}")
        if m > best[1]:
            best = (w, m)
    m_cand = macro(y, rb)
    print(f"  {1.0:>6.2f}  {m_cand:>8.4f}  {m_cand-m_base:>+9.4f}   <- candidate alone")

    w, m = best
    print()
    if w == 0.0:
        print("  VERDICT: no weight helps. The candidate does not earn a slot.")
    else:
        d, sd = boot_delta(y, ra, (1 - w) * ra + w * rb)
        print(f"  best w={w:.2f}: {m:.4f}  paired delta {d:+.4f} +-{sd:.4f} "
              f"({abs(d)/sd if sd else 0:.1f} sigma)")
        print("  Remember the reference leans toward the candidate -- confirm on gold-58 "
              "before believing a positive.")


if __name__ == "__main__":
    main()
