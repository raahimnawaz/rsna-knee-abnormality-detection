"""Score an arm on the 58 GOLD studies -- the only local instrument that measures what the
leaderboard measures -- and convert that reading into an LB estimate.

    python fusion/score_gold.py                      # pilkwang's shipped OOF (the anchor)
    python fusion/score_gold.py fusion/runs_port     # any run with an oof_all.csv
    python fusion/score_gold.py --paired A B         # two arms, paired bootstrap

WHY THIS EXISTS, AND WHY IT IS NOT `score_oof.py`.

`score_oof.py` scores against `lixin_gpt56`, a **report-derived** source. That is the right
instrument for a fixed-target question (architecture, pooling, resolution) because both arms
share the reference's bias, and it has n=2,612 so it is precise. It is the WRONG instrument for
the question "is this closer to the truth", because the leaderboard scores against expert
**image** reads and reports and images genuinely disagree. 3e measured that gap on our own
anchor without naming it: pilkwang reads 0.8434 through `score_oof.py` and 0.891 on the board.

The consequence, which 3l makes measurable: when a model gets better at SEEING the knee, it
departs from the report labels precisely on the studies where the report was wrong. A real
vision gain is therefore partly booked by `score_oof.py` as disagreement with the teacher.
**The report instrument systematically understates exactly the kind of gain that matters.**

WHAT THIS BUYS. gold-58 + a constant offset predicts the leaderboard:

    LB_estimate = gold58_macro + GOLD_TO_LB

measured at **+0.046** across four independent systems (3l). That converts a ~2 h submission
into a ~0 s local read for any question big enough to clear the noise floor.

WHAT IT DOES NOT BUY, AND THIS IS 3b's RULE, UNREPEALED. n=58 gives a macro half-width of
**+-0.038**. So:

  * **Never SELECT on it.** 3b: a weight chosen on these 58 claims +0.0137 and delivers -0.0034,
    negative in 92% of draws. Selection needs n~400. Nothing here changes that.
  * **Do use it to judge a DIRECTION.** One pre-registered decision, effect >= ~0.02, sign only.
    "Is a second model family worth adding" is such a question. "Is this +0.002 tweak worth it"
    is not, and no local instrument can answer that one -- that is what the board is for.

PER-LABEL READINGS ARE INDICATIVE ONLY. 9-35 positives per label means half-widths of +-0.10 to
+-0.24. 3l uses them only for the one claim they can carry: that four labels put the report
reading OUTSIDE the gold CI, which is more than the ~0.6 labels chance would give, so the two
instruments differ **systematically** rather than noisily.
"""
from __future__ import annotations

import argparse
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

#: gold-58 -> public LB, measured over four systems in 3l. Spread +-0.005, which is well
#: inside the +-0.038 the 58 studies themselves impose, i.e. consistent with a constant.
#: Re-derive with --anchor if a fifth system ever gives a (gold, LB) pair.
GOLD_TO_LB = 0.046

PILKWANG_OOF = D / "external" / "pilkwang_weights" / "oof.npz"


def gold_frame() -> pd.DataFrame:
    """The 58 image-read studies, indexed by StudyInstanceUID. These are the ONLY labels in
    this project not derived from a report."""
    t = pd.read_csv(D / "train.csv").set_index("StudyInstanceUID")
    g = t[L].dropna(how="all")
    if len(g) != 58:
        raise SystemExit(f"expected 58 gold studies, found {len(g)} -- has train.csv changed?")
    return g


def load_arm(spec: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, pred[n,12]) for a run directory or, by default, pilkwang's shipped OOF."""
    if spec is None:
        d = np.load(PILKWANG_OOF, allow_pickle=True)
        order = [str(x) for x in d["targets"]]
        if order != L:
            raise SystemExit(f"pilkwang target order changed: {order}")
        return d["ids"], d["pred"]
    p = Path(spec)
    csv = p / "oof_all.csv" if p.is_dir() else p
    df = pd.read_csv(csv)
    return df["StudyInstanceUID"].values, df[L].values.astype(float)


def macro(y: np.ndarray, p: np.ndarray) -> float:
    v = [auc(y[:, i], p[:, i]) for i in range(len(L))]
    return float(np.nanmean(v))


def boot(y: np.ndarray, p: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI over STUDIES. Resamples studies, not labels -- the 12 AUCs share studies."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        v = macro(y[k], p[k])
        if not np.isnan(v):
            out.append(v)
    return tuple(np.percentile(out, [2.5, 97.5]))  # type: ignore[return-value]


def align(ids: np.ndarray, pred: np.ndarray, g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pos = {u: i for i, u in enumerate(ids)}
    keep = [u for u in g.index if u in pos]
    if len(keep) < len(g):
        print(f"  note: arm covers {len(keep)}/{len(g)} gold studies")
    return g.loc[keep, L].values.astype(float), pred[[pos[u] for u in keep]]


def report(name: str, y: np.ndarray, p: np.ndarray, per_label: bool) -> float:
    m = macro(y, p)
    lo, hi = boot(y, p)
    print(f"\n{name}")
    print(f"  gold-58 macro   {m:.4f}   95% CI [{lo:.4f}, {hi:.4f}]  (+-{(hi - lo) / 2:.4f})")
    print(f"  LB estimate     {m + GOLD_TO_LB:.3f}   (= gold + {GOLD_TO_LB:.3f}, 3l)")
    if per_label:
        print(f"  {'label':18s} {'AUC':>6s} {'pos':>4s}   <- indicative only, see docstring")
        for i, lab in enumerate(L):
            print(f"  {lab:18s} {auc(y[:, i], p[:, i]):6.3f} {int(np.nansum(y[:, i])):4d}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", nargs="?", help="run dir with oof_all.csv (default: pilkwang OOF)")
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"), help="paired A-B on shared studies")
    ap.add_argument("--per-label", action="store_true")
    ap.add_argument("--anchor", type=float, help="known LB for this arm; prints implied offset")
    a = ap.parse_args()

    g = gold_frame()

    if a.paired:
        ya, pa = align(*load_arm(a.paired[0]), g)
        yb, pb = align(*load_arm(a.paired[1]), g)
        if len(ya) != len(yb):
            raise SystemExit("arms cover different gold studies; paired read is invalid")
        ma = report(a.paired[0], ya, pa, a.per_label)
        mb = report(a.paired[1], yb, pb, a.per_label)
        rng = np.random.default_rng(0)
        d = [macro(ya[k], pa[k]) - macro(yb[k], pb[k])
             for k in (rng.integers(0, len(ya), len(ya)) for _ in range(4000))]
        d = np.array([x for x in d if not np.isnan(x)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"\nPAIRED  A-B = {ma - mb:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
        print("  Sign only. 3b still forbids SELECTING on this set.")
        return

    ids, pred = load_arm(a.run)
    y, p = align(ids, pred, g)
    m = report(a.run or "pilkwang 20-member (shipped oof.npz)", y, p, a.per_label or a.run is None)
    if a.anchor is not None:
        print(f"\n  implied offset gold->LB = {a.anchor - m:+.4f}   (running estimate {GOLD_TO_LB:+.3f})")


if __name__ == "__main__":
    main()
