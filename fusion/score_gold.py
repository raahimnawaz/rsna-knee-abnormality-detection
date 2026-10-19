"""Score a run on the 58 radiologist-graded (gold) studies.

    python fusion/score_gold.py fusion/runs_cnxs_teach1_f0
    python fusion/score_gold.py --paired <run A> <run B>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "fusion"))
from metrics import auc  # noqa: E402

D = PROJ / "data"
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

GOLD_TO_LB = 0.039

PILKWANG_OOF = D / "external" / "pilkwang_weights" / "oof.npz"


def gold_frame() -> pd.DataFrame:
    """The 58 image-read studies, indexed by StudyInstanceUID."""
    t = pd.read_csv(D / "train.csv").set_index("StudyInstanceUID")
    g = t[L].dropna(how="all")
    if len(g) != 58:
        raise SystemExit(f"expected 58 gold studies, found {len(g)} -- has train.csv changed?")
    return g


def load_arm(spec: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, pred[n,12]) for a run directory or, by default, the public baseline's OOF."""
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
    """Percentile CI over STUDIES."""
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
    print(f"  LB estimate     {m + GOLD_TO_LB:.3f}   (= gold + {GOLD_TO_LB:.3f}, 3v)")
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
        ia, pa_all = load_arm(a.paired[0])
        ib, pb_all = load_arm(a.paired[1])
        shared = g.index.intersection(pd.Index(ia)).intersection(pd.Index(ib))
        if len(shared) < 20:
            raise SystemExit(f"only {len(shared)} shared gold studies; too few to read")
        gs = g.loc[shared]
        print(f"paired on {len(shared)} shared gold studies")
        ya, pa = align(ia, pa_all, gs)
        yb, pb = align(ib, pb_all, gs)
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
