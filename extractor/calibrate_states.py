"""Fit the soft-target constants against gold instead of guessing them. IMPROVEMENTS.md 1.2/1.3.

>>> THE FIT WAS TESTED AND IT LOST: macro 0.743 -> 0.699 (IMPROVEMENTS.md 1.3a). Do NOT enable
>>> `fusion/train.py --calibrated-targets` as a setting. This script stays because the
>>> MEASUREMENT is sound and reusable; the CONCLUSION drawn from it was not.
>>>
>>> The `absent` bucket is heterogeneous -- it mixes true positives and true negatives -- so
>>> re-targeting it to its mean teaches the model the mean rather than the discrimination. The
>>> guessed 0.08 is badly calibrated and strongly SEPARATING, and macro AUC only rewards
>>> separation. How far each label's `absent` was raised predicts how much AUC it lost:
>>> corr = -0.776. Better calibration, worse ranking.

`rule_extractor.SCORE` maps the five extractor states to training targets:

    pos 0.95 / hedged 0.65 / weak 0.45 / neg 0.03 / absent 0.08

Those five numbers were chosen by reasoning and never fitted (IMPROVEMENTS.md 1.3), and they
set the target for every one of the 4,349 studies the vision model trains on. Measured against
the 58 gold studies, the ladder is **monotone in the right direction but badly spaced**:

    state    cells   P(gold=1)        95% CI     SCORE
    pos        182       0.747  [0.679,0.805]     0.95
    hedged      52       0.558  [0.423,0.684]     0.65
    weak        42       0.238  [0.135,0.385]     0.45
    neg         55       0.073  [0.029,0.173]     0.03
    absent     365       0.167  [0.132,0.209]     0.08

Three things that table says, worst first.

  1. `absent` is the whole ballgame and it is set to less than half its measured value.
     **62.1% of the entire 4,407 x 12 target matrix is `absent`** -- silence, not negation.
     Scored 0.08 against a measured 0.167. And `absent` is not a mild `neg`: silence is 2.3x
     more likely to hide a true positive than an explicit "no tear" is. The repo separates them
     by 0.05; the data separates them by 0.094 in the same direction but twice as far.

  2. `absent` is wildly heterogeneous across labels, so one constant cannot be right.
     0.031 for ACL (radiologists comment on the ACL when it matters) up to 0.372 for Synovitis
     and 0.500 for Effusion. Synovitis is the case IMPROVEMENTS.md 2.1 already called the
     report-only ceiling -- and this is the mechanism: 87.6% of reports never mention it, and
     37% of those knees have it. A flat 0.08 there tells the model "no synovitis" on 43 of 58
     gold studies when 16 of them are positive. That is not a weak label, it is a wrong one.

  3. `pos` at 0.95 is overconfident, and the reason is not extractor error.
     Gold is an independent *image* read. A report that says "ACL tear" is confirmed by the
     image reader 74.7% of the time. The remaining quarter is genuine report-vs-image
     disagreement, and no amount of extractor polish removes it. 0.95 asserts a certainty the
     modality does not have.

WHY GOLD AND NOT THE HAND LABELS. IMPROVEMENTS.md 1.3 proposes fitting these "once hand-labels
exist". For `pos`/`hedged`/`weak` that works. For `absent` it measures the wrong quantity: the
hand labels are read from the *report*, so P(hand=1 | absent) asks "when the extractor saw
nothing, did a careful human reading the same text also see nothing" -- extractor recall. The
training target's job is to predict what the *image* shows, and only gold is an independent
image read. The two questions have different answers precisely where it matters.

WHY THIS IS CROSS-FITTED. Fitting on gold and then reporting gold macro-AUC is the leak that
turns a held-out number into a hyperparameter. `--folds` emits one table per fold, each fitted
on the gold studies *outside* that fold, so the model that predicts fold f's gold studies was
trained on targets no fold-f gold study contributed to. train.py --calibrated-targets consumes
that. The pooled table is printed for reading, never for training.

    python extractor/calibrate_states.py                 # print the tables
    python extractor/calibrate_states.py --write         # + data/state_calibration.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "extractor"))
from rule_extractor import LABELS, SCORE                        # noqa: E402

D = PROJ / "data"
STATES = ["pos", "hedged", "weak", "neg", "absent"]

# Candidate prior strengths for the per-label shrink, in pseudo-counts. 0 = trust the per-label
# rate outright, inf = ignore it and use the pooled rate. The per-label `absent` cells run 4-50,
# so neither end is defensible a priori and the grid is resolved by leave-one-out below.
SHRINK_GRID = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, np.inf]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Normal-approximation CIs go out of [0,1] at these counts."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(states_path: Path, train_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gold studies only, aligned: (extractor state, gold label) per study x label."""
    for p in (states_path, train_path):
        if not p.exists():
            sys.exit(f"{p} not found -- run extractor/run_extract.py first")
    st = pd.read_csv(states_path).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    tr = pd.read_csv(train_path).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    gold = tr.dropna(subset=LABELS)
    common = gold.index.intersection(st.index)
    return st.loc[common, LABELS], gold.loc[common, LABELS].astype(float)


def counts(state: pd.DataFrame, y: pd.DataFrame) -> dict:
    """(k, n) per state pooled, and per (label, state)."""
    pooled = {s: (int(y.values[(state == s).values].sum()), int((state == s).values.sum()))
              for s in STATES}
    per_label = {lab: {s: (int(y[lab].values[(state[lab] == s).values].sum()),
                           int((state[lab] == s).values.sum()))
                       for s in STATES} for lab in LABELS}
    return {"pooled": pooled, "per_label": per_label}


def shrink(k: int, n: int, prior: float, m: float) -> float:
    """Beta-prior posterior mean: the per-label rate pulled toward the pooled rate.

    m is in pseudo-counts, so a label with 4 `absent` cells barely moves off the pooled value
    while one with 50 mostly keeps its own. m = inf collapses to the pooled rate.
    """
    if not np.isfinite(m):
        return prior
    return (k + m * prior) / (n + m) if (n + m) > 0 else prior


def loo_logloss(c: dict, m: float) -> float:
    """Leave-one-out log loss over gold cells at prior strength m.

    Each cell is scored by a table fitted without it, so a per-label rate that is really just
    that one cell repeating itself cannot win. This is what picks m rather than taste.
    """
    tot, n_cells = 0.0, 0
    for lab in LABELS:
        for s in STATES:
            k, n = c["per_label"][lab][s]
            if n == 0:
                continue
            pk, pn = c["pooled"][s]
            for y in ([1] * k + [0] * (n - k)):
                # Drop this cell from both the per-label and the pooled counts.
                prior = (pk - y) / (pn - 1) if pn > 1 else 0.5
                p = shrink(k - y, n - 1, prior, m)
                p = min(max(p, 1e-6), 1 - 1e-6)
                tot += -np.log(p if y else 1 - p)
                n_cells += 1
    return tot / max(n_cells, 1)


def fit(state: pd.DataFrame, y: pd.DataFrame, m: float | None = None) -> dict:
    """-> {"pooled": {state: p}, "per_label": {label: {state: p}}, "m": m}."""
    c = counts(state, y)
    pooled = {s: (k / n if n else float("nan")) for s, (k, n) in c["pooled"].items()}
    if m is None:
        m = min(SHRINK_GRID, key=lambda g: loo_logloss(c, g))
    per_label = {lab: {s: shrink(k, n, pooled[s] if np.isfinite(pooled[s]) else 0.5, m)
                       for s, (k, n) in c["per_label"][lab].items()}
                 for lab in LABELS}
    return {"pooled": pooled, "per_label": per_label, "m": (None if not np.isfinite(m) else m),
            "n_studies": int(len(y))}


def print_pooled(state: pd.DataFrame, y: pd.DataFrame, f: dict) -> None:
    c = counts(state, y)
    n_total = sum(n for _, n in c["pooled"].values())
    print(f"P(gold=1 | extractor state) -- pooled over 12 labels, {len(y)} gold studies\n")
    print(f"{'state':<9}{'cells':>7}{'share':>8}{'gold=1':>8}{'P(1|s)':>9}{'95% CI':>17}"
          f"{'SCORE':>8}{'delta':>8}")
    for s in STATES:
        k, n = c["pooled"][s]
        lo, hi = wilson(k, n)
        p = f["pooled"][s]
        print(f"{s:<9}{n:>7}{n / n_total:>7.1%}{k:>8}{p:>9.3f}   [{lo:.3f},{hi:.3f}]"
              f"{SCORE[s]:>8.2f}{p - SCORE[s]:>+8.3f}")
    print(f"\n  {c['pooled']['absent'][1] / n_total:.1%} of the target matrix is `absent`. "
          f"It carries the largest error in the table.")


def print_per_label(f: dict) -> None:
    m = f["m"]
    print(f"\nPer-label, shrunk toward the pooled rate at m={m if m is not None else 'inf'} "
          f"pseudo-counts (leave-one-out)\n")
    print(f"{'label':<18}" + "".join(f"{s:>9}" for s in STATES))
    for lab in LABELS:
        print(f"{lab:<18}" + "".join(f"{f['per_label'][lab][s]:>9.3f}" for s in STATES))
    print(f"\n{'SCORE (current)':<18}" + "".join(f"{SCORE[s]:>9.2f}" for s in STATES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", default=str(D / "extract_states.csv"))
    ap.add_argument("--train", default=str(D / "train.csv"))
    ap.add_argument("--folds", default=str(D / "folds.csv"))
    ap.add_argument("--out", default=str(D / "state_calibration.json"))
    ap.add_argument("--m", type=float, default=None,
                    help="prior strength; default picks it by leave-one-out log loss")
    ap.add_argument("--write", action="store_true", help="write the cross-fitted JSON")
    args = ap.parse_args()

    state, y = load(Path(args.states), Path(args.train))
    if len(y) < 20:
        sys.exit(f"only {len(y)} gold studies matched -- nothing to fit")

    pooled_fit = fit(state, y, args.m)
    print_pooled(state, y, pooled_fit)
    print_per_label(pooled_fit)

    print(f"\n{'-' * 72}\nn={len(y)} gold studies. Per-label `absent` cells run 4-50, so the "
          f"per-label column is\ndirectional -- the shrink is what keeps a 4-cell rate from "
          f"being read as a measurement.\nThe pooled column is the well-supported part.")

    # ---- cross-fitted tables ----------------------------------------------------------
    out = {"pooled": pooled_fit, "per_fold": {}, "source": Path(args.states).name}
    folds_p = Path(args.folds)
    if folds_p.exists():
        folds = pd.read_csv(folds_p).drop_duplicates("StudyInstanceUID").set_index(
            "StudyInstanceUID")
        fold_of = folds.reindex(state.index)["fold"]
        print("\nCross-fitted tables (each fold fitted on the gold OUTSIDE it):")
        for fo in sorted(folds["fold"].unique()):
            keep = fold_of.index[fold_of != fo]
            if len(keep) < 20:
                print(f"  fold {fo}: only {len(keep)} gold outside -- skipped")
                continue
            ff = fit(state.loc[keep], y.loc[keep], args.m)
            out["per_fold"][str(int(fo))] = ff
            held = int((fold_of == fo).sum())
            print(f"  fold {fo}: fitted on {len(keep)} gold, holds out {held}   "
                  f"pos {ff['pooled']['pos']:.3f} absent {ff['pooled']['absent']:.3f}")
    else:
        print(f"\n{folds_p} not found -- run fusion/folds.py for the cross-fitted tables. "
              f"Without them\nthese constants must not be used for anything that reports gold "
              f"AUC.")

    if args.write:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
        print("Consume it with: python fusion/train.py --calibrated-targets "
              f"{Path(args.out).name}")
    else:
        print(f"\n(--write to save {Path(args.out).name})")


if __name__ == "__main__":
    main()
