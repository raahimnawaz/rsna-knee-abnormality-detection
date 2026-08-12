"""Score any run's `oof_all.csv` on the SAME scale as the 0.7229 baseline. One definition.

WHY THIS EXISTS. `IMPROVEMENTS.md` 2j's baseline -- **0.7229 +- 0.0048, site-grouped, n=2,612**
-- is not "OOF predictions scored against the training targets". It is scored against
**`lixin73/labels_llm_gpt56sol`, a third label source the model never trained on**, over the
NON-GOLD studies only, with the graded reference binarised at 0.5. That is what
`instrument_test.py::evaluate` does, and the number only means what 2j says it means when it is
reproduced exactly.

`fusion/train_port.py` originally printed a report-OOF macro against `data/targets.csv` --
`steven_v2`, its own training targets. That is a different and **upward-biased** quantity: a
model is rewarded for reproducing the idiosyncrasies of the exact label source it fit, not just
the signal in it. Printed beside "baseline to beat: 0.7229" it would have looked like a
comparison and been a category error. Caught 2026-08-10 before the first fold finished, by
asking why the baseline was 0.7229 and not the 0.89 elsewhere in the docs.

So: **one scorer, used by every arm.** If a number is going to sit next to 0.7229 it comes from
here.

    python fusion/score_oof.py fusion/runs_port
    python fusion/score_oof.py fusion/runs_port fusion/runs_baseline    # paired A/B

THE REFERENCE IS NOT NEUTRAL, AND THAT IS FINE HERE -- 2g's caveat applies and is narrower than
it looks. `steven_v4` predicts `lixin` at AUC 0.9998, so this reference cannot arbitrate label
SOURCES; an arm trained on a near-copy wins by construction. It can arbitrate everything at
FIXED targets -- architecture, pooling, resolution, the port -- because both arms then share the
reference's bias. The port is a fixed-target comparison: same `steven_v2` targets, same
site-grouped folds, frozen encoder against a fine-tuned one.
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
NEUTRAL = D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56" / \
    "labels_llm_gpt56sol.csv"
BASELINE = 0.7229
BASELINE_SD = 0.0048


def boot_macro(y: np.ndarray, p: np.ndarray, n: int = 400, seed: int = 0) -> tuple[float, float]:
    """Bootstrap SD of the macro over studies. Same routine as instrument_test, same seed."""
    rng = np.random.default_rng(seed)
    base = float(np.nanmean([auc(y[:, i], p[:, i]) for i in range(len(L))]))
    out = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        v = [auc(y[k, i], p[k, i]) for i in range(len(L))]
        if not np.all(np.isnan(v)):
            out.append(np.nanmean(v))
    return base, float(np.std(out))


def load_oof(run: Path) -> pd.DataFrame:
    return pd.read_csv(run / "oof_all.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")


def score(run: Path, ref: pd.DataFrame, gold_ids: set,
          restrict: set | None = None) -> tuple[float, float, int, dict]:
    """`restrict` forces two arms onto the SAME studies, which is what makes an A/B paired.

    It matters more here than it looks: the frozen-embedding cache covers 2,650 studies and the
    336 tile cache covers 3,599, and a single-fold port run covers ~700. Scoring each arm over
    whatever it happens to hold compares difficulty as much as method -- the same category of
    error as scoring a model against its own training targets, one level up.
    """
    oof = load_oof(run)
    ids = [u for u in oof.index if u in ref.index and u not in gold_ids
           and (restrict is None or u in restrict)]
    y = ref.loc[ids, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    p = oof.loc[ids, L].to_numpy(float)
    keep = ~np.isnan(y).any(axis=1)
    y, p = (y[keep] > 0.5).astype(float), p[keep]
    if len(y) < 50:
        raise SystemExit(f"{run}: only {len(y)} scorable studies -- too few to mean anything")
    m, sd = boot_macro(y, p)
    per = {lab: auc(y[:, i], p[:, i]) for i, lab in enumerate(L)}
    return m, sd, len(y), per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run dirs containing oof_all.csv")
    a = ap.parse_args()

    if not NEUTRAL.exists():
        raise SystemExit(f"missing {NEUTRAL}\n  run extractor/bench_public_labels.py --download")
    ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    gold_ids = set(pd.read_csv(D / "train.csv").dropna(subset=L).StudyInstanceUID)

    print(__doc__.splitlines()[0])
    print(f"reference: {NEUTRAL.name} (held out from training) · gold excluded ({len(gold_ids)})\n")

    # Paired by construction: with more than one arm, every arm is scored on the intersection.
    restrict = None
    if len(a.runs) > 1:
        sets = [set(load_oof(Path(r)).index) for r in a.runs]
        restrict = set.intersection(*sets)
        print("  arms cover " + " / ".join(f"{len(s):,}" for s in sets) +
              f" studies -> scoring all of them on the {len(restrict):,} they share\n")

    res = {}
    for r in a.runs:
        run = Path(r)
        m, sd, n, per = score(run, ref, gold_ids, restrict)
        res[run.name] = (m, sd, n, per)
        delta = m - BASELINE
        sig = abs(delta) / np.hypot(sd, BASELINE_SD)
        print(f"  {run.name:<22} n={n:>5}  macro {m:.4f} +-{sd:.4f}   "
              f"vs 0.7229 baseline: {delta:+.4f} ({sig:.1f} sigma)")

    first = res[Path(a.runs[0]).name][3]
    print(f"\n{'label':<18}" + "".join(f"{Path(r).name[:11]:>12}" for r in a.runs))
    for lab in L:
        print(f"{lab:<18}" + "".join(f"{res[Path(r).name][3][lab]:>12.3f}" for r in a.runs))
    print("-" * (18 + 12 * len(a.runs)))
    print(f"{'MACRO':<18}" + "".join(f"{res[Path(r).name][0]:>12.4f}" for r in a.runs))

    if len(a.runs) == 2:
        # PAIRED bootstrap: resample studies once and score BOTH arms on that resample, so the
        # shared-study variance cancels instead of being added twice.
        #
        # The first version of this printed hypot(sA, sB) and called it "conservative". It is
        # conservative to the point of being the wrong test: the arms are scored on identical
        # studies by construction (see `restrict` above), most of the macro's variance is which
        # studies were drawn, and that component is COMMON. Measured 2026-08-11 on the first real
        # A/B, the unpaired SE was +-0.0138 against a paired +-0.0088 -- the same +0.0171 delta
        # reading 1.2 sigma instead of 1.9. Discarding the pairing throws away most of what
        # pairing was for.
        rA, rB = Path(a.runs[0]).name, Path(a.runs[1]).name
        oA, oB = load_oof(Path(a.runs[0])), load_oof(Path(a.runs[1]))
        # sorted(), not set order: `restrict` is a set of UID strings, and str hashing is
        # randomised per process, so iterating it laid `yy`/`pA`/`pB` out in a different order
        # every run while `default_rng(0)` drew the SAME integer indices into them -- a different
        # resample each time from a fixed seed. Measured before the fix over four PYTHONHASHSEEDs:
        # the delta was IDENTICAL at +0.0171 every time (it is order-invariant, being computed on
        # all studies at once), but the SD ranged +-0.0086 .. +-0.0091, i.e. 1.9-2.0 sigma and
        # P(delta>0) 0.969-0.980. So this never threatened +0.0171 or the 1.9 sigma headline; it
        # moved the third decimal of the interval. Fixed anyway -- the single scoring definition
        # should not move at all, and "reproduced exactly" has to be literally true here.
        ids = sorted(u for u in restrict if u in ref.index and u not in gold_ids)
        yy = ref.loc[ids, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        keep = ~np.isnan(yy).any(axis=1)
        ids = list(np.asarray(ids)[keep])
        yy = (yy[keep] > 0.5).astype(float)
        pA, pB = oA.loc[ids, L].to_numpy(float), oB.loc[ids, L].to_numpy(float)

        def mac(p, k):
            return float(np.nanmean([auc(yy[k, i], p[k, i]) for i in range(len(L))]))

        base = np.arange(len(yy))
        d0 = mac(pB, base) - mac(pA, base)
        rng = np.random.default_rng(0)
        ds = []
        for _ in range(2000):
            k = rng.integers(0, len(yy), len(yy))
            try:
                ds.append(mac(pB, k) - mac(pA, k))
            except Exception:                                          # noqa: BLE001, S112
                continue
        ds = np.asarray(ds)
        sd = float(ds.std())
        print(f"\n  PAIRED delta ({rB} - {rA}) = {d0:+.4f} +-{sd:.4f}"
              f"  ->  {abs(d0) / sd:.1f} sigma,  P(delta>0) = {(ds > 0).mean():.3f}")
        print(f"  (unpaired hypot SE would be +-{np.hypot(res[rA][1], res[rB][1]):.4f} = "
              f"{abs(d0) / np.hypot(res[rA][1], res[rB][1]):.1f} sigma -- the WRONG test here)")
        up = sum(res[rB][3][lab] > res[rA][3][lab] for lab in L)
        print(f"  per-label: {up}/12 up. The macro rides on the largest few, so read the "
              f"table, not the count.")
    else:
        print(f"\n  NOTE: a single arm against the recorded 0.7229 is NOT a paired comparison.")
        print("  The baseline's own oof_all.csv was never kept (fusion/runs*/ hold oof_gold.csv")
        print("  only), so the honest A/B is to re-run the frozen-cache arm under folds_site")
        print("  and score both here. ~32 min on mps, per data/_comparison.log.")
    del first


if __name__ == "__main__":
    main()
