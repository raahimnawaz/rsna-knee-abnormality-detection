"""Is a large-n report-derived OOF actually a working instrument?

Three questions, in order of how much they matter:

  1. How much tighter is it? (arithmetic -- guaranteed, but worth seeing the size)
  2. Does it agree in LEVEL with cross-fitted gold-37?
  3. Does it RESOLVE an A/B that gold-37 could not?  <- the only one that decides anything

Question 3 has to be asked without circularity. Arm A trains on our labels, arm B trains on
steven_v4; scoring each against its own training targets would just measure which arm's targets
are easier to fit. So both arms are scored against a THIRD source neither of them saw
(lixin_gpt56), over every non-gold cached study.

>>> AND THAT DESIGN FAILED. READ THIS BEFORE QUOTING Q3. <<<

lixin_gpt56 is not a neutral reference: `steven_v4` predicts it at AUC **0.9998**, and the
public readers sit at mean |r| 0.87-0.95 against each other (IMPROVEMENTS.md 2g). Arm B is
therefore scored against a near-copy of its own training targets, and the +0.039 / 6.0 sigma
this script prints for Q3 is an artefact. It is left in, running, because the number is how the
confound was found and deleting it would invite someone to rebuild the same test.

What survives is Q1, which is what the plan actually needs: the large-n instrument is ~6.7x
tighter than gold-37, so it can resolve ~0.01 where gold cannot resolve 0.04. That licenses
comparisons **at fixed targets** -- architecture, pooling, slice count, augmentation, the port's
reproduction gate. It does not license comparing label SOURCES, because the reference is itself
a label source and the arm whose targets resemble it wins by construction. Gold-58 keeps that
job.
"""
import os
import sys

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extractor"))
from metrics import auc  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
REPO = str(PROJ)
# Where the two arms' run directories live. Overridable because these are throwaway A/B runs
# that do not belong beside the checkpoints in fusion/runs*.
S = os.environ.get("RUNS_DIR", str(PROJ / "fusion"))
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

NEUTRAL = f"{REPO}/data/public_llm_labels/lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv"

gold = pd.read_csv(f"{REPO}/data/train.csv").dropna(subset=L).set_index("StudyInstanceUID")[L]
neutral = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")


def boot_macro(y: np.ndarray, p: np.ndarray, n: int = 400, seed: int = 0) -> tuple[float, float]:
    """Bootstrap SD of the macro over studies -- the honest noise floor for a macro."""
    rng = np.random.default_rng(seed)
    base = float(np.nanmean([auc(y[:, i], p[:, i]) for i in range(len(L))]))
    out = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        v = [auc(y[k, i], p[k, i]) for i in range(len(L))]
        if not np.all(np.isnan(v)):
            out.append(np.nanmean(v))
    return base, float(np.std(out))


def load(run: str) -> pd.DataFrame:
    return pd.read_csv(f"{S}/{run}/oof_all.csv").set_index("StudyInstanceUID")


def evaluate(oof: pd.DataFrame, ref: pd.DataFrame, ids, tag: str):
    idx = [u for u in ids if u in oof.index and u in ref.index]
    y = ref.loc[idx, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    p = oof.loc[idx, L].to_numpy(float)
    keep = ~np.isnan(y).any(axis=1)
    y, p = y[keep], p[keep]
    # binarise a graded reference at its midpoint -- validation only, per the 0.899 notebook
    yb = (y > 0.5).astype(float)
    m, sd = boot_macro(yb, p)
    print(f"  {tag:<44} n={len(yb):>5}  macro {m:.4f}  +-{sd:.4f}")
    return m, sd


print(__doc__)
runs = {"A: trained on OURS (0.777 labels)": os.environ.get("ARM_A", "runs_ours"),
        "B: trained on steven_v4 (0.893 labels)": os.environ.get("ARM_B", "runs_inst")}
oofs = {k: load(v) for k, v in runs.items()}
any_oof = next(iter(oofs.values()))
gold_ids = [u for u in gold.index if u in any_oof.index]
nongold = [u for u in any_oof.index if u not in set(gold.index)]

print(f"cached studies with OOF: {len(any_oof)}   gold covered: {len(gold_ids)}   "
      f"non-gold: {len(nongold)}\n")

print("Q1+Q2 -- the two instruments, same predictions, arm B:")
b = oofs["B: trained on steven_v4 (0.893 labels)"]
g_m, g_sd = evaluate(b, gold, gold_ids, "gold-37 (image-read, current instrument)")
r_m, r_sd = evaluate(b, neutral, nongold, "report-OOF vs neutral reader (proposed)")
print(f"\n  noise floor ratio: gold +-{g_sd:.4f} vs report +-{r_sd:.4f}"
      f"  ->  {g_sd / r_sd:.1f}x tighter")
print(f"  level agreement:   {abs(g_m - r_m):.3f} apart"
      f"  (they measure different references, so this is context, not a gate)")

print("\nQ3 -- does it resolve the A/B that gold-37 could not?")
print("  scored against lixin_gpt56, which NEITHER arm trained on:")
res = {}
for k, o in oofs.items():
    res[k] = evaluate(o, neutral, nongold, k)
print()
ks = list(res)
d = res[ks[1]][0] - res[ks[0]][0]
se = np.hypot(res[ks[0]][1], res[ks[1]][1])
print(f"  delta B-A = {d:+.4f}  +-{se:.4f} (unpaired, conservative)  ->  {abs(d) / se:.1f} sigma")
print("\n  same A/B on gold-37:")
for k, o in oofs.items():
    evaluate(o, gold, gold_ids, k)
