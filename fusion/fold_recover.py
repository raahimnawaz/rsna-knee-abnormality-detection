"""Recover a published arm's fold assignment when it ships neither a split nor an OOF.

    python fusion/fold_recover.py --validate     # prove the method on pilkwang, where truth exists
    python fusion/fold_recover.py --arm ft_b     # honest OOF for rsna-ft-b on the gold studies

WHY THIS EXISTS. §3n left `ft_b` at "loads and runs, not reproduced". Its gate was an ALL-fold
read, and an all-fold read on gold is inflated — measured at **+0.1474** on our validated pilkwang
path (0.8516 honest OOF → 0.9990 all-20). To price an arm for F6 we need the prediction from the
fold that HELD EACH STUDY OUT, and `ft_b` publishes no split. Neither does the DINOv3 arm, and
RadImageNet ships a `fold_sha256` whose `folds_v1.csv` is not published anywhere (§9e).

**§3i's recovery does not transfer.** It matched fold-means against pilkwang's *shipped OOF*, and
there is no shipped OOF here.

THE METHOD, which is §3i's physics inverted. For a study, four of the five folds trained on it and
one did not. The four that trained on it were fitted to the same target and agree; **the fold that
held it out is the outlier.** So per study, score each fold by its L2 distance from the mean of the
other four and take the argmax.

WHY THIS IS TRUSTWORTHY HERE AND NOT JUST PLAUSIBLE — it is validated where the answer is known.
`--validate` runs it against §3i's recovered partition for pilkwang on the same 47 gold studies:

  * **5 fold-means (4 seeds each): 97.9%**, 46/47, against 20% chance. Recovered partition
    [10 9 5 8 15] against truth [9 10 5 8 15]. L1 agrees exactly; a logit-space distance is worse
    at 93.6%, so plain L2 is used.
  * **ONE member per fold, which is `ft_b`'s actual shape: 87.8% mean** over the four seed slots
    (80.9 / 91.5 / 83.0 / 95.7). **This is the number that applies to `ft_b`,** and the residual
    misassignment biases the result *upward* — a study given the wrong fold is scored by a model
    that memorised it. `--validate` also prints what that costs at the AUC level.

WHAT THIS DOES NOT DO. It does not recover folds for studies outside the scored set, and it does
not make a single-model OOF comparable to a four-seed-mean OOF. **pilkwang's 0.8516 is a mean of
the 4 members that held each study out; `ft_b`'s OOF is ONE model.** That handicaps `ft_b` by
whatever seed-averaging is worth, so the two numbers are not a like-for-like ensemble comparison
and must not be read as one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "extractor"))
D = PROJ / "data"

from metrics import auc  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def recover(fm: np.ndarray) -> np.ndarray:
    """(n_folds, n_studies, 12) probabilities -> held-out fold per study."""
    n_f, n_s, _ = fm.shape
    out = np.zeros(n_s, int)
    for i in range(n_s):
        out[i] = int(np.argmax([np.linalg.norm(fm[k, i] - np.delete(fm, k, 0)[:, i].mean(0))
                                for k in range(n_f)]))
    return out


def macro(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean([auc(y[:, j], p[:, j]) for j in range(12)]))


def gold_labels(ids) -> np.ndarray:
    return (pd.read_csv(D / "train.csv").set_index("StudyInstanceUID")
            .reindex(list(ids))[LABELS].to_numpy())


def validate() -> int:
    """Prove the method on pilkwang, where §3i already established the partition."""
    d = np.load(D / "_crop_ab_gold.npz", allow_pickle=True)
    P, truth, y = d["P130"], d["best"], d["y"]
    folds = np.array([m["fold"] for m in
                      json.load(open(D / "external" / "pilkwang_weights" / "manifest.json"))["members"]])
    fm = np.stack([P[folds == f].mean(0) for f in range(5)])

    r = recover(fm)
    print(f"5 fold-means (4 seeds each): {(r == truth).mean():.1%}  "
          f"({(r == truth).sum()}/{len(truth)}), chance 20%")
    print(f"  recovered {np.bincount(r, minlength=5)}   truth {np.bincount(truth, minlength=5)}")

    print("\nONE member per fold -- ft_b's actual shape:")
    accs, deltas = [], []
    oof_true = np.stack([P[folds == truth[i], i].mean(0) for i in range(len(truth))])
    m_true = macro(y, oof_true)
    for s in range(4):
        idx = [np.where(folds == f)[0][s] for f in range(5)]
        rs = recover(P[idx])
        accs.append((rs == truth).mean())
        # what does misassignment cost at the AUC level? score with recovered vs true folds
        a = macro(y, np.stack([P[folds == rs[i], i].mean(0) for i in range(len(rs))]))
        deltas.append(a - m_true)
        print(f"  seed slot {s}: {(rs == truth).mean():5.1%} accuracy   "
              f"macro {a:.4f} vs true-fold {m_true:.4f}  ({a - m_true:+.4f})")
    print(f"\n  mean accuracy {np.mean(accs):.1%}   "
          f"mean AUC cost of misassignment {np.mean(deltas):+.4f} "
          f"(range {min(deltas):+.4f} to {max(deltas):+.4f})")
    print("  Sign is the point: misassignment biases UPWARD, so a recovered-fold OOF is a "
          "CEILING\n  on the honest number, not an unbiased estimate of it.")
    return 0


def arm_ft_b() -> int:
    f = np.load(D / "_ft_b_gold.npz", allow_pickle=True)
    P, ids = f["pred"], f["ids"]            # (n_studies, 5 folds, 12)
    fm = np.transpose(P, (1, 0, 2))         # -> (5, n_studies, 12)
    y = gold_labels(ids)
    keep = ~np.isnan(y).any(1) & ~np.isnan(fm).any((0, 2))
    fm, y, ids = fm[:, keep], y[keep].astype(int), np.array(ids)[keep]

    r = recover(fm)
    print(f"ft_b on {len(ids)} gold studies")
    print(f"  recovered partition {np.bincount(r, minlength=5)}  (flat would be ~{len(ids)/5:.1f})")
    chi = ((np.bincount(r, minlength=5) - len(ids) / 5) ** 2 / (len(ids) / 5)).sum()
    print(f"  chi2 vs flat = {chi:.2f} on 4 df   "
          f"({'consistent with flat' if chi < 9.49 else 'NOT flat -- suspect'})")

    oof = np.stack([fm[r[i], i] for i in range(len(ids))])
    allf = fm.mean(0)
    print(f"\n  ALL-fold (§3n's gate)      {macro(y, allf):.4f}   inflated, memorised")
    print(f"  held-out fold only (OOF)   {macro(y, oof):.4f}   <- the honest read")
    print(f"  inflation                  {macro(y, allf) - macro(y, oof):+.4f}")
    print("\n  reference on the SAME studies: pilkwang honest OOF 0.8516 (§3m arm A)")
    print("  NOT like-for-like -- pilkwang's is a 4-seed mean, this is ONE model.")
    for j, t in enumerate(LABELS):
        print(f"    {t:18s} {auc(y[:, j], oof[:, j]):.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--arm", choices=["ft_b"])
    a = ap.parse_args()
    if a.validate:
        return validate()
    if a.arm == "ft_b":
        return arm_ft_b()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
