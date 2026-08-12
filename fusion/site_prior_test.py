"""If harmonising site AWAY costs 0.013-0.032, does adding site signal GAIN? The inverse test.

WHY THIS EXISTS. `fusion/harmonise_test.py` found that removing between-scanner score differences
*hurts* (site_id -0.0319 at 18σ), and the prevalence table says why: case mix genuinely differs by
scanner -- Medial OA runs 0.479 at one manufacturer against 0.338 at another. The between-group
ordering is signal. This asks the obvious next question: is that signal fully exploited already,
or does an EXPLICIT per-site prevalence prior add on top?

    python fusion/site_prior_test.py

FOLD SCHEME -- AND THIS IS THE POINT OF THE FILE. It uses **random study-level folds, not
`data/folds_site.csv`**, and that is deliberate. Site-grouped folds put every study from a site in
one fold, so a site's prevalence can never be estimated from the training folds -- the scheme makes
this quantity structurally invisible. The two schemes estimate different things:

  * site-grouped answers *"how would this do at a NEW hospital?"*
  * random answers *"how would this do on held-out studies from THESE hospitals?"*

The competition's test split is studies, not sites, so the leaderboard asks the second question.
**§2j's "nothing gets compared except under site-grouped folds" is the right rule for reporting
generalisation and the wrong rule for predicting this leaderboard**, and that distinction has been
invisible in this project until now.

Prevalence is shrunk toward the global rate (empirical-Bayes, strength `K`) because §3b is the
standing warning: 265 fingerprints over ~4,300 studies is ~16 studies each, and an unshrunk
per-site rate on 16 studies is exactly the regime that reports a gain and delivers a loss.
"""
from __future__ import annotations

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


def macro(y, P):
    return float(np.nanmean([auc(y[:, i], P[:, i]) for i in range(len(L))]))


def rank01(v):
    return rankdata(v) / len(v)


def main() -> None:
    oof = pd.read_csv(PROJ / "fusion" / "runs_pilkwang" / "oof_all.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")
    ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    gold = set(pd.read_csv(D / "train.csv").dropna(subset=L).StudyInstanceUID)
    site = pd.read_csv(D / "site_fingerprint.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")

    ids = [u for u in oof.index if u in ref.index and u not in gold and u in site.index]
    yv = ref.loc[ids, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~np.isnan(yv).any(axis=1)
    ids = [u for u, k in zip(ids, ok) if k]
    y = (yv[ok] > 0.5).astype(float)
    P = oof.loc[ids, L].to_numpy(float)
    s = site.loc[ids, "site_id"].astype(str).to_numpy()
    print(f"reference {NEUTRAL.name} · gold excluded · n={len(ids)} · {len(np.unique(s))} sites")
    print(f"  baseline macro: {macro(y, P):.4f}\n")

    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(ids))            # RANDOM folds, per the docstring
    print(f"  {'K (shrinkage)':<16}{'w':>6}{'macro':>9}{'delta':>9}")
    base_oof = np.column_stack([rank01(P[:, i]) for i in range(len(L))])
    b = macro(y, base_oof)
    best = (None, None, b)
    for K in (10, 25, 50, 100):
        for w in (0.05, 0.10, 0.20, 0.30):
            out = base_oof.copy()
            for f in range(5):
                te = np.where(fold == f)[0]
                tr = np.where(fold != f)[0]
                for i in range(len(L)):
                    glob = y[tr, i].mean()
                    df = pd.DataFrame({"s": s[tr], "y": y[tr, i]})
                    agg = df.groupby("s")["y"].agg(["sum", "count"])
                    prior = ((agg["sum"] + K * glob) / (agg["count"] + K)).to_dict()
                    pv = np.array([prior.get(x, glob) for x in s[te]])
                    out[te, i] = (1 - w) * base_oof[te, i] + w * rank01(pv)
            m = macro(y, out)
            print(f"  {K:<16}{w:>6.2f}{m:>9.4f}{m-b:>+9.4f}")
            if m > best[2]:
                best = (K, w, m)
    print(f"\n  baseline (rank of OOF): {b:.4f}")
    if best[0] is None:
        print("  VERDICT: no (K, w) beats the baseline. The site signal is already exploited.")
    else:
        print(f"  BEST: K={best[0]} w={best[1]:.2f} -> {best[2]:.4f} "
              f"({best[2]-b:+.4f})")
        print("  Confirm on the leaderboard before believing it (§3b): this is selected over "
              "16 (K, w) pairs on the same studies it is scored on.")


if __name__ == "__main__":
    main()
