"""⛔ F1 IS DEAD. This file is the measurement that killed it — see `IMPROVEMENTS.md` §3u.

    python fusion/site_prior_ship.py --diagnose    # the 2x2 that settles it
    python fusion/site_prior_ship.py --validate    # the operational A/B
    python fusion/site_prior_ship.py --build       # the table, kept only for reference

**§3f's "+0.0023, free, unshipped" was not free and does not exist.** Shipped in the only
configuration a submission can actually run, it is **−0.0057, positive in 0% of 2,000 bootstrap
draws**, CI [−0.0063, −0.0051].

WHY THIS IS NOT JUST `site_prior_test.py` WITH AN EXPORT. That file measured +0.0023 by fitting
the per-site prevalence on `y[train]` where `y` is the *reference* label source it was also scored
against. **A submission cannot do that** — at test time there are no labels, so the prior must be
built from a label table we own and applied to studies whose only observable is a DICOM header.

**THE DIAGNOSTIC, and it disproves the obvious hypothesis.** The first guess was source-matching:
that fitting and scoring on the same labels flattered it. `--diagnose` varies the two
independently and the split is **not** along the diagonal:

| prior source → scored against | delta |
|---|--:|
| reference → reference | **+0.0022** ← §3f as measured |
| targets.csv → targets.csv | **−0.0042** ← same-source, and NEGATIVE |
| targets.csv → reference | **−0.0057** ← operational |
| reference → targets.csv | **+0.0008** ← cross-source, and POSITIVE |

**It splits by PRIOR SOURCE, not by matching.** A prior built from `lixin_gpt56` helps in both
columns; one built from `targets.csv` (= `steven_v2`) hurts in both. So §3f measured a property of
*one particular label table's* per-site prevalence, not a property of site prevalence.

**And the best honest number for even the good source is +0.0008** — the cross-source cell — which
is a fifth of the ±0.005 instrument precision. There is no configuration worth a submission slot:
what we own is negative, and what is positive is unmeasurably small and only positive when scored
against a source it was not fitted on.

**WHAT IS SELECTED AND WHAT IS NOT.** `site_prior_test.py` swept 16 (K, w) pairs and warned that
its winner was selected on the studies it scored on. Re-reading that sweep: **`K` is inert**
(10→100 moves the result by 0.0003 at fixed `w`) and **`w` is the whole story** (+0.0017 at 0.05,
+0.0023 at 0.10, −0.0000 at 0.20, −0.0079 at 0.30). Positive in **8/8** cells at w ≤ 0.10. That is
a smooth curve with one effective free parameter, not a lucky cell among sixteen — which is why
this is being shipped at all. **`K=10, w=0.10` is carried over unchanged; nothing is re-tuned
here.** Re-tuning on this script's output would be exactly §3b's error.

THE TEST-TIME FALLBACK, which is the part most likely to bite. A test study whose fingerprint was
never seen in training gets the **global** prevalence for every label, i.e. a mid-rank prior, which
shrinks it slightly toward the middle and cannot inject a wrong site's case mix. `--validate`
reports how often that happens under a simulated split so the rate is known before, not after.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "extractor"))
D = PROJ / "data"

from metrics import auc  # noqa: E402

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
NEUTRAL = (D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56"
           / "labels_llm_gpt56sol.csv")
K_SHRINK, W_BLEND = 10, 0.10          # §3f / site_prior_test.py. NOT re-tuned here.


def rank01(v):
    return rankdata(v) / len(v)


def macro(y, P):
    return float(np.nanmean([auc(y[:, i], P[:, i]) for i in range(len(L))]))


def _load():
    fp = pd.read_csv(D / "site_fingerprint.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")
    tgt = pd.read_csv(D / "targets.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")
    return fp, tgt


def _prior_from(tgt: pd.DataFrame, fps: pd.Series, ids) -> tuple[dict, np.ndarray]:
    """Empirical-Bayes prevalence per fingerprint, shrunk toward the global rate."""
    y = tgt.loc[ids, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    s = fps.loc[ids].to_numpy()
    glob = np.nanmean(y, axis=0)
    table = {}
    for f in np.unique(s):
        k = s == f
        cnt = (~np.isnan(y[k])).sum(0)
        tot = np.nansum(y[k], axis=0)
        table[str(f)] = ((tot + K_SHRINK * glob) / (cnt + K_SHRINK)).round(6).tolist()
    return table, glob


def validate() -> int:
    fp, tgt = _load()
    ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    oof = pd.read_csv(PROJ / "fusion" / "runs_pilkwang" / "oof_all.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")
    gold = set(pd.read_csv(D / "train.csv").dropna(subset=L).StudyInstanceUID)

    ids = [u for u in oof.index if u in ref.index and u in fp.index and u in tgt.index
           and u not in gold]
    yv = ref.loc[ids, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~np.isnan(yv).any(1)
    ids = [u for u, k in zip(ids, ok) if k]
    y = (yv[ok] > 0.5).astype(float)
    P = oof.loc[ids, L].to_numpy(float)
    fps = fp["fingerprint"].astype(str)

    print(f"reference {NEUTRAL.name} (held out of BOTH arms) · gold excluded · n={len(ids)}")
    print(f"prior fitted on targets.csv — the operational source, NOT the reference\n")

    base = np.column_stack([rank01(P[:, i]) for i in range(len(L))])
    b = macro(y, base)

    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(ids))       # random study folds -- the LB splits by study
    out = base.copy()
    unseen = 0
    for f in range(5):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        tr_ids = [ids[i] for i in tr]
        table, glob = _prior_from(tgt, fps, tr_ids)
        pv = np.zeros((len(te), len(L)))
        for j, i in enumerate(te):
            row = table.get(str(fps.loc[ids[i]]))
            if row is None:
                unseen += 1
                row = glob
            pv[j] = row
        for i_l in range(len(L)):
            out[te, i_l] = (1 - W_BLEND) * base[te, i_l] + W_BLEND * rank01(pv[:, i_l])

    m = macro(y, out)
    print(f"  baseline (rank of pilkwang OOF)      {b:.4f}")
    print(f"  + site prior, K={K_SHRINK} w={W_BLEND:.2f}  {m:.4f}")
    print(f"  DELTA                                 {m-b:+.4f}")
    print(f"    (site_prior_test.py, prior fitted on the REFERENCE: +0.0023)")
    print(f"\n  unseen-fingerprint fallbacks: {unseen}/{len(ids)} = {100*unseen/len(ids):.1f}%")
    print("    those studies get the global rate, i.e. a mid-rank prior -- it cannot inject")
    print("    another site's case mix, only shrink them slightly toward the middle.")

    rng2 = np.random.default_rng(1)
    dd = np.array([macro(y[k], out[k]) - macro(y[k], base[k])
                   for k in (rng2.integers(0, len(y), len(y)) for _ in range(2000))])
    dd = dd[~np.isnan(dd)]
    print(f"\n  bootstrap over studies: positive in {100*(dd>0).mean():.0f}% of draws, "
          f"95% CI [{np.percentile(dd,2.5):+.4f}, {np.percentile(dd,97.5):+.4f}]")
    print("\n  SHIP ONLY IF the operational delta is still positive and the CI is not "
          "dominated by zero.\n  Do NOT re-tune K or w against this number (§3b).")
    return 0


def diagnose() -> int:
    """The 2x2 that kills F1: vary the prior's label SOURCE and the scoring source separately."""
    fp, tgt = _load()
    ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    oof = pd.read_csv(PROJ / "fusion" / "runs_pilkwang" / "oof_all.csv").drop_duplicates(
        "StudyInstanceUID").set_index("StudyInstanceUID")
    gold = set(pd.read_csv(D / "train.csv").dropna(subset=L).StudyInstanceUID)
    ids = [u for u in oof.index if u in ref.index and u in fp.index and u in tgt.index
           and u not in gold]
    yv = ref.loc[ids, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~np.isnan(yv).any(1)
    ids = [u for u, k in zip(ids, ok) if k]
    P = oof.loc[ids, L].to_numpy(float)
    fps = fp["fingerprint"].astype(str)
    bin_ = lambda df: (df.loc[ids, L].apply(  # noqa: E731
        pd.to_numeric, errors="coerce").to_numpy(float) > 0.5).astype(float)
    Yref, Ytgt = bin_(ref), bin_(tgt)

    def run(prior_src, score_y):
        base = np.column_stack([rank01(P[:, i]) for i in range(len(L))])
        out = base.copy()
        fold = np.random.default_rng(0).integers(0, 5, len(ids))
        for f in range(5):
            te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
            tbl, glob = _prior_from(prior_src, fps, [ids[i] for i in tr])
            pv = np.array([tbl.get(str(fps.loc[ids[i]]), glob) for i in te])
            for j in range(len(L)):
                out[te, j] = (1 - W_BLEND) * base[te, j] + W_BLEND * rank01(pv[:, j])
        return macro(score_y, out) - macro(score_y, base)

    print(f"n={len(ids)} · K={K_SHRINK} w={W_BLEND} · instrument precision ±0.005\n")
    print("  prior source -> scored against      delta")
    print(f"  reference    -> reference        {run(ref, Yref):+.4f}   <- §3f's +0.0023")
    print(f"  targets.csv  -> targets.csv      {run(tgt, Ytgt):+.4f}   <- same-source, other pair")
    print(f"  targets.csv  -> reference        {run(tgt, Yref):+.4f}   <- OPERATIONAL")
    print(f"  reference    -> targets.csv      {run(ref, Ytgt):+.4f}   <- cross, other direction")
    print("\n  The split is by PRIOR SOURCE, not by source-matching: a prior built from the")
    print("  reference helps in both columns, one built from targets.csv hurts in both.")
    print("  So §3f's +0.0023 is a property of `lixin_gpt56`'s per-site prevalence, and the")
    print("  honest cross-source value of even THAT is +0.0008 -- inside the ±0.005 noise.")
    print("\n  VERDICT: F1 does not ship in any configuration. See IMPROVEMENTS.md §3u.")
    return 0


def build() -> int:
    """The whole-train table the kernel ships. Keyed by fingerprint STRING, not site_id --
    site_id is a `factorize` index and is meaningless outside the frame that produced it."""
    fp, tgt = _load()
    ids = [u for u in tgt.index if u in fp.index]
    table, glob = _prior_from(tgt, fp["fingerprint"].astype(str), ids)
    obj = {"labels": L, "K": K_SHRINK, "w": W_BLEND,
           "global": [round(float(x), 6) for x in glob], "sites": table}
    out = D / "site_prior_table.json"
    out.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  {len(table)} fingerprints over {len(ids)} training studies, K={K_SHRINK}")
    print(f"  global prevalence: " + ", ".join(f"{l} {g:.3f}" for l, g in zip(L, glob)))
    print("\n  The kernel keys on the fingerprint STRING:")
    print("    Manufacturer|ManufacturerModelName|SoftwareVersions|round(median(ImagingFrequency),3)|ReceiveCoilName")
    print("  with 'NA' for any missing field, and the per-study MODE over its series.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.validate:
        return validate()
    if a.diagnose:
        return diagnose()
    if a.build:
        return build()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
