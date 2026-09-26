"""The pre-registered gold-58 read of our Qwen labels (`RUNBOOK.md`, the Qwen labels block).

    python extractor/bench_qwen_labels.py                       # data/qwen_labels/qwen35_9b_v1
    python extractor/bench_qwen_labels.py --tag <other>         # a later candidate: spends budget

Exactly two candidates touch gold, each once, fixed before the run finished:
  C1  the Qwen table alone                                    vs `steven_v2` (our training targets)
  C2  equal rank-mean of `steven_v2` and the Qwen table,
      mapped back onto `steven_v2`'s own per-label values     vs `steven_v2`
C2's mapping is what makes it TRAINABLE: a raw rank-mean has mean 0.5 on every label, which would
teach BCE a 50% prevalence. Mapping ranks onto steven_v2's sorted values keeps its marginal
exactly and changes only the ORDER -- and order is all the AUROC below reads.

THE RULE (DATA.md §3's Stage L gate, one change made before any result, stated in the RUNBOOK):
paired bootstrap over the 58 gold studies, 4,000 resamples, seed 20260906.
  PASS  point estimate >= +0.010 AND the 95% CI excludes 0
  GUARD the macro of the six frozen labels (ACL, MCL, Medial Meniscus, Medial OA, PF OA, Baker's)
        may not fall more than 0.02 below steven_v2's
If both pass, C2 is preferred (it keeps the incumbent's calibration). Everything else printed --
per-label deltas, the 86-study hand set, mention rates -- is a diagnostic, not a second chance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_external import gold_rows_differ, leaks_gold  # noqa: E402
from metrics import auc  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
D = PROJ / "data"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FROZEN = ["ACL", "MCL", "Medial Meniscus", "Medial OA", "PF OA", "Baker's"]
STEVEN = D / "public_llm_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv"
N_BOOT, SEED = 4000, 20260906


def table(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    return d[LABELS].apply(pd.to_numeric, errors="coerce")


def rank_blend(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Equal rank-mean of two tables over ALL their shared studies, then each label's ranks mapped
    onto `base`'s sorted values: base's marginal exactly, the blend's order."""
    idx = base.index.intersection(other.index)
    out = pd.DataFrame(index=idx)
    for lab in LABELS:
        b, o = base.loc[idx, lab].to_numpy(float), other.loc[idx, lab].to_numpy(float)
        r = (rankdata(b) + rankdata(o)) / 2
        order = np.argsort(rankdata(r, method="ordinal"))
        v = np.empty(len(idx))
        v[order] = np.sort(b)
        out[lab] = v
    return out


def macro(Y: np.ndarray, V: np.ndarray, cols=None) -> float:
    cols = range(len(LABELS)) if cols is None else cols
    return float(np.nanmean([auc(Y[:, k], V[:, k]) if len(set(Y[:, k])) > 1 else np.nan for k in cols]))


def paired(Y, A, B, cols=None):
    """B - A on the same resampled studies; returns point, lo, hi, P(>0)."""
    rng = np.random.default_rng(SEED)
    ds = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(Y), len(Y))
        ds.append(macro(Y[i], B[i], cols) - macro(Y[i], A[i], cols))
    ds = np.asarray(ds)
    ds = ds[np.isfinite(ds)]
    return macro(Y, B, cols) - macro(Y, A, cols), np.percentile(ds, 2.5), np.percentile(ds, 97.5), (ds > 0).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="qwen35_9b_v1")
    ap.add_argument("--write", action="store_true",
                    help="also write both candidates as training tables: <tag>/targets_c1.csv, targets_c2.csv")
    a = ap.parse_args()

    tr = pd.read_csv(D / "train.csv")
    gold = tr.dropna(subset=LABELS).set_index("StudyInstanceUID")[LABELS].astype(float)
    gidx = gold.index
    Y = gold.to_numpy(float)
    rest = pd.Index(tr.StudyInstanceUID[~tr.StudyInstanceUID.isin(gidx)])

    base = table(STEVEN)
    q = table(D / "qwen_labels" / a.tag / "targets.csv")
    if len(q) < 4000:
        raise SystemExit(f"{a.tag}: {len(q)} studies -- the run has not finished; C2's ranks need the corpus")
    blend = rank_blend(base, q)
    cands = {"C1 qwen alone": q, "C2 rank-mean -> steven_v2 marginal": blend}

    if a.write:
        tr_ids = tr.StudyInstanceUID.astype(str)
        for fn, T in (("targets_c1.csv", q), ("targets_c2.csv", blend)):
            out = T.reindex(tr_ids)
            if out[LABELS].isna().any().any():
                raise SystemExit(f"{fn}: {int(out[LABELS].isna().any(axis=1).sum())} studies without a score")
            out.reset_index().rename(columns={"index": "StudyInstanceUID"}).to_csv(
                D / "qwen_labels" / a.tag / fn, index=False)
        print(f"wrote targets_c1.csv, targets_c2.csv under data/qwen_labels/{a.tag}/ ({len(tr_ids)} studies)")
    B = base.reindex(gidx).to_numpy(float)
    print(f"gold-58: steven_v2 {macro(Y, B):.4f}  (the incumbent; DATA.md §11 reads 0.8873)\n")
    for name, T in cands.items():
        vg, vn = T.reindex(gidx).to_numpy(float), T.reindex(rest).to_numpy(float)
        if not np.isfinite(vg).all():
            raise SystemExit(f"{name}: {int((~np.isfinite(vg)).sum())} gold cells missing")
        if leaks_gold(vg, Y) or gold_rows_differ(vg, vn):
            raise SystemExit(f"{name}: fails the leak check -- not a result")
        d, lo, hi, p = paired(Y, B, vg)
        f0, flo, fhi, _ = paired(Y, B, vg, [LABELS.index(x) for x in FROZEN])
        ok, guard = d >= 0.010 and lo > 0, f0 >= -0.02
        print(f"{name}")
        print(f"  macro {macro(Y, vg):.4f}   paired {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {p:.3f}"
              f"   -> {'PASS' if ok else 'fail'}")
        print(f"  GUARD frozen-six {f0:+.4f} [{flo:+.4f}, {fhi:+.4f}]  -> {'ok' if guard else 'BREACHED'}")
        print(f"  VERDICT: {'PASS' if ok and guard else 'FAIL'}")
        per = [auc(Y[:, k], vg[:, k]) - auc(Y[:, k], B[:, k]) for k in range(len(LABELS))]
        print("  per label (diagnostic): " + "  ".join(f"{l} {x:+.3f}" for l, x in zip(LABELS, per)) + "\n")

    # ---- diagnostics: none of these gate anything
    print("=== diagnostics ===")
    rho = [spearmanr(q.reindex(rest)[l], base.reindex(rest)[l], nan_policy="omit").statistic for l in LABELS]
    print(f"Spearman vs steven_v2 on non-gold rows: mean {np.nanmean(rho):.3f}  "
          + " ".join(f"{l[:6]} {r:.2f}" for l, r in zip(LABELS, rho)))
    probs = D / "qwen_labels" / a.tag / "probs.jsonl"
    if probs.exists():
        import json
        P = np.array([json.loads(s)["probs"] for s in probs.read_text().splitlines()])
        print("P(mentioned) = 1 - P(F), corpus mean. Qwen counts an explicit denial as a mention, so this runs above\n"
              "  DATA.md §2's regex mention rates (Synovitis 0.124, Fracture 0.205, Effusion 0.832):")
        print("  " + "  ".join(f"{l[:9]} {1 - P[:, k, 5].mean():.3f}" for k, l in enumerate(LABELS)))
    hand = pd.read_csv(PROJ / "labeling/model_labels.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    hidx = hand.index.intersection(q.index)
    H = hand.loc[hidx, LABELS].to_numpy(float)
    print(f"hand set (n={len(hidx)}, binary, one reader -- the iteration instrument, not truth): "
          f"steven_v2 {macro(H, base.reindex(hidx).to_numpy(float)):.4f}  "
          f"qwen {macro(H, q.reindex(hidx).to_numpy(float)):.4f}  "
          f"C2 {macro(H, blend.reindex(hidx).to_numpy(float)):.4f}")


if __name__ == "__main__":
    main()
