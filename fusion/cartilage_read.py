"""Read the cartilage-measurement test by its rule (LOG 10-16 ~10:30).

    python fusion/cartilage_read.py <dir with the shards' measurements.csv, searched recursively>

Rule: the route continues only if, on Lateral OA or PF OA, a boosted model on the measurements alone reaches held-out
AUROC >= 0.85 AND within-class Spearman rho <= 0.5 with our five teacher folds' held-out predictions. Held out = five
folds by site (`folds_site_stage_b.csv`); targets `targets_consensus_v2` >= 0.5. Gold-58 is printed, never used.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[1]
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
RULE_FINDINGS = ["Lateral OA", "PF OA"]
REPORT_FINDINGS = ["Lateral Meniscus", "Synovitis", "Medial OA"]
AUC_MIN, RHO_MAX = 0.85, 0.5
NOT_FEATURES = {"StudyInstanceUID", "series", "side", "side_src", "error", "decode_s", "segment_s", "measure_s"}
GBM = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, l2_regularization=1.0, random_state=0)


def our_oof() -> pd.DataFrame:
    R = PROJ / "fusion"
    parts = [R / "runs_cnxs_teach1_f0/runs_convnext_small/fold0/avg3/oof_all.csv"] + \
        sorted(R.glob("runs_cnxs_teach1_f12/runs_convnext_small/fold*/avg3/oof_all.csv")) + \
        sorted(R.glob("runs_cnxs_teach1_f34/runs_convnext_small/fold*/avg3/oof_all.csv"))
    assert len(parts) == 5, parts
    return pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[L]


def our_gold() -> pd.DataFrame:
    R = PROJ / "fusion"
    ds = [pd.read_csv(R / f"runs_cnxs_teach1_{p}/runs_convnext_small/oof_gold_foldmean.csv").set_index("StudyInstanceUID")[L]
          for p in ("f0", "f12", "f34")]
    return (ds[0] + 2 * ds[1] + 2 * ds[2]) / 5


def within_rho(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([spearmanr(a[y == c], b[y == c]).correlation for c in (0, 1)]))


def main() -> None:
    files = sorted(glob.glob(str(Path(sys.argv[1]) / "**" / "measurements.csv"), recursive=True))
    m = pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in files]).drop_duplicates("StudyInstanceUID")
    m = m.set_index("StudyInstanceUID")
    n_err = int(m["error"].notna().sum()) if "error" in m else 0
    ok = m[(m.get("seg_ok", 0) == 1) & (m["error"].isna() if "error" in m else True)]
    print(f"{len(files)} files | {len(m)} studies | errors {n_err} | segmentation ok {len(ok)}")
    if "error" in m and n_err:
        print("  errors:", m["error"].dropna().str.slice(0, 60).value_counts().head(5).to_dict())
    for c in ("decode_s", "segment_s", "measure_s"):
        if c in ok:
            print(f"  {c}: median {ok[c].median():.1f} s, p90 {ok[c].quantile(.9):.1f} s")
    feats = [c for c in ok.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(ok[c])]
    print(f"  {len(feats)} features")

    t = pd.read_csv(PROJ / "data/train.csv").set_index("StudyInstanceUID")
    gold_ids = t.index[t[L].notna().any(axis=1)]
    tg = pd.read_csv(PROJ / "data/student/targets_consensus_v2.csv").set_index("StudyInstanceUID")[L]
    folds = pd.read_csv(PROJ / "data/student/folds_site_stage_b.csv").set_index("StudyInstanceUID")["fold"]
    oof, og = our_oof(), our_gold()
    tr = ok.index.difference(gold_ids).intersection(folds.index).intersection(oof.index)
    gd = ok.index.intersection(gold_ids)
    X, Xg = ok.loc[tr, feats].to_numpy(float), ok.loc[gd, feats].to_numpy(float)
    fo = folds.loc[tr].to_numpy()
    print(f"held-out set: {len(tr)} studies (folds {np.bincount(fo).tolist()}) | gold with measurements: {len(gd)}\n")

    verdicts = {}
    print(f"{'finding':17s} {'pos':>4s} {'meas AUROC':>10s} {'ours AUROC':>10s} {'rho':>6s}  {'gold meas / ours':>17s}  rule")
    for f in RULE_FINDINGS + REPORT_FINDINGS:
        y = (tg.loc[tr, f] >= 0.5).to_numpy().astype(int)
        p = np.zeros(len(tr))
        for k in np.unique(fo):
            te = fo == k
            clf = HistGradientBoostingClassifier(**GBM).fit(X[~te], y[~te])
            p[te] = clf.predict_proba(X[te])[:, 1]
        a_m, a_o = roc_auc_score(y, p), roc_auc_score(y, oof.loc[tr, f].to_numpy())
        rho = within_rho(p, oof.loc[tr, f].to_numpy(), y)
        clf = HistGradientBoostingClassifier(**GBM).fit(X, y)
        yg = t.loc[gd, f].to_numpy().astype(int)
        g_m = roc_auc_score(yg, clf.predict_proba(Xg)[:, 1]) if 0 < yg.sum() < len(yg) else np.nan
        g_o = roc_auc_score(yg, og.loc[gd, f].to_numpy()) if 0 < yg.sum() < len(yg) else np.nan
        passed = a_m >= AUC_MIN and rho <= RHO_MAX
        tag = ("PASS" if passed else "fail") if f in RULE_FINDINGS else "(report only)"
        if f in RULE_FINDINGS:
            verdicts[f] = passed
        print(f"{f:17s} {int(y.sum()):4d} {a_m:10.3f} {a_o:10.3f} {rho:6.2f}  {g_m:8.3f} / {g_o:.3f}  {tag}")
    go = any(verdicts.values())
    print(f"\nVERDICT: {'CONTINUE' if go else 'CLOSED'} -- rule: AUROC >= {AUC_MIN} and rho <= {RHO_MAX} on "
          f"Lateral OA or PF OA ({', '.join(f'{k} {v}' for k, v in verdicts.items())})")


if __name__ == "__main__":
    main()
