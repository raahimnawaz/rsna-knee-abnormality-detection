"""Read the cartilage-measurement test by its rule (LOG 10-16 ~10:30).

    python fusion/cartilage_read.py <shard output dir or measurements.csv> [...]

Rule: the route continues only if, on Lateral OA or PF OA, a boosted model on the measurements alone reaches held-out
AUROC >= 0.85 AND within-class Spearman rho <= 0.5 with our five teacher folds' held-out predictions. Held out = five
folds by site (`folds_site_stage_b.csv`); targets `targets_consensus_v2` >= 0.5. Gold-58 is printed, never used.

Scored as it would ship (review 10-16, finding 1): EVERY sampled non-gold study counts. A study the segmenter could not
read (an error, or `seg_ok` 0) keeps blank features, so the model ranks it from nothing, as a submission would have
to. The readable-only numbers are printed beside them, look only. A study that appears in two input files stops the
read (finding 2): pass the six shards' outputs, not a folder that also holds a smoke run.
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
sys.path.insert(0, str(PROJ / "notebooks"))
from build_cartilage_kernel import sample_uids  # noqa: E402

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
RULE_FINDINGS = ["Lateral OA", "PF OA"]
REPORT_FINDINGS = ["Lateral Meniscus", "Synovitis", "Medial OA"]
AUC_MIN, RHO_MAX = 0.85, 0.5
NOT_FEATURES = {"StudyInstanceUID", "series", "side", "side_src", "error", "decode_s", "segment_s", "measure_s",
                "seg_ok", "source"}
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


def load(paths: list[str]) -> pd.DataFrame:
    """All measurements.csv under the given dirs / files. A study in two files is an error, not a silent pick."""
    files = []
    for p in paths:
        files += [p] if p.endswith(".csv") else sorted(glob.glob(str(Path(p) / "**" / "measurements.csv"), recursive=True))
    if not files:
        raise SystemExit(f"no measurements.csv under {paths}")
    frames = []
    for f in files:
        d = pd.read_csv(f, dtype={"StudyInstanceUID": str})
        d["source"] = f
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    dup = m[m.StudyInstanceUID.duplicated(keep=False)]
    if len(dup):
        where = dup.groupby("StudyInstanceUID").source.apply(lambda s: " + ".join(sorted(set(s)))).value_counts()
        raise SystemExit(f"{dup.StudyInstanceUID.nunique()} studies appear in more than one file:\n"
                         + "\n".join(f"  {n} in {w}" for w, n in where.items())
                         + "\npass only the six shards' outputs")
    print(f"{len(files)} files")
    return m.set_index("StudyInstanceUID")


def fit_cv(X: np.ndarray, y: np.ndarray, fo: np.ndarray) -> np.ndarray:
    p = np.zeros(len(y))
    for k in np.unique(fo):
        te = fo == k
        p[te] = HistGradientBoostingClassifier(**GBM).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    return p


def main() -> None:
    m = load(sys.argv[1:])
    t = pd.read_csv(PROJ / "data/train.csv").set_index("StudyInstanceUID")
    gold_ids = set(t.index[t[L].notna().any(axis=1)])
    sampled = sample_uids()
    rest = [u for u in sampled if u not in gold_ids]
    missing = [u for u in sampled if u not in m.index]
    err = m["error"].notna() if "error" in m else pd.Series(False, index=m.index)
    seg = (m["seg_ok"] == 1) if "seg_ok" in m else pd.Series(False, index=m.index)
    readable = seg & ~err
    n_meas = len(m.index.intersection(sampled))
    print(f"sampled {len(sampled)} (gold 58 + {len(rest)}) | measured {n_meas} | missing {len(missing)} | "
          f"errors {int(err.sum())} | segmenter failed (seg_ok 0) {int((~seg & ~err).sum())} | "
          f"FAILURE RATE {1 - readable.loc[readable.index.intersection(sampled)].mean():.1%} of measured")
    if err.any():
        print("  errors:", m.loc[err, "error"].str.slice(0, 60).value_counts().head(5).to_dict())
    for c in ("decode_s", "segment_s", "measure_s"):
        if c in m:
            print(f"  {c}: median {m.loc[readable, c].median():.1f} s, p90 {m.loc[readable, c].quantile(.9):.1f} s")
    feats = [c for c in m.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(m[c])]
    F = m[feats].astype(float)
    F.loc[~readable] = np.nan                                 # a failed study ships with nothing
    print(f"  {len(feats)} features")

    tg = pd.read_csv(PROJ / "data/student/targets_consensus_v2.csv").set_index("StudyInstanceUID")[L]
    folds = pd.read_csv(PROJ / "data/student/folds_site_stage_b.csv").set_index("StudyInstanceUID")["fold"]
    oof, og = our_oof(), our_gold()
    tr = [u for u in rest if u in m.index and u in folds.index and u in oof.index]
    gd = [u for u in sampled if u in gold_ids and u in m.index]
    X, Xg = F.loc[tr].to_numpy(), F.loc[gd].to_numpy()
    fo = folds.loc[tr].to_numpy()
    rd = readable.loc[tr].to_numpy()
    print(f"held-out set: {len(tr)} studies, {rd.sum()} readable (folds {np.bincount(fo).tolist()}) | gold {len(gd)}\n")

    verdicts = {}
    print(f"{'finding':17s} {'pos':>4s} {'meas AUROC':>10s} {'ours':>6s} {'rho':>6s} | {'readable only':>18s} | "
          f"{'gold meas / ours':>16s} | rule")
    for f in RULE_FINDINGS + REPORT_FINDINGS:
        y = (tg.loc[tr, f] >= 0.5).to_numpy().astype(int)
        o = oof.loc[tr, f].to_numpy()
        p = fit_cv(X, y, fo)
        a_m, a_o, rho = roc_auc_score(y, p), roc_auc_score(y, o), within_rho(p, o, y)
        r_auc, r_rho = roc_auc_score(y[rd], p[rd]), within_rho(p[rd], o[rd], y[rd])
        yg = t.loc[gd, f].to_numpy().astype(int)
        pg = HistGradientBoostingClassifier(**GBM).fit(X, y).predict_proba(Xg)[:, 1]
        g_m = roc_auc_score(yg, pg) if 0 < yg.sum() < len(yg) else np.nan
        g_o = roc_auc_score(yg, og.loc[gd, f].to_numpy()) if 0 < yg.sum() < len(yg) else np.nan
        passed = a_m >= AUC_MIN and rho <= RHO_MAX
        tag = ("PASS" if passed else "fail") if f in RULE_FINDINGS else "(report only)"
        if f in RULE_FINDINGS:
            verdicts[f] = passed
        print(f"{f:17s} {int(y.sum()):4d} {a_m:10.3f} {a_o:6.3f} {rho:6.2f} | AUROC {r_auc:.3f} rho {r_rho:4.2f} | "
              f"{g_m:7.3f} / {g_o:.3f} | {tag}")
    rule = (f"rule: AUROC >= {AUC_MIN} and rho <= {RHO_MAX} on Lateral OA or PF OA, all sampled studies "
            f"({', '.join(f'{k} {v}' for k, v in verdicts.items())})")
    if missing:
        print(f"\nVERDICT: INCOMPLETE -- {len(missing)} sampled studies have no row yet; {rule}")
    else:
        print(f"\nVERDICT: {'CONTINUE' if any(verdicts.values()) else 'CLOSED'} -- {rule}")


if __name__ == "__main__":
    main()
