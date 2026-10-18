"""The cell screen (owner 10-18, LOG): boosted trees on location-fixed cells of K's knee-centred crops.

    python fusion/cell_screen.py feats              # data/kcache -> data/cell_feats.npz (CPU, all cores)
    python fusion/cell_screen.py fit [--labels dc]  # xgboost per weak finding, fold 0 -> the 847 held-out; rule + importance

K's cache puts every knee in the same canonical frame around the joint and patella, so a cell index is a fixed
anatomical place. Per study and region (lat_sag, lat_cor, pf_ax, ant_sag): intensities z-scored over the filled slices
(zero = outside the field of view, ignored); two depth bins (slices 0-3, 4-7); an 8 x 8 grid of 32-px cells; per cell
the mean, s.d. and 90th percentile of intensity and the mean |Laplacian| (fine detail: small lesions, edges).
4 regions x 2 bins x 64 cells x 4 = 2,048 features; a cell with no data is NaN (xgboost routes it).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
CACHE = PROJ / "data" / "kcache"
FEATS = PROJ / "data" / "cell_feats.npz"
REGIONS = ["lat_sag", "lat_cor", "pf_ax", "ant_sag"]
STATS = ["mean", "sd", "p90", "lap"]
G, CELL, PER = 8, 32, 8
WEAK4 = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis"]
FOLD0 = PROJ / "fusion" / "runs_k_f0" / "runs_convnext_small" / "fold0"
TEACH = PROJ / "fusion" / "runs_cnxs_teach1_f0" / "runs_convnext_small" / "fold0" / "oof_all.csv"


def names() -> list[str]:
    return [f"{r}|d{b}|r{i}c{j}|{s}" for r in REGIONS for b in (0, 1) for i in range(G) for j in range(G) for s in STATS]


def study_feats(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full((len(REGIONS), 2, G, G, len(STATS)), np.nan, np.float32)
    for r in range(len(REGIONS)):
        x = vol[r * PER:(r + 1) * PER].astype(np.float32)
        m = mask[r * PER:(r + 1) * PER].astype(bool)
        if not m.any():
            continue
        x[~m] = 0
        valid = x > 0
        if valid.sum() < 1000:
            continue
        mu, sd = x[valid].mean(), x[valid].std() + 1e-6
        z = np.where(valid, (x - mu) / sd, np.nan)
        lap = np.full_like(z, np.nan)
        c = z[:, 1:-1, 1:-1]
        lap[:, 1:-1, 1:-1] = np.abs(4 * c - z[:, :-2, 1:-1] - z[:, 2:, 1:-1] - z[:, 1:-1, :-2] - z[:, 1:-1, 2:])
        for b, sl in enumerate((slice(0, 4), slice(4, 8))):
            keep = m[sl]
            if not keep.any():
                continue
            zb, lb = z[sl][keep], lap[sl][keep]                          # (k, 256, 256)
            k = len(zb)
            zc = zb.reshape(k, G, CELL, G, CELL).transpose(1, 3, 0, 2, 4).reshape(G, G, -1)
            lc = lb.reshape(k, G, CELL, G, CELL).transpose(1, 3, 0, 2, 4).reshape(G, G, -1)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out[r, b, :, :, 0] = np.nanmean(zc, -1)
                    out[r, b, :, :, 1] = np.nanstd(zc, -1)
                    out[r, b, :, :, 2] = np.nanpercentile(zc, 90, axis=-1)
                    out[r, b, :, :, 3] = np.nanmean(lc, -1)
    return out.reshape(-1)


def _shard(args: tuple[Path, int, int]) -> tuple[list[str], np.ndarray]:
    d, lo, hi = args
    v = np.load(d / "vols.npy", mmap_mode="r")
    m = np.load(d / "masks.npy")
    ids = np.load(d / "ids.npy", allow_pickle=True)
    return [str(u) for u in ids[lo:hi]], np.stack([study_feats(np.asarray(v[i]), m[i]) for i in range(lo, hi)])


def cmd_feats() -> None:
    jobs = []
    for d in sorted(CACHE.glob("s*/k_cache")):
        n = len(np.load(d / "ids.npy", allow_pickle=True))
        jobs += [(d, lo, min(lo + 50, n)) for lo in range(0, n, 50)]
    with Pool() as pool:
        parts = pool.map(_shard, jobs)
    ids = sum((p[0] for p in parts), [])
    X = np.concatenate([p[1] for p in parts])
    if len(set(ids)) != len(ids):
        raise SystemExit("a study appears twice in the cache")
    np.savez_compressed(FEATS, X=X, ids=np.array(ids), names=np.array(names()))
    print(f"{X.shape} -> {FEATS} | NaN share {np.isnan(X).mean():.3f}")


def cmd_fit(labels: str) -> None:
    import xgboost as xgb
    from scipy.stats import spearmanr
    sys.path.insert(0, str(PROJ / "fusion"))
    sys.path.insert(0, str(PROJ / "extractor"))
    import score_oof as so
    from metrics import auc
    z = np.load(FEATS, allow_pickle=True)
    X = pd.DataFrame(z["X"], index=z["ids"], columns=z["names"])
    src = {"v3": "targets_teacher_v3.csv", "dc": "targets_teacher_v3_dc.csv"}[labels]
    T = pd.read_csv(PROJ / "data" / "student" / src).set_index("StudyInstanceUID")
    tr = pd.read_csv(FOLD0 / "train_studies.csv").StudyInstanceUID.astype(str)
    va = pd.read_csv(FOLD0 / "val_studies.csv").StudyInstanceUID.astype(str)
    ref = pd.read_csv(so.NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    teach = pd.read_csv(TEACH).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    rng = np.random.default_rng(20261018)
    print(f"labels {src} | features {X.shape[1]} | train {len(tr)} | held-out {len(va)} | xgboost {xgb.__version__}")
    params = {"objective": "binary:logistic", "tree_method": "hist", "eta": 0.05, "max_depth": 4, "subsample": 0.8,
              "colsample_bytree": 0.3, "min_child_weight": 5, "lambda": 1.0, "eval_metric": "auc", "seed": 20261018,
              "nthread": 0}
    rows, preds = [], {}
    for f in WEAK4:
        y = T.loc[tr, f]
        y = y[y.notna()]
        ids = y.index.to_numpy()
        inner = rng.random(len(ids)) < 0.10
        dtr = xgb.DMatrix(X.loc[ids[~inner]], label=(y[~inner] >= 0.5).astype(int))
        din = xgb.DMatrix(X.loc[ids[inner]], label=(y[inner] >= 0.5).astype(int))
        bst = xgb.train(params, dtr, 600, evals=[(din, "inner")], early_stopping_rounds=50, verbose_eval=False)
        va_ok = [u for u in va if u in ref.index and u in teach.index]
        p = pd.Series(bst.predict(xgb.DMatrix(X.loc[va_ok]), iteration_range=(0, bst.best_iteration + 1)), index=va_ok)
        yr = pd.to_numeric(ref.loc[va_ok, f], errors="coerce")
        ok = yr.notna()
        yb = (yr[ok] > 0.5).astype(int).to_numpy()
        a = auc(yb, p[ok].to_numpy())
        t = teach.loc[p[ok].index, f].to_numpy()
        rn, rp = spearmanr(p[ok][yb == 0], t[yb == 0])[0], spearmanr(p[ok][yb == 1], t[yb == 1])[0]
        rho = (rn + rp) / 2
        verdict = "PASS" if a >= 0.84 and rho <= 0.5 else "fail"
        rows.append((f, len(ids), int((y >= 0.5).sum()), bst.best_iteration + 1, int(ok.sum()), int(yb.sum()), a,
                     auc(yb, t), rn, rp, rho, verdict))
        preds[f] = (bst, va_ok, p, yr)
    print(f"\n{'finding':<17}{'train':>7}{'pos':>6}{'rounds':>7}{'held':>6}{'pos':>5}{'AUROC':>8}{'teacher':>9}"
          f"{'rho-':>7}{'rho+':>7}{'rho':>7}  rule (>= 0.84 and <= 0.5)")
    for r in rows:
        print(f"{r[0]:<17}{r[1]:>7}{r[2]:>6}{r[3]:>7}{r[4]:>6}{r[5]:>5}{r[6]:>8.3f}{r[7]:>9.3f}{r[8]:>7.2f}{r[9]:>7.2f}"
              f"{r[10]:>7.2f}  {r[11]}")
    # where each finding is read from: gain importance summed by region x depth, and the top cells
    print("\nshare of total gain by region x depth bin (look only)")
    for f in WEAK4:
        g = preds[f][0].get_score(importance_type="total_gain")
        s = pd.Series(g, dtype=float)
        if s.empty:
            continue
        grp = s.groupby([k.split("|")[0] + "|" + k.split("|")[1] for k in s.index]).sum() / s.sum()
        top = s.groupby(["|".join(k.split("|")[:3]) for k in s.index]).sum().sort_values(ascending=False) / s.sum()
        print(f"  {f:<17} " + "  ".join(f"{k} {v:.2f}" for k, v in grp.sort_values(ascending=False).head(4).items())
              + " | top cells: " + ", ".join(f"{k} {v:.2f}" for k, v in top.head(4).items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["feats", "fit"])
    ap.add_argument("--labels", choices=["v3", "dc"], default="v3")
    a = ap.parse_args()
    cmd_feats() if a.cmd == "feats" else cmd_fit(a.labels)


if __name__ == "__main__":
    main()
