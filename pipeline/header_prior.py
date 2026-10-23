"""Header-only prior of the report labels, and the de-bias that subtracts a share of it from a submission.

Train labels are read from reports; the test is scored against annotators. A model that sees only DICOM headers
(scanner, software, coil, protocol, series names) can only learn which sites report what, so its ranks estimate the
report-style component that does not transfer.

    python pipeline/header_prior.py cv <train_series_headers.csv> <report_labels.csv>
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CAT = ["Manufacturer", "ManufacturerModelName", "SoftwareVersions", "ReceiveCoilName", "MagneticFieldStrength"]
NUM = ["RepetitionTime", "EchoTime", "SliceThickness", "PixelBandwidth", "EchoTrainLength", "FlipAngle", "Rows"]
COLUMNS = ["StudyInstanceUID", "SeriesInstanceUID", "n_files"] + CAT + NUM + ["SeriesDescription", "error"]
N_TOKENS = 40
MAX_LEVELS = 200
DEBIAS_W = {c: -0.10 for c in LABELS} | {"Synovitis": -0.25}


def _text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        v = "\\".join(str(x).strip() for x in v)
    s = str(v).strip()
    return s or None


def _num(v) -> float | None:
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        v = v[0] if len(v) else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_series(series_dir: Path) -> dict:
    """The header of a series' first DICOM file, as one row."""
    import pydicom
    files = sorted(series_dir.glob("*.dcm"))
    row = {"StudyInstanceUID": series_dir.parent.name, "SeriesInstanceUID": series_dir.name, "n_files": len(files)}
    if not files:
        return row | {"error": "no dcm"}
    h = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
    for t in CAT + ["SeriesDescription"]:
        row[t] = _text(h.get(t))
    f = _num(h.get("MagneticFieldStrength"))
    if f is not None:
        row["MagneticFieldStrength"] = f"{f:.1f}"
    for t in NUM:
        row[t] = _num(h.get(t))
    return row


def read_split(root: Path, split: str, studies=None, workers: int = 16) -> pd.DataFrame:
    """One row per series listed in `{split}_series.csv`, read from `{split}_series/<study>/<series>/`."""
    root = Path(root)
    ser = pd.read_csv(root / f"{split}_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    if studies is not None:
        ser = ser[ser.StudyInstanceUID.isin(set(studies))]
    dirs = [root / f"{split}_series" / s / r for s, r in zip(ser.StudyInstanceUID, ser.SeriesInstanceUID)]

    def one(d: Path) -> dict:
        try:
            return read_series(d)
        except Exception as e:                                     # noqa: BLE001
            return {"StudyInstanceUID": d.parent.name, "SeriesInstanceUID": d.name, "error": repr(e)[:200]}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, dirs))
    return pd.DataFrame(rows).reindex(columns=COLUMNS)


def load_table(path: Path) -> pd.DataFrame:
    """A table `read_split` wrote, with its text columns kept as text (a field strength "3.0" stays a string)."""
    text = {c: str for c in ["StudyInstanceUID", "SeriesInstanceUID", "SeriesDescription", "error"] + CAT}
    return pd.read_csv(path, dtype=text, keep_default_na=False, na_values=[""]).reindex(columns=COLUMNS)


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def study_features(rows: pd.DataFrame, vocab: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-study features; the vocabulary (category levels, series-name tokens) is fixed on the training rows."""
    rows = rows.assign(StudyInstanceUID=rows.StudyInstanceUID.astype(str))
    g = rows.groupby("StudyInstanceUID")
    X = pd.DataFrame({c: g[c].agg(_mode) for c in CAT})
    for c in NUM:
        X[c + "_med"] = pd.to_numeric(rows[c], errors="coerce").groupby(rows.StudyInstanceUID).median()
    X["n_series"] = g.size().astype(float)
    toks = {u: set() for u in X.index}
    for u, d in zip(rows.StudyInstanceUID, rows.SeriesDescription.fillna("").astype(str).str.upper()):
        toks[u].update(pd.Series([d]).str.findall(r"[A-Z]{2,}").iloc[0])
    if vocab is None:
        vocab = {}
        for c in CAT:
            n = X[c].dropna().astype(str).value_counts()
            vocab[c] = sorted(n.index, key=lambda v: (-n[v], v))[:MAX_LEVELS]
        n = pd.Series([w for s in toks.values() for w in s]).value_counts()
        vocab["tokens"] = sorted(n.index, key=lambda w: (-n[w], w))[:N_TOKENS]
    for c in CAT:
        X[c] = X[c].astype(object).map({v: float(i) for i, v in enumerate(vocab[c])}).astype(float)
    for w in vocab["tokens"]:
        X["tok_" + w] = [float(w in toks[u]) for u in X.index]
    return X, vocab


def _model(X: pd.DataFrame, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=seed,
                                          categorical_features=[c in CAT for c in X.columns])


def fit_predict(Xtr: pd.DataFrame, Y: pd.DataFrame, Xte: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One model per finding on the studies whose report reads it; the prior for each test study."""
    out = {}
    for c in LABELS:
        m = Y[c].notna().to_numpy()
        out[c] = _model(Xtr, seed).fit(Xtr[m], Y[c][m].astype(int)).predict_proba(Xte)[:, 1]
    return pd.DataFrame(out, index=Xte.index)


def cv_auroc(X: pd.DataFrame, Y: pd.DataFrame, folds: int = 5, seed: int = 0) -> dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    res = {}
    for c in LABELS:
        m = Y[c].notna().to_numpy()
        Xm, ym = X[m], Y[c][m].astype(int).to_numpy()
        oof = np.zeros(len(ym))
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xm, ym):
            oof[va] = _model(X, seed).fit(Xm.iloc[tr], ym[tr]).predict_proba(Xm.iloc[va])[:, 1]
        res[c] = round(float(roc_auc_score(ym, oof)), 4)
    return res


def debias(sub: pd.DataFrame, prior: pd.DataFrame, w: dict) -> pd.DataFrame:
    """Per finding: (1 - w) rank(sub) + w rank(prior), w < 0, rescaled to [0, 1]; a prior-less study ranks mid."""
    out = sub.copy()
    pr = prior.reindex(sub.StudyInstanceUID.astype(str))
    for c, wc in w.items():
        rs = sub[c].rank(method="average", pct=True).to_numpy(np.float64)
        rp = pr[c].rank(method="average", pct=True).fillna(0.5).to_numpy(np.float64)
        z = (1.0 - wc) * rs + wc * rp
        lo, hi = z.min(), z.max()
        out[c] = (z - lo) / (hi - lo) if hi > lo else rs
    return out


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "cv":
    tab = load_table(Path(sys.argv[2]))
    Y = pd.read_csv(sys.argv[3], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    X, _ = study_features(tab)
    for k, v in cv_auroc(X.reindex(Y.index), Y).items():
        print(f"{k:18s} {v:.3f}")
