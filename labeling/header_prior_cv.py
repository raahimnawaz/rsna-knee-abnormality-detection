import re, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
R = sys.argv[1]
h = pd.read_parquet(f"{R}/data/external/metadata_probe/headers.parquet")
t = pd.read_csv(f"{R}/data/student/targets_consensus_v2.csv").set_index("StudyInstanceUID")
CAT = ["Manufacturer", "ManufacturerModelName", "SoftwareVersions", "ReceiveCoilName", "MagneticFieldStrength"]
NUM = ["RepetitionTime", "EchoTime", "SliceThickness", "PixelBandwidth", "EchoTrainLength", "FlipAngle", "Rows"]
def mode(s):
    s = s.dropna().astype(str); return s.mode().iloc[0] if len(s) else "NA"
g = h.groupby("StudyInstanceUID")
X = pd.DataFrame({c: g[c].agg(mode) for c in CAT})
for c in NUM:
    v = pd.to_numeric(h[c], errors="coerce"); X[c + "_med"] = v.groupby(h.StudyInstanceUID).median()
X["n_series"] = g.size()
tok = h.assign(tk=h.SeriesDescription.fillna("").str.upper().str.findall(r"[A-Z]{2,}")).groupby("StudyInstanceUID").tk.sum()
top = pd.Series([w for l in tok for w in l]).value_counts().head(40).index
for w in top: X["tok_" + w] = tok.apply(lambda l: int(w in l))
for c in CAT: X[c] = X[c].astype("category").cat.codes
X = X.loc[t.index]
skf = StratifiedKFold(5, shuffle=True, random_state=0)
for c in t.columns:
    y = t[c]; m = y.notna(); yb = (y[m] >= 0.5).astype(int).values; Xm = X[m.values]
    oof = np.zeros(len(yb))
    for tr, va in skf.split(Xm, yb):
        mdl = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                             categorical_features=list(range(len(CAT))), random_state=0)
        mdl.fit(Xm.iloc[tr], yb[tr]); oof[va] = mdl.predict_proba(Xm.iloc[va])[:, 1]
    print(f"{c:18s} n={m.sum():5d} pos={yb.mean():.3f}  header-only CV AUROC {roc_auc_score(yb, oof):.3f}")
