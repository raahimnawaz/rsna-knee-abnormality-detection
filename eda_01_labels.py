"""Phase-1 EDA: how much supervision actually exists, and in what languages."""
import pandas as pd, numpy as np, re, collections, sys

D = r"C:\Users\Raahim\rsna-knee-mri\data"
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

tr = pd.read_csv(f"{D}/train.csv")
ts = pd.read_csv(f"{D}/train_series.csv")
te = pd.read_csv(f"{D}/test.csv")
tes = pd.read_csv(f"{D}/test_series.csv")

print("="*70); print("SHAPES")
print(f"train.csv          {tr.shape}   cols: {list(tr.columns)}")
print(f"train_series.csv   {ts.shape}   cols: {list(ts.columns)}")
print(f"test.csv           {te.shape}")
print(f"test_series.csv    {tes.shape}")

print("="*70); print("LABEL COVERAGE  <-- the number that sets the ceiling")
lab = tr[LABELS]
n_any = lab.notna().any(axis=1).sum()
n_all = lab.notna().all(axis=1).sum()
print(f"studies total                 : {len(tr)}")
print(f"studies with >=1 label present: {n_any}  ({100*n_any/len(tr):.1f}%)")
print(f"studies with ALL 12 present   : {n_all}  ({100*n_all/len(tr):.1f}%)")
print(f"studies with NO labels        : {len(tr)-n_any}  ({100*(len(tr)-n_any)/len(tr):.1f}%)")

print("\nper-label non-null count and positive rate (among labelled):")
rows=[]
for c in LABELS:
    nn = tr[c].notna().sum()
    pos = tr[c].sum() if nn else 0
    rows.append((c, nn, int(pos), 100*pos/nn if nn else np.nan))
print(pd.DataFrame(rows, columns=["label","n_labelled","n_pos","pos_%"]).to_string(index=False))

print("="*70); print("REPORTS")
rep = tr["Report"]
print(f"non-null reports: {rep.notna().sum()} / {len(tr)}")
print(f"char length: median {rep.dropna().str.len().median():.0f}, "
      f"p05 {rep.dropna().str.len().quantile(.05):.0f}, "
      f"p95 {rep.dropna().str.len().quantile(.95):.0f}, "
      f"max {rep.dropna().str.len().max():.0f}")

# crude language fingerprinting via stopword hits + script detection
PROBES = {
 "english":   r"\b(the|and|with|there|is|no|joint|tear|normal)\b",
 "spanish":   r"\b(el|la|los|las|con|sin|de la|rodilla|derrame|menisco)\b",
 "portuguese":r"\b(do|da|dos|das|com|sem|joelho|derrame|menisco)\b",
 "french":    r"\b(le|la|les|avec|sans|genou|epanchement|m[eé]nisque)\b",
 "german":    r"\b(der|die|das|und|mit|ohne|kniegelenk|erguss|meniskus)\b",
 "italian":   r"\b(il|lo|la|con|senza|ginocchio|versamento|menisco)\b",
 "dutch":     r"\b(de|het|een|met|zonder|knie|vocht|meniscus)\b",
 "turkish":   r"\b(ve|ile|yok|diz|eklem|y[ıi]rt[ıi]k)\b",
}
SCRIPTS = {
 "cyrillic": r"[\u0400-\u04FF]", "greek": r"[\u0370-\u03FF]",
 "arabic":   r"[\u0600-\u06FF]", "hebrew": r"[\u0590-\u05FF]",
 "cjk":      r"[\u4E00-\u9FFF]", "hangul": r"[\uAC00-\uD7AF]",
 "kana":     r"[\u3040-\u30FF]", "thai":   r"[\u0E00-\u0E7F]",
 "devanagari": r"[\u0900-\u097F]",
}
def sniff(t):
    if not isinstance(t,str) or not t.strip(): return "EMPTY"
    for name,pat in SCRIPTS.items():
        if len(re.findall(pat,t)) > 5: return name.upper()
    low = t.lower()
    best, bn = None, 0
    for name,pat in PROBES.items():
        n = len(re.findall(pat, low))
        if n > bn: best, bn = name, n
    return best if bn >= 3 else "latin-unknown"

tr["lang_guess"] = rep.map(sniff)
print("\nlanguage fingerprint (heuristic, latin-script guesses are approximate):")
vc = tr["lang_guess"].value_counts()
for k,v in vc.items(): print(f"  {k:16s} {v:6d}  ({100*v/len(tr):5.1f}%)")

print("\nlabel coverage BY language  <-- checks for a language with no gold labels:")
cov = tr.groupby("lang_guess").apply(
    lambda g: pd.Series({"studies":len(g),
                         "labelled":g[LABELS].notna().any(axis=1).sum()}))
cov["labelled_%"] = (100*cov["labelled"]/cov["studies"]).round(1)
print(cov.sort_values("studies",ascending=False).to_string())

print("="*70); print("SERIES")
spst = ts.groupby("StudyInstanceUID").size()
print(f"series per study: median {spst.median():.0f}, mean {spst.mean():.2f}, "
      f"min {spst.min()}, max {spst.max()}")
print("\nplane x fluid-sensitive x fat-sup counts:")
print(ts.groupby(["Anatomical_Plane","Fluid_Sensitive","Fat_Suppression"]).size()
        .rename("n").reset_index().to_string(index=False))
print(f"\ntest studies: {len(te)}, test series rows: {len(tes)}")

print("="*70); print("LABEL CO-OCCURRENCE (phi, gold subset only)")
gold = tr[tr[LABELS].notna().all(axis=1)]
if len(gold) > 10:
    print(f"gold subset n={len(gold)}")
    print(gold[LABELS].corr().round(2).to_string())
else:
    print("gold subset too small for a correlation matrix")

tr[["StudyInstanceUID","lang_guess"]].to_csv(
    r"C:\Users\Raahim\rsna-knee-mri\data\lang_guess.csv", index=False)
print("\nwrote lang_guess.csv")
