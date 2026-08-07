"""Is the model-vs-gold disagreement CALIBRATION or genuine report/image divergence?

If errors are one-directional, the reports do carry the signal and my threshold is simply
set wrong -- that is recoverable by moving a threshold. If errors are balanced, the report
genuinely underdetermines the label and no amount of tuning helps.

This distinction decides whether 84.7% is a ceiling or just my current operating point.
"""
import pandas as pd, numpy as np
from pathlib import Path

R = Path(r"C:\Users\Raahim\rsna-knee-mri")
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
     "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

samp = pd.read_csv(R/"labeling"/"labeling_sample.csv")
mine = pd.read_csv(R/"labeling"/"model_labels.csv")
m = samp.merge(mine, on="item_id", suffixes=("_gold", "_mine"))
g = m[m.is_gold == True]
print(f"gold studies labelled: {len(g)}\n")

rows = []
for c in L:
    y = g[f"{c}_gold"].values.astype(int)
    p = g[f"{c}_mine"].values.astype(int)
    fp = int(((p == 1) & (y == 0)).sum())   # I said yes, gold said no
    fn = int(((p == 0) & (y == 1)).sum())   # I said no,  gold said yes
    tot = fp + fn
    skew = abs(fp - fn) / tot if tot else np.nan
    verdict = ("-" if tot == 0 else
               "CALIBRATION (I over-call)"  if tot >= 3 and fp >= 3 * max(fn, 1) else
               "CALIBRATION (I under-call)" if tot >= 3 and fn >= 3 * max(fp, 1) else
               "divergence / mixed")
    rows.append((c, fp, fn, tot, skew, verdict))

d = pd.DataFrame(rows, columns=["label","FP_I_said_yes","FN_I_said_no","errors","skew","verdict"])
print(d.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

tot_fp = d.FP_I_said_yes.sum(); tot_fn = d.FN_I_said_no.sum()
n_cells = len(g) * 12
print(f"\ntotal errors {tot_fp+tot_fn} / {n_cells} cells "
      f"({100*(tot_fp+tot_fn)/n_cells:.1f}%)   FP={tot_fp}  FN={tot_fn}")

print("\n" + "="*70)
print("RECOVERABLE vs IRREDUCIBLE")
print("="*70)
cal = d[d.verdict.str.startswith("CALIBRATION")].errors.sum()
div = d[d.verdict == "divergence / mixed"].errors.sum()
print(f"errors in one-directional (calibratable) labels : {cal}")
print(f"errors in mixed-direction  (divergent)   labels : {div}")
if cal + div:
    print(f"\nIf every calibratable label were perfectly re-thresholded, agreement would rise")
    print(f"from {100*(1-(tot_fp+tot_fn)/n_cells):.1f}% to ~{100*(1-div/n_cells):.1f}% "
          f"-- THAT is the closer estimate of the true report-only ceiling.")
