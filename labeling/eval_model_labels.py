"""Evaluate the model-produced labels against (a) gold, (b) the rule extractor.

Only (a) is a real validation -- gold is independent evidence. (b) is agreement between
two methods that share a designer, so treat it as a consistency check, not accuracy.
"""
import pandas as pd, numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

ROOT = PROJ
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
     "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

samp = pd.read_csv(ROOT/"labeling"/"labeling_sample.csv")
mine = pd.read_csv(ROOT/"labeling"/"model_labels.csv")
states = pd.read_csv(ROOT/"data"/"extract_states.csv")
scores = pd.read_csv(ROOT/"data"/"pseudo_labels.csv")

m = samp.merge(mine, on="item_id", suffixes=("_gold", "_mine"))
print(f"labelled so far: {len(m)} / {len(samp)}")

# ---------- duplicates: intra-rater consistency ----------
dups = m[m.dup_of.notna() & (m.dup_of != "")]
if len(dups):
    pairs = 0; agree = 0
    for _, d in dups.iterrows():
        orig = m[(m.StudyInstanceUID == d.dup_of) & (m.item_id != d.item_id)]
        if len(orig):
            pairs += 1
            agree += sum(int(d[f"{c}_mine"]) == int(orig.iloc[0][f"{c}_mine"]) for c in L)
    if pairs:
        print(f"\nINTRA-RATER: {pairs} duplicate pairs, "
              f"{100*agree/(pairs*12):.1f}% label-level agreement")

# ---------- vs gold ----------
gold = m[m.is_gold == True]
print(f"\n{'='*66}\nvs GOLD (independent evidence) -- n={len(gold)}\n{'='*66}")
if len(gold) >= 5:
    rows = []
    for c in L:
        y = gold[f"{c}_gold"].values.astype(float)
        p = gold[f"{c}_mine"].values.astype(float)
        if len(set(y)) < 2:
            rows.append((c, int(y.sum()), np.nan, (p == y).mean())); continue
        tp = ((p==1)&(y==1)).sum(); fp = ((p==1)&(y==0)).sum(); fn = ((p==0)&(y==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else np.nan
        rows.append((c, int(y.sum()), f1, (p == y).mean()))
    d = pd.DataFrame(rows, columns=["label","n_pos_gold","F1","agreement"])
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    allm = gold[[f"{c}_mine" for c in L]].values
    allg = gold[[f"{c}_gold" for c in L]].values
    print(f"\noverall label-level agreement with gold: {(allm==allg).mean():.3f}")
    print(f"  ceiling implication: this is roughly how well ANY report-only extractor")
    print(f"  can do against image-derived labels.")

# ---------- vs rule extractor ----------
r = m.merge(states, on="StudyInstanceUID", suffixes=("", "_rule"))
print(f"\n{'='*66}\nvs RULE EXTRACTOR -- n={len(r)}\n{'='*66}")
rows = []
for c in L:
    rule_pos = r[f"{c}_rule"].isin(["pos", "hedged"]).astype(float).values if f"{c}_rule" in r \
               else r[c].isin(["pos","hedged"]).astype(float).values
    mine_pos = r[f"{c}_mine"].values.astype(float)
    agree = (rule_pos == mine_pos).mean()
    rows.append((c, int(mine_pos.sum()), int(rule_pos.sum()), agree))
d = pd.DataFrame(rows, columns=["label","n_pos_model","n_pos_rule","agreement"])
print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nmean agreement: {d.agreement.mean():.3f}")
print("\nlargest disagreements are the highest-value studies to review by hand.")
