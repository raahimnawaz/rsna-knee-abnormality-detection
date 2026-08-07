"""Run the rule extractor over all 4,407 reports; evaluate on the 58 gold studies.

Outputs
  data/pseudo_labels.csv    soft targets for every study (this is the training table)
  data/extract_states.csv   raw states, for adjudication against method B later
  data/extract_evidence.csv the clause behind each positive, for spot-checking
"""
import pandas as pd, numpy as np, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rule_extractor import RuleExtractor, LABELS

PROJ = Path(__file__).resolve().parents[1]

ROOT = PROJ
D = ROOT / "data"

ex = RuleExtractor(ROOT / "labeling" / "glossary.json")
tr = (pd.read_csv(D / "train.csv")
        .merge(pd.read_csv(D / "lang_detected.csv"), on="StudyInstanceUID"))
tr["is_gold"] = tr[LABELS].notna().all(axis=1)
print(f"reports {len(tr)}  gold {tr.is_gold.sum()}")

res = [ex.extract(t, l) for t, l in zip(tr["Report"], tr["lang"])]
scores = pd.DataFrame([r["scores"] for r in res], index=tr.index)
states = pd.DataFrame([r["states"] for r in res], index=tr.index)
evid = pd.DataFrame([{L: r["evidence"].get(L, "") for L in LABELS} for r in res],
                    index=tr.index)


def auc(y, s):
    """Rank-based AUC; returns nan if only one class present."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


g = tr.is_gold.values
print("\n" + "=" * 74)
print("EVALUATION ON THE 58 GOLD STUDIES")
print("=" * 74)
rows = []
for L in LABELS:
    y = tr.loc[g, L].values.astype(float)
    s = scores.loc[g, L].values
    pred = (s >= 0.5).astype(float)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else np.nan
    rows.append((L, int(y.sum()), auc(y, s), prec, rec, f1, (pred == y).mean()))

ev = pd.DataFrame(rows, columns=["label", "n_pos", "AUC", "prec", "recall", "F1", "acc"])
print(ev.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nMACRO AUC on gold: {ev.AUC.mean():.4f}   (n=58, wide CIs -- directional only)")

print("\n" + "=" * 74)
print("STATE DISTRIBUTION ACROSS ALL 4,407 REPORTS")
print("=" * 74)
dist = pd.DataFrame({L: states[L].value_counts() for L in LABELS}).T.fillna(0).astype(int)
for c in ["pos", "hedged", "weak", "neg", "absent"]:
    if c not in dist:
        dist[c] = 0
dist = dist[["pos", "hedged", "weak", "neg", "absent"]]
dist["pos_rate_%"] = (100 * dist["pos"] / len(tr)).round(1)
print(dist.to_string())

print("\n" + "=" * 74)
print("POSITIVE RATE BY LANGUAGE  <-- flat rows mean the glossary is failing there")
print("=" * 74)
bylang = pd.DataFrame({
    L: tr.groupby("lang").apply(lambda x: (states.loc[x.index, L] == "pos").mean() * 100,
                                include_groups=False)
    for L in LABELS})
bylang["n"] = tr.groupby("lang").size()
print(bylang.round(1).to_string())

out = tr[["StudyInstanceUID", "lang"]].copy()
for L in LABELS:
    out[L] = scores[L].values
out.to_csv(D / "pseudo_labels.csv", index=False)

st = tr[["StudyInstanceUID", "lang"]].copy()
for L in LABELS:
    st[L] = states[L].values
st.to_csv(D / "extract_states.csv", index=False)

evo = tr[["StudyInstanceUID", "lang"]].copy()
for L in LABELS:
    evo[L] = evid[L].values
evo.to_csv(D / "extract_evidence.csv", index=False)

print(f"\nwrote pseudo_labels.csv / extract_states.csv / extract_evidence.csv to {D}")
