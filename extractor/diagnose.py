"""Bootstrap CIs on the gold eval, plus targeted diagnostics for the weakest labels."""
import pandas as pd, numpy as np, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rule_extractor import RuleExtractor, LABELS, norm

PROJ = Path(__file__).resolve().parents[1]

ROOT = PROJ; D = ROOT / "data"
ex = RuleExtractor(ROOT / "labeling" / "glossary.json")
tr = pd.read_csv(D / "train.csv").merge(pd.read_csv(D / "lang_detected.csv"), on="StudyInstanceUID")
tr["is_gold"] = tr[LABELS].notna().all(axis=1)
sc = pd.read_csv(D / "pseudo_labels.csv")
stt = pd.read_csv(D / "extract_states.csv")
evd = pd.read_csv(D / "extract_evidence.csv").fillna("")

def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    p, n = y == 1, y == 0
    if p.sum() == 0 or n.sum() == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[p].sum() - p.sum()*(p.sum()+1)/2) / (p.sum()*n.sum())

g = tr.is_gold.values
gy = tr.loc[g, LABELS].values.astype(float)
gs = sc.loc[g, LABELS].values
rng = np.random.default_rng(0)

print("="*72); print("BOOTSTRAP 95% CI ON GOLD (n=58, 2000 resamples)"); print("="*72)
macro = []
rows = []
for j, L in enumerate(LABELS):
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(gy), len(gy))
        a = auc(gy[idx, j], gs[idx, j])
        if not np.isnan(a): boots.append(a)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append((L, int(gy[:, j].sum()), auc(gy[:, j], gs[:, j]), lo, hi, hi-lo))
for _ in range(2000):
    idx = rng.integers(0, len(gy), len(gy))
    macro.append(np.nanmean([auc(gy[idx, j], gs[idx, j]) for j in range(12)]))
df = pd.DataFrame(rows, columns=["label","n_pos","AUC","lo95","hi95","width"])
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
mlo, mhi = np.percentile(macro, [2.5, 97.5])
print(f"\nMACRO AUC 0.757  95% CI [{mlo:.3f}, {mhi:.3f}]  -> +-{(mhi-mlo)/2:.3f}")
print("Per-label CIs span ~0.2-0.4 AUC. Differences below ~0.15 are NOT resolvable here.")

print("\n"+"="*72); print("DIAG 1: Effusion over-fires (recall .91, precision .67)"); print("="*72)
sev = {"trace/minimal/small":r"\b(trace|minimal|small|mild|scant|tiny|minim|pequeñ|leve|az |küçük|malen|klein|petite|discret|gering)",
       "moderate":r"\b(moderate|moderat|orta|umjeren|matig|modéré|умерен)",
       "large":r"\b(large|gross|tense|marked|abundant|grand|groß|büyük|velik|голям)"}
eff = evd.loc[stt["Effusion"]=="pos","Effusion"]
for k,p in sev.items():
    print(f"  {k:22s} {eff.str.contains(p, case=False, regex=True).sum():5d} / {len(eff)}")
print("  -> qualifier is ignored entirely; a trace effusion scores the same as a tense one.")

print("\n"+"="*72); print("DIAG 2: Greek cannot separate medial vs lateral meniscus"); print("="*72)
gr = tr.lang=="greek"
same = (stt.loc[gr,"Medial Meniscus"]==stt.loc[gr,"Lateral Meniscus"]).mean()
print(f"  identical state for both menisci in {100*same:.1f}% of Greek reports")
print(f"  Greek term lists -> medial {ex.F['Medial Meniscus']['greek']}")
print(f"                     lateral {ex.F['Lateral Meniscus']['greek']}")
print("  -> 'μηνίσκ' was added to BOTH lists (right for highlighting, wrong for extraction)")

print("\n"+"="*72); print("DIAG 3: Croatian OA compartments never fire (0.0%)"); print("="*72)
hr = tr[tr.lang=="croatian"]
pats = ["medijaln","lateraln","odjelj","femorotibijaln","kompartm","zglobn prostor","medijalnom","lateralnom"]
for p in pats:
    n = hr.Report.map(lambda t: p in norm(str(t))).sum()
    print(f"  '{p}' appears in {n:4d} / {len(hr)} Croatian reports")
print(f"  glossary medial-compartment terms: {ex.F['_compartment_medial']['croatian']}")
print("  -> phrasing in-corpus does not match the guessed compartment terms")

print("\n"+"="*72); print("DIAG 4: Bulgarian Baker's over-fires (62.7% vs ~24% elsewhere)"); print("="*72)
bg = tr[tr.lang=="bulgarian"]
for p in ["киста","бейк","поплитеал","подколянн","ганглий","менискал"]:
    n = bg.Report.map(lambda t: p in norm(str(t))).sum()
    print(f"  '{p}' appears in {n:4d} / {len(bg)} Bulgarian reports")
print("  -> bare 'киста' matches ANY cyst (ganglion, meniscal, subchondral), not just Baker's")

print("\n"+"="*72); print("DIAG 5: Synovitis is under-detected (recall .37)"); print("="*72)
print("  gold positive rate  47%   corpus 'pos' rate  11.5%")
print("  by language:", {l: round(100*(stt.loc[tr.lang==l,'Synovitis']=='pos').mean(),1)
                          for l in sorted(tr.lang.unique())})
print("  -> synovitis is usually INFERRED from effusion/synovial thickening, rarely named;")
print("     German 0.8% and Bulgarian 1.4% suggest the word simply is not used there.")
