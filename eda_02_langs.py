"""Identify the 'latin-unknown' bucket and confirm series-type structure."""
import pandas as pd, re, collections

D = r"C:\Users\Raahim\rsna-knee-mri\data"
tr = pd.read_csv(f"{D}/train.csv").merge(pd.read_csv(f"{D}/lang_guess.csv"), on="StudyInstanceUID")
ts = pd.read_csv(f"{D}/train_series.csv")

unk = tr[tr.lang_guess == "latin-unknown"]["Report"].dropna()
print(f"latin-unknown reports: {len(unk)}")
print(f"median length: {unk.str.len().median():.0f} chars\n")

tok = collections.Counter()
for t in unk:
    tok.update(re.findall(r"[a-zà-ÿąćęłńóśźżčďěňřšťůžăâîșţğıİöüşç]{3,}", t.lower()))
print("top 60 tokens in the latin-unknown bucket (language fingerprints):")
for w, n in tok.most_common(60):
    print(f"  {w:18s} {n}")

print("\n" + "="*60)
print("series-type structure check: is (Fluid_Sensitive, Fat_Suppression) collapsible?")
print(pd.crosstab(ts.Fluid_Sensitive, ts.Fat_Suppression))
print("\nseries-type combos actually present:",
      sorted(set(zip(ts.Anatomical_Plane, ts.Fluid_Sensitive, ts.Fat_Suppression))))

print("\n" + "="*60)
print("series-type coverage per study (how often is a plane/type missing?):")
ts["combo"] = ts.Anatomical_Plane + "_" + ts.Fluid_Sensitive.map({0:"nonFS",1:"FS"})
piv = ts.pivot_table(index="StudyInstanceUID", columns="combo",
                     values="SeriesInstanceUID", aggfunc="count").fillna(0)
print((piv > 0).mean().round(3).to_string())
print(f"\nstudies missing >=1 of the 6 combos: {(piv == 0).any(axis=1).mean():.1%}")
