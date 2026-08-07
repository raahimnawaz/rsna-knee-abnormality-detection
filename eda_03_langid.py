"""Proper language ID with lingua, replacing the stopword heuristic.

The heuristic misfired: 'de' appeared in both the Dutch and Spanish probe lists, and
Spanish medical prose is dense with it, so Spanish reports scored as Dutch.
"""
import pandas as pd
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder

D = Path(r"C:\Users\Raahim\rsna-knee-mri\data")

# LATIN deliberately excluded: it only ever captured terse, telegraphic ENGLISH reports
# ("ACL normal. MCL normal. Medial meniscus tear.") that are too function-word-poor to
# place. Verified by inspection of all 53 -- every one was English.
CANDIDATES = ["ENGLISH","SPANISH","PORTUGUESE","ITALIAN","FRENCH","GERMAN","DUTCH",
              "TURKISH","GREEK","RUSSIAN","CROATIAN","BOSNIAN","SERBIAN","SLOVENE",
              "POLISH","CZECH","SLOVAK","ROMANIAN","HUNGARIAN","DANISH","SWEDISH",
              "NORWEGIAN_BOKMAL","FINNISH","UKRAINIAN","BULGARIAN","CATALAN"]
langs = [getattr(Language, n) for n in CANDIDATES if hasattr(Language, n)]
print(f"detecting among {len(langs)} candidate languages")

det = LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()

# South-Slavic variants are the same language for our purposes
MERGE = {"bosnian":"croatian", "serbian":"croatian"}

tr = pd.read_csv(D / "train.csv")
old = pd.read_csv(D / "lang_guess.csv").rename(columns={"lang_guess":"heuristic"})

def detect(t):
    if not isinstance(t, str) or not t.strip():
        return "unknown", 0.0
    vals = det.compute_language_confidence_values(t[:3000])
    if not vals:
        return "unknown", 0.0
    top = vals[0]
    name = top.language.name.lower()
    return MERGE.get(name, name), round(top.value, 3)

res = [detect(t) for t in tr["Report"]]
tr["lang"] = [r[0] for r in res]
tr["lang_conf"] = [r[1] for r in res]

print("\n=== corrected language distribution ===")
vc = tr["lang"].value_counts()
for k, v in vc.items():
    med = tr.loc[tr.lang == k, "lang_conf"].median()
    print(f"  {k:14s} {v:5d}  ({100*v/len(tr):5.1f}%)   median conf {med:.2f}")

LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
tr["is_gold"] = tr[LABELS].notna().all(axis=1)
print("\n=== gold coverage by corrected language ===")
g = tr.groupby("lang").agg(studies=("lang","size"), gold=("is_gold","sum"))
print(g.sort_values("studies", ascending=False).to_string())

m = tr[["StudyInstanceUID","lang","lang_conf"]].merge(old, on="StudyInstanceUID")
print("\n=== where the heuristic disagreed with lingua ===")
x = pd.crosstab(m.heuristic, m.lang)
disagree = (m.heuristic.map({"latin-unknown":"croatian","GREEK":"greek","CYRILLIC":"russian"})
              .fillna(m.heuristic) != m.lang).sum()
print(f"disagreements: {disagree} / {len(m)}  ({100*disagree/len(m):.1f}%)")
print(x.to_string())

low = tr[tr.lang_conf < 0.6]
print(f"\nlow-confidence (<0.60) detections: {len(low)}  -> worth eyeballing in the labeler")

tr[["StudyInstanceUID","lang","lang_conf"]].to_csv(D / "lang_detected.csv", index=False)
print(f"\nwrote {D/'lang_detected.csv'}")
