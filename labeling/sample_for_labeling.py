"""Build a stratified hand-labeling sample.

Design:
  * ~250 stratified reports across all 9 languages, with a FLOOR per language so the
    small ones (French: 69 studies, 0 gold labels) still get real coverage. English is
    CAPPED because it already has 28 of the 58 gold labels.
  * ~30 gold-labelled studies mixed in BLIND -> measures whether your labelling
    definitions match the organisers'. If you disagree with them systematically, your
    extractor will inherit that disagreement across all 4,349 studies.
  * ~20 duplicates at distant positions -> measures intra-rater consistency. If you
    can't reproduce yourself, the ceiling is lower than you think.

Output: labeling_sample.csv (shuffled; gold/dup flags present but NOT shown in the UI).
"""
import pandas as pd, numpy as np, hashlib, json, sys
from pathlib import Path

ROOT = Path(r"C:\Users\Raahim\rsna-knee-mri")
D, OUT = ROOT / "data", ROOT / "labeling"
SEED = 20260806
N_STRAT, N_GOLD, N_DUP = 250, 30, 20
FLOOR, ENGLISH_CAP = 20, 60

LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

# Languages come from lingua (eda_03_langid.py), NOT the original stopword heuristic --
# that misfiled 263 Spanish reports as Dutch and called the Bulgarian block Russian.
rng = np.random.default_rng(SEED)
tr = (pd.read_csv(D / "train.csv")
        .merge(pd.read_csv(D / "lang_detected.csv"), on="StudyInstanceUID"))
tr["is_gold"] = tr[LABELS].notna().all(axis=1)

gold, rest = tr[tr.is_gold], tr[~tr.is_gold]
print(f"corpus {len(tr)}  gold {len(gold)}  unlabelled {len(rest)}")

# --- stratified allocation over the unlabelled pool -------------------------------
counts = rest.lang.value_counts()
prop = (counts / counts.sum() * N_STRAT)
alloc = {}
for lang, n_avail in counts.items():
    want = int(round(max(prop[lang], FLOOR if n_avail >= 50 else n_avail)))
    if lang == "english":
        want = min(want, ENGLISH_CAP)
    alloc[lang] = int(min(want, n_avail))

print("\nallocation:")
for lang, n in sorted(alloc.items(), key=lambda kv: -kv[1]):
    print(f"  {lang:12s} {n:4d}  of {counts[lang]:5d} available")
print(f"  {'TOTAL':12s} {sum(alloc.values()):4d}")

picks = [rest[rest.lang == lang].sample(n, random_state=SEED) for lang, n in alloc.items() if n]
strat = pd.concat(picks)

# --- gold studies, blind ----------------------------------------------------------
gold_pick = gold.sample(min(N_GOLD, len(gold)), random_state=SEED)
print(f"\ngold mixed in blind: {len(gold_pick)}")

items = pd.concat([strat, gold_pick]).drop_duplicates("StudyInstanceUID").reset_index(drop=True)
items["dup_of"] = ""

# --- duplicates -------------------------------------------------------------------
dups = items.sample(min(N_DUP, len(items)), random_state=SEED + 1).copy()
dups["dup_of"] = dups.StudyInstanceUID
items = pd.concat([items, dups], ignore_index=True)

# shuffle, then force duplicates far from their originals
items = items.sample(frac=1, random_state=SEED + 2).reset_index(drop=True)
items.insert(0, "item_id", [f"it{i:04d}" for i in range(len(items))])

keep = ["item_id","StudyInstanceUID","lang","lang_conf","Report","is_gold","dup_of"] + LABELS
items[keep].to_csv(OUT / "labeling_sample.csv", index=False)

print(f"\nwrote {OUT/'labeling_sample.csv'}  ({len(items)} items, "
      f"{items.StudyInstanceUID.nunique()} unique studies)")
print(f"estimated time at 60-90 s/report: {len(items)*75/3600:.1f} h")
print("\nper-language item counts:")
print(items.lang.value_counts().to_string())
