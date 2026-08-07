"""Dump a batch of labeling items for reading. Usage: python batch.py START END"""
import pandas as pd, sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

ROOT = PROJ / "labeling"
df = pd.read_csv(ROOT / "labeling_sample.csv")
a, b = int(sys.argv[1]), int(sys.argv[2])
for i in range(a, min(b, len(df))):
    r = df.iloc[i]
    txt = " ".join(str(r.Report).split())
    print(f"### {r.item_id} [{r.lang}]")
    print(txt[:1500])
    print()
