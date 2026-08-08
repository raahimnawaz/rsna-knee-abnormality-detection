"""Stress-test the claim: 'gold labels are not derived from the reports'.

Four ways that claim could be an artefact rather than a fact:
  A  TRUNCATION   -- I read truncated reports. Evidence may sit past the cut.
  B  UNDER-DETERMINATION -- if identical report text maps to DIFFERENT label sets, the
                    reports cannot be the label source. This is the strongest test.
  C  DUPLICATION  -- same report reused across studies (bilateral knees, template reports)
                    could pair text with another study's labels.
  D  DATA HYGIENE -- non-binary / unexpected values in the label columns.
"""
import pandas as pd, re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

D = PROJ / "data"
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
     "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
tr = pd.read_csv(D/"train.csv")
g = tr[tr[L].notna().all(axis=1)].reset_index(drop=True)

print("="*74); print("D  DATA HYGIENE"); print("="*74)
vals = pd.unique(g[L].values.ravel())
print("distinct values in gold label columns:", sorted(vals))
print("report null/blank in gold:", g.Report.isna().sum(), "/", len(g))
print("full-report length: min %d  median %d  max %d"
      % (g.Report.str.len().min(), g.Report.str.len().median(), g.Report.str.len().max()))
print("gold reports longer than my 1100-char dump cut:",
      (g.Report.str.len() > 1100).sum(), "/", len(g), "  <-- TRUNCATION RISK")

print("\n" + "="*74); print("C  DUPLICATION"); print("="*74)
print("duplicate StudyInstanceUID in train.csv:", tr.StudyInstanceUID.duplicated().sum())
dup_txt = tr[tr.Report.duplicated(keep=False)]
print(f"studies sharing an identical report with another study: {len(dup_txt)} / {len(tr)}")

print("\n" + "="*74); print("B  UNDER-DETERMINATION (strongest test)"); print("="*74)
gd = g[g.Report.duplicated(keep=False)]
if len(gd) == 0:
    print("no identical reports WITHIN the 58 gold -- widening to gold-vs-any-study:")
    shared = tr[tr.Report.isin(g.Report) & tr[L].notna().all(axis=1)]
    gd = shared[shared.Report.duplicated(keep=False)]
if len(gd):
    n_pairs = n_diff = 0
    for txt, grp in gd.groupby("Report"):
        if len(grp) < 2: continue
        arr = grp[L].values.astype(int)
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                n_pairs += 1
                if not (arr[i] == arr[j]).all():
                    n_diff += 1
                    if n_diff <= 3:
                        print(f"\n  IDENTICAL TEXT, DIFFERENT LABELS  (diff on "
                              f"{[L[k] for k in range(12) if arr[i][k]!=arr[j][k]]})")
                        print("   text:", " ".join(str(txt).split())[:260])
                        print("   A:", dict(zip(L, arr[i])))
                        print("   B:", dict(zip(L, arr[j])))
    print(f"\n  gold pairs with identical report text: {n_pairs}")
    print(f"  of those, pairs with DIFFERENT labels : {n_diff}")
else:
    print("  no identical report text among gold studies -- test not available")

print("\n" + "="*74); print("A  TRUNCATION -- full text of the contradiction cases"); print("="*74)
probes = [("Medial Meniscus", r"medial meniscus tear|rotura de menisco interno"),
          ("Synovitis", r"synovit")]
for lab, pat in probes:
    for _, r in g.iterrows():
        txt = " ".join(str(r.Report).split())
        has = bool(re.search(pat, txt, re.I))
        if has != bool(r[lab]):
            print(f"\n--- {lab}: text_says={has}  label={int(r[lab])}  "
                  f"len={len(txt)} chars  (FULL TEXT BELOW)")
            print(txt)
            print("   ALL LABELS:", {k: int(r[k]) for k in L})
            break
