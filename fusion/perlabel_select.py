"""Per-label model selection vs averaging, on the 691 fold-0 studies, scored exactly as score_oof.py.

    .venv/bin/python fusion/perlabel_select.py        # from the repo root; CPU, ~2 min

QUESTION (2026-09-24): with two of our own models scored out of fold on the same studies, is it
better to use each one only on the labels where it is strongest (or to use the second only as a
"verification" of the first) than to average them? Choices are made on a random half and scored
on the other half, 1,000 times, so a per-label pick is never scored on the studies that chose it.
The 0.03 "clearly ahead" threshold was fixed before the first run and has not been tuned.
This only works between models with honest out-of-fold scores on the same studies -- our own. The
public stack's members trained on these 691, so no per-label comparison against them exists here.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "fusion"); sys.path.insert(0, "extractor")
from score_oof import NEUTRAL, D, L
from metrics import auc

ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
gold = set(pd.read_csv(D / "train.csv").dropna(subset=L).StudyInstanceUID)
runs = {"ortho": "fusion/runs_c6_ortho", "coatnet": "fusion/runs_ab_coatnet", "fork_input": "fusion/runs_forksample_kaggle"}
oof = {k: pd.read_csv(f"{v}/oof_all.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID") for k, v in runs.items()}
ids = sorted(set.intersection(*[set(o.index) for o in oof.values()]) & set(ref.index) - gold)
y = ref.loc[ids, L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
keep = ~np.isnan(y).any(1); ids = list(np.asarray(ids)[keep]); y = (y[keep] > 0.5).astype(float)
R = {k: o.loc[ids, L].rank(pct=True).to_numpy() for k, o in oof.items()}     # ranks: what a blend uses
n = len(y); print(f"{n} studies, reference {NEUTRAL.name}, gold excluded\n")

def per_label(p, k): return np.array([auc(y[k, i], p[k, i]) for i in range(12)])
def mac(p, k): return float(np.nanmean(per_label(p, k)))
A, B = R["ortho"], R["coatnet"]
avg = (A + B) / 2
def select(winner):                       # winner[i] True -> ortho for label i
    return np.where(winner[None, :], A, B)
def gated(thr=0.25):                      # "verification": ortho, unless the two disagree by > thr in rank
    return np.where(np.abs(A - B) > thr, avg, A)

allk = np.arange(n)
pa, pb = per_label(A, allk), per_label(B, allk)
print("per label (all 681):  " + "  ".join(f"{l[:9]} {a:.3f}/{b:.3f}" for l, a, b in zip(L, pa, pb)))
oracle = select(pa >= pb)
print(f"\nON ALL {n} (in-sample for anything chosen):")
for name, p in [("ortho alone", A), ("coatnet alone", B), ("average of both", avg),
                ("per-label winner, chosen on these same studies", oracle), ("verification (ortho, avg where they disagree)", gated())]:
    print(f"  {name:48s} {mac(p, allk):.4f}")

# honest version: choose on a random half, score on the other half, 1,000 times
rng = np.random.default_rng(0)
res = {k: [] for k in ["ortho", "average", "select", "select_if_clear", "verification"]}
for _ in range(1000):
    perm = rng.permutation(n); c, t = perm[: n // 2], perm[n // 2:]
    da, db = per_label(A, c), per_label(B, c)
    win = da >= db
    clear = np.abs(da - db) > 0.03          # take a winner only when it is clearly ahead, else average
    sel_c = np.where(clear[None, :], np.where(win[None, :], A, B), avg)
    res["ortho"].append(mac(A, t)); res["average"].append(mac(avg, t))
    res["select"].append(mac(select(win), t)); res["select_if_clear"].append(mac(sel_c, t))
    res["verification"].append(mac(gated(), t))
base = np.array(res["average"])
print(f"\nHONEST: chosen on a random half, scored on the other half, 1,000 splits (vs plain average):")
for k, v in res.items():
    v = np.array(v); d = v - base
    print(f"  {k:16s} mean {v.mean():.4f}   vs average {d.mean():+.4f} +-{d.std():.4f}   beats average in {(d > 0).mean():.0%} of splits")

# a third, weaker family: does averaging it in help?
three = (A + B + R["fork_input"]) / 3
print(f"\nadding the weaker 'fork_input' arm ({mac(R['fork_input'], allk):.4f}) as a third, equal member: "
      f"{mac(three, allk):.4f} vs two-model average {mac(avg, allk):.4f}")

# the winning rule against ORTHO ALONE, and what it chose per label
rng = np.random.default_rng(1)
d_vs_ortho, use = [], np.zeros((12, 3))           # columns: ortho / coatnet / average
for _ in range(1000):
    perm = rng.permutation(n); c, t = perm[: n // 2], perm[n // 2:]
    da, db = per_label(A, c), per_label(B, c)
    clear, win = np.abs(da - db) > 0.03, da >= db
    sel = np.where(clear[None, :], np.where(win[None, :], A, B), avg)
    d_vs_ortho.append(mac(sel, t) - mac(A, t))
    for i in range(12):
        use[i, 2 if not clear[i] else (0 if win[i] else 1)] += 1
d = np.array(d_vs_ortho)
print(f"\nselect_if_clear vs ORTHO ALONE: {d.mean():+.4f} +-{d.std():.4f}, better in {(d > 0).mean():.0%} of splits")
print("how often each label went to ortho / coatnet / average (of 1,000):")
for l, u in zip(L, use.astype(int)):
    print(f"  {l:17s} {u[0]:5d} {u[1]:5d} {u[2]:5d}")
