"""F2-cheap, the measurement: does a tighter crop as an EXTRA TTA window reorder studies?

§3j established that the crop moves predictions by more than a member swap does, with per-label
signs that are anatomy — central findings up, peripheral fluid findings down. **None of that is
worth a submission yet**, because §3a's filter is absolute: macro-AUROC is invariant to per-label
monotone transforms, so a uniform shift scores exactly zero. This file asks the only question that
pays: does the crop **reorder** studies.

    python fusion/crop_ab.py --n 600 --out data/_crop_ab_n600.npz

THREE ARMS, and the third is the one that would actually be submitted:

    A   crop 130 only                     what is banked today
    B   crop 90 only                      the tighter box alone, expected to LOSE on the
                                          peripheral labels since it crops them out of frame
    C   130 + 90 pooled per target        the proposal: the wide view stays in the pool and the
                                          tight one is an ADDITIONAL window, so no member is ever
                                          asked to predict from an out-of-distribution input alone

Arm C pools by §3d's rule extended with §3j's table: the five central labels (both menisci, the
three OA) take **max** over the two crops, since a finding visible in the tight box is being
diluted by the wide one; the three peripheral fluid labels (Effusion, Synovitis, Baker's) stay on
**130 only**, because 90 mm puts them out of frame; the rest take the **mean**. That rule is fixed
HERE, before the AUC is seen, and it is derived from anatomy plus a measurement made on a different
quantity (the shift, §3j) rather than from this file's own output. **Choosing it after seeing these
AUCs would be selection on the evaluation set, which is §3b's entire lesson.**

SCORING. `extractor/metrics.auc`, paired bootstrap over studies, against `lixin_gpt56` — a third
label source held out from what these members trained on, per §2o. The reference is neutral by
construction here in a way it has never been on this project: **all three arms are the same twenty
members on the same studies**, differing in one config value. There is no arm the reference can
favour.

OOF, NOT ALL-MEMBER. Each study is scored from the four members that HELD IT OUT, recovered by the
§3i fold procedure (partition clean, χ² p ≈ 0.7). Using all twenty would fold in sixteen members
that trained on the study, and memorised studies are exactly where a crop cannot help — the model
recalls the answer rather than reading the pixels — so an all-member read is biased toward
**"the crop does nothing"**, which is the expensive direction to be wrong in.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "extractor"))
D = PROJ / "data"
W = D / "external" / "pilkwang_weights"

from metrics import auc  # noqa: E402
from fusion.pilkwang_model import load_member, manifest, pick_device  # noqa: E402
from fusion.pilkwang_pixels import NII, build_cache  # noqa: E402
from fusion.pilkwang_gate import predict_member  # noqa: E402

TARGETS = manifest()["targets"]
CENTRAL = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA"]
PERIPHERAL = ["Effusion", "Synovitis", "Baker's"]


def pool_arm_c(p130: np.ndarray, p90: np.ndarray) -> np.ndarray:
    """§3d's per-target rule extended by §3j's anatomy table. Fixed before any AUC is seen."""
    out = np.empty_like(p130)
    for j, t in enumerate(TARGETS):
        if t in CENTRAL:
            out[:, j] = np.maximum(p130[:, j], p90[:, j])
        elif t in PERIPHERAL:
            out[:, j] = p130[:, j]
        else:
            out[:, j] = 0.5 * (p130[:, j] + p90[:, j])
    return out


def macro(y: np.ndarray, p: np.ndarray) -> tuple[float, list[float]]:
    v = [auc(y[:, j], p[:, j]) for j in range(len(TARGETS))
         if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(v)), v


def paired_boot(y, pa, pb, n_boot=2000, seed=0):
    """Paired bootstrap over STUDIES -- §2q: the unpaired hypot SE read 1.9σ as 1.2σ."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        d[b] = macro(y[i], pb[i])[0] - macro(y[i], pa[i])[0]
    return float(d.mean()), float(d.std()), float((d > 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--crop-b", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    members = manifest()["members"]
    folds = np.array([m["fold"] for m in members])
    ref = np.load(W / "oof.npz", allow_pickle=True)
    pos = {s: i for i, s in enumerate(list(ref["ids"]))}
    ref_pred = ref["pred"]

    slots = pd.read_csv(D / "slots_pilkwang.csv")
    have = {p.name.split("_")[1].replace(".nii", "") for p in NII.glob("*.nii*")}
    slots = slots[slots["SeriesInstanceUID"].isin(have)]
    full = pd.read_csv(D / "slots_pilkwang.csv").groupby("StudyInstanceUID").size()
    got = slots.groupby("StudyInstanceUID").size()
    ok = sorted(s for s in got.index if got[s] == full[s] and s in pos)
    rng = np.random.default_rng(a.seed)
    studies = list(rng.permutation(ok)[:a.n])
    print(f"{len(studies)} studies of {len(ok):,} eligible")

    P = {}
    for crop in (130.0, a.crop_b):
        t = time.time()
        cache, mask, _ = build_cache(studies, slots, crop_mm=crop, verbose=False)
        print(f"\ncrop {crop:g} mm: cache {cache.nbytes / 1e9:.2f} GB in {time.time() - t:.0f}s")
        dev = pick_device()
        Q = np.zeros((len(members), len(studies), 12), np.float32)
        for k, m in enumerate(members):
            t0 = time.time()
            model, _, _ = load_member(m, dev=dev)
            Q[k] = predict_member(model, cache, mask, dev, int(m["config"]["img"]))
            del model
            print(f"  [{k + 1:>2}/20] {m['id']} fold {m['fold']}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        P[crop] = Q
        del cache

    # Fold recovery on the 130 arm -- the config the shipped OOF was produced at (§3i).
    y_ref = ref_pred[[pos[s] for s in studies]]
    cand = np.stack([P[130.0][folds == f].mean(0) for f in range(5)])
    best = np.abs(cand - y_ref[None]).mean(2).argmin(0)
    print("\nrecovered fold partition: " +
          "  ".join(f"{f}:{int((best == f).sum())}" for f in range(5)))

    def oof(Q):
        return np.stack([Q[folds == best[i], i].mean(0) for i in range(len(studies))])

    A, B = oof(P[130.0]), oof(P[a.crop_b])
    C = pool_arm_c(A, B)

    # THE path from fusion/score_oof.py, not a lookalike. `report_labels_gpt56sol.csv` sits in the
    # same directory and is a different table; §2o's live bug was scoring against the wrong file.
    NEUTRAL = (D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56" /
               "labels_llm_gpt56sol.csv")
    lix = pd.read_csv(NEUTRAL).set_index("StudyInstanceUID").reindex(studies)
    y = lix[TARGETS].to_numpy()
    # Gold studies are excluded exactly as score_oof.py excludes them: they are the neutral
    # arbiter for label-source questions and this is not one, so they stay unspent.
    gold = set(np.array(ref["ids"])[ref["gold_mask"]].tolist())
    keep = (~np.isnan(y).any(1)) & np.array([s not in gold for s in studies])
    y, A, B, C = y[keep].astype(int), A[keep], B[keep], C[keep]
    print(f"scored on {len(y)} non-gold studies with a {NEUTRAL.name} row")

    mA, _ = macro(y, A)
    mB, _ = macro(y, B)
    mC, _ = macro(y, C)
    print("\n" + "=" * 66)
    print(f"  A  crop 130 only            {mA:.4f}")
    print(f"  B  crop {a.crop_b:g} only             {mB:.4f}   {mB - mA:+.4f}")
    print(f"  C  130 + {a.crop_b:g}, per target   {mC:.4f}   {mC - mA:+.4f}")
    for name, arm in (("B", B), ("C", C)):
        d, se, pw = paired_boot(y, A, arm)
        print(f"\n  {name} vs A paired: {d:+.4f} ± {se:.4f}  ({d / se if se else 0:.1f}σ), "
              f"positive in {pw:.0%} of draws")
    print("\nper label, A -> C:")
    for j, t in enumerate(TARGETS):
        if 0 < y[:, j].sum() < len(y):
            print(f"  {t:<18} {auc(y[:, j], A[:, j]):.4f} -> {auc(y[:, j], C[:, j]):.4f}  "
                  f"{auc(y[:, j], C[:, j]) - auc(y[:, j], A[:, j]):+.4f}")
    print("=" * 66)
    if a.out:
        np.savez_compressed(a.out, A=A, B=B, C=C, y=y, ids=np.array(studies)[keep],
                            best=best, P130=P[130.0], P90=P[a.crop_b])
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
