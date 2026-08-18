"""§3z, THE MEASUREMENT: does pilkwang's slice BAND throw away the labels we are worst at?

`ARCHITECTURES.md`'s trap table is the whole motivation. Every arm that rivals or beats pilkwang
looks at more of the volume than pilkwang does:

    arm        slices/series   band
    pilkwang        12         (0.20, 0.80)    <- what we ship
    ft_b            32         FULL STACK      <- 0.883 solo
    DINOv3          16         (0.12, 0.88)
    tonylica         9         (0.20, 0.80)

On SAGITTAL the slice axis IS medial-lateral (`pipeline/slot_cache.py`, and §2l's canonical
132/132 axes). So a band of (0.20, 0.80) means the twenty members we ship have never been shown
the outer 20% of any sagittal stack -- the far medial and far lateral compartments -- and
**Lateral Meniscus is our worst label (0.720) with our largest gap to the teacher (+0.146)**.
§3y's own `sag_lat` slot sits at depth 0.75, just inside a band the members were never trained
past. That is a hypothesis, it is free to test, and it has never been tested.

    python fusion/band_ab.py --n 600 --out data/_band_ab_n600.npz     # the real one
    python fusion/band_ab.py --gold --out data/_band_ab_gold.npz      # the confirmatory read

WHY THIS IS FREE, AND WHERE THE RISK ACTUALLY IS. §3g measured that `fingerprint()` takes
`img_size` and NOT `SLICE_BAND`, so the members load and run at any band. `SlotHead` has
`slot_emb` (per-SLOT) and no per-slice positional embedding, so unlike K19 there is no learned
index to misalign -- the windows are independent forward passes pooled afterwards. Nothing here
can silently score at the wrong indices. The one real risk is DOMAIN SHIFT: weights fitted inside
(0.2, 0.8) may simply degrade on slices outside it. §3g named that risk and called it "an
empirical question with a cheap paired local answer". This file is that answer.

FOUR ARMS, AND THE POINT OF THE MIDDLE TWO IS THAT THEY DECOMPOSE THE LEVER. §3y is being run
right now with `sag_med`/`sag_lat` (depth only, no box) and four boxed slots (box only, depth
0.5) in ONE bundle, so a Stage 1 gain there cannot say whether it bought coverage or resolution.
This file does not repeat that mistake:

    A   band (0.20, 0.80), 12 slices -> 10 windows    the banked control
    B   band (0.12, 0.88), 12 slices -> 10 windows    COVERAGE only: same window count, same TTA
                                                      density, strictly more anatomy in frame
    C   band (0.20, 0.80), 16 slices -> 14 windows    DENSITY only: same anatomy, more windows
    D   A + B pooled per target                       the arm that would actually ship

B-vs-A and C-vs-A are one variable each. If both move, they are separable. If only C moves, the
lever is TTA variance reduction and is subject to §3v's sub-additivity; if only B moves, it is
new anatomy and is not.

THE POOLING RULE FOR D, FIXED HERE, BEFORE ANY AUC IS SEEN. It is §3d's confirmed mechanism --
*"a focal finding appears in SOME windows, so a mean over windows dilutes its evidence and a max
does not; a diffuse finding is present in all of them and the mean is the better estimator"* --
applied to the axis B actually moves, which is position along the medial-lateral slice axis:

  * **max(A, B)** for findings living at the medial/lateral EXTREMES, which is the anatomy B adds
    and A cannot see: both menisci, Medial OA, Lateral OA, MCL.
  * **A only** for ACL, which lives in the intercondylar notch at the CENTRE of the slice axis.
    B's extra windows are pure out-of-distribution noise for it and must not be allowed to dilute
    a label already at 0.919.
  * **mean(A, B)** for everything else -- Effusion, Synovitis, Baker's, Contusion, Fracture, PF OA
    -- which are either diffuse or anterior rather than lateral.

That rule comes from anatomy plus a mechanism measured on a DIFFERENT quantity (§3d, on window
pooling), never from this file's output. **§3b is unrepealed: this run may evaluate the rule, it
may never choose it.** Re-tuning `pool_arm_d` after seeing these AUCs would make the result
unreadable, and §3u is what that error costs when it reaches a submission.

⚠️ MEMORY. Each arm holds ONE cache at a time (`del` between arms, as `crop_ab.py` does), but at
n=600 that is ~4.9 GB for A/B and ~6.5 GB for C on a 17.2 GB box that already swaps at 336
(§2p, §2v). **Do not run this while a training job holds MPS** -- §3y Stage 0 is a 2.4 h fold-0
run and the two together will page. Check `ps` first.
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

# The three arms, as (band, n_slice). A is the shipped config and is what oof.npz was produced at,
# so it is also the arm the §3i fold recovery runs on.
ARM_A = ((0.20, 0.80), 12)
ARM_B = ((0.12, 0.88), 12)
ARM_C = ((0.20, 0.80), 16)

# Findings at the medial/lateral extremes of the slice axis -- the anatomy arm B adds.
LATERAL = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "MCL"]
# Findings in the intercondylar notch, at the centre of the slice axis.
MIDLINE = ["ACL"]


def pool_arm_d(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """§3d's mechanism on the slice axis. Fixed before any AUC is seen -- see the docstring."""
    out = np.empty_like(pa)
    for j, t in enumerate(TARGETS):
        if t in LATERAL:
            out[:, j] = np.maximum(pa[:, j], pb[:, j])
        elif t in MIDLINE:
            out[:, j] = pa[:, j]
        else:
            out[:, j] = 0.5 * (pa[:, j] + pb[:, j])
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


def run_arm(name, band, n_slice, studies, slots, members, verbose=False):
    """Build one arm's pixels, run all twenty members, return [20, n_study, 12]."""
    t = time.time()
    cache, mask, stats = build_cache(studies, slots, band=band, n_slice=n_slice,
                                     verbose=verbose)
    n_win = max(cache.shape[2] - 3 + 1, 1)
    print(f"\narm {name}: band {band[0]:.2f}-{band[1]:.2f}, {n_slice} slices "
          f"-> {n_win} windows · cache {cache.nbytes / 1e9:.2f} GB in {time.time() - t:.0f}s "
          f"· {stats['filled']:,} slots, {stats['reversed']} reversed")
    dev = pick_device()
    Q = np.zeros((len(members), len(studies), 12), np.float32)
    for k, m in enumerate(members):
        t0 = time.time()
        model, _, _ = load_member(m, dev=dev)
        Q[k] = predict_member(model, cache, mask, dev, int(m["config"]["img"]))
        del model
        print(f"  [{k + 1:>2}/20] {m['id']} fold {m['fold']}  {time.time() - t0:.0f}s",
              flush=True)
    del cache
    return Q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gold", action="store_true",
                    help="confirmatory read on the image-read studies; sign only, §3b")
    ap.add_argument("--skip-c", action="store_true",
                    help="drop the density arm; B-vs-A is the coverage question on its own")
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
    if a.gold:
        # Fixed id order, no sampling -- there is no seed to vary and nothing to re-draw if the
        # answer is unwelcome. Same construction as crop_ab.py's §3l-4 read.
        gold_ids = set(np.array(ref["ids"])[ref["gold_mask"]].tolist())
        studies = sorted(s for s in ok if s in gold_ids)
        print(f"GOLD READ: {len(studies)} of {len(gold_ids)} gold studies have full NIfTI "
              f"coverage. SIGN ONLY -- this set may evaluate one fixed decision, never choose.")
    else:
        rng = np.random.default_rng(a.seed)
        studies = list(rng.permutation(ok)[:a.n])
        print(f"{len(studies)} studies of {len(ok):,} eligible")

    P = {"A": run_arm("A", *ARM_A, studies, slots, members),
         "B": run_arm("B", *ARM_B, studies, slots, members)}
    if not a.skip_c:
        P["C"] = run_arm("C", *ARM_C, studies, slots, members)

    # Fold recovery runs on arm A -- the config the shipped OOF was produced at (§3i). Doing it
    # on B would ask a wrongly-banded prediction to identify the holdout group.
    y_ref = ref_pred[[pos[s] for s in studies]]
    cand = np.stack([P["A"][folds == f].mean(0) for f in range(5)])
    best = np.abs(cand - y_ref[None]).mean(2).argmin(0)
    print("\nrecovered fold partition: " +
          "  ".join(f"{f}:{int((best == f).sum())}" for f in range(5)))

    def oof(Q):
        """Each study from the four members that HELD IT OUT -- §3i. An all-member read folds in
        sixteen members that trained on the study, and a memorised study is exactly where a wider
        band cannot help, which biases toward 'the band does nothing'."""
        return np.stack([Q[folds == best[i], i].mean(0) for i in range(len(studies))])

    arms = {k: oof(v) for k, v in P.items()}
    arms["D"] = pool_arm_d(arms["A"], arms["B"])

    if a.gold:
        y = (pd.read_csv(D / "train.csv").set_index("StudyInstanceUID")
             .reindex(studies)[TARGETS].to_numpy())
        keep = ~np.isnan(y).any(1)
        label = f"{int(keep.sum())} GOLD studies (image reads, train.csv)"
    else:
        NEUTRAL = (D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56" /
                   "labels_llm_gpt56sol.csv")
        lix = pd.read_csv(NEUTRAL).set_index("StudyInstanceUID").reindex(studies)
        y = lix[TARGETS].to_numpy()
        # Gold studies stay unspent: they are the neutral arbiter and this is not a label
        # question. Exactly score_oof.py's exclusion.
        gold = set(np.array(ref["ids"])[ref["gold_mask"]].tolist())
        keep = (~np.isnan(y).any(1)) & np.array([s not in gold for s in studies])
        label = f"{int(keep.sum())} non-gold studies with a {NEUTRAL.name} row"
    y = y[keep].astype(int)
    arms = {k: v[keep] for k, v in arms.items()}
    print(f"scored on {label}")

    mA = macro(y, arms["A"])[0]
    print("\n" + "=" * 72)
    print(f"  A  band .20-.80 x12   {mA:.4f}   control, the shipped config")
    for k, what in (("B", "band .12-.88 x12   "), ("C", "band .20-.80 x16   "),
                    ("D", "A+B per target     ")):
        if k not in arms:
            continue
        m = macro(y, arms[k])[0]
        d, se, pw = paired_boot(y, arms["A"], arms[k])
        print(f"  {k}  {what}{m:.4f}   {m - mA:+.4f}  |  paired {d:+.4f} ± {se:.4f} "
              f"({d / se if se else 0:>4.1f}σ), positive in {pw:.0%}")
    print("\nper label, A -> D   (B adds far-medial/far-lateral anatomy; watch the menisci)")
    for j, t in enumerate(TARGETS):
        if 0 < y[:, j].sum() < len(y):
            va, vd = auc(y[:, j], arms["A"][:, j]), auc(y[:, j], arms["D"][:, j])
            rule = "max" if t in LATERAL else ("A only" if t in MIDLINE else "mean")
            print(f"  {t:<18} {va:.4f} -> {vd:.4f}  {vd - va:+.4f}   [{rule}]")
    print("=" * 72)
    print("§3z decision rule: see IMPROVEMENTS.md. This run may NOT retune pool_arm_d.")

    if a.out:
        np.savez_compressed(a.out, y=y, ids=np.array(studies)[keep], best=best,
                            **{f"arm_{k}": v for k, v in arms.items()},
                            **{f"raw_{k}": v for k, v in P.items()})
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
