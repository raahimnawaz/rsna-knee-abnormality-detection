"""THE GATE: does our rebuilt pixel path feed the fork's members the pixels they were fitted on?

Nothing downstream is worth running until this passes. The F2-cheap crop A/B (`IMPROVEMENTS.md`
§3g) measures a difference between two pixel configurations put through frozen members. If our
reconstruction is itself off, that error is a per-study noise term sitting on BOTH arms, and noise
dilutes a paired delta toward zero. **A false "the crop does nothing" is the expensive way to be
wrong here, and it is indistinguishable from a true one without this file.**

    python fusion/pilkwang_gate.py --n 60           # first read, ~15 min
    python fusion/pilkwang_gate.py --n 250          # the real one

WHY THE GATE IS NOT "REPRODUCE THE HOLDOUT AUC". That was the plan until the fork's source turned
out to publish **no fold assignment** -- the kernel is inference-only for the members, and the
study-to-fold map lived in the training run, which is not published. So there is no way to look up
which studies a member held out.

The way through is better than what it replaced: **fold recovery and the pixel gate are the same
measurement.** Their OOF prediction for a study is the mean of exactly the four members that held
it out (20 members = 5 folds x 4 seeds). So run all twenty, and for each study ask which of the
five four-member groups reproduces the shipped OOF value. Three readouts fall out, and they fail
in different ways:

  1. PARTITION. A faithful path recovers a clean ~20/20/20/20/20 split over studies. A broken one
     produces an argmin that is noise, and noise does not partition evenly by fold.
  2. MARGIN. Best fold against second best, per study. This is the identifiability check: if the
     five candidate means sit on top of each other the argmin means nothing regardless of how the
     partition looks.
  3. RESIDUAL. How closely the winning group's mean matches the shipped value. This is the actual
     fidelity number, and unlike the other two it does not benefit from the memorisation gap.

READOUT 3 HAS A SCALE THAT IS NOT INVENTED, which is the part worth keeping. The weights dataset
ships two prediction matrices of the same lineage -- `oof.npz::pred` and `merge_gain.npz::ours` --
and they differ by **mean |delta| 0.0165** at correlation 0.99. That is how well the fork
reproduces ITSELF across two of its own runs. A residual well under 0.0165 means our path is as
close to theirs as theirs is to itself; a residual far above it means pixels.

WHY THE PARTITION IS EASY AND THE RESIDUAL IS NOT -- read them in that order. For a study, the
four members that held it out never saw it; the other sixteen trained on it. That memorisation gap
is large and survives a mediocre reconstruction, so readout 1 can pass on pixels that are merely
approximately right. Only readout 3 is sensitive to being exactly right. **A pass on 1 and 2 with
a bad 3 means the members are recognisable but not reproduced, and the crop A/B is not yet safe.**

The target is `oof.npz::pred`, NOT `merge_gain.npz::ours`. `pred` ships beside the manifest whose
twenty members we are running. `ours` is an experiment record and is not a blend of anything --
swept in probability and in logit space, the best fit to `pred` is w = 0.00 with mean |delta| still
0.0165, so the two are different runs of one lineage rather than components of each other.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
D = PROJ / "data"
W = D / "external" / "pilkwang_weights"

from fusion.pilkwang_model import (GROUP, load_member, manifest,  # noqa: E402
                                   pick_device)
from fusion.pilkwang_pixels import IMG, N_SLICE, NII, build_cache  # noqa: E402

EVAL_BATCH = 8
SELF_CONSISTENCY = 0.0165      # mean |delta| between the fork's own two matrices


def window_starts(n_slice: int, group: int = GROUP) -> list[int]:
    """Theirs. With `slices` 12 and `group` 3 this is ten overlapping windows, not one."""
    if n_slice >= group:
        return list(range(n_slice - group + 1))
    return [0]


@torch.no_grad()
def predict_member(model, cache, mask, dev, img_size, pool="prob"):
    """Their `predict_member`, minus the time-budget machinery which is a platform concern.

    `pool="prob"` is the fork's shipped default (`TTA_POOL = "prob"`), so it is what produced
    `oof.npz`. Our 0.899 submission changed it to logit with per-target overrides -- that is a
    DIFFERENT estimator and must not be used to reproduce their file.
    """
    starts = window_starts(cache.shape[2])
    model.eval()
    out = []
    for b in range(0, len(cache), EVAL_BATCH):
        c = cache[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[b:b + EVAL_BATCH]).to(dev)
        acc = None
        for st in starts:
            rows = torch.from_numpy(np.ascontiguousarray(c[:, :, st:st + GROUP])).to(dev)
            z = model(rows, m, img_size).float()
            v = z if pool == "logit" else torch.sigmoid(z)
            acc = v if acc is None else acc + v
        v = acc / len(starts)
        out.append((torch.sigmoid(v) if pool == "logit" else v).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="studies to run")
    ap.add_argument("--crop-mm", type=float, default=130.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None, help="npz to write member predictions to")
    ap.add_argument("--no-k16", action="store_true",
                    help="ablate the measured slice-direction bit (see pilkwang_pixels)")
    a = ap.parse_args()

    man = manifest()
    members = man["members"]
    ref = np.load(W / "oof.npz", allow_pickle=True)
    ref_ids, ref_pred = list(ref["ids"]), ref["pred"]
    pos = {s: i for i, s in enumerate(ref_ids)}

    slots = pd.read_csv(D / "slots_pilkwang.csv")
    have = {p.name.split("_")[1].replace(".nii", "") for p in NII.glob("*.nii*")}
    slots = slots[slots["SeriesInstanceUID"].isin(have)]
    # Studies holding every slot they were assigned, so a missing file never explains a miss.
    full = pd.read_csv(D / "slots_pilkwang.csv").groupby("StudyInstanceUID").size()
    got = slots.groupby("StudyInstanceUID").size()
    ok = [s for s in got.index if got[s] == full[s] and s in pos]
    rng = np.random.default_rng(a.seed)
    studies = list(rng.permutation(sorted(ok))[:a.n])
    print(f"{len(studies)} studies with every assigned slot on disk, of {len(ok):,} eligible")

    t0 = time.time()
    print(f"\nbuilding pixels at crop {a.crop_mm:g} mm, "
          f"K16 {'OFF (ablation)' if a.no_k16 else 'on'} ...")
    cache, mask, stats = build_cache(studies, slots, crop_mm=a.crop_mm,
                                     use_direction=not a.no_k16, verbose=True)
    print(f"  {cache.nbytes / 1e9:.2f} GB, {int(mask.sum())} slots, "
          f"{stats['reversed']} reversed, {time.time() - t0:.0f}s")

    dev = pick_device()
    P = np.zeros((len(members), len(studies), 12), np.float32)
    for k, m in enumerate(members):
        t = time.time()
        model, _, d = load_member(m, dev=dev)
        P[k] = predict_member(model, cache, mask, dev, int(m["config"]["img"]))
        del model
        print(f"  [{k + 1:>2}/{len(members)}] {m['id']} fold {m['fold']} "
              f"fp {d:.1e}  {time.time() - t:.0f}s", flush=True)

    y = ref_pred[[pos[s] for s in studies]]
    folds = np.array([m["fold"] for m in members])
    cand = np.stack([P[folds == f].mean(0) for f in range(5)])       # [5, n, 12]
    err = np.abs(cand - y[None]).mean(2)                             # [5, n]
    best = err.argmin(0)
    srt = np.sort(err, axis=0)
    margin = srt[1] - srt[0]
    resid = srt[0]

    print("\n" + "=" * 70)
    print("READOUT 1 -- PARTITION (a faithful path recovers ~20% per fold)")
    for f in range(5):
        k = int((best == f).sum())
        print(f"  fold {f}: {k:>4} / {len(studies)}  ({k / len(studies):5.1%})")
    print("\nREADOUT 2 -- MARGIN, best fold vs second best (identifiability)")
    print(f"  median {np.median(margin):.4f}   mean {margin.mean():.4f}   "
          f"frac > residual {np.mean(margin > resid):.1%}")
    print("\nREADOUT 3 -- RESIDUAL vs the shipped OOF (pixel fidelity)")
    print(f"  median {np.median(resid):.4f}   mean {resid.mean():.4f}")
    print(f"  the fork reproduces ITSELF at mean |delta| {SELF_CONSISTENCY:.4f}")
    v = "PASS" if resid.mean() < SELF_CONSISTENCY else "MISS"
    print(f"  --> {v}: our residual is {resid.mean() / SELF_CONSISTENCY:.2f}x their "
          f"self-consistency")
    print("=" * 70)

    if a.out:
        np.savez_compressed(a.out, pred=P, ids=np.array(studies), folds=folds,
                            ref=y, best=best, resid=resid, margin=margin)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
