"""Is the reconstruction defect NEUTRAL to the crop comparison? The test that decides F2-cheap.

THE SITUATION. §3i measured our rebuilt pixel path against the fork's own OOF and found a residual
of **0.0168** against a ~1e-5 floor, with slice ORDER as the mechanism: a random permutation moves
predictions 0.0501, as much as swapping in a different member. K16 fixes sagittal and buys 0.0017;
coronal bounds at ~0.0014 even if measured perfectly. The rest looks like roughly a third of series
being *permuted* rather than merely *reversed*, and that cannot be resolved from anything on disk --
`direction_thumbs.npz` carries first/mid/last only, and the NIfTI converter publishes no code
(checked: `davidadekanmi/rsna-knee-nifti-part1` has no companion kernel and empty metadata).

THE EXPENSIVE ANSWER would be a Kaggle CPU kernel shipping a thumbnail per slice in geometric
order -- ~730k thumbnails, a few hundred MB -- so the permutation could be recovered locally by
matching. Affordable but slow, and it is a large build to commission on the assumption that the
defect matters.

THE QUESTION THIS FILE ASKS INSTEAD, which is cheaper and is the one the project's own rule
demands. §2s: **before any A/B, state the reference and show it is NEUTRAL to both arms.** Four
measurements on this project were lost to an instrument entangled with the thing it measured. Here
the potential entanglement is explicit and testable: both crop arms run through the SAME imperfect
slice order, so the defect is a shared perturbation. What would invalidate the A/B is not the
defect itself but an INTERACTION -- the crop delta coming out different under a different ordering.

So: measure the crop delta under our order, and again under a deliberately perturbed order. If the
delta is the same, the reconstruction error is orthogonal to the crop axis and F2-cheap can be run
locally today. If it is not, the Kaggle export is necessary and this file has bought the reason.

There is a mechanical prior that this will pass -- crop is in-plane and order is through-plane, so
they act on different axes of the tensor -- but a prior is not a measurement, and the whole point
of the rule is that the four losses each had a plausible prior too.

    python fusion/crop_neutrality_test.py --n 60 --members 5

READOUT. Per (study, label) let d = P(crop 90) - P(crop 130). Under two orderings we get d_a and
d_b. Three numbers:

  * corr(d_a, d_b)        -- does the crop move the SAME studies the same way? This is the one
                             that matters, because the A/B is a paired per-study comparison.
  * mean d_a vs mean d_b  -- does the crop's average effect survive?
  * |d| vs |d_a - d_b|    -- is the crop effect large against its own instability?

A high correlation with a small spread means the defect is a constant the paired test differences
away. Anything else and F2-cheap waits for the export.
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
D = PROJ / "data"

from fusion.pilkwang_model import load_member, manifest, pick_device  # noqa: E402
from fusion.pilkwang_pixels import NII, build_cache  # noqa: E402
from fusion.pilkwang_gate import predict_member  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--members", type=int, default=5, help="one per fold is enough for a delta")
    ap.add_argument("--crop-b", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    slots = pd.read_csv(D / "slots_pilkwang.csv")
    have = {p.name.split("_")[1].replace(".nii", "") for p in NII.glob("*.nii*")}
    slots = slots[slots["SeriesInstanceUID"].isin(have)]
    full = pd.read_csv(D / "slots_pilkwang.csv").groupby("StudyInstanceUID").size()
    got = slots.groupby("StudyInstanceUID").size()
    ok = sorted(s for s in got.index if got[s] == full[s])
    rng = np.random.default_rng(a.seed)
    studies = list(rng.permutation(ok)[:a.n])

    # One member per fold: the crop DELTA is a within-member quantity, so members buy precision
    # on the mean rather than identification, and five is enough to see an interaction.
    man = manifest()["members"]
    seen, members = set(), []
    for m in man:
        if m["fold"] not in seen:
            seen.add(m["fold"])
            members.append(m)
        if len(members) >= a.members:
            break

    print(f"{len(studies)} studies, {len(members)} members (one per fold), "
          f"crop 130 vs {a.crop_b:g} mm")

    caches = {}
    for crop in (130.0, a.crop_b):
        t = time.time()
        c, mask, _ = build_cache(studies, slots, crop_mm=crop, verbose=False)
        caches[crop] = c
        print(f"  built crop {crop:g} mm in {time.time() - t:.0f}s")

    dev = pick_device()
    # "ours" is the order we actually have; "reversed" is a deliberate, uniform perturbation of it
    # -- not a candidate convention, a probe. It stands in for the unknown permutation because it
    # is the largest single-parameter move available and it is exactly reproducible.
    P = {}
    for k, m in enumerate(members):
        model, _, _ = load_member(m, dev=dev)
        img = int(m["config"]["img"])
        for crop, c in caches.items():
            for order in ("ours", "reversed"):
                cc = c if order == "ours" else c[:, :, ::-1].copy()
                P[(m["id"], crop, order)] = predict_member(model, cc, mask, dev, img)
        del model
        print(f"  [{k + 1}/{len(members)}] {m['id']} fold {m['fold']} done", flush=True)

    da = np.stack([P[(m["id"], a.crop_b, "ours")] - P[(m["id"], 130.0, "ours")]
                   for m in members]).mean(0)
    db = np.stack([P[(m["id"], a.crop_b, "reversed")] - P[(m["id"], 130.0, "reversed")]
                   for m in members]).mean(0)

    print("\n" + "=" * 68)
    print(f"CROP DELTA  d = P({a.crop_b:g} mm) - P(130 mm), averaged over {len(members)} members")
    print(f"  our order       mean {da.mean():+.5f}   mean|d| {np.abs(da).mean():.5f}")
    print(f"  reversed order  mean {db.mean():+.5f}   mean|d| {np.abs(db).mean():.5f}")
    print(f"\nNEUTRALITY")
    print(f"  corr(d_ours, d_reversed)      {np.corrcoef(da.ravel(), db.ravel())[0, 1]:.4f}")
    print(f"  mean |d_ours - d_reversed|    {np.abs(da - db).mean():.5f}")
    print(f"  ... against mean|d| of        {np.abs(da).mean():.5f}")
    ratio = np.abs(da - db).mean() / max(np.abs(da).mean(), 1e-9)
    print(f"  instability / effect          {ratio:.2f}")
    print("\nper label, mean d (ours / reversed):")
    T = manifest()["targets"]
    for j, t in enumerate(T):
        print(f"  {t:<18} {da[:, j].mean():+.5f}  {db[:, j].mean():+.5f}")
    print("=" * 68)
    print("A high correlation with instability well under 1 means the reconstruction defect")
    print("differences away and F2-cheap can be measured locally. Otherwise it waits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
