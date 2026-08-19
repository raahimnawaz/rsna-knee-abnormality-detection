"""Is our decode faithful enough that the fork's weights still RECOGNISE their training data?

    python fusion/pixel_fidelity.py --arms none,permute,reverse --out data/_pixel_fidelity.npz
    python fusion/pixel_fidelity.py --arms none,permute --limit-members 6      # ~14 min first read

THE QUESTION. Everything trained here scores 0.6924-0.7710; everything downloaded scores 0.8469
-0.8522. Two causes were proposed -- our pixels, or our training -- and neither is established.
§3i-6 killed the argument from the reconstruction residual (0.0165 of perturbation is worth
+0.0013 of macro, under 2% of the gap), so the pixel half needs an instrument in AUROC.

⛔ THE OBVIOUS INSTRUMENT DOES NOT WORK, AND FINDING OUT COST ONE SMOKE TEST. Each member declares
`annot`, its AUC on the 58 gold studies, mean 0.8375 -- so "run the same weights on our pixels and
compare" looks free. **It is not the same studies.** Every member trained on 4/5 of the corpus, so
~80% of those gold studies are in its TRAINING set; their `annot` must be a holdout-only figure.
Measured here: member `44128e3ff3` reads **0.9960** against a declared 0.8669. That is
memorisation, not fidelity. And the fold map cannot be recovered honestly -- §2y records that
their scheme "is not reproduced" and that "the 20 members cannot be re-weighted individually from
anything published", while `band_ab`'s `argmin |cand - y_ref|` recovery selects toward the very
predictions we would compare against.

✅ SO THE MEMORISATION IS THE INSTRUMENT, NOT THE NUISANCE. **A network cannot score 0.996 on
inputs whose slice order has been scrambled relative to the ones it memorised.** Recognition of
training data is a *sensitive, fold-free* probe of whether our decode reproduces the pixels these
weights were fitted on -- and it needs no holdout map, because being memorised is the point.

What it lacks is a scale, so this file supplies one INTERNALLY, which is the whole design:

    none      our native decode                                    the arm under test
    permute   slice order randomly permuted within each slot       what "scrambled" costs
    reverse   slice stack reversed                                 the milder, K16-shaped error

**Read the SPREAD, never the level.** An absolute AUC has no meaning here -- it is inflated by
memorisation by construction. What means something is how far `none` sits from `permute`:

    none >> permute    our decode is at the faithful end; the band is TRAINING -> Phase C
    none ~= permute    the probe is insensitive and proves nothing -- report that, do not
                       reinterpret it as a pass
    none ~= permute and both low    the weights do not recognise our pixels at all -> §3i-4

⚠️ This bounds DECODE fidelity. It does not bound the training pipeline, the targets, or the
`slot_cache` tiles `train_port` actually trains on (which `contract_audit.py` shows are a 160 mm
crop against the fork's 130 mm). Pixel config here is `pilkwang_pixels`' defaults, i.e. the fork's
native contract, deliberately not the port's.
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

from metrics import auc                                                    # noqa: E402
from fusion.pilkwang_model import load_member, manifest, pick_device       # noqa: E402
from fusion.pilkwang_pixels import NII, build_cache                        # noqa: E402
from fusion.pilkwang_gate import predict_member                            # noqa: E402
from fusion.band_ab import _free_accel                                     # noqa: E402

TARGETS = manifest()["targets"]


def macro(y: np.ndarray, p: np.ndarray) -> float:
    v = [auc(y[:, j], p[:, j]) for j in range(len(TARGETS)) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(v))


def scramble(cache: np.ndarray, how: str, seed: int = 2026) -> np.ndarray:
    """Perturb the SLICE axis only. cache is [study, slot, slice, H, W].

    Per (study, slot) rather than one global permutation, because a single shared permutation
    would be a relabelling every window sees identically and the model could partly absorb it.
    The failure mode we are pricing is per-series disorder (§3i-3: interleaved and multi-echo
    acquisitions do not number slices in spatial order), so it is applied per series.
    """
    if how == "none":
        return cache
    out = cache.copy()
    if how == "reverse":
        return out[:, :, ::-1].copy()
    rng = np.random.default_rng(seed)
    n_st, n_sl, n_z = out.shape[:3]
    for i in range(n_st):
        for j in range(n_sl):
            out[i, j] = out[i, j][rng.permutation(n_z)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", default="none,permute",
                    help="comma list from none,permute,reverse")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-members", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    members = manifest()["members"]
    if a.limit_members:
        members = members[:a.limit_members]
    ref = np.load(W / "oof.npz", allow_pickle=True)
    gold_ids = set(np.array(ref["ids"])[ref["gold_mask"]].tolist())

    slots = pd.read_csv(D / "slots_pilkwang.csv")
    have = {p.name.split("_")[1].replace(".nii", "") for p in NII.glob("*.nii*")}
    slots = slots[slots["SeriesInstanceUID"].isin(have)]
    full = pd.read_csv(D / "slots_pilkwang.csv").groupby("StudyInstanceUID").size()
    got = slots.groupby("StudyInstanceUID").size()
    ok = {s for s in got.index if got[s] == full[s]}
    studies = sorted(gold_ids & ok)

    tr = pd.read_csv(D / "train.csv").set_index("StudyInstanceUID")
    y = tr.reindex(studies)[TARGETS].to_numpy()
    keep = ~np.isnan(y).any(1)
    studies = [s for s, k in zip(studies, keep) if k]
    y = y[keep].astype(int)

    print(__doc__.splitlines()[0])
    print(f"\n{len(studies)} gold studies with full NIfTI coverage · {len(members)} members")
    print("⚠️ Levels are inflated by memorisation BY DESIGN. Read the spread between arms.\n")

    t = time.time()
    base, mask, stats = build_cache(studies, slots)
    print(f"cache {base.nbytes / 1e9:.2f} GB in {time.time() - t:.0f}s · "
          f"{stats['filled']:,} slots, {stats['reversed']} reversed")

    dev = pick_device()
    arms = [s.strip() for s in a.arms.split(",")]
    res: dict[str, np.ndarray] = {}
    for arm in arms:
        cache = scramble(base, arm, a.seed)
        vals = []
        print(f"\narm {arm}:")
        for k, m in enumerate(members):
            t0 = time.time()
            model, _, _ = load_member(m, dev=dev)
            p = predict_member(model, cache, mask, dev, int(m["config"]["img"]))
            del model
            _free_accel()
            vals.append(macro(y, p))
            print(f"  [{k + 1:>2}/{len(members)}] {m['id']} fold {m['fold']}  "
                  f"{vals[-1]:.4f}   {time.time() - t0:.0f}s", flush=True)
        res[arm] = np.array(vals)
        if cache is not base:
            del cache
            _free_accel()

    print("\n" + "=" * 66)
    print(f"  {'member':<12}{'fold':>5}" + "".join(f"{x:>12}" for x in arms))
    for i, m in enumerate(members):
        print(f"  {m['id']:<12}{m['fold']:>5}" + "".join(f"{res[x][i]:>12.4f}" for x in arms))
    print("-" * 66)
    print(f"  {'MEAN':<17}" + "".join(f"{res[x].mean():>12.4f}" for x in arms))

    if "none" in res:
        for arm in arms:
            if arm == "none":
                continue
            d = res["none"] - res[arm]
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            print(f"\n  none − {arm}: {d.mean():+.4f} ± {se:.4f} "
                  f"({d.mean() / se if se and se == se else float('nan'):.1f}σ over "
                  f"{len(d)} members, {(d > 0).sum()}/{len(d)} positive)")
    print("=" * 66)
    print("  large positive spread -> our decode is at the faithful end; band is TRAINING")
    print("  spread ~ 0            -> probe insensitive; it proves NOTHING either way")

    if a.out:
        np.savez_compressed(a.out, ids=np.array(studies), y=y,
                            member=np.array([m["id"] for m in members]),
                            fold=np.array([m["fold"] for m in members]),
                            declared=np.array([m["annot"] for m in members]),
                            **{f"arm_{k}": v for k, v in res.items()})
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
