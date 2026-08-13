"""`tonylica/rsna2026-models` — F6 arm 4, which needs no new architecture at all.

    python fusion/tonylica_arm.py --check      # 4 folds, strict load
    python fusion/tonylica_arm.py --gate       # score on the gold studies

WHY THIS ONE IS CHEAP. Its four checkpoints load **strict** into our already-verified
`fusion/pilkwang_model.build_model(pool='cls_mean_focal', prior=True)` — 233 of 234 keys matched
the pilkwang architecture on first comparison, the only extra being `head.slot_prior`. It uses
**the same six slots**, so `fusion/pilkwang_pixels.build_cache` is the pixel path too. No
transcription; only a different pixel contract.

THE CONTRACT, from the checkpoint's own `cache_metadata` rather than from any description:

    img 224 · crop_mm 160.0 · cache_slices 9 · group 3 · n_group 3 · n_group_max 3
    train_shape [4407, 6, 9, 224, 224]

**`n_group == n_group_max == 3` over 9 slices at group 3 means THREE NON-OVERLAPPING windows**
(starts 0, 3, 6), not the ten sliding windows `pilkwang_gate.window_starts` produces for their
12-slice cache. That is the one place this arm departs from the shipped helper, and it is read
off the metadata, not guessed.

⚠️ **THIS IS A WEAK ARM AND §2y APPLIES TO IT.** Its own shipped per-fold gold (`annot`) is
**0.7992 / 0.8068 / 0.7339 / 0.7070 — mean ≈ 0.762**, against pilkwang's per-member mean of
**0.8375**. §2y closed *our* port at 0.7323 for being too weak despite a real diversity of 0.639.
The public notebooks fold this arm into their vote and never measured whether it earns the slot.
**Score it, then decide.** Being free to run is not a reason to include it.

THE GATE, stated before running. An honest single-member gold read should land near **0.76**.
A large miss means the pooling is wrong: `cls_mean_focal`'s third part is `topk(n/8).mean`, and a
strict shape match does NOT prove tonylica used that rather than a plain `amax` — both give
`dim*3`. Reproducing a published number is a reproduction question, so trying the alternative
third part *if the first misses* is legitimate; **choosing between them on our blend delta would
not be.**
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
sys.path.insert(0, str(PROJ / "extractor"))
D = PROJ / "data"

from metrics import auc  # noqa: E402
from fusion.pilkwang_model import build_model, pick_device  # noqa: E402
from fusion.pilkwang_pixels import build_cache  # noqa: E402
from fusion.ft_b_model import LABELS  # noqa: E402

CKPT = D / "external" / "tonylica_4fold" / "rsna_20260807_v1.pt"
IMG, CROP_MM, N_SLICE, GROUP = 224, 160.0, 9, 3
STARTS = [0, 3, 6]          # n_group == n_group_max == 3: non-overlapping, not sliding
EVAL_BATCH = 8


def load_all(dev):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    out = []
    for fs in ck["fold_states"]:
        m = build_model(0, source="facebook/dinov2-small", variant="small",
                        pool="cls_mean_focal", prior=True, quiet=True)
        m.load_state_dict(fs["state_dict"], strict=True)
        out.append((fs["fold"], m.to(dev).eval(), float(fs["score"])))
    return out, ck


@torch.no_grad()
def predict(model, cache, mask, dev):
    """Their three non-overlapping groups, pooled in probability space (the shipped default)."""
    out = []
    for b in range(0, len(cache), EVAL_BATCH):
        c = cache[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[b:b + EVAL_BATCH]).to(dev)
        acc = None
        for st in STARTS:
            rows = torch.from_numpy(np.ascontiguousarray(c[:, :, st:st + GROUP])).to(dev)
            v = torch.sigmoid(model(rows, m, IMG).float())
            acc = v if acc is None else acc + v
        out.append((acc / len(STARTS)).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--out", default="data/_tonylica_gold.npz")
    a = ap.parse_args()

    dev = pick_device()
    models, ck = load_all(dev)
    print(f"device {dev} · {len(models)} folds · contract img {IMG} crop {CROP_MM:g} "
          f"slices {N_SLICE} starts {STARTS}")
    for f, _, s in models:
        print(f"  fold {f}: strict OK · its selection score {s:.4f}")
    if not a.gate:
        print("\nStrict load proves the architecture, not the pooling. Run --gate for that.")
        return 0

    ref = np.load(D / "external" / "pilkwang_weights" / "oof.npz", allow_pickle=True)
    gold = set(np.array(ref["ids"])[ref["gold_mask"]].tolist())
    slots = pd.read_csv(D / "slots_pilkwang.csv")
    have = {p.name.split("_")[1].replace(".nii", "")
            for p in (D / "nifti" / "nifti_train").glob("*.nii*")}
    ok = slots[slots["SeriesInstanceUID"].isin(have)]
    full = slots.groupby("StudyInstanceUID").size()
    got = ok.groupby("StudyInstanceUID").size()
    studies = sorted(s for s in got.index if got[s] == full[s] and s in gold)
    print(f"\nGATE: {len(studies)} gold studies with full slot coverage")

    t = time.time()
    cache, mask, _ = build_cache(studies, ok, crop_mm=CROP_MM, out_size=IMG,
                                 n_slice=N_SLICE, verbose=False)
    print(f"  cache {cache.nbytes / 1e9:.2f} GB in {time.time() - t:.0f}s")

    P = np.zeros((len(models), len(studies), 12), np.float32)
    for k, (f, m, _) in enumerate(models):
        t0 = time.time()
        P[k] = predict(m, cache, mask, dev)
        print(f"  fold {f}: {time.time() - t0:.0f}s")
    np.savez_compressed(a.out, pred=P, ids=np.array(studies), folds=np.array([f for f, _, _ in models]))

    y = (pd.read_csv(D / "train.csv").set_index("StudyInstanceUID")
         .reindex(studies)[LABELS].to_numpy())
    keep = ~np.isnan(y).any(1)
    y = y[keep].astype(int)
    macro = lambda p: float(np.nanmean([auc(y[:, j], p[:, j]) for j in range(12)]))
    print("\n" + "=" * 62)
    for k, (f, _, _) in enumerate(models):
        print(f"  fold {f} alone (all-member, memorised) {macro(P[k][keep]):.4f}")
    print(f"  4-fold mean                          {macro(P.mean(0)[keep]):.4f}")
    print(f"  their shipped per-fold gold (annot): 0.7992 / 0.8068 / 0.7339 / 0.7070, mean 0.762")
    print("  GATE: a single member near ~0.76 means the pooling is right.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
