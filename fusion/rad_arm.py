"""§C-3 Stage 1: does the PUBLIC RadImageNet arm earn a blend slot on OUR instrument?

§4b measured **+0.0135** from a RadImageNet arm and it is the reason this file exists. ⛔ **That
number was measured on the `e11` variant** (`_RAD_E11_CROP_MM = 130.0`), and the weights we can
legally ship are **`folds_v1`, full-frame** (`rad_model`'s docstring). **Different arms, §2o's error
class, so +0.0135 does not transfer and this run has to produce its own number.**

    .venv/bin/python fusion/rad_arm.py --gold          # the §4b comparison, n=47
    .venv/bin/python fusion/rad_arm.py --n 600         # large-n, report instrument

THE GATE, PRE-REGISTERED IN `PLAN.md` §C-3 AND §9f-C, BEFORE ANY NUMBER EXISTED:

  * **STRENGTH FIRST, and it is the binding constraint (§4b).** Every arm below ~0.83 lost -- port
    0.7323, our own R50 0.6924, tonylica 0.788, DINOv3 0.8025 -- and both winners were at parity,
    `ft_b` 0.8522 and the e11 RadImageNet 0.8514. **Below 0.83 on gold-47 -> park it**, and do not
    reach for a better readout, which is the reasoning §3w-2 refuted at 4.8σ.
  * **Then the blend delta** under §9e's pre-registered rule: equal-weight rank-mean over FAMILIES,
    nothing fitted, against the banked two-family blend.
  * **§3b binds: this run may EVALUATE the arm, it may never TUNE it.** No weight search, no per-
    label selection, no crop sweep. 47 studies cannot choose anything.

⚠️ WHAT THIS RUN CANNOT DO, STATED IN ADVANCE. The arm ships **no fold map** -- the manifest pins a
`fold_sha256` for a `folds_v1.csv` we do not have -- so the held-out fold is **recovered** by
§3i's procedure. `fold_recover` depends on the memorisation gap being large, and
[[diversity-is-not-the-constraint]]'s corollary records it **collapsing** when an arm's
all-member-vs-held-out inflation was +0.05 rather than pilkwang's +0.1474. **Both reads are printed
and disagreement between them is the finding, not an inconvenience:** a recovered-OOF number far
below the all-fold number means recovery worked; the two sitting on top of each other means it did
not, and then only the all-fold read is interpretable -- inflated, and inflated in the arm's favour.
"""
from __future__ import annotations

import argparse
import json
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
W = D / "external" / "pilkwang_weights"

from metrics import auc  # noqa: E402
from fusion.pilkwang_model import manifest as pilk_manifest, pick_device  # noqa: E402
from fusion.pilkwang_pixels import NII  # noqa: E402
from fusion.rad_model import load_encoder, load_head  # noqa: E402
from fusion.rad_pixels import RAD_SLOTS, build_rad_cache  # noqa: E402
from fusion.fold_recover import recover  # noqa: E402

TARGETS = pilk_manifest()["targets"]
STRENGTH_BAR = 0.83          # §C-3, fixed before the run


def macro(y, p):
    v = [auc(y[:, j], p[:, j]) for j in range(12) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(v)), v


def rank(A):
    return np.stack([pd.Series(A[:, j]).rank().to_numpy() / len(A) for j in range(A.shape[1])], 1)


def paired_boot(y, pa, pb, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d[b] = macro(y[i], pb[i])[0] - macro(y[i], pa[i])[0]
    return float(d.mean()), float(d.std()), float((d > 0).mean()), \
        float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


@torch.no_grad()
def run_arm(studies, slots, dev, batch=48, verbose=True):
    """-> fold predictions [5, n_study, 12] as probabilities."""
    t = time.time()
    cache, mask, stats = build_rad_cache(studies, slots, verbose=verbose)
    print(f"  pixels {cache.shape} = {cache.nbytes / 1e9:.2f} GB in {time.time() - t:.0f}s · "
          f"{stats['filled']:,} planes, {stats['reversed']} reversed, "
          f"{stats['no_file']} missing")

    enc, info = load_encoder(dev)
    n, n_slot, n_slice = cache.shape[:3]
    flat = cache.reshape(-1, cache.shape[-2], cache.shape[-1])
    tok_mask = np.repeat(mask[:, :, None], n_slice, axis=2).reshape(n, -1)
    valid = np.flatnonzero(tok_mask.reshape(-1) > 0)
    feats = np.zeros((n * n_slot * n_slice, 2048), np.float16)
    t = time.time()
    for s in range(0, len(valid), batch):
        idx = valid[s:s + batch]
        # _rad_encode, verbatim: uint8 -> x/127.5-1 -> grey repeated to 3 channels.
        img = torch.from_numpy(flat[idx]).to(dev).float().div_(127.5).sub_(1.0)
        img = img.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        f = enc(img).float().cpu().numpy()
        if not np.isfinite(f).all():
            raise RuntimeError("non-finite RadImageNet feature")
        feats[idx] = f.astype(np.float16)
        if verbose and (s // batch) % 40 == 0 and s:
            print(f"    encoded {s:,}/{len(valid):,}  {s / (time.time() - t):.0f} img/s",
                  flush=True)
    feats = feats.reshape(n, n_slot * n_slice, 2048)
    print(f"  encoded {len(valid):,} images in {time.time() - t:.0f}s")

    P = np.zeros((5, n, 12), np.float32)
    for f in range(5):
        head, _ = load_head(f, dev)
        out = []
        for s in range(0, n, 64):
            x = torch.from_numpy(feats[s:s + 64]).to(dev)
            m = torch.from_numpy(tok_mask[s:s + 64].astype(np.float32)).to(dev)
            out.append(torch.sigmoid(head(x, m)).float().cpu().numpy())
        P[f] = np.concatenate(out)
        del head
    return P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true", help="the §4b comparison set")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ref = np.load(W / "oof.npz", allow_pickle=True)
    ids_all, gm = list(ref["ids"]), ref["gold_mask"]
    slots = pd.read_csv(D / "slots_pilkwang.csv")
    have = {p.name.split("_")[1].replace(".nii", "") for p in NII.glob("*.nii*")}
    slots = slots[slots["SeriesInstanceUID"].isin(have)]

    if a.gold:
        # The SAME 47 studies §4b scored, in the same order, taken from the same file.
        d = np.load(D / "_crop_ab_gold.npz", allow_pickle=True)
        studies = list(d["ids"])
        print(f"§4b comparison set: {len(studies)} gold studies")
    else:
        gold = set(np.array(ids_all)[gm].tolist())
        got = slots.groupby("StudyInstanceUID").size()
        ok = sorted(s for s in got.index if s in set(ids_all) and s not in gold)
        studies = list(np.random.default_rng(a.seed).permutation(ok)[:a.n])
        print(f"{len(studies)} non-gold studies")

    dev = pick_device()
    print(f"device {dev}")
    P = run_arm(studies, slots, dev)

    fold_oof = np.stack([P[recover(P)[i], i] for i in range(len(studies))])
    allfold = P.mean(0)

    print("\n" + "=" * 74)
    if a.gold:
        y = pd.read_csv(D / "train.csv").set_index("StudyInstanceUID").reindex(studies)[
            TARGETS].to_numpy()
        keep = ~np.isnan(y).any(1)
        y = y[keep].astype(int)
        fo, af = fold_oof[keep], allfold[keep]
        m_oof, m_all = macro(y, fo)[0], macro(y, af)[0]
        print(f"STRENGTH on gold-{len(y)}   (bar {STRENGTH_BAR}, §C-3, fixed before the run)")
        print(f"  recovered-OOF   {m_oof:.4f}      <- the honest read")
        print(f"  all-5-fold      {m_all:.4f}      <- INFLATED (§3n: +0.1474 for pilkwang)")
        print(f"  inflation       {m_all - m_oof:+.4f}   "
              f"{'recovery WORKED' if m_all - m_oof > 0.02 else 'recovery SUSPECT -- see docstring'}")
        print(f"\n  reference: pilkwang 0.8516 · ft_b 0.8522 · e11 Rad 0.8514 · DINOv3 0.8025")
        verdict = "PASS -> proceed to the blend" if m_oof >= STRENGTH_BAR else "PARK IT"
        print(f"  --> {m_oof:.4f} vs bar {STRENGTH_BAR}: **{verdict}**")

        if m_oof >= STRENGTH_BAR:
            import json as _json
            dd = np.load(D / "_crop_ab_gold.npz", allow_pickle=True)
            folds = np.array([m["fold"] for m in _json.load(open(W / "manifest.json"))["members"]])
            pilk = np.stack([dd["P130"][folds == dd["best"][i], i].mean(0)
                             for i in range(len(dd["best"]))])
            z = np.load(D / "_ft_b_gold.npz", allow_pickle=True)
            fm = np.transpose(z["pred"], (1, 0, 2))
            ftb = np.stack([fm[recover(fm)[i], i] for i in range(fm.shape[1])])
            pilk, ftb = pilk[keep], ftb[keep]
            b2 = (rank(pilk) + rank(ftb)) / 2
            b3 = (rank(pilk) + rank(ftb) + rank(fo)) / 3
            m2, m3 = macro(y, b2)[0], macro(y, b3)[0]
            dm, se, pw, lo, hi = paired_boot(y, b2, b3)
            print(f"\nBLEND, §9e equal-weight rank-mean over families")
            print(f"  2-family (banked)      {m2:.4f}")
            print(f"  3-family (+ this arm)  {m3:.4f}   {m3 - m2:+.4f}")
            print(f"  paired {dm:+.4f} ± {se:.4f}, positive in {pw:.0%}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
            print(f"  §4b's e11 arm on this same set: +0.0135, 98%, CI [+0.0004, +0.0262]")
    else:
        NEU = (D / "public_llm_labels" / "lixin73_rsna-knee-llm-report-labels-sol56" /
               "labels_llm_gpt56sol.csv")
        y = pd.read_csv(NEU).set_index("StudyInstanceUID").reindex(studies)[TARGETS].to_numpy()
        keep = ~np.isnan(y).any(1)
        y = y[keep].astype(int)
        print(f"LARGE-N report instrument, n={len(y)}")
        print(f"  recovered-OOF {macro(y, fold_oof[keep])[0]:.4f}   "
              f"all-5-fold {macro(y, allfold[keep])[0]:.4f}")
    print("=" * 74)
    print("§3b: this run may EVALUATE the arm. It may not tune it.")
    if a.out:
        np.savez_compressed(a.out, P=P, ids=np.array(studies))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
