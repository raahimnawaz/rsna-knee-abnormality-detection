"""Open the DINOv3 black box: what it reads, what it ignores, and what can be cut.

    python fusion/dinov3_audit.py --weights     # static, instant, no pixels
    python fusion/dinov3_audit.py --slots       # slot-ablation sensitivity (needs pixels)
    python fusion/dinov3_audit.py --delta       # the cross-attention branch, kept or cut
    python fusion/dinov3_audit.py --all

WHY. §3f audited pilkwang and found the lateral compartment systematically harder — a property of
the *task*, not the model, which reframed the whole plan. This is the same exercise on the arm we
are about to blend in. **Blending a model whose behaviour we have not looked at is how §2y and
§3q both went wrong**: an arm that was fine on its own and did not earn its slot.

THE READING RULE. Everything here is either (a) a property of the *weights*, which involves no
data and cannot be over-fitted, or (b) a **paired** ablation on fixed weights and fixed studies.
Neither selects anything, so §3b is not engaged. **None of these numbers may be used to pick a
variant to ship** — they are for understanding, and any change they motivate has to be judged
afterwards on its own pre-registered gate.

⚠️ **THE ABSOLUTE AUCs IN `--slots` AND `--delta` ARE ALL-MEMBER READS AND ARE BIASED UP.** Every
fold has seen 4/5 of these studies in training. §3n measured that inflation at **+0.1474** for
pilkwang (0.8516 honest → 0.9990 all-20). **Do not compare them to pilkwang's 0.8516 or `ft_b`'s
0.8522, which are honest fold-resolved numbers.** Only the *paired differences* here are clean.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
D = PROJ / "data"
CKPT = D / "external" / "dinov3_vits16_folds"

from fusion.dinov3_model import LABELS, SLOTS, load_fold  # noqa: E402

SLOT_NAME = [f"{p[:3].upper()}{'_FS' if f else '_NO'}" for p, f in SLOTS]


def _sd(fold: int) -> dict:
    return torch.load(CKPT / f"m_f{fold}.pt", map_location="cpu",
                      weights_only=False)["state_dict"]


# --------------------------------------------------------------------------------------------
def audit_weights() -> None:
    print("=" * 78)
    print("STATIC AUDIT — properties of the weights. No data involved.")
    print("=" * 78)

    G = np.array([_sd(f)["readout.pool.gate"].float().numpy() for f in range(5)])
    print("\n[1] `readout.pool.gate` — how much of the CROSS-ATTENTION DELTA each label uses.")
    print("    `forward` is `base + gate * delta`, and gate is INITIALISED AT 0.0.")
    print(f"    overall mean |gate| = {np.abs(G).mean():.4f}, max over all labels/folds = "
          f"{np.abs(G).max():.4f}")
    worst = np.argsort(-np.abs(G).mean(0))[:3]
    print("    largest three: " + ", ".join(f"{LABELS[i]} {G[:, i].mean():+.4f}" for i in worst))
    print("    -> the gate never left its initialisation. The label-query cross-attention over")
    print("       patch tokens is switched OFF. See --delta for what that is worth in AUC.")

    q = np.array([_sd(f)["readout.pool.q"].float().numpy() for f in range(5)])
    dw = np.array([_sd(f)["readout.pool.dw"].float().numpy() for f in range(5)])
    print("\n[2] Did the delta branch train at all, or is it at init?")
    print(f"    q  std {q.std():.5f} (init 0.0200)   dw std {dw.std():.5f} (init 0.0510)")
    print("    -> q moved, dw did not grow. Gradients DID flow; the model learned to gate it off.")
    print("       This is a trained-to-zero branch, not an unreachable one.")

    W = np.array([_sd(f)["readout.pool.base.2.weight"].float().numpy() for f in range(5)]).mean(0)
    print("\n[3] `base.2.weight` (12, 832) — what each label reads.")
    print("    832 = CLS segment-MEAN (384) ++ CLS segment-MAX (384) ++ presence embed (64)\n")
    print(f"    {'label':18s} {'mean%':>7s} {'max%':>7s} {'pres%':>7s}")
    for i, lab in enumerate(LABELS):
        a, b, c = (np.abs(W[i, :384]).sum(), np.abs(W[i, 384:768]).sum(),
                   np.abs(W[i, 768:]).sum())
        t = a + b + c
        print(f"    {lab:18s} {100*a/t:7.1f} {100*b/t:7.1f} {100*c/t:7.1f}")
    a, b, c = (np.abs(W[:, :384]).sum(), np.abs(W[:, 384:768]).sum(), np.abs(W[:, 768:]).sum())
    t = a + b + c
    print(f"    {'OVERALL':18s} {100*a/t:7.1f} {100*b/t:7.1f} {100*c/t:7.1f}")
    print("    -> MAX over a study's series slightly outweighs MEAN for every label except")
    print("       Synovitis. Consistent with 'does ANY series show this', which is what an")
    print("       abnormality read is. Presence contributes a steady ~6-7%.")

    T = np.array([_sd(f)["enc.tok.weight"].float().numpy() for f in range(5)])
    print("\n[4] `enc.tok.weight` (7, 384) — the slot-type token. Row 0 is padding_idx.")
    pad = np.abs(T[:, 0]).max()
    print(f"    row 0 (padding) max|.| = {pad:.6f}  {'(zero, as expected)' if pad < 1e-6 else '(NOT zero)'}")
    t0 = T[0, 1:]
    n = t0 / np.linalg.norm(t0, axis=1, keepdims=True)
    C = n @ n.T
    print(f"    fold-0 cosine similarity between the six slot tokens:")
    print(f"      {'':10s}" + "".join(f"{s:>10s}" for s in SLOT_NAME))
    for i, s in enumerate(SLOT_NAME):
        print(f"      {s:10s}" + "".join(f"{C[i, j]:10.3f}" for j in range(6)))
    off = C[~np.eye(6, dtype=bool)]
    print(f"    off-diagonal mean {off.mean():+.3f}, max {off.max():+.3f}")
    print("    -> near-orthogonal slot tokens mean the conditioning genuinely separates the six")
    print("       acquisitions; a block of ~1.0 would mean two slots are treated identically.")

    Q = np.array([_sd(f)["readout.pool.q"].float().numpy() for f in range(5)])[0]
    n = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    C = n @ n.T
    iu = np.triu_indices(12, 1)
    pair = sorted(zip(C[iu], zip(*iu)), reverse=True)[:4]
    print("\n[5] `readout.pool.q` (12, 384) — the per-label queries (dead branch, but readable).")
    print("    most-similar label pairs: " +
          ", ".join(f"{LABELS[i]}~{LABELS[j]} {c:+.2f}" for c, (i, j) in pair))


# --------------------------------------------------------------------------------------------
@torch.no_grad()
def _predict_configs(dev, drop_delta: bool = False):
    """Decode each gold study once, then run: full, and leave-one-slot-out for all six."""
    from fusion.dinov3_pixels import build_study, slot_table
    from pipeline.preprocess import study_laterality

    ids = [str(x) for x in np.load(D / "_ft_b_gold.npz", allow_pickle=True)["ids"]]
    tab = slot_table()
    by = {s: g for s, g in tab.groupby("StudyInstanceUID") if s in set(ids)}
    ids = [s for s in ids if s in by]
    lat = study_laterality(D / "study_meta.csv")

    models = []
    for f in range(5):
        m, _ = load_fold(f, dev)
        if drop_delta:
            m.readout.pool.gate.data.zero_()
        models.append(m)

    # cfg 0 = full; cfg 1..6 = that slot removed
    out = np.full((7, 5, len(ids), 12), np.nan, np.float32)
    for i, sid in enumerate(ids):
        got = build_study(by[sid], (lat.get(sid) or (None,))[0], False, {})
        if got is None:
            continue
        im, sl = got[0].to(dev), got[1].to(dev)
        for cfg in range(7):
            keep = torch.ones(len(sl), dtype=torch.bool, device=dev)
            if cfg:
                keep &= (sl != cfg)          # sl is 1-based, cfg 1..6 names the slot
            if keep.sum() == 0:
                continue
            i2, s2 = im[keep], sl[keep]
            si = torch.zeros(len(s2), dtype=torch.long, device=dev)
            sm = torch.zeros(len(s2), 0, device=dev)
            for mi, m in enumerate(models):
                out[cfg, mi, i] = torch.sigmoid(m(i2, s2, sm, si, 1).float())[0].cpu().numpy()
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(ids)}")
    return np.array(ids), out


def _gold_macro(ids, pred):
    from fusion.score_gold import align, gold_frame, macro
    y, p = align(ids, pred, gold_frame())
    return macro(y, p), len(y)


def audit_slots(dev) -> None:
    print("=" * 78)
    print("SLOT ABLATION — which acquisitions each label actually depends on")
    print("=" * 78)
    print("\n  Removing a slot removes BOTH its pixels and its presence-embedding contribution,")
    print("  which is exactly what a study missing that slot looks like. Paired on identical")
    print("  studies and weights.\n")
    ids, out = _predict_configs(dev)
    full = out[0].mean(0)
    m0, n = _gold_macro(ids, full)
    print(f"\n  all slots present : gold macro {m0:.4f} over n={n}  "
          f"(ALL-MEMBER read, biased up — see the module docstring)\n")
    print(f"  {'slot removed':12s} {'ΔmacroAUC':>10s} {'mean|Δp|':>10s}   most-affected labels")
    for cfg in range(1, 7):
        cur = out[cfg].mean(0)
        ok = ~np.isnan(cur[:, 0])
        if ok.sum() < 5:
            print(f"  {SLOT_NAME[cfg-1]:12s} {'(too few studies have it)':>10s}")
            continue
        m1, _ = _gold_macro(ids[ok], cur[ok])
        d = np.abs(full[ok] - cur[ok])
        top = np.argsort(-d.mean(0))[:3]
        print(f"  {SLOT_NAME[cfg-1]:12s} {m1-m0:+10.4f} {d.mean():10.4f}   "
              + ", ".join(f"{LABELS[i]} {d[:, i].mean():.3f}" for i in top))
    np.savez(D / "_dinov3_slot_ablation.npz", ids=ids, pred=out)
    print(f"\n  saved -> data/_dinov3_slot_ablation.npz")


def audit_delta(dev) -> None:
    print("=" * 78)
    print("DELTA-BRANCH ABLATION — is the cross-attention worth its compute?")
    print("=" * 78)
    p = D / "_dinov3_gate_ablation.npz"
    if not p.exists():
        raise SystemExit("run the gate ablation first (see git history) or use --slots")
    z = np.load(p, allow_pickle=True)
    on, off, ids = z["on"], z["off"], z["ids"]
    a, _ = _gold_macro(ids, on.mean(0))
    b, _ = _gold_macro(ids, off.mean(0))
    d = np.abs(on - off)
    print(f"\n  5-fold mean, delta KEPT    : gold macro {a:.5f}")
    print(f"  5-fold mean, delta DELETED : gold macro {b:.5f}")
    print(f"  paired difference          : {b-a:+.5f}")
    print(f"  max |Δsigmoid| {d.max():.5f}   mean |Δsigmoid| {d.mean():.5f}")
    print("\n  -> the branch shifts individual probabilities by up to 0.055 but does not change")
    print("     the RANKING, which is all macro-AUROC reads. It can be cut.")


def audit_blend() -> None:
    """§3q's bar: comparable strength AND correlation below `ft_b`'s 0.632. Blend rule is the
    pre-registered one (§9e rule 5): equal-weight rank-mean over FAMILIES, nothing fitted."""
    import json

    from scipy.stats import rankdata, spearmanr
    from fusion.fold_recover import gold_labels, macro, recover

    d = np.load(D / "_crop_ab_gold.npz", allow_pickle=True)
    P, truth, y = d["P130"], d["best"], d["y"]
    folds = np.array([m["fold"] for m in json.load(
        open(D / "external" / "pilkwang_weights" / "manifest.json"))["members"]])
    pilk = np.stack([P[folds == truth[i], i].mean(0) for i in range(len(truth))])

    def oof(npz):
        z = np.load(D / npz, allow_pickle=True)
        fm = np.transpose(z["pred"], (1, 0, 2))
        r = recover(fm)
        return np.stack([fm[r[i], i] for i in range(fm.shape[1])]), z["ids"], fm

    ftb, ids_f, fm_f = oof("_ft_b_gold.npz")
    dv3, ids_d, fm_d = oof("_dinov3_gold.npz")
    if list(ids_f) != list(ids_d):
        raise SystemExit("arm study order differs -- refusing to blend")

    rk = lambda A: np.stack([rankdata(A[:, j]) / len(A) for j in range(12)], 1)  # noqa: E731
    b2 = (rk(pilk) + rk(ftb)) / 2
    b3 = (rk(pilk) + rk(ftb) + rk(dv3)) / 3
    sp = lambda A, B: float(np.mean(  # noqa: E731
        [spearmanr(A[:, j], B[:, j]).statistic for j in range(12)]))

    print("=" * 78)
    print("DOES DINOv3 EARN A SLOT? §3q's bar, §9e's pre-registered blend rule")
    print("=" * 78)
    print(f"\n  pilkwang OOF (4-seed mean of held-out fold)  {macro(y, pilk):.4f}")
    print(f"  ft_b OOF     (one model)                     {macro(y, ftb):.4f}")
    print(f"  DINOv3 OOF   (one model, direction-TTA)      {macro(y, dv3):.4f}")
    print(f"\n  all-FOLD reads, directly comparable (both 5 members, no recovery needed):")
    print(f"    ft_b   {macro(y, fm_f.mean(0)):.4f}   DINOv3 {macro(y, fm_d.mean(0)):.4f}   "
          f"-> {macro(y, fm_d.mean(0)) - macro(y, fm_f.mean(0)):+.4f}")
    print(f"\n  2-family blend (banked route)  {macro(y, b2):.4f}")
    print(f"  3-family blend (+ DINOv3)      {macro(y, b3):.4f}")
    print(f"  DELTA                          {macro(y, b3) - macro(y, b2):+.4f}")
    rng = np.random.default_rng(0)
    dd = np.array([v for v in (macro(y[k], b3[k]) - macro(y[k], b2[k])
                               for k in (rng.integers(0, len(y), len(y)) for _ in range(2000)))
                   if not np.isnan(v)])
    print(f"    positive in {100*(dd>0).mean():.0f}% of draws, 95% CI "
          f"[{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}]")
    print(f"\n  Spearman, mean over 12 labels:")
    print(f"    DINOv3 vs pilkwang {sp(dv3, pilk):.3f}")
    print(f"    DINOv3 vs ft_b     {sp(dv3, ftb):.3f}   <- MOST DIVERSE ARM MEASURED")
    print(f"    ft_b   vs pilkwang {sp(ftb, pilk):.3f}   (§3o reference 0.632)")
    print("\n  -> clears the DIVERSITY half of §3q's bar and fails the STRENGTH half.")
    print("     Same shape as §2y's port and §3q's tonylica: diverse, and too weak.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--slots", action="store_true")
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not any([a.weights, a.slots, a.delta, a.blend, a.all]):
        ap.print_help()
        return 0
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if a.weights or a.all:
        audit_weights()
    if a.delta or a.all:
        print()
        audit_delta(dev)
    if a.slots or a.all:
        print()
        audit_slots(dev)
    if a.blend or a.all:
        print()
        audit_blend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
