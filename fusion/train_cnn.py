"""Workstream C: RadImageNet ResNet-50, FULLY fine-tuned, on the port's own slot cache.

    python fusion/train_cnn.py --check                       # shapes + strict load, no training
    python fusion/train_cnn.py --probe-steps 150             # is the LR sane? ~9 min, no gold
    caffeinate -i .venv/bin/python fusion/train_cnn.py --run-folds 0 --verbose

WHY THIS FILE EXISTS. `PLAN.md` 9f-C decided the trained arm should be a fine-tuned CNN: every
member of the 0.908 blend is a ViT (DINOv2 x2 families), the one CNN the public field runs
(RadImageNet R50) is **frozen**, and 2e measured frozen as the configuration that cannot adapt.
So a fine-tuned CNN is the one family nobody in this competition is running.

=================================================================================================
DEVIATION FROM 9f-C's EXECUTABLE SPEC, DELIBERATE, AND IT MAKES THE EXPERIMENT BETTER
=================================================================================================

The spec says *"input: reuse `ft_b_pixels.py` unchanged"*. **That is not affordable and it is not
the sharper experiment.**

*Not affordable:* `ft_b_pixels.load_series_nifti` decodes NIfTI live. That is right for a one-shot
gate over 47 studies; for training it would decode ~3,599 studies x ~5.5 series x 32 slices every
epoch, and there is no cache in that convention on disk. Building one costs 35-78 GB (K=32 at 336,
uint8) against 245 GB free, plus the build itself.

*Not the sharper experiment:* `fusion/runs_port` already trained **dinov2-small** on
`data/tiles336`, site-grouped folds, `data/targets.csv`, the same `SlotHead`. Running the CNN
through the identical pipeline turns 9f-C item 3 -- *"a CNN has locality and translation
equivariance built in while a ViT must learn them from 4,407 studies; that is a sample-efficiency
claim, and it is testable"* -- into a **paired A/B with exactly one variable changed**. Reusing
`ft_b_pixels` would change the backbone AND the pixel convention AND the head at once, which is
the thing 9f-C's own head-reuse argument warns against.

**So: same cache, same folds, same targets, same head, same loop. Backbone ViT -> CNN. That is
the whole diff.** The control is `fusion/runs_port/summary.json`: fold 0, report-OOF **0.7298**,
gold **0.7574**, 220 min.

Note `GROUP = 3` is **pilkwang's own constant**, not a thin approximation of it -- the 6x3x336x336
tile is the fork's representation, one 3-slice window per slot at depth 0.5. What the cache cannot
do is *window augmentation*: the fork sees many windows per series at inference (that is what
per-target TTA pooling is), this cache holds one. That is a real handicap and it applies equally
to both arms of the A/B, so it does not contaminate the comparison.

=================================================================================================
THE BATCHNORM TRAP, AND IT IS SILENT -- this is why the swap is not a one-line change
=================================================================================================

`SlotDataset` zeroes absent slots (`x = x * m.view(-1,1,1,1)`) and passes a mask the head honours.
For a ViT that is harmless: LayerNorm is per-token, so an all-zero image contributes nothing to
any other image's statistics.

**ResNet-50 has BatchNorm.** `SlotNet.forward` flattens (B, K, 3, H, W) to (B*K, 3, H, W) and
forwards every slot including the absent ones, so those all-zero images would enter the batch
statistics of every BN layer. **87.2% of studies are missing at least one series type** and axial
non-fluid exists for only 19% (`FINDINGS.md` 3.2), so a large and *study-dependent* fraction of
each batch would be zeros. Nothing raises; the arm just trains against corrupted normalisation.

`RadSlotNet.forward` therefore **gathers the present slots, forwards only those, and scatters the
embeddings back**. It is also strictly less compute. Added to the trap table's spirit: the
conventions that break when crossed are not only pixel conventions.

=================================================================================================
LEARNING RATE -- the one number 9f-C does not give, chosen by probe, NOT by gold
=================================================================================================

`LR_BACKBONE = 8e-6` is the fork's value for a ViT with **six blocks** open. This arm unfreezes
**all** of a 23.5M-param CNN (2e: a frozen encoder cannot adapt, and RadImageNet frozen is the
field's cautionary example), so 8e-6 is very likely far too low.

**`--probe-steps` settles it in ~9 min per candidate on TRAINING LOSS ONLY**, read against the
floor/prior references `loss_references` already prints. This is not a 3b violation: 3b bans
selecting on the 47/58 gold studies, and the probe never touches gold, OOF, or any label source
other than the training targets it is already optimising. A learning rate that does not move the
training loss off the constant prior is broken, and that is all the probe is asked to detect.
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
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "extractor"))

from metrics import auc  # noqa: E402
from fusion.train_port import (  # noqa: E402
    LABELS, SlotDataset, SlotHead, TileStore, loss_references, weighted_bce,
)

D = PROJ / "data"
ENCODER = D / "external" / "radimagenet_r50_encoder" / "ResNet50.pt"

#: 9f-C: fine-tune ALL blocks. The head keeps the fork's 1e-3; the backbone gets a standard
#: full-fine-tune rate rather than the fork's six-blocks-open 8e-6. Confirm with --probe-steps.
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
EPOCHS = 10


class RadSlotNet(nn.Module):
    """RadImageNet ResNet-50 trunk -> GAP -> the port's SlotHead.

    `d_in = 2048`, NOT 2x2048: a CNN has no CLS token, so there is nothing to concatenate a
    patch-mean to. 9f-C says as much.
    """

    def __init__(self, n_slots: int, dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet50(weights=None)
        self.trunk = nn.Sequential(*list(r.children())[:-2])   # -> (B, 2048, H/32, W/32)
        if pretrained:
            if not ENCODER.exists():
                raise SystemExit(
                    f"missing {ENCODER} -- kaggle datasets download "
                    "mattiaangeli/radimagenet-resnet50-encoder --unzip")
            sd = torch.load(ENCODER, map_location="cpu", weights_only=False)
            # The checkpoint is an nn.Sequential dump under a `backbone.` prefix, i.e. numeric
            # keys (`0.weight`), not torchvision's named ones (`conv1.weight`). ARCHITECTURES.md
            # 5 says "loads strict into a torchvision resnet50 trunk" -- true only after this
            # strip, and only against `children()[:-2]`. Verified 318/318, 0 missing, 0 unexpected.
            sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
            self.trunk.load_state_dict(sd, strict=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.n_features = 2048
        self.head = SlotHead(self.n_features, n_slots, dropout=dropout)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x [B,K,3,H,W] · m [B,K] bool -> logits [B,12].

        Only the PRESENT slots are forwarded. See the BatchNorm section in this file's docstring:
        forwarding the zeroed absent slots would poison every BN layer's batch statistics with a
        study-dependent fraction of all-zero images, silently.
        """
        b, k = x.shape[:2]
        flat = x.flatten(0, 1)                                  # [B*K, 3, H, W]
        sel = m.flatten()                                       # [B*K]
        e = x.new_zeros(b * k, self.n_features)
        if sel.any():
            e[sel] = self.pool(self.trunk(flat[sel])).flatten(1)
        return self.head(e.view(b, k, -1), m)

    def param_groups(self) -> list[dict]:
        return [{"params": list(self.trunk.parameters()), "lr": LR_BACKBONE},
                {"params": list(self.head.parameters()), "lr": LR_HEAD}]


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _fold_cfg(a, store: TileStore, fold: int, n_train: int) -> dict:
    """Everything a resumed checkpoint must agree with. Same contract as train_port._fold_cfg,
    plus `arch` -- resuming a ViT checkpoint into this file would be the 2s error class arriving
    through a convenience feature, and the state_dict keys happen not to collide loudly."""
    return {"arch": "radimagenet_r50", "labels": str(a.labels), "folds": str(a.folds),
            "cache": str(a.cache), "tag": a.tag, "fold": int(fold), "slots": list(store.slots),
            "n_train": int(n_train), "img": a.img, "epochs": a.epochs, "batch": a.batch,
            "lr_backbone": a.lr_backbone, "lr_head": LR_HEAD, "wd": a.wd, "dropout": a.dropout,
            "slot_dropout": a.slot_dropout, "seed": a.seed, "limit": a.limit,
            "preprocess_version": store.manifest.get("preprocess_version")}


def build_loaders(fold: int, folds: pd.DataFrame, store: TileStore, targets: pd.DataFrame, a):
    have_any = store.have.any(1)
    usable = set(store.index.StudyInstanceUID[have_any])
    tr = [u for u in folds[(folds.fold != fold) & (~folds.is_gold)].StudyInstanceUID
          if u in usable]
    va = [u for u in folds[folds.fold == fold].StudyInstanceUID if u in usable]
    if a.limit:
        tr, va = tr[:a.limit], va[:max(a.limit // 4, 8)]
    tr_ds = SlotDataset(store, tr, targets, train=True, slot_dropout=a.slot_dropout,
                        seed=a.seed + fold)
    va_ds = SlotDataset(store, va, targets, train=False)
    nw = a.workers
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                        num_workers=nw, persistent_workers=nw > 0)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=a.batch, num_workers=nw,
                                        persistent_workers=nw > 0)
    return tr, va, tr_ds, tr_dl, va_dl


def run_fold(fold: int, folds: pd.DataFrame, store: TileStore, targets: pd.DataFrame,
             a, dev: torch.device):
    tr, va, tr_ds, tr_dl, va_dl = build_loaders(fold, folds, store, targets, a)

    model = RadSlotNet(len(store.slots), dropout=a.dropout).to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  fold {fold}: train {len(tr):,} / val {len(va):,} studies · "
          f"{n_tr / 1e6:.1f}M of {n_all / 1e6:.1f}M params trainable (ALL, 9f-C)")
    lo, hi = loss_references(tr_ds.y)
    print(f"  loss scale: floor {lo:.4f} (perfect, p=y) .. {hi:.4f} (constant prior). "
          f"It CANNOT reach 0 -- the targets are soft.")

    opt = torch.optim.AdamW(model.param_groups(), weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr_backbone, LR_HEAD], total_steps=max(a.epochs * len(tr_dl), 1),
        pct_start=0.25)

    ck = Path(a.out) / f"fold{fold}_last.pt"
    start_ep = 0
    if a.resume and ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        if st.get("cfg") != _fold_cfg(a, store, fold, len(tr)):
            raise SystemExit(
                f"{ck} was written under a different configuration:\n"
                f"  checkpoint {st.get('cfg')}\n  this run   {_fold_cfg(a, store, fold, len(tr))}\n"
                "Resuming across a config change would silently mix two experiments.")
        model.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_ep = st["epoch"]
        print(f"  RESUMED from {ck.name} at epoch {start_ep}/{a.epochs}", flush=True)

    for ep in range(start_ep, a.epochs):
        model.train()
        tot = n = 0
        t0 = time.time()
        for i, b in enumerate(tr_dl):
            opt.zero_grad(set_to_none=True)
            logits = model(b["x"].to(dev), b["m"].to(dev))
            loss = weighted_bce(logits, b["y"].to(dev), b["w"].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b["uid"])
            n += len(b["uid"])
            if a.verbose and (i + 1) % 25 == 0:
                ips = n * len(store.slots) / (time.time() - t0)
                print(f"    ep {ep + 1} step {i + 1}/{len(tr_dl)}  loss {tot / n:.4f}  "
                      f"{ips:.1f} img/s", flush=True)
        print(f"    ep {ep + 1}/{a.epochs}  loss {tot / max(n, 1):.4f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

        tmp = ck.with_suffix(".pt.tmp")
        torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep + 1,
                    "cfg": _fold_cfg(a, store, fold, len(tr))}, tmp)
        tmp.replace(ck)

    model.eval()
    uids, preds = [], []
    with torch.no_grad():
        for b in va_dl:
            preds.append(torch.sigmoid(model(b["x"].to(dev),
                                             b["m"].to(dev))).float().cpu().numpy())
            uids += b["uid"]
    return model, uids, np.concatenate(preds) if preds else np.zeros((0, 12)), len(tr)


def probe(a, store: TileStore, folds: pd.DataFrame, targets: pd.DataFrame, dev: torch.device):
    """Short training-loss-only LR probe. Never reads gold, OOF, or any held-out label."""
    tr, va, tr_ds, tr_dl, _ = build_loaders(0, folds, store, targets, a)
    lo, hi = loss_references(tr_ds.y)
    print(f"\nloss scale: floor {lo:.4f} .. prior {hi:.4f}   (prior = 'learned nothing')")
    print(f"train {len(tr):,} studies · {len(tr_dl)} steps/epoch · probing {a.probe_steps} steps\n")
    for lr in [float(x) for x in a.probe_lrs.split(",")]:
        torch.manual_seed(a.seed)
        model = RadSlotNet(len(store.slots), dropout=a.dropout).to(dev)
        opt = torch.optim.AdamW(
            [{"params": list(model.trunk.parameters()), "lr": lr},
             {"params": list(model.head.parameters()), "lr": LR_HEAD}], weight_decay=a.wd)
        model.train()
        run, t0, first = [], time.time(), None
        for i, b in enumerate(tr_dl):
            if i >= a.probe_steps:
                break
            opt.zero_grad(set_to_none=True)
            loss = weighted_bce(model(b["x"].to(dev), b["m"].to(dev)),
                                b["y"].to(dev), b["w"].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(loss.item())
            if first is None:
                first = loss.item()
            if a.verbose and (i + 1) % 25 == 0:
                print(f"    lr {lr:.0e} step {i + 1}/{a.probe_steps} "
                      f"last25 {np.mean(run[-25:]):.4f}", flush=True)
        last = float(np.mean(run[-25:]))
        print(f"  lr {lr:.0e}: first {first:.4f} -> last25 {last:.4f}   "
              f"({(hi - last) / max(hi - lo, 1e-9) * 100:5.1f}% of the way prior->floor)  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        del model, opt
    print("\nPick the lowest loss that is not diverging. This chose an LR on TRAINING loss only;")
    print("3b is untouched -- no gold, no OOF, no held-out label source was read.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=str(D / "tiles336"),
                    help="tiles336 by default -- the SAME cache runs_port used, so the A/B is paired")
    ap.add_argument("--tag", default="protocol")
    ap.add_argument("--folds", default=str(D / "folds_site.csv"))
    ap.add_argument("--labels", default=str(D / "targets.csv"))
    ap.add_argument("--run-folds", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img", type=int, default=336)
    ap.add_argument("--lr-backbone", type=float, default=LR_BACKBONE)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--slot-dropout", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(PROJ / "fusion" / "runs_cnn"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check", action="store_true", help="shapes + strict load, no training")
    ap.add_argument("--probe-steps", type=int, default=0, help="LR probe, training loss only")
    ap.add_argument("--probe-lrs", default="3e-5,1e-4,3e-4")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = pick_device()

    if a.check:
        import torchvision
        print(f"device {dev} · torchvision {torchvision.__version__}")
        store = TileStore(Path(a.cache), a.tag)
        print(f"cache {a.cache} · {len(store):,} studies · slots {store.slots}")
        m = RadSlotNet(len(store.slots)).to(dev).eval()
        x = torch.randn(2, len(store.slots), 3, a.img, a.img, device=dev)
        msk = torch.ones(2, len(store.slots), dtype=torch.bool, device=dev)
        msk[0, 1] = False
        msk[1, 3] = False
        with torch.no_grad():
            out = m(x, msk)
        print(f"  strict load OK · trunk 2048-d · forward {tuple(x.shape)} -> {tuple(out.shape)}")
        # The BN check: an absent slot must not change the present slots' embeddings.
        m2 = RadSlotNet(len(store.slots)).to(dev).eval()
        m2.load_state_dict(m.state_dict())
        with torch.no_grad():
            a_all = m2(x, torch.ones_like(msk))
            x2 = x.clone()
            x2[0, 1] = 999.0                      # corrupt an ABSENT slot
            b_all = m2(x2, msk)
            c_all = m2(x, msk)
        d = float((b_all - c_all).abs().max())
        print(f"  masked-forward check: corrupting an absent slot moves logits by {d:.2e} "
              f"(must be 0 -- BN pollution is what this catches)")
        print(f"  (all-slots-present logits differ from masked, as they should: "
              f"{float((a_all - c_all).abs().max()):.3e})")
        print(f"  params {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M, all trainable")
        return

    store = TileStore(Path(a.cache), a.tag)
    folds = pd.read_csv(a.folds)
    targets = pd.read_csv(a.labels).set_index("StudyInstanceUID")
    print(f"device {dev} · cache {a.cache} · {len(store):,} studies · "
          f"labels {Path(a.labels).name}")

    if a.probe_steps:
        probe(a, store, folds, targets, dev)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    which = ([int(x) for x in a.run_folds.split(",")] if a.run_folds
             else sorted(folds.fold.unique()))
    rows, t0 = [], time.time()
    for f in which:
        model, uids, preds, n_train = run_fold(f, folds, store, targets, a, dev)
        torch.save({"state_dict": model.state_dict(),
                    "cfg": _fold_cfg(a, store, f, n_train)}, out / f"fold{f}.pt")
        rows.append(pd.DataFrame(preds, columns=LABELS).assign(
            StudyInstanceUID=uids, fold=f))
        del model

    oof = pd.concat(rows, ignore_index=True)
    oof.to_csv(out / "oof_all.csv", index=False)
    summary = {"arch": "radimagenet_r50", "folds": which, "epochs": a.epochs,
               "batch": a.batch, "img": a.img, "lr_backbone": a.lr_backbone,
               "lr_head": LR_HEAD, "cache": str(a.cache), "n_oof": int(len(oof)),
               "minutes": (time.time() - t0) / 60}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/oof_all.csv  ({len(oof):,} studies)  "
          f"{summary['minutes']:.1f} min")
    print("Score it with:  .venv/bin/python fusion/score_oof.py " + str(out))
    print("Then the GATE:  see the pre-registered block in this run's directory / IMPROVEMENTS 3w")


if __name__ == "__main__":
    raise SystemExit(main())
