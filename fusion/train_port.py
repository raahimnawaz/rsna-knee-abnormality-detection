"""The training port: DINOv2-small at 336 with the last six blocks OPEN. README Phase 0 step 4.

This is the file `IMPROVEMENTS.md` 2e argues for. `fusion/train.py` trains a head on frozen
1536-d embeddings and therefore cannot adapt the encoder at any resolution, under any head, with
any labels -- which is why every macro this project has produced sits between 0.695 and 0.744
while a 0.891 fork opens `UNFREEZE_LAST = 6` blocks at `LR_BACKBONE = 8e-6`. The encoder runs
inside the loop here, on the uint8 tiles from `pipeline/slot_cache.py`.

WHAT IT REPRODUCES, AND THE ONE PLACE IT CANNOT. Read from `pilkwang`'s code, not its
description (`IMPROVEMENTS.md` 2e):

    UNFREEZE_LAST = 6     LR_BACKBONE = 8e-6     LR_HEAD = 1e-3     EPOCHS = 10
    N_SLOT = 6            GROUP = 3              backbone dinov2 SMALL, runs at 224 and 336
    SlotHead: per-diagnosis attention over slot embeddings, SLOT_PRIOR_STRENGTH = 0.55
    confidence-weighted targets, W = 0.25 + 0.75 * conf

⛔ THREE OF THOSE FOUR LINES WERE READ OFF THEIR NOTEBOOK AND ARE NOT WHAT PRODUCED THEIR WEIGHTS
`2026-08-19`. `fusion/contract_audit.py` diffs them against `pilkwang_weights/manifest.json`, which
records the config behind a mean member holdout of 0.8398:

    EPOCHS = 10                  the 20 shipped members did 20, 24, 25, 27, 29, 30, 37, 60
    SLOT_PRIOR_STRENGTH = 0.55   the manifest records `prior: false`
    backbone (timm, reg4)        the manifest names `facebook/dinov2-small`, which has NO registers
    crop 160 mm (slot_cache)     the manifest records `crop_mm: 130.0` -- 0.476 vs 0.387 mm/px

§2y measured this file at 0.7323 against the fork's 0.8434 and closed §2w step 4 on it. That
comparison bundled all five divergences, so it never measured what it claimed. **The model is now
built by `pilkwang_model.build_model` -- their object, not our reconstruction of it.**

The fork's six slots are `SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1,
SAG_T1` -- it separates fat-suppression from fluid-sensitivity and carries two sagittal
fluid-sensitive variants but no axial non-fluid slot. **That split is not reconstructable from
the competition metadata**: `Fluid_Sensitive` and `Fat_Suppression` are byte-identical over all
24,371 series (measured here; FINDINGS.md 3.1), so those columns yield exactly 3 planes x 2 and
nothing finer. Ours are therefore `ax_fs, ax_nf, cor_fs, cor_nf, sag_fs, sag_nf`. This is a
KNOWN, RECORDED divergence and the reproduction gate in step 5 has to read it as one: the finer
split is recoverable from `SeriesDescription` / `EchoTime` / `ScanningSequence`, which we already
hold for all 24,371 series in `data/external/dicom_headers_zhukovoleksiy.parquet`, and doing that
is the first thing to try if the gate misses.

The other standing decision this file implements: **reproduce the fork's configuration, in our
own code** (README "Two standing decisions"). Nothing here is copied from the fork's file; the
gate is the test that the reconstruction is faithful.

    python fusion/train_port.py --run-folds 0 --out fusion/runs_port      # ~2.6 h, one fold
    python fusion/train_port.py --out fusion/runs_port                    # all five, ~13 h
    python fusion/train_port.py --run-folds 0 --epochs 1 --limit 60       # smoke test
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
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "extractor"))

from metrics import auc  # noqa: E402

D = PROJ / "data"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# The fork's constants. Changing one of these means the reproduction gate no longer applies.
UNFREEZE_LAST = 6
LR_BACKBONE = 8e-6
LR_HEAD = 1e-3
EPOCHS = 30           # median of the members' 20-60; the old 10 was their notebook, not their weights
# Both now match the manifest. They are read by the superseded SlotNet below and by
# contract_audit.py, which must see what the live path uses.
SLOT_PRIOR_STRENGTH = 0.0             # manifest: `prior: false`
BACKBONE = "facebook/dinov2-small"    # manifest: NOT timm's reg4 variant

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- data
class TileStore:
    """The memmapped uint8 cache from pipeline/slot_cache.py. One open file, not 4,407."""

    def __init__(self, cache_dir: Path, tag: str = "protocol"):
        self.arr = np.load(cache_dir / f"tiles_{tag}.npy", mmap_mode="r")
        self.index = pd.read_csv(cache_dir / f"index_{tag}.csv")
        self.manifest = json.loads((cache_dir / f"manifest_{tag}.json").read_text())
        self.slots = [s["name"] for s in self.manifest["slots"]]
        self.row_of = dict(zip(self.index.StudyInstanceUID, self.index.row))
        self.have = self.index[[f"has_{s}" for s in self.slots]].to_numpy(bool)

    def __len__(self) -> int:
        return len(self.index)


class SlotDataset(torch.utils.data.Dataset):
    """One study -> (tiles [K,3,336,336] float, slot mask [K], targets [12], weights [12]).

    Augmentation is deliberately thin, and one common augmentation is deliberately ABSENT: a
    horizontal flip would mirror medial against lateral, which is the exact axis `canonicalise`
    spends the laterality tag to fix and which four of the twelve labels depend on. Free
    accuracy on ImageNet, silent label corruption here.

    Slot dropout IS in the test distribution rather than being defensive padding -- 87.2% of
    studies are missing at least one series type and axial non-fluid exists for only 19% of them
    (FINDINGS.md 3.2) -- so dropping a present slot at train time is sampling the real thing.
    """

    def __init__(self, store: TileStore, uids: list[str], targets: pd.DataFrame,
                 train: bool = False, slot_dropout: float = 0.2, seed: int = 0):
        self.store, self.uids, self.train = store, uids, train
        self.slot_dropout = slot_dropout
        self.y = targets.loc[uids, LABELS].to_numpy(np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid = self.uids[i]
        r = self.store.row_of[uid]
        # np.array not np.asarray: a memmap slice is read-only, and torch.from_numpy on a
        # read-only buffer hands back a tensor it warns is unsafe to write. The copy is the
        # 2 MB the loader was going to make anyway.
        x = torch.from_numpy(np.array(self.store.arr[r], dtype=np.uint8)).float() / 255.0
        m = torch.from_numpy(self.store.have[r].copy())

        if self.train:
            if self.slot_dropout > 0 and m.sum() > 1:
                drop = torch.from_numpy(
                    self.rng.random(len(m)) < self.slot_dropout) & m
                if (m & ~drop).any():                 # never drop the last surviving slot
                    m = m & ~drop
            # Intensity only: gamma and gain. The percentile normalisation upstream already
            # removed scanner-to-scanner scale, so this is jitter around a fixed reference
            # rather than an attempt to model it.
            g = float(self.rng.uniform(0.85, 1.18))
            s = float(self.rng.uniform(0.9, 1.1))
            x = (x.clamp(0, 1) ** g * s).clamp(0, 1)

        # ⛔ NORMALISATION MOVED INTO THE MODEL, 2026-08-19. `pilkwang_model.Model.forward` does
        # `.div_(255)` and then `(x - mean) / std` itself -- it is the fork's own object, verified
        # 20/20 at 7e-06 -- so normalising here as well would apply it twice. That runs perfectly
        # and scores wrongly, which is §9h exactly. Augmentation stays in [0,1] where the gamma is
        # meaningful; the tensor is handed back on the 0-255 scale the fork's forward expects.
        x = (x * 255.0) * m.view(-1, 1, 1, 1)         # absent slots are exactly zero, not noise

        y = torch.from_numpy(self.y[i])
        # The fork's confidence weighting. Our targets are soft (0.5 means "the report does not
        # say"), so |2p-1| is literally how much the label source committed, and a 0.5 row
        # contributes at 0.25 instead of teaching the model that the finding is absent.
        w = 0.25 + 0.75 * (2 * y - 1).abs()
        return {"x": x, "m": m, "y": y, "w": w, "uid": uid}


# ----------------------------------------------------------------------------- model
class SlotHead(nn.Module):
    """Per-diagnosis attention over slot embeddings, with a learned per-diagnosis slot prior.

    One query token per label, so 'trust the sagittal slot for ACL, the axial one for PF OA' is
    representable rather than something a shared pool has to average away. The prior is the
    fork's `SLOT_PRIOR_STRENGTH = 0.55`: attention is mixed with a learned static distribution
    over slots, which stops a 12-way attention from having to discover plane/finding anatomy
    from scratch on 3,599 studies.
    """

    def __init__(self, d_in: int, n_slots: int, n_labels: int = 12, d: int = 256,
                 prior_strength: float = SLOT_PRIOR_STRENGTH, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU())
        self.q = nn.Parameter(torch.zeros(n_labels, d))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.prior = nn.Parameter(torch.zeros(n_labels, n_slots))
        self.prior_strength = prior_strength
        self.drop = nn.Dropout(dropout)
        self.out = nn.Parameter(torch.zeros(n_labels, d))
        nn.init.trunc_normal_(self.out, std=0.02)
        self.bias = nn.Parameter(torch.zeros(n_labels))
        self.norm = nn.LayerNorm(d)

    def forward(self, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """s [B,K,d_in] · mask [B,K] bool -> logits [B,12]."""
        h = self.proj(s)                                        # [B,K,d]
        k, v = self.k(h), self.v(h)                             # [B,K,d]
        att = torch.einsum("td,bkd->btk", self.q, k) / (h.shape[-1] ** 0.5)   # [B,12,K]

        neg = torch.finfo(att.dtype).min
        mk = mask[:, None, :].expand_as(att)
        att = torch.softmax(att.masked_fill(~mk, neg), dim=-1)
        pri = torch.softmax(self.prior[None].expand_as(att).masked_fill(~mk, neg), dim=-1)
        a = (1 - self.prior_strength) * att + self.prior_strength * pri
        # A study with zero usable slots cannot occur (filtered upstream), but a masked-out row
        # would softmax to a uniform vector over nothing; zero it rather than let it contribute.
        a = a * mask[:, None, :].float()

        pooled = self.norm(torch.einsum("btk,bkd->btd", a, v))  # [B,12,d]
        return (self.drop(pooled) * self.out[None]).sum(-1) + self.bias


class SlotNet(nn.Module):
    def __init__(self, n_slots: int, img: int = 336, unfreeze_last: int = UNFREEZE_LAST,
                 dropout: float = 0.1):
        super().__init__()
        import timm
        self.backbone = timm.create_model(BACKBONE, pretrained=True, img_size=img,
                                          num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks[len(self.backbone.blocks) - unfreeze_last:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        self.head = SlotHead(self.backbone.num_features, n_slots, dropout=dropout)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x [B,K,3,H,W] -> logits [B,12]. The encoder sees B*K images; that is the whole cost."""
        b, k = x.shape[:2]
        e = self.backbone(x.flatten(0, 1)).view(b, k, -1)
        return self.head(e, m)

    def param_groups(self) -> list[dict]:
        return [{"params": [p for p in self.backbone.parameters() if p.requires_grad],
                 "lr": LR_BACKBONE},
                {"params": list(self.head.parameters()), "lr": LR_HEAD}]


class OrthoNet(nn.Module):
    """OrthoFoundation-L in the fork's harness. The one configuration nobody public is running.

    ⛔ WHY THIS EXISTS AND WHY IT IS NOT ANOTHER ARM. Measured 2026-08-19 over all 1,992 leaderboard
    rows: every author publishing an ensemble kernel sits in **0.917-0.922** -- Tony Li 0.922,
    Mattia Angeli 0.920, Aman Atar 0.920, Kunal Desale 0.920 -- and every one of them runs the same
    **22 M dinov2-small pretrained on natural images**, because they all forked the same baseline.
    The leaders at 0.942-0.952 publish nothing. So the public ceiling is 0.922, blending public
    arms lands inside it by construction, and the way out is not another arm.

    This is a **303 M DINOv3-L continued-pretrained on 1,251,655 knee images** (§C-5), 14x the
    parameters with the domain the architecture was missing. §3t's generic DINOv3 measured 0.8025
    and §3w-2's narrowed verdict is the point: *our training was refuted, the architecture was
    not.* It scored 0.7710 FROZEN this morning and failed its gate -- and a frozen cache cannot
    fine-tune at any resolution, under any head (§2e), which is exactly what that measured.

    Head, pooling and slot geometry are the fork's, unchanged, so the ONLY variable against the
    reproduction is the encoder. Pooling is `preprocess.embed`'s verified convention --
    `num_prefix_tokens`, not a hardcoded 1 -- because this backbone carries **4 register tokens**
    after CLS and slicing at `[:, 1:]` would fold four registers into the patch mean.
    """

    def __init__(self, n_slots: int, img: int = 336, unfreeze_last: int = UNFREEZE_LAST,
                 dropout: float = 0.1):
        super().__init__()
        from fusion.orthofoundation_model import build as build_of
        from fusion.pilkwang_model import SlotHead as ForkSlotHead
        self.backbone, _ = build_of(img=img)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks[len(self.backbone.blocks) - unfreeze_last:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        dim = self.backbone.num_features
        self.head = ForkSlotHead(dim * 2, n_slots, len(LABELS), prior=False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        print(f"  backbone: OrthoFoundation-L, {len(self.backbone.blocks)} blocks, "
              f"last {unfreeze_last} trainable, dim {dim}, prefix {self.backbone.num_prefix_tokens}")

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        b, k = x.shape[:2]
        z = x.flatten(0, 1).float().div(255.0)
        z = (z - self.mean) / self.std
        tok = self.backbone.forward_features(z)
        cls, patch = tok[:, 0], tok[:, self.backbone.num_prefix_tokens:].mean(1)
        return self.head(torch.cat([cls, patch], -1).view(b, k, -1), m)


def weighted_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()


def loss_references(y: np.ndarray) -> tuple[float, float]:
    """-> (irreducible floor, constant-prior loss). Printed so the loss is READABLE.

    **This loss cannot reach zero and it is a mistake to read it as though it could.** The
    targets are soft -- 19.7% of values are exactly 0.5, meaning "the report does not say" -- and
    a model predicting p = y perfectly still pays the entropy H(y), which is ln 2 at 0.5. So the
    whole meaningful range is [floor, prior], measured on the fold-0 training set as
    **0.2040 to 0.4640**, and a raw "0.33" is 50% of the way across it rather than a bad number.

    The constant per-label prior is the "learned nothing" end: it is what a model scores by
    ignoring the images and emitting each label's base rate. Epoch 1 landing at 0.4523 against a
    prior of 0.4640 is the honest picture of how little one epoch buys.

    Added 2026-08-10 after "the loss is so high, should we keep going?" -- a fair question that
    the printed number alone could not answer, and which cost a measurement to settle. Printing
    both ends means it never costs one again. Neither number decides whether to continue: only
    the OOF does, because training headroom can be consumed by memorising.
    """
    # float64 BEFORE the clip, and this is not cosmetic: `y` arrives float32 from
    # SlotDataset, where `1 - 1e-9` rounds to exactly 1.0, so the clip is a no-op on the upper
    # end and the 24 targets that sit at exactly 1.0 take log(0) = -inf -> floor NaN. Caught
    # 2026-08-17 when 3w's LR probe printed `floor nan`. Display-only -- the training loss is
    # weighted_bce on logits and never calls this -- but a NaN floor makes the probe's
    # "% of the way prior->floor" readout unreadable, which is the number it exists to give.
    y = np.asarray(y, dtype=np.float64)
    w = 0.25 + 0.75 * np.abs(2 * y - 1)
    p = np.clip(y, 1e-9, 1 - 1e-9)
    floor = float((w * -(p * np.log(p) + (1 - p) * np.log(1 - p))).mean())
    q = np.clip(np.repeat(y.mean(0)[None, :], len(y), 0), 1e-9, 1 - 1e-9)
    prior = float((w * -(y * np.log(q) + (1 - y) * np.log(1 - q))).mean())
    return floor, prior


# ----------------------------------------------------------------------------- loop
def _fold_cfg(a, store: TileStore, fold: int, n_train: int) -> dict:
    """Everything a resumed checkpoint must agree with to be the SAME experiment.

    Deliberately includes the label path: resuming a `targets.csv` run into a `report_labels_v2`
    run would silently produce an arm trained on two different label sources and no output would
    say so. That is the §2s error class -- a comparison whose provenance is not what it claims --
    arriving through the back door of a convenience feature.
    """
    return {"labels": str(a.labels), "folds": str(a.folds), "cache": str(a.cache),
            "tag": a.tag, "fold": int(fold), "slots": list(store.slots),
            # n_train is in here because OneCycleLR's total_steps is epochs x len(tr_dl). The
            # corpus downloads incrementally (81.5% and rising), so resuming after more studies
            # land would restore a scheduler built for a different total. That surfaces as
            # "Tried to step N times. The specified number of total steps is M" at the END of the
            # run -- loud, but hours late. Catching it here fails in the first second instead.
            "n_train": int(n_train),
            "img": a.img, "epochs": a.epochs, "batch": a.batch,
            "unfreeze_last": a.unfreeze_last, "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
            "wd": a.wd, "dropout": a.dropout, "slot_dropout": a.slot_dropout,
            "seed": a.seed, "limit": a.limit,
            "preprocess_version": store.manifest.get("preprocess_version")}


def run_fold(fold: int, folds: pd.DataFrame, store: TileStore, targets: pd.DataFrame,
             a, dev: torch.device):
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

    # ⛔ THE FORK'S OWN OBJECT, NOT OUR RECONSTRUCTION OF IT `2026-08-19`. `contract_audit.py`
    # found this port diverging from the config that produced the fork's 0.8398 on five axes;
    # three of them -- backbone, prior, pool -- are fixed by simply building THEIR model, which
    # `pilkwang_model.py` already exposes and which `--check` verifies at 20/20, 7e-06:
    #
    #     backbone   timm vit_small_patch14_reg4_dinov2  ->  facebook/dinov2-small (no registers)
    #     prior      SLOT_PRIOR_STRENGTH = 0.55          ->  prior=False, as the manifest records
    #     pool       (implicit in SlotNet)               ->  cls_mean, CLS || patch-mean
    #
    # SlotNet below is kept as the record of what was measured at 0.7323 and is no longer used.
    # `img_size=None` at the call sites is correct: the tiles are already `a.img`, so the fork's
    # forward skips its interpolate rather than resampling a resampled image.
    if a.backbone == "orthofoundation":
        model = OrthoNet(len(store.slots), img=a.img, unfreeze_last=a.unfreeze_last,
                         dropout=a.dropout).to(dev)
    else:
        from fusion.pilkwang_model import build_model
        model = build_model(a.unfreeze_last, prior=False, pool="cls_mean").to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  fold {fold}: train {len(tr):,} / val {len(va):,} studies · "
          f"{n_tr / 1e6:.1f}M of {n_all / 1e6:.1f}M params trainable")
    lo, hi = loss_references(tr_ds.y)
    print(f"  loss scale: floor {lo:.4f} (perfect, p=y) .. {hi:.4f} (constant prior). "
          f"It CANNOT reach 0 -- the targets are soft.")

    # The fork's Model has no `param_groups`; the two-rate split is the port's own contribution
    # and is preserved exactly (LR_BACKBONE on the opened blocks, LR_HEAD on the head).
    groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad],
               "lr": LR_BACKBONE},
              {"params": list(model.head.parameters()), "lr": LR_HEAD}]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=max(a.epochs * len(tr_dl), 1),
        pct_start=0.25)

    # Resume point. `fold{f}.pt` is only written when a fold RETURNS, so before 2026-08-12 a run
    # that died mid-fold left nothing at all -- the gate arm lost 7 of 10 epochs and ~3 h that way
    # (§2u). On a box that thrashes (§2p) and a budget of ~20 five-fold experiments (§2t-3), an
    # unrecoverable long run is the expensive kind of bug. The per-epoch checkpoint is 175 MB
    # (measured -- it carries AdamW's two moments as well as the full state_dict) and is
    # OVERWRITTEN each epoch, so the cost is 175 MB of disk and about a second of wall clock.
    ck = Path(a.out) / f"fold{fold}_last.pt"
    start_ep = 0
    if a.resume and ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        if st.get("cfg") != _fold_cfg(a, store, fold, len(tr)):
            raise SystemExit(
                f"{ck} was written under a different configuration:\n"
                f"  checkpoint {st.get('cfg')}\n  this run   {_fold_cfg(a, store, fold, len(tr))}\n"
                "Resuming across a config change would silently mix two experiments. Delete it "
                "or drop --resume.")
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

        # Written to a temp path and renamed, so a kill DURING the save cannot leave a truncated
        # checkpoint that then fails to load -- rename is atomic on the same filesystem.
        tmp = ck.with_suffix(".pt.tmp")
        torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep + 1,
                    "cfg": _fold_cfg(a, store, fold, len(tr))}, tmp)
        tmp.replace(ck)

    model.eval()
    uids, preds = [], []
    with torch.no_grad():
        for b in va_dl:
            preds.append(torch.sigmoid(model(b["x"].to(dev), b["m"].to(dev))).float().cpu().numpy())
            uids += b["uid"]
    return model, uids, np.concatenate(preds) if preds else np.zeros((0, 12)), len(tr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=str(D / "tiles336"))
    ap.add_argument("--tag", default="protocol")
    ap.add_argument("--backbone", default="pilkwang",
                    choices=["pilkwang", "orthofoundation"],
                    help="pilkwang = the fork's dinov2-small (the reproduction); "
                         "orthofoundation = DINOv3-L on 1.25M knee images, 303M params")
    ap.add_argument("--folds", default=str(D / "folds_site.csv"),
                    help="site-grouped by default: ungrouped folds inflate by +0.024 (2j)")
    ap.add_argument("--labels", default=str(D / "targets.csv"))
    ap.add_argument("--run-folds", default=None, help="e.g. '0' or '0,1'; default all")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=8, help="STUDIES per step; x6 slots = images")
    ap.add_argument("--img", type=int, default=336)
    ap.add_argument("--unfreeze-last", type=int, default=UNFREEZE_LAST)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--slot-dropout", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--limit", type=int, default=0, help="studies per fold, for a smoke test")
    ap.add_argument("--out", default=str(PROJ / "fusion" / "runs_port"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue from fold{N}_last.pt if it matches this run's config")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    store = TileStore(Path(a.cache), a.tag)
    folds = pd.read_csv(a.folds)
    targets = pd.read_csv(a.labels).drop_duplicates("StudyInstanceUID").set_index(
        "StudyInstanceUID")

    print(__doc__.splitlines()[0])
    print(f"device {dev} · cache {Path(a.cache).name}/{a.tag} "
          f"{store.arr.shape} ({store.arr.nbytes / 1e9:.2f} GB)")
    print(f"slots {store.slots}")
    print(f"folds {Path(a.folds).name} · labels {Path(a.labels).name} · "
          f"epochs {a.epochs} · batch {a.batch} studies")
    if store.manifest.get("direction_bits", 0) == 0:
        print("NOTE: cache built with NO K16 direction bit. Fine for the protocol slots "
              "(depth 0.5 is\n      near direction-invariant) but sagittal anatomical slabs "
              "are not buildable yet.")

    which = ([int(x) for x in a.run_folds.split(",")] if a.run_folds
             else sorted(folds.fold.unique()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    oof: dict[str, np.ndarray] = {}
    for f in which:
        model, uids, preds, n_tr = run_fold(f, folds, store, targets, a, dev)
        oof.update(dict(zip(uids, preds)))
        torch.save({"state_dict": model.state_dict(), "fold": int(f), "labels": LABELS,
                    "slots": store.slots, "img": a.img,
                    "unfreeze_last": a.unfreeze_last,
                    "cache_manifest": store.manifest}, out / f"fold{f}.pt")
        print(f"  fold {f} done: trained on {n_tr:,}, predicted {len(uids):,} "
              f"({(time.time() - t0) / 60:.1f} min cumulative)", flush=True)

    uids = list(oof)
    pd.DataFrame(np.stack([oof[u] for u in uids]), columns=LABELS).assign(
        StudyInstanceUID=uids).to_csv(out / "oof_all.csv", index=False)

    # ---- score on the SAME scale as the baseline, via the one shared definition --------------
    # NOT against `targets` -- that is this model's own training source (steven_v2), and scoring
    # against it rewards reproducing that reader's idiosyncrasies rather than the signal. The
    # 0.7229 in 2j is scored against lixin_gpt56, a THIRD source, over non-gold studies only.
    # An earlier version of this file printed the targets-scored number directly beneath
    # "baseline to beat: 0.7229", which reads as a comparison and is a category error.
    # fusion/score_oof.py is now the single definition; this just calls it.
    sys.path.insert(0, str(PROJ / "fusion"))
    from score_oof import BASELINE, BASELINE_SD, NEUTRAL, score as score_run  # noqa: E402
    macro, aucs = float("nan"), {}
    if NEUTRAL.exists():
        ref = pd.read_csv(NEUTRAL).drop_duplicates("StudyInstanceUID").set_index(
            "StudyInstanceUID")
        gold_ids = set(pd.read_csv(D / "train.csv").dropna(subset=LABELS).StudyInstanceUID)
        try:
            macro, sd, n, aucs = score_run(out, ref, gold_ids)
            print(f"\n{'=' * 62}\nreport-OOF vs {NEUTRAL.name} (held out), n={n:,}\n{'=' * 62}")
            for lab in LABELS:
                print(f"{lab:<18}{aucs[lab]:>9.3f}")
            print(f"{'-' * 27}\n{'MACRO':<18}{macro:>9.4f}  +-{sd:.4f}")
            d = macro - BASELINE
            print(f"  baseline {BASELINE} +-{BASELINE_SD} (site-grouped, frozen cache, 2j): "
                  f"{d:+.4f} ({abs(d) / float(np.hypot(sd, BASELINE_SD)):.1f} sigma)")
            print("  NOT a paired A/B -- see fusion/score_oof.py's closing note.")
        except SystemExit as e:                                        # noqa: BLE001
            print(f"\nscoring skipped: {e}")
    else:
        print(f"\nscoring skipped: {NEUTRAL} missing; run "
              "extractor/bench_public_labels.py --download")

    gold = pd.read_csv(D / "train.csv").dropna(subset=LABELS)
    gold = gold[gold.StudyInstanceUID.isin(oof)]
    gmacro = float("nan")
    if len(gold) >= 10:
        G = np.stack([oof[u] for u in gold.StudyInstanceUID])
        ga = {lab: auc(gold[lab].to_numpy(float), G[:, i]) for i, lab in enumerate(LABELS)
              if min(gold[lab].sum(), len(gold) - gold[lab].sum()) >= 2}
        gmacro = float(np.mean(list(ga.values())))
        print(f"\ngold-{len(gold)} check (NOT the arbiter, +-0.031): macro {gmacro:.3f}")

    (out / "summary.json").write_text(json.dumps(
        {"macro_report_oof": macro, "per_label": aucs, "macro_gold": gmacro,
         # len(oof), NOT len(ref): `ref` is the neutral label TABLE (4,407 rows) and is bound
         # only inside the `if NEUTRAL.exists()` branch above, so reading it here recorded the
         # wrong count on every run and raised NameError on the path where the reference file is
         # missing -- after every fold had trained. Existing summaries read n_oof: 691, which is
         # len(oof) and correct; they predate b3387e8.
         "n_oof": len(oof), "folds": which, "epochs": a.epochs, "batch": a.batch,
         "img": a.img, "unfreeze_last": a.unfreeze_last, "slots": store.slots,
         "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
         "minutes": (time.time() - t0) / 60}, indent=2))
    print(f"\n{(time.time() - t0) / 60:.1f} min total · wrote {out}/oof_all.csv, summary.json")


if __name__ == "__main__":
    main()
