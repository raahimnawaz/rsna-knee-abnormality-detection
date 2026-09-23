"""Train one encoder on 2.5D windows of a fixed slice stack -- the frontier's recipe, as a harness.

    python fusion/train_windows.py --arch ortho   --corpus /kaggle/input --train train.csv \
        --val val.csv --labels targets.csv --out runs_ortho --epochs 10
    python fusion/train_windows.py --arch tiny --synthetic 24 --out /tmp/tw --epochs 1   # wiring

WHY THIS EXISTS. `MODEL.md` §0: the public frontier (dreaddevelopment, LB 0.928 single model)
publishes its training recipe, and C6 copied its input but not its training. This file
re-implements that recipe as a harness in which the ENCODER is the only thing an arm changes:

    ortho     OrthoFoundation-L, loaded strict with its own RoPE periods (orthofoundation_model.py)
    dinov3l   stock `vit_large_patch16_dinov3.lvd1689m` -- Ortho's architecture without the knee
              corpus, so ortho - dinov3l is what knee pretraining is worth (Stage C's C1 - C0)
    coatnet   `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` -- the frontier's own encoder, at 384
    tiny      `vit_tiny_patch16_224` without weights -- wiring checks only

THE RECIPE, read from their script (not its description), and where this differs:

  * A window is three consecutive slices as the three channels. Train: `--k` windows drawn per
    study per step. Eval: EVERY window the study holds.
  * Encoder fully fine-tuned, AdamW, lr 3e-5 encoder / 1e-3 head, wd 0.02, OneCycle with a 15%
    warm-up, grad-norm clip 3.0. Augmentation is a per-study intensity gain of +-10% and nothing
    else: no flips, laterality is signal.
  * Head: per-finding attention MIL. LayerNorm, then Linear(F,256)-Tanh-Dropout(0.2)-Linear(256,12)
    gives each finding its own softmax over windows; a per-finding linear reads its pooled vector.
  * Loss: BCE with pos_weight = clip((1-p)/p, 1, 10), p = clip(train prevalence, 0.03, 0.7).
  DIFFERENCES, each deliberate:
  * Windows never straddle a slot boundary. Their `_centers` ranges over the whole stack, so a
    window can put the last sagittal slice beside the first coronal one. Here a window's three
    slices come from one series. 44 = 12+10+8+6+8 gives at most 34 windows per study.
  * The confidence weight `0.25 + 0.75|2y-1|` multiplies the loss, because `steven_v2`'s 0.5
    means "the report does not say", not "hedged" (train_port.py:173). Their labels differ.
  * NO epoch is selected. They keep the best epoch on gold-58, which is `RUNBOOK.md` §3 rule 1.
    Primary is the LAST epoch; secondary, fixed in advance, is the weight average of the last
    three. Gold-58 is scored and written, never used.
  * fp16 + GradScaler on CUDA: a T4 has no bf16.

Normalisation is ImageNet's for every arm, and for Ortho that is not an assumption: its own
pretraining code applies exactly these constants (`PREMORTEM.md` §9, resolved 2026-09-23).

Outputs in --out: `oof_all.csv` (the --val studies, what `score_oof.py` reads), `avg3/oof_all.csv`
(so `score_oof.py <out>/avg3` scores the secondary the same way), `oof_gold.csv`, `summary.json` with timings, and `bench.json` -- the per-step cost of THIS encoder
on THIS machine, which is what the full run is priced from (PREMORTEM §1: bench, never extrapolate).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "extractor"))

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOT_SIZES = (12, 10, 8, 6, 8)          # the corpus's 44: SAG-fluid, SAG-other, COR-fluid, COR-2, AX
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ARCHS = {
    "ortho":   dict(res=336, pool="token"),
    "dinov3l": dict(res=336, pool="token", timm="vit_large_patch16_dinov3.lvd1689m"),
    "coatnet": dict(res=384, pool="avg", timm="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"),
    "tiny":    dict(res=224, pool="token", timm="vit_tiny_patch16_224"),
}
ORTHO_URL = "https://media.githubusercontent.com/media/ytrsk/OrthoFoundation/main/OrthoFoudation-L.pth"
ORTHO_BYTES = 1213056638


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- corpus
class Corpus:
    """Row-ordered parts as one (N, D, 336, 336) uint8 view. ROW ORDER IS THE CONTRACT: within a
    part, ids[i] names vols[i] and masks[i]; nothing here sorts any of the three.

    Two layouts, found by what is attached:
      raptor44  dreaddevelopment's CC0 corpus, two parts, D = 44 = 12+10+8+6+8 (15-85%)
      stage_a   our `pipeline/build_slots64.py` shards, D = 64 = 18+14+12+8+12, span in the manifest.
                Every shard of one build must be present, or the run refuses.
    """

    RAPTOR_PARTS = (("all_vols.npy", "all_masks.npy", "all_ids.npy"),
                    ("extra_vols.npy", "extra_masks.npy", "extra_ids.npy"))

    def __init__(self, root: Path, layout: str = "auto"):
        found, c6 = {}, []
        for d, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if x not in ("train_series", "test_series", "train_images")]
            for f in fs:
                found.setdefault(f, Path(d) / f)
            if "manifest.json" in fs and "vols.npy" in fs:
                try:
                    j = json.loads((Path(d) / "manifest.json").read_text())
                except (OSError, ValueError):
                    continue
                if j.get("spec") == "c6-64slice":
                    c6.append((int(j.get("shard", 0)), Path(d), j))
        if layout == "stage_a" or (layout == "auto" and c6):
            if not c6:
                raise SystemExit(f"no Stage A shard (manifest spec c6-64slice) under {root}")
            c6.sort(key=lambda t: t[0])
            n_sh = {j["n_shards"] for _, _, j in c6}
            got = [s for s, _, _ in c6]
            if len(n_sh) != 1 or got != list(range(next(iter(n_sh)))):
                raise SystemExit(f"Stage A shards incomplete or mixed: have {got} of {n_sh}")
            if len({tuple(j["depth_range"]) for _, _, j in c6}) != 1:
                raise SystemExit("Stage A shards were built with different spans")
            self.slot_sizes = tuple(s["n_slices"] for s in c6[0][2]["slots"])
            self.layout = f"stage_a span {c6[0][2]['depth_range']}"
            parts = [(d / "vols.npy", d / "masks.npy", d / "ids.npy") for _, d, _ in c6]
        else:
            self.slot_sizes = SLOT_SIZES
            self.layout = "raptor44"
            parts = []
            for v, m, i in self.RAPTOR_PARTS:
                if v not in found:
                    raise SystemExit(f"corpus part {v} not found under {root} -- attach "
                                     f"dreaddevelopment/knee-raptor-corpus and knee-raptor-corpus-ext")
                parts.append((found[v], found[m], found[i]))
        depth = sum(self.slot_sizes)
        self.vols, masks, ids, self.starts = [], [], [], []
        for v, m, i in parts:
            vol = np.load(v, mmap_mode="r")
            msk = np.load(m)
            idv = np.load(i, allow_pickle=True).astype(str)
            if not (len(vol) == len(msk) == len(idv)) or vol.shape[1:] != (depth, 336, 336):
                raise SystemExit(f"{v}: shapes {vol.shape} / {msk.shape} / {idv.shape} break the contract")
            self.starts.append(sum(len(x) for x in self.vols))
            self.vols.append(vol)
            masks.append(msk)
            ids.append(idv)
            log(f"corpus part {v.name}: {vol.shape} {vol.dtype}  from {v.parent}")
        self.masks = np.concatenate(masks)
        self.ids = np.concatenate(ids)
        self.row = {u: k for k, u in enumerate(self.ids)}
        if len(self.row) != len(self.ids):
            raise SystemExit("duplicate StudyInstanceUIDs in the corpus ids")
        log(f"corpus layout {self.layout}: {len(self.ids)} studies, slots {self.slot_sizes}")

    def study(self, uid: str) -> tuple[np.ndarray, np.ndarray]:
        r = self.row[uid]
        p = int(np.searchsorted(self.starts, r, side="right")) - 1
        return self.vols[p][r - self.starts[p]], self.masks[r]


class SyntheticCorpus:
    """Random stacks in the raptor44 layout, for a wiring check off Kaggle."""

    def __init__(self, uids: list[str], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.slot_sizes, self.layout = SLOT_SIZES, "synthetic raptor44"
        self.ids = np.asarray(uids)
        self.row = {u: k for k, u in enumerate(uids)}
        self.vol = rng.integers(0, 255, (len(uids), 44, 336, 336), dtype=np.uint8)
        self.masks = np.ones((len(uids), 44), np.uint8)
        self.masks[::3, 34:36] = 0                          # a short last coronal, some studies
        self.masks[1::4, 36:44] = 0                         # and a missing axial slot

    def study(self, uid: str) -> tuple[np.ndarray, np.ndarray]:
        r = self.row[uid]
        return self.vol[r], self.masks[r]


def window_centres(mask: np.ndarray, slot_sizes=SLOT_SIZES) -> list[int]:
    """Centres whose three slices are all valid AND inside one slot."""
    out, o = [], 0
    for n in slot_sizes:
        for c in range(o + 1, o + n - 1):
            if mask[c - 1] and mask[c] and mask[c + 1]:
                out.append(c)
        o += n
    return out


class Windows(torch.utils.data.Dataset):
    """One study -> (uint8 windows [K,3,336,336], targets [12]). Normalisation happens on device."""

    def __init__(self, corpus, uids: list[str], y: np.ndarray, k: int | None, seed: int = 0):
        self.c, self.uids, self.y, self.k, self.seed = corpus, uids, y, k, seed
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        vol, mask = self.c.study(self.uids[i])
        cs = window_centres(mask, self.c.slot_sizes)
        if not cs:                                          # nothing valid: the stack's middle
            cs = [len(mask) // 2]
        if self.k is not None:                              # train: k draws, no repeats if possible
            cs = self.rng.sample(cs, self.k) if len(cs) >= self.k else \
                [self.rng.choice(cs) for _ in range(self.k)]
        w = np.stack([vol[c - 1:c + 2] for c in cs])        # (K,3,H,W)
        return torch.from_numpy(np.ascontiguousarray(w)), torch.from_numpy(self.y[i])


def seed_worker(_wid: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(hash((info.dataset.seed, info.id)))


def pad_collate(batch):
    """Eval batches hold studies with different window counts: pad, and carry the mask."""
    kmax = max(b[0].shape[0] for b in batch)
    x = torch.zeros((len(batch), kmax) + tuple(batch[0][0].shape[1:]), dtype=torch.uint8)
    m = torch.zeros((len(batch), kmax), dtype=torch.bool)
    for j, (w, _) in enumerate(batch):
        x[j, :len(w)] = w
        m[j, :len(w)] = True
    return x, m, torch.stack([b[1] for b in batch])


# ----------------------------------------------------------------------------- model
def build_encoder(arch: str, weights_dir: Path) -> nn.Module:
    import timm
    spec = ARCHS[arch]
    if arch == "ortho":
        import hashlib
        import urllib.request
        import orthofoundation_model as om
        dst = weights_dir / "OrthoFoudation-L.pth"
        if not dst.exists() or dst.stat().st_size != ORTHO_BYTES:
            log(f"downloading OrthoFoundation-L from {ORTHO_URL}")
            urllib.request.urlretrieve(ORTHO_URL, dst)
        if dst.stat().st_size != ORTHO_BYTES:
            raise SystemExit(f"{dst}: {dst.stat().st_size} bytes, expected {ORTHO_BYTES} (an LFS pointer?)")
        h = hashlib.sha256(dst.read_bytes()).hexdigest()
        if h != om.SHA256:
            raise SystemExit(f"OrthoFoundation sha256 {h} != recorded {om.SHA256}")
        m, periods = om.build(dst, img=spec["res"], global_pool=spec["pool"])
        if not torch.equal(m.rope.periods.float(), periods.float()):
            raise SystemExit("RoPE periods were not applied -- the §9h trap")
        log(f"ortho: strict load, sha256 ok, RoPE periods = theirs")
        return m
    kw = dict(pretrained=arch != "tiny", num_classes=0, global_pool=spec["pool"])
    if spec["pool"] == "token" and arch != "tiny":
        kw["img_size"] = spec["res"]
    return timm.create_model(spec["timm"], **kw)


class WindowMIL(nn.Module):
    """Encoder + per-finding attention over windows. Input is uint8; the device does the rest."""

    def __init__(self, encoder: nn.Module, res: int, n: int = len(LABELS), drop: float = 0.2):
        super().__init__()
        self.encoder, self.res = encoder, res
        f = encoder.num_features
        self.norm = nn.LayerNorm(f)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.w = nn.Parameter(torch.zeros(n, f))
        self.b = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.w, std=0.02)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def prep(self, x: torch.Tensor, gain: torch.Tensor | None) -> torch.Tensor:
        b, k = x.shape[:2]
        x = x.flatten(0, 1).float().div_(255.0)
        if gain is not None:                                # per-study intensity jitter, [0,1] space
            x = (x * gain.repeat_interleave(k).view(-1, 1, 1, 1)).clamp_(0, 1)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x, mask=None, gain=None):
        b, k = x.shape[:2]
        h = self.norm(self.encoder(self.prep(x, gain)).view(b, k, -1).float())
        a = self.att(h)                                     # (B,K,12)
        if mask is not None:
            a = a.masked_fill(~mask[..., None], float("-inf"))
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.w).sum(-1) + self.b


# ----------------------------------------------------------------------------- run
def load_split(path: Path) -> list[str]:
    return pd.read_csv(path).StudyInstanceUID.astype(str).drop_duplicates().tolist()


@torch.no_grad()
def predict(model, ds, dev, amp, bs) -> np.ndarray:
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=ds_workers(),
                                     collate_fn=pad_collate)
    out = []
    for x, m, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out.append(torch.sigmoid(model(x.to(dev), m.to(dev)).float()).cpu().numpy())
    return np.concatenate(out)


def ds_workers() -> int:
    return int(os.environ.get("TW_WORKERS", "4"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=sorted(ARCHS), required=True)
    ap.add_argument("--corpus", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--layout", choices=["auto", "raptor44", "stage_a"], default="auto")
    ap.add_argument("--synthetic", type=int, default=0, help="N random studies instead of a corpus")
    ap.add_argument("--train", type=Path, help="csv with StudyInstanceUID")
    ap.add_argument("--val", type=Path, help="csv with StudyInstanceUID -- scored, never selected on")
    ap.add_argument("--gold", type=Path, help="train.csv; its fully-labelled rows are scored, never used")
    ap.add_argument("--labels", type=Path, help="targets.csv (StudyInstanceUID + 12 columns)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, default=Path("/tmp"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--k", type=int, default=12, help="windows per study per training step")
    ap.add_argument("--bs", type=int, default=8, help="studies per optimiser step")
    ap.add_argument("--micro", type=int, default=2, help="studies per forward; bs/micro = accumulation")
    ap.add_argument("--eval-bs", type=int, default=2)
    ap.add_argument("--bb-lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop an epoch early (benching)")
    ap.add_argument("--time-budget-h", type=float, default=10.5)
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.bs % a.micro:
        raise SystemExit("--bs must be a multiple of --micro")

    t_start = time.time()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    a.out.mkdir(parents=True, exist_ok=True)
    log(f"arch {a.arch} {ARCHS[a.arch]} | device {dev}"
        + (f" {torch.cuda.get_device_name(0)}" if dev == "cuda" else "") + f" | amp fp16 {amp}")

    # ---- studies and targets
    gold_ids: list[str] = []
    if a.synthetic:
        uids = [f"syn{i:04d}" for i in range(a.synthetic)]
        corpus = SyntheticCorpus(uids)
        rng = np.random.default_rng(1)
        tgt = pd.DataFrame(rng.choice([0.0, 0.5, 1.0], (len(uids), 12), p=[.6, .2, .2]),
                           index=uids, columns=LABELS)
        tr, va = uids[: a.synthetic * 2 // 3], uids[a.synthetic * 2 // 3:]
    else:
        corpus = Corpus(a.corpus, a.layout)
        tgt = pd.read_csv(a.labels).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        tr, va = load_split(a.train), load_split(a.val)
        if a.gold:
            g = pd.read_csv(a.gold).dropna(subset=LABELS)
            gold_ids = [u for u in g.StudyInstanceUID.astype(str) if u in corpus.row]
            gtab = g.set_index("StudyInstanceUID")
        if set(tr) & set(va) or set(tr) & set(gold_ids):
            raise SystemExit("train overlaps val or gold -- refusing")
    miss_tr = [u for u in tr if u not in corpus.row]
    miss_va = [u for u in va if u not in corpus.row]
    tr = [u for u in tr if u in corpus.row and tgt.loc[u, LABELS].notna().all()]
    va = [u for u in va if u in corpus.row]
    if a.limit_train:
        tr = tr[: a.limit_train]
    if a.limit_val:
        va = va[: a.limit_val]
    log(f"train {len(tr)} (missing from corpus {len(miss_tr)}) | val {len(va)} "
        f"(missing {len(miss_va)}) | gold {len(gold_ids)}")
    ytr = tgt.loc[tr, LABELS].to_numpy(np.float32, copy=True)   # pandas CoW hands back read-only
    yva = np.zeros((len(va), 12), np.float32)
    nwin = [len(window_centres(corpus.study(u)[1], corpus.slot_sizes)) for u in va]
    log(f"val windows per study: median {int(np.median(nwin))}, min {min(nwin)}, max {max(nwin)}")

    # ---- model
    model = WindowMIL(build_encoder(a.arch, a.weights_dir), ARCHS[a.arch]["res"]).to(dev)
    if hasattr(model.encoder, "set_grad_checkpointing") and dev == "cuda":
        model.encoder.set_grad_checkpointing(True)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    log(f"encoder {n_enc / 1e6:.1f}M params, num_features {model.encoder.num_features}")

    prev = np.clip(ytr.mean(0), 0.03, 0.7)
    pos_w = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": a.bb_lr},
                             {"params": [p for n_, p in model.named_parameters()
                                         if not n_.startswith("encoder.")], "lr": a.head_lr}],
                            weight_decay=a.wd)
    tr_ds = Windows(corpus, tr, ytr, k=a.k, seed=a.seed)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=a.micro, shuffle=True, drop_last=True,
                                        num_workers=ds_workers(), worker_init_fn=seed_worker,
                                        persistent_workers=ds_workers() > 0)
    accum = a.bs // a.micro
    steps_per_epoch = len(tr_dl) // accum
    if a.max_steps:
        steps_per_epoch = min(steps_per_epoch, a.max_steps)
    total = max(steps_per_epoch * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],
                                                total_steps=total, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log(f"steps/epoch {steps_per_epoch} (bs {a.bs} = {a.micro} x {accum}) | epochs {a.epochs} | "
        f"k {a.k} | lr {a.bb_lr}/{a.head_lr}")

    # ---- train
    hist, last3, step_times = [], [], []
    stopped_early = None
    for ep in range(a.epochs):
        model.train()
        t_ep, tot, n_opt = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(tr_dl):
            if n_opt >= steps_per_epoch:
                break
            t0 = time.time()
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            gain = 1.0 + (torch.rand(len(x), device=dev) - 0.5) * 0.2
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x, gain=gain)
            w = 0.25 + 0.75 * (2 * y - 1).abs()
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_w,
                                                       reduction="none") * w).mean()
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {ep} iter {it}")
            scaler.scale(loss / accum).backward()
            tot += float(loss.detach())
            if (it + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                n_opt += 1
            if dev == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.time() - t0)
            if it % 50 == 0:
                log(f"  ep{ep} it{it} loss {float(loss.detach()):.4f} "
                    f"({np.mean(step_times[-50:]):.2f} s per {a.micro}-study forward+backward)")
            if time.time() - t_start > a.time_budget_h * 3600:
                stopped_early = f"time budget at epoch {ep} iter {it}"
                break
        hist.append({"epoch": ep, "loss": tot / max(it + 1, 1), "sec": time.time() - t_ep,
                     "opt_steps": n_opt})
        log(f"epoch {ep}: loss {hist[-1]['loss']:.4f}  {hist[-1]['sec'] / 60:.1f} min")
        if ep >= a.epochs - 3:
            last3.append({k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
        if stopped_early:
            log(f"⛔ stopped early: {stopped_early} -- the schedule did not finish")
            break

    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None
    per_fwd = float(np.median(step_times)) if step_times else None
    bench = {"arch": a.arch, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
             "micro_studies": a.micro, "k": a.k, "res": ARCHS[a.arch]["res"],
             "sec_per_micro_step_median": per_fwd,
             "sec_per_window_train": per_fwd / (a.micro * a.k) if per_fwd else None,
             "train_epoch_min_measured": hist[0]["sec"] / 60 if hist else None,
             "steps_in_measured_epoch": hist[0]["opt_steps"] if hist else None,
             "full_epoch_min_projected": (per_fwd * len(tr_dl) / 60) if per_fwd else None,
             "peak_cuda_gb": peak}

    # ---- evaluate: last epoch (primary), then the fixed last-3 average (secondary)
    t_ev = time.time()
    va_ds = Windows(corpus, va, yva, k=None)
    p_last = predict(model, va_ds, dev, amp, a.eval_bs)
    bench["eval_sec_per_study"] = (time.time() - t_ev) / max(len(va), 1)
    pd.DataFrame(p_last, columns=LABELS).assign(StudyInstanceUID=va).to_csv(a.out / "oof_all.csv", index=False)
    if gold_ids:
        gds = Windows(corpus, gold_ids, np.zeros((len(gold_ids), 12), np.float32), k=None)
        pg = predict(model, gds, dev, amp, a.eval_bs)
        pd.DataFrame(pg, columns=LABELS).assign(StudyInstanceUID=gold_ids).to_csv(a.out / "oof_gold.csv", index=False)
    if a.save_weights:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   a.out / "last.pt")
    if len(last3) >= 2:
        avg = {k: (sum(sd[k].double() for sd in last3) / len(last3)).to(last3[0][k].dtype)
               if last3[0][k].is_floating_point() else last3[0][k] for k in last3[0]}
        model.load_state_dict(avg)
        pa = predict(model, va_ds, dev, amp, a.eval_bs)
        d3 = a.out / "avg3"
        d3.mkdir(exist_ok=True)
        pd.DataFrame(pa, columns=LABELS).assign(StudyInstanceUID=va).to_csv(d3 / "oof_all.csv", index=False)

    gold_auc = None
    if gold_ids:
        from metrics import auc
        yg = gtab.loc[gold_ids, LABELS].to_numpy(float)
        gold_auc = float(np.nanmean([auc((yg[:, i] > 0.5).astype(float), pg[:, i]) for i in range(12)]))
    summary = {"arch": a.arch, "layout": corpus.layout, "epochs": a.epochs, "epochs_done": len(hist), "stopped_early": stopped_early,
               "n_train": len(tr), "n_val": len(va), "n_gold": len(gold_ids), "k": a.k, "bs": a.bs,
               "bb_lr": a.bb_lr, "head_lr": a.head_lr, "wd": a.wd, "seed": a.seed,
               "primary": "last epoch", "secondary": "mean of last-3 epoch weights (fixed, not selected)",
               "gold_macro_report_only": gold_auc, "hist": hist, "bench": bench,
               "wall_min": (time.time() - t_start) / 60}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (a.out / "bench.json").write_text(json.dumps(bench, indent=2))
    log(f"BENCH {json.dumps(bench)}")
    log(f"gold-58 macro {gold_auc} -- REPORTED ONLY, never selected on (RUNBOOK §3 rule 1)")
    log(f"done in {summary['wall_min']:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
