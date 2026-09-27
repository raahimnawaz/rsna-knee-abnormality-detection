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
    # wiring only, random weights: Ortho's own architecture family (Eva, RoPE, registers) at 22M params,
    # so gradient checkpointing and DDP are exercised on the code path Ortho takes, on a CPU
    "smalleva": dict(res=224, pool="token", timm="vit_small_patch16_dinov3"),
    # A VLM reading the whole study as one sequence (fusion/qwen_vlm.py): same forward signature as
    # WindowMIL, so everything below is shared. qwentiny is its wiring check at a toy size.
    "qwen35vl": dict(res=384, pool="qwen", hf="Qwen/Qwen3.5-2B"),
    "qwentiny": dict(res=64, pool="qwen"),
}
UNTRAINED = {"tiny", "smalleva", "qwentiny"}
ORTHO_URL = "https://media.githubusercontent.com/media/ytrsk/OrthoFoundation/main/OrthoFoudation-L.pth"
ORTHO_BYTES = 1213056638


def log(msg: str) -> None:
    if os.environ.get("RANK", "0") == "0":                  # under torchrun, rank 0 speaks for all
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- corpus
class Corpus:
    """Row-ordered parts as one (N, D, 336, 336) uint8 view. ROW ORDER IS THE CONTRACT: within a
    part, ids[i] names vols[i] and masks[i]; nothing here sorts any of the three.

    Three layouts, found by what is attached:
      raptor44  dreaddevelopment's CC0 corpus, two parts, D = 44 = 12+10+8+6+8 (15-85%)
      stage_a   our `pipeline/build_slots64.py` shards, D = 64 = 18+14+12+8+12, span in the manifest.
      stage_b   `build_slots64.py --spec c7 --canon`: D = 96 over seven slots, 384 px, canonical.
                Every shard of one build must be present, or the run refuses.
    """

    RAPTOR_PARTS = (("all_vols.npy", "all_masks.npy", "all_ids.npy"),
                    ("extra_vols.npy", "extra_masks.npy", "extra_ids.npy"))
    SPECS = {"stage_a": "c6-64slice", "stage_b": "c7-96slice-384"}

    def __init__(self, root: Path, layout: str = "auto"):
        found, shards = {}, {k: [] for k in self.SPECS}
        for d, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if x not in ("train_series", "test_series", "train_images")]
            for f in fs:
                found.setdefault(f, Path(d) / f)
            if "manifest.json" in fs and "vols.npy" in fs:
                try:
                    j = json.loads((Path(d) / "manifest.json").read_text())
                except (OSError, ValueError):
                    continue
                for k, spec in self.SPECS.items():
                    if j.get("spec") == spec:
                        shards[k].append((int(j.get("shard", 0)), Path(d), j))
        if layout == "auto":
            have = [k for k in self.SPECS if shards[k]]
            if len(have) > 1:
                raise SystemExit(f"both {have} are attached -- pass --layout to say which")
            layout = have[0] if have else "raptor44"
        self.img = 336
        if layout in self.SPECS:
            c6 = shards[layout]
            if not c6:
                raise SystemExit(f"no {layout} shard (manifest spec {self.SPECS[layout]}) under {root}")
            c6.sort(key=lambda t: t[0])
            n_sh = {j["n_shards"] for _, _, j in c6}
            got = [s for s, _, _ in c6]
            if len(n_sh) != 1 or got != list(range(next(iter(n_sh)))):
                raise SystemExit(f"{layout} shards incomplete or mixed: have {got} of {n_sh}")
            if len({tuple(j["depth_range"]) for _, _, j in c6}) != 1:
                raise SystemExit(f"{layout} shards were built with different spans")
            if len({bool(j.get("canon")) for _, _, j in c6}) != 1:
                raise SystemExit(f"{layout} shards mix canonical and stored orientation")
            self.slot_sizes = tuple(s["n_slices"] for s in c6[0][2]["slots"])
            self.slot_meta = [dict(s) for s in c6[0][2]["slots"]]         # plane, fluid: the VLM's slot headers
            self.img = int(c6[0][2].get("img", 336))
            self.layout = f"{layout} span {c6[0][2]['depth_range']} img {self.img} canon {bool(c6[0][2].get('canon'))}"
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
            if not (len(vol) == len(msk) == len(idv)) or vol.shape[1:] != (depth, self.img, self.img):
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


def thin_windows(cs: list[int], slot_sizes, n: int) -> list[int]:
    """Evenly spaced within each slot, each slot keeping its share of `n` -- `window_curve.thin`'s rule,
    with the full count taken from the slot sizes instead of Stage A's 54 (identical on Stage A)."""
    full = sum(max(s - 2, 0) for s in slot_sizes)
    if not cs or n <= 0 or n >= full:
        return cs
    out, o = [], 0
    for size in slot_sizes:
        mine = [c for c in cs if o <= c < o + size]
        if mine:
            k = max(1, int(round(len(mine) * n / full)))
            idx = np.unique(np.round(np.linspace(0, len(mine) - 1, k)).astype(int))
            out += [mine[i] for i in idx]
        o += size
    return out


def window_meta(centres, slot_sizes) -> tuple[np.ndarray, np.ndarray]:
    """Each window's slot index and its relative depth inside that slot, 0 (first) .. 1 (last).
    Depth only means the same anatomy across studies on a CANONICAL cache (Stage B): on Stage A a
    sagittal slot runs lateral->medial on one knee and medial->lateral on the other."""
    starts = np.cumsum((0,) + tuple(slot_sizes))[:-1]
    c = np.asarray(centres)
    slot = np.searchsorted(starts, c, side="right") - 1
    n = np.asarray(slot_sizes)[slot]
    pos = np.clip((c - starts[slot] - 1) / np.maximum(n - 3, 1), 0.0, 1.0)
    return slot.astype(np.int64), pos.astype(np.float32)


class Windows(torch.utils.data.Dataset):
    """One study -> (uint8 windows [K,3,H,W], slot [K], depth [K], targets [12]). Normalisation on device."""

    def __init__(self, corpus, uids: list[str], y: np.ndarray, k: int | None, seed: int = 0, thin: int = 0):
        self.c, self.uids, self.y, self.k, self.seed, self.thin = corpus, uids, y, k, seed, thin
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        vol, mask = self.c.study(self.uids[i])
        cs = window_centres(mask, self.c.slot_sizes)
        if not cs:                                          # nothing valid: the stack's middle
            cs = [len(mask) // 2]
        elif self.k is None and self.thin:                  # eval, a fixed evenly spaced subset
            cs = thin_windows(cs, self.c.slot_sizes, self.thin)
        if self.k is not None:                              # train: k draws, no repeats if possible
            cs = self.rng.sample(cs, self.k) if len(cs) >= self.k else \
                [self.rng.choice(cs) for _ in range(self.k)]
        w = np.stack([vol[c - 1:c + 2] for c in cs])        # (K,3,H,W)
        slot, pos = window_meta(cs, self.c.slot_sizes)
        return (torch.from_numpy(np.ascontiguousarray(w)), torch.from_numpy(slot), torch.from_numpy(pos),
                torch.from_numpy(self.y[i]))


def seed_worker(_wid: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(hash((info.dataset.seed, info.id)))


def pad_collate(batch):
    """Eval batches hold studies with different window counts: pad, and carry the mask."""
    kmax = max(b[0].shape[0] for b in batch)
    x = torch.zeros((len(batch), kmax) + tuple(batch[0][0].shape[1:]), dtype=torch.uint8)
    m = torch.zeros((len(batch), kmax), dtype=torch.bool)
    sl = torch.zeros((len(batch), kmax), dtype=torch.int64)
    ps = torch.zeros((len(batch), kmax), dtype=torch.float32)
    for j, (w, s, p, _) in enumerate(batch):
        x[j, :len(w)], sl[j, :len(w)], ps[j, :len(w)] = w, s, p
        m[j, :len(w)] = True
    return x, m, sl, ps, torch.stack([b[3] for b in batch])


# ----------------------------------------------------------------------------- model
def build_encoder(arch: str, weights_dir: Path, res: int | None = None, drop_path: float = 0.0) -> nn.Module:
    import timm
    spec = dict(ARCHS[arch])
    spec["res"] = res or spec["res"]
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
        m, periods = om.build(dst, img=spec["res"], global_pool=spec["pool"],
                              **({"drop_path_rate": drop_path} if drop_path else {}))
        if not torch.equal(m.rope.periods.float(), periods.float()):
            raise SystemExit("RoPE periods were not applied -- the §9h trap")
        log(f"ortho: strict load, sha256 ok, RoPE periods = theirs | img {spec['res']} | drop_path {drop_path}")
        return m
    kw = dict(pretrained=arch not in UNTRAINED, num_classes=0, global_pool=spec["pool"])
    if drop_path:
        kw["drop_path_rate"] = drop_path
    if spec["pool"] == "token" and arch != "tiny":         # smalleva gets img_size too: RoPE at res
        kw["img_size"] = spec["res"]
    return timm.create_model(spec["timm"], **kw)


class WindowMIL(nn.Module):
    """Encoder + per-finding attention over windows. Input is uint8; the device does the rest."""

    def __init__(self, encoder: nn.Module, res: int, n: int = len(LABELS), drop: float = 0.2,
                 n_slots: int = 0):
        super().__init__()
        self.encoder, self.res, self.n_slots = encoder, res, n_slots
        f = encoder.num_features
        self.norm = nn.LayerNorm(f)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.w = nn.Parameter(torch.zeros(n, f))
        self.b = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.w, std=0.02)
        if n_slots:
            # Which slot a window came from and how deep inside it, added to its feature. Both start at
            # ZERO, so at initialisation this is exactly the model without them; training decides.
            self.slot_emb = nn.Embedding(n_slots, f)
            self.depth_proj = nn.Linear(8, f)
            for t in (self.slot_emb.weight, self.depth_proj.weight, self.depth_proj.bias):
                nn.init.zeros_(t)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def where(self, slot: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        ang = pos.float()[..., None] * (torch.arange(1, 5, device=pos.device, dtype=torch.float32) * torch.pi)
        return self.slot_emb(slot) + self.depth_proj(torch.cat([ang.sin(), ang.cos()], -1))

    def prep(self, x: torch.Tensor, gain: torch.Tensor | None) -> torch.Tensor:
        b, k = x.shape[:2]
        x = x.flatten(0, 1).float().div_(255.0)
        if gain is not None:                                # per-study intensity jitter, [0,1] space
            x = (x * gain.repeat_interleave(k).view(-1, 1, 1, 1)).clamp_(0, 1)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x, mask=None, gain=None, slot=None, pos=None):
        b, k = x.shape[:2]
        h = self.norm(self.encoder(self.prep(x, gain)).view(b, k, -1).float())
        if self.n_slots:
            if slot is None or pos is None:
                raise ValueError("this model was built with slot/depth embeddings: pass slot and pos")
            h = h + self.where(slot, pos)
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
    # `--compile` compiles the TRAINING step only. Eval feeds a new window count per study (42-62), each a new
    # shape to compile, and on 12 studies the 09-26 bench's eval ran 1.4-2.3x slower -- so eval runs eager.
    with torch.compiler.set_stance("force_eager"):
        for x, m, sl, ps, _ in dl:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out.append(torch.sigmoid(model(x.to(dev), m.to(dev), slot=sl.to(dev), pos=ps.to(dev)).float())
                           .cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)), np.float32)


def ds_workers() -> int:
    return int(os.environ.get("TW_WORKERS", "4"))


# ----------------------------------------------------------------------------- distributed
def dist_setup() -> tuple[int, int, int]:
    """(world, rank, local_rank): one process, unless launched by `torchrun --nproc_per_node N`."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 1, 0, 0
    import torch.distributed as dist
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return world, rank, local


def barrier(world: int) -> None:
    if world > 1:
        import torch.distributed as dist
        dist.barrier()


def all_any(flag: bool, world: int, dev) -> bool:
    """True on every rank if it is true on ANY -- so all ranks leave a loop on the same step."""
    if world == 1:
        return flag
    import torch.distributed as dist
    t = torch.tensor([1.0 if flag else 0.0], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


def predict_sharded(model, corpus, uids, dev, amp, bs, world, rank, thin: int = 0) -> np.ndarray:
    """Rank r scores uids[r::world]; every rank gets all of them back, in `uids` order."""
    mine = uids[rank::world]
    p = predict(model, Windows(corpus, mine, np.zeros((len(mine), 12), np.float32), k=None, thin=thin),
                dev, amp, bs)
    if world == 1:
        return p
    import torch.distributed as dist
    parts = [None] * world
    dist.all_gather_object(parts, (mine, p))
    got = {u: row for us, ps in parts for u, row in zip(us, ps)}
    return np.stack([got[u] for u in uids]) if uids else p


def param_groups(model: WindowMIL, bb_lr: float, head_lr: float, llrd: float) -> list[dict]:
    """The encoder at bb_lr and everything else at head_lr. With layer-wise decay (llrd < 1), block i
    of n runs at bb_lr * llrd^(n - i), the patch embedding and tokens at llrd^(n + 1) and the final
    norm at bb_lr -- early, general features move least."""
    enc = model.encoder
    head = [p for n_, p in model.named_parameters() if not n_.startswith("encoder.")]
    blocks = getattr(enc, "blocks", None)
    if not llrd or llrd >= 1.0 or blocks is None:
        return [{"params": list(enc.parameters()), "lr": bb_lr}, {"params": head, "lr": head_lr}]
    nb, groups = len(blocks), {}
    for name, p in enc.named_parameters():
        if name.startswith("blocks."):
            lid = int(name.split(".")[1]) + 1
        elif name.startswith(("norm", "fc_norm", "head")):
            lid = nb + 1
        else:
            lid = 0
        groups.setdefault(lid, {"params": [], "lr": bb_lr * llrd ** (nb + 1 - lid)})["params"].append(p)
    return [groups[k] for k in sorted(groups)] + [{"params": head, "lr": head_lr}]


def main() -> None:
    from contextlib import nullcontext
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=sorted(ARCHS), required=True)
    ap.add_argument("--corpus", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--layout", choices=["auto", "raptor44", "stage_a", "stage_b"], default="auto")
    ap.add_argument("--synthetic", type=int, default=0, help="N random studies instead of a corpus")
    ap.add_argument("--train", type=Path, help="csv with StudyInstanceUID")
    ap.add_argument("--val", type=Path, help="csv with StudyInstanceUID -- scored, never selected on")
    ap.add_argument("--gold", type=Path, help="train.csv; its fully-labelled rows are scored, never used")
    ap.add_argument("--labels", type=Path, help="targets.csv (StudyInstanceUID + 12 columns)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, default=Path("/tmp"))
    ap.add_argument("--epochs", type=int, default=10, help="with --fit-budget: the most it may run")
    ap.add_argument("--k", type=int, default=12, help="windows per study per training step")
    ap.add_argument("--bs", type=int, default=8, help="studies per optimiser step, over all GPUs")
    ap.add_argument("--micro", type=int, default=2, help="studies per forward per GPU")
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
    # Stage B's package (RUNBOOK); every one of these defaults to the behaviour before it existed
    ap.add_argument("--res", type=int, default=0, help="encoder input size; 0 = the arch's own (ortho 336)")
    ap.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth rate")
    ap.add_argument("--llrd", type=float, default=1.0, help="layer-wise LR decay per ViT block; 1 = off")
    ap.add_argument("--pos-emb", action="store_true", help="slot + depth embedding per window")
    ap.add_argument("--fit-budget", action="store_true",
                    help="time real training steps first, then cut --epochs to what fits the budget")
    ap.add_argument("--eval-reserve-h", type=float, default=0.0, help="with --fit-budget: hours kept for eval")
    ap.add_argument("--lora-r", type=int, default=16, help="qwen archs: LoRA rank on the language model")
    ap.add_argument("--lora-lr", type=float, default=1e-4, help="qwen archs: LoRA learning rate")
    ap.add_argument("--ckpt-blocks", type=int, default=-1,
                    help="gradient-checkpoint only the first N encoder blocks; -1 = all (the default), 0 = none")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the encoder in place, for training steps only (eval runs eager)")
    ap.add_argument("--eval-windows", type=int, default=0,
                    help="score an evenly spaced ~N windows per study (thin_windows); 0 = every window")
    a = ap.parse_args()

    world, rank, local = dist_setup()
    if a.bs % (a.micro * world):
        raise SystemExit(f"--bs {a.bs} must be a multiple of --micro {a.micro} x {world} GPU(s)")
    t_start = time.time()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    amp = dev.startswith("cuda")
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=True)
    log(f"arch {a.arch} {ARCHS[a.arch]} | device {dev}"
        + (f" {torch.cuda.get_device_name(local)}" if amp else "") + f" | amp fp16 {amp} | world {world}")

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
    nwin = [len(thin_windows(window_centres(corpus.study(u)[1], corpus.slot_sizes), corpus.slot_sizes,
                             a.eval_windows)) for u in va]
    log(f"val windows per study: median {int(np.median(nwin))}, min {min(nwin)}, max {max(nwin)}"
        + (f" (thinned to ~{a.eval_windows})" if a.eval_windows else ""))
    if a.pos_emb and not str(getattr(corpus, "layout", "")).startswith("stage_b"):
        log("⚠️ --pos-emb on a non-canonical corpus: a slot's depth means lateral on one knee, medial on another")

    # ---- model. Rank 0 fetches the encoder weights first; the others read the same file after it.
    res = a.res or ARCHS[a.arch]["res"]
    is_qwen = ARCHS[a.arch]["pool"] == "qwen"
    if is_qwen:
        import qwen_vlm

        def make():
            return qwen_vlm.build(a.arch, a.weights_dir, res, getattr(corpus, "slot_meta", None),
                                  len(corpus.slot_sizes), lora_r=a.lora_r, fp16_frozen=amp,
                                  grad_ckpt=bool(amp or os.environ.get("TW_FORCE_CKPT")),
                                  tokenizer_dir=os.environ.get("TW_QWEN_TOKENIZER"))
        # Each rank moves its copy to its device BEFORE the next rank loads: two host copies at once is how
        # the 09-26 CPU smoke was killed.
        model = make().to(dev) if rank == 0 else None
        barrier(world)
        if rank != 0:
            model = make().to(dev)
        log(f"qwen: {model.n_lora} LoRA linears r {a.lora_r}, lr {a.lora_lr} | {model.tokens_per_image} tokens "
            f"per window | slot headers {model.slot_names} | frozen LM fp16 {amp}")
    else:
        enc = build_encoder(a.arch, a.weights_dir, res, a.drop_path) if rank == 0 else None
        barrier(world)
        if rank != 0:
            enc = build_encoder(a.arch, a.weights_dir, res, a.drop_path)
        model = WindowMIL(enc, res, n_slots=len(corpus.slot_sizes) if a.pos_emb else 0).to(dev)
    if not is_qwen and hasattr(model.encoder, "set_grad_checkpointing") and (amp or os.environ.get("TW_FORCE_CKPT")):
        # NON-reentrant checkpointing: re-entrant checkpoints and DDP's gradient hooks do not mix
        # (DDP then needs static_graph, which cannot start under no_sync -- measured locally 09-25).
        import timm.layers as _tl
        if hasattr(_tl, "set_reentrant_ckpt"):
            _tl.set_reentrant_ckpt(False)
        elif world > 1:
            raise SystemExit("this timm cannot switch off re-entrant checkpointing; DDP would break")
        if a.ckpt_blocks < 0:
            model.encoder.set_grad_checkpointing(True)
        else:
            # Checkpoint only the first N blocks: peak memory was 6.9 / 8.5 GB of 15 with all 24 (AUDIT 09-26
            # pass 4), and every checkpointed block costs its forward twice. The forward is overridden per
            # INSTANCE, so state_dict keys -- and last.pt -- are unchanged.
            import torch.utils.checkpoint as _cp
            model.encoder.set_grad_checkpointing(False)
            for blk in list(model.encoder.blocks)[: a.ckpt_blocks]:
                blk.forward = (lambda x, _f=blk.forward, **kw: _cp.checkpoint(_f, x, use_reentrant=False, **kw))
        log(f"gradient checkpointing: {'all' if a.ckpt_blocks < 0 else a.ckpt_blocks} of "
            f"{len(model.encoder.blocks)} encoder blocks")
    if a.compile and not is_qwen:
        model.encoder.compile()          # in place: keys stay `encoder.*`, unlike torch.compile(module)
        log("encoder compiled (torch.compile, in place)")
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable {n_trn / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    log(f"encoder {n_enc / 1e6:.1f}M params, num_features {model.encoder.num_features} | res {res} | "
        f"drop_path {a.drop_path} | llrd {a.llrd} | pos_emb {a.pos_emb}")

    prev = np.clip(ytr.mean(0), 0.03, 0.7)
    pos_w = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    groups = (model.param_groups(a.bb_lr, a.head_lr, a.lora_lr) if is_qwen
              else param_groups(model, a.bb_lr, a.head_lr, a.llrd))
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    if a.llrd < 1.0:
        log(f"layer-wise LR: {len(groups) - 1} encoder groups, {groups[0]['lr']:.2e} (embeddings) .. "
            f"{groups[-2]['lr']:.2e} (top) | head {groups[-1]['lr']:.2e}")
    tr_ds = Windows(corpus, tr, ytr, k=a.k, seed=a.seed + 1000 * rank)
    sampler = (torch.utils.data.distributed.DistributedSampler(tr_ds, num_replicas=world, rank=rank,
                                                                shuffle=True, seed=a.seed, drop_last=True)
               if world > 1 else None)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=a.micro, shuffle=sampler is None, sampler=sampler,
                                        drop_last=True, num_workers=ds_workers(), worker_init_fn=seed_worker,
                                        persistent_workers=ds_workers() > 0)
    if world > 1:
        # gradient_as_bucket_view: the gradients live inside DDP's all-reduce buckets instead of beside them,
        # one fewer full copy per rank (~1.2 GB for ViT-L). The 09-25 CPU smoke was OOM-killed at step 2.
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if amp else None,
                                                        gradient_as_bucket_view=True)
    else:
        net = model
    accum = a.bs // (a.micro * world)
    steps_per_epoch = len(tr_dl) // accum
    if a.max_steps:
        steps_per_epoch = min(steps_per_epoch, a.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def micro_step(x, sl, ps, y, sync: bool):
        gain = 1.0 + (torch.rand(len(x), device=dev) - 0.5) * 0.2
        with (nullcontext() if sync or world == 1 else net.no_sync()):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = net(x, gain=gain, slot=sl, pos=ps)
            w = 0.25 + 0.75 * (2 * y - 1).abs()
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_w,
                                                       reduction="none") * w).mean()
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            scaler.scale(loss / accum).backward()
        return loss

    def to_dev(batch):
        return [t.to(dev, non_blocking=True) for t in batch]

    # ---- how many epochs fit: time real micro-steps (data loading included), never extrapolate
    epochs, probe = a.epochs, None
    if a.fit_budget:
        net.train()
        if sampler is not None:
            sampler.set_epoch(0)
        it, ts = iter(tr_dl), []
        for j in range(8):
            t0 = time.time()
            x, sl, ps, y = to_dev(next(it))
            micro_step(x, sl, ps, y, sync=False)
            if amp:
                torch.cuda.synchronize()
            ts.append(time.time() - t0)
        opt.zero_grad(set_to_none=True)
        del it
        probe = float(np.median(ts[2:]))
        # x1.35, not the +10% first written: Stage B's probe read 2.79 s per micro-step and the run measured 3.52
        # (+26%: the probe's 8 steps miss the optimiser, the all-reduce and loader contention). It planned 13
        # epochs and stopped at 11.2 -- the schedule cut at ~86% (RUNBOOK, Stage B, 09-26).
        epoch_s = probe * steps_per_epoch * accum * 1.35
        avail = a.time_budget_h * 3600 - (time.time() - t_start) - a.eval_reserve_h * 3600
        fit = int(max(1, min(a.epochs, avail // epoch_s)))
        if world > 1:
            import torch.distributed as dist
            t = torch.tensor([fit], device=dev)
            dist.broadcast(t, 0)
            fit = int(t.item())
        epochs = fit
        log(f"fit-budget: {probe:.2f} s per {a.micro}-study micro-step -> {epoch_s / 60:.1f} min/epoch; "
            f"{avail / 3600:.2f} h left after {a.eval_reserve_h} h eval reserve -> {epochs} epoch(s) (cap {a.epochs})")
    total = max(steps_per_epoch * epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                total_steps=total, pct_start=0.15)
    log(f"steps/epoch {steps_per_epoch} (bs {a.bs} = {a.micro} x {world} GPU x {accum}) | epochs {epochs} | "
        f"k {a.k} | lr {a.bb_lr}/{a.head_lr}")

    # ---- train
    hist, last3, step_times, recent = [], [], [], []
    waits, loss_trace, t_end = [], [], None       # data wait between micro-steps; the first losses, for A/B equality
    stopped_early, gnorm = None, float("nan")
    for ep in range(epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        net.train()
        t_ep, tot, n_opt, stop = time.time(), 0.0, 0, False
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(tr_dl):
            if n_opt >= steps_per_epoch:
                break
            t0 = time.time()
            if t_end is not None:
                waits.append(t0 - t_end)
            x, sl, ps, y = to_dev(batch)
            last_micro = (it + 1) % accum == 0
            loss = micro_step(x, sl, ps, y, sync=last_micro)
            tot += float(loss.detach())
            if len(loss_trace) < 60:
                loss_trace.append(round(float(loss.detach()), 5))
            recent = (recent + [float(loss.detach())])[-50:]
            if last_micro:
                scaler.unscale_(opt)
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                n_opt += 1
                stop = all_any(time.time() - t_start > a.time_budget_h * 3600, world, dev)
            if amp:
                torch.cuda.synchronize()
            step_times.append(time.time() - t0)
            t_end = time.time()
            if it % 50 == 0:
                log(f"  ep{ep} it{it} loss {float(loss.detach()):.4f} (mean of last {len(recent)}: "
                    f"{np.mean(recent):.4f}) | grad-norm {gnorm:.3f} | scale {scaler.get_scale() if amp else 1:.0f} | "
                    f"{np.mean(step_times[-50:]):.2f} s per {a.micro}-study micro-step")
            if stop:
                stopped_early = f"time budget at epoch {ep} iter {it}"
                break
        hist.append({"epoch": ep, "loss": tot / max(it + 1, 1), "sec": time.time() - t_ep,
                     "opt_steps": n_opt})
        log(f"epoch {ep}: loss {hist[-1]['loss']:.4f}  {hist[-1]['sec'] / 60:.1f} min")
        if ep >= epochs - 3:
            keep = ({n_ for n_, p in model.named_parameters() if p.requires_grad} if is_qwen else None)
            last3.append({k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                          if keep is None or k in keep})
        if stopped_early:
            log(f"⛔ stopped early: {stopped_early} -- the schedule did not finish")
            break

    if world > 1:
        # DDP's invariant, checked rather than assumed: every rank must end with the SAME weights. A
        # mis-synchronised accumulation step would let them drift, and only rank 0's copy is saved.
        import torch.distributed as dist
        # Only trainable weights can drift between ranks. Summing the VLM's frozen 2B in fp64 would allocate
        # ~4 GB for the embedding alone, after a full session of training (AUDIT.md, 09-26).
        chk = torch.tensor([sum(float(p.detach().double().sum()) for p in model.parameters() if p.requires_grad)],
                           device=dev, dtype=torch.float64)
        both = [torch.zeros_like(chk) for _ in range(world)]
        dist.all_gather(both, chk)
        vals = [float(t.item()) for t in both]
        log(f"DDP weight checksum per rank: {vals}")
        if max(vals) - min(vals) > 1e-6 * max(1.0, abs(vals[0])):
            raise SystemExit(f"ranks ended with different weights: {vals}")
    peak = torch.cuda.max_memory_allocated(dev) / 1e9 if amp else None
    per_fwd = float(np.median(step_times)) if step_times else None
    bench = {"arch": a.arch, "device": torch.cuda.get_device_name(local) if amp else "cpu", "world": world,
             "micro_studies": a.micro, "k": a.k, "res": res, "eval_windows": a.eval_windows,
             "sec_per_micro_step_median": per_fwd,
             "sec_per_window_train": per_fwd / (a.micro * a.k) if per_fwd else None,
             "train_epoch_min_measured": hist[0]["sec"] / 60 if hist else None,
             "steps_in_measured_epoch": hist[0]["opt_steps"] if hist else None,
             "full_epoch_min_projected": (per_fwd * len(tr_dl) / 60) if per_fwd else None,
             "fit_budget_probe_sec": probe, "epochs_run": len(hist), "peak_cuda_gb": peak,
             "data_wait_sec_median": float(np.median(waits)) if waits else None,
             "data_wait_frac": float(np.sum(waits) / max(np.sum(waits) + np.sum(step_times), 1e-9)) if waits else None,
             "ckpt_blocks": a.ckpt_blocks, "compile": a.compile, "loss_trace": loss_trace}

    # ---- evaluate: last epoch (primary), then the fixed last-3 average (secondary). Sharded over GPUs.
    t_ev = time.time()
    p_last = predict_sharded(model, corpus, va, dev, amp, a.eval_bs, world, rank, a.eval_windows)
    bench["eval_sec_per_study"] = (time.time() - t_ev) * world / max(len(va), 1)     # per study, per GPU
    if rank == 0:
        pd.DataFrame(p_last, columns=LABELS).assign(StudyInstanceUID=va).to_csv(a.out / "oof_all.csv", index=False)
    pg = None
    if gold_ids:
        pg = predict_sharded(model, corpus, gold_ids, dev, amp, a.eval_bs, world, rank, a.eval_windows)
        if rank == 0:
            pd.DataFrame(pg, columns=LABELS).assign(StudyInstanceUID=gold_ids).to_csv(a.out / "oof_gold.csv", index=False)
    if a.save_weights and rank == 0:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   a.out / "last.pt")
        # ⛔ What the state dict does NOT carry. timm holds `rope.periods` as a NON-PERSISTENT buffer,
        # so an inference kernel that rebuilds the encoder and loads last.pt gets timm's periods, not
        # the ones this model trained with (§9h). Written here so nothing downstream has to know that --
        # and, since Stage B, how to rebuild the model at all: its size, its slots, its embeddings.
        rope = getattr(model.encoder, "rope", None)
        (a.out / "last_meta.json").write_text(json.dumps({
            "arch": a.arch, "res": res, "pool": ARCHS[a.arch]["pool"],
            "timm": ARCHS[a.arch].get("timm", "vit_large_patch16_dinov3"),
            "layout": corpus.layout, "slot_sizes": list(corpus.slot_sizes), "img": getattr(corpus, "img", 336),
            "pos_emb": bool(a.pos_emb), "n_slots": model.n_slots, "llrd": a.llrd, "drop_path": a.drop_path,
            "rope_periods": rope.periods.float().tolist() if rope is not None and hasattr(rope, "periods") else None,
            "mean": MEAN, "std": STD,
            **({"hf": ARCHS[a.arch].get("hf"), "qwen_config": model.qwen.config.to_dict(), "qwen_ids": model.ids,
                "lora_r": a.lora_r, "slot_headers": model.slot_names, "mean": list(qwen_vlm.QWEN_MEAN),
                "std": list(qwen_vlm.QWEN_STD)} if is_qwen else {})}, indent=1, default=str))
    if len(last3) >= 2:
        avg = {k: (sum(sd[k].double() for sd in last3) / len(last3)).to(last3[0][k].dtype)
               if last3[0][k].is_floating_point() else last3[0][k] for k in last3[0]}
        missing = model.load_state_dict(avg, strict=not is_qwen)
        if is_qwen and missing.unexpected_keys:
            raise SystemExit(f"avg3: unexpected keys {missing.unexpected_keys[:3]}")
        pa = predict_sharded(model, corpus, va, dev, amp, a.eval_bs, world, rank, a.eval_windows)
        if rank == 0:
            d3 = a.out / "avg3"
            d3.mkdir(exist_ok=True)
            pd.DataFrame(pa, columns=LABELS).assign(StudyInstanceUID=va).to_csv(d3 / "oof_all.csv", index=False)

    if rank == 0:
        gold_auc = None
        if gold_ids:
            from metrics import auc
            yg = gtab.loc[gold_ids, LABELS].to_numpy(float)
            gold_auc = float(np.nanmean([auc((yg[:, i] > 0.5).astype(float), pg[:, i]) for i in range(12)]))
        summary = {"arch": a.arch, "layout": corpus.layout, "epochs": epochs, "epochs_cap": a.epochs,
                   "epochs_done": len(hist), "stopped_early": stopped_early, "world": world,
                   "n_train": len(tr), "n_val": len(va), "n_gold": len(gold_ids), "k": a.k, "bs": a.bs,
                   "bb_lr": a.bb_lr, "head_lr": a.head_lr, "wd": a.wd, "seed": a.seed, "res": res,
                   "drop_path": a.drop_path, "llrd": a.llrd, "pos_emb": a.pos_emb, "eval_windows": a.eval_windows,
                   "labels_sha256": __import__("hashlib").sha256(Path(a.labels).read_bytes()).hexdigest()
                   if a.labels else None,
                   "primary": "last epoch", "secondary": "mean of last-3 epoch weights (fixed, not selected)",
                   "gold_macro_report_only": gold_auc, "hist": hist, "bench": bench,
                   "wall_min": (time.time() - t_start) / 60}
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
        (a.out / "bench.json").write_text(json.dumps(bench, indent=2))
        log(f"BENCH {json.dumps(bench)}")
        log(f"gold-58 macro {gold_auc} -- REPORTED ONLY, never selected on (RUNBOOK §3 rule 1)")
        log(f"done in {summary['wall_min']:.1f} min -> {a.out}")
    if world > 1:
        import torch.distributed as dist
        barrier(world)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
