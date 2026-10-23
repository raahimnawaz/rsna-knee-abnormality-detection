r"""Train one encoder on 2.5D windows of a fixed slice stack, with a per-finding attention-MIL head.

    python fusion/train_windows.py --arch convnext_small --corpus <cache> --train train.csv --val val.csv \
        --labels targets.csv --out runs --epochs 16
    python fusion/train_windows.py --arch tiny --synthetic 24 --out /tmp/tw --epochs 1    # wiring check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOT_SIZES = (12, 10, 8, 6, 8)
WEAK4 = ("Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ARCHS = {
    "coatnet": dict(res=384, pool="avg", timm="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"),
    "resnet34": dict(res=224, pool="avg", timm="resnet34.a1_in1k", ckpt=False),
    "convnext_tiny": dict(res=384, pool="avg", timm="convnext_tiny.in12k_ft_in1k_384", ckpt=False),
    "convnext_small": dict(res=384, pool="avg", timm="convnext_small.in12k_ft_in1k_384", ckpt=False),
    "tiny":    dict(res=224, pool="token", timm="vit_tiny_patch16_224"),
    "cnxatto": dict(res=224, pool="avg", timm="convnext_atto", ckpt=False),
}
UNTRAINED = {"tiny", "cnxatto"}


MIRROR_PAIRS = (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))
MIRROR_PERM = list(range(len(LABELS)))
for _a, _b in MIRROR_PAIRS:
    MIRROR_PERM[LABELS.index(_a)], MIRROR_PERM[LABELS.index(_b)] = LABELS.index(_b), LABELS.index(_a)
MIRROR_UNKNOWN = LABELS.index("MCL")


def log(msg: str) -> None:
    if os.environ.get("RANK", "0") == "0":
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slot_is_sagittal(corpus) -> torch.Tensor:
    """One bool per slot, for `mirror_windows`."""
    meta = getattr(corpus, "slot_meta", None)
    if not meta or "canon True" not in str(getattr(corpus, "layout", "")):
        raise SystemExit(f"mirroring needs a canonical cache with slot planes (Stage B / C); this is {getattr(corpus, 'layout', '?')}")
    planes = [m.get("plane") for m in meta]
    if not set(planes) <= {"Sagittal", "Coronal", "Axial"}:
        raise SystemExit(f"mirroring: unknown slot planes {planes}")
    return torch.tensor([p == "Sagittal" for p in planes])


def mirror_windows(w: torch.Tensor, slot: torch.Tensor, pos: torch.Tensor,
                   sagittal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Windows (..., K, 3, H, W) of a CANONICAL stack -> the same knee mirrored medial <-> lateral, and its depths."""
    sag = sagittal.to(slot.device)[slot]
    return (torch.where(sag[..., None, None, None], w.flip(-3), w.flip(-1)),
            torch.where(sag, 1.0 - pos, pos))


def mirror_targets(y: np.ndarray) -> np.ndarray:
    """Targets (..., 12) of the mirrored knee: the medial / lateral pairs swapped, MCL unknown (NaN)."""
    m = np.array(y[..., MIRROR_PERM], dtype=np.float32, copy=True)
    m[..., MIRROR_UNKNOWN] = np.nan
    return m


def mirror_average(plain, mirrored):
    """Probabilities (..., 12) of the plain and the mirrored pass -> one prediction: the mirrored pass read back into
    the plain knee's columns and averaged in; MCL from the plain pass alone."""
    out = 0.5 * (plain + mirrored[..., MIRROR_PERM])
    out[..., MIRROR_UNKNOWN] = plain[..., MIRROR_UNKNOWN]
    return out


class Corpus:
    """Row-ordered parts as one (N, D, 336, 336) uint8 view."""

    RAPTOR_PARTS = (("all_vols.npy", "all_masks.npy", "all_ids.npy"),
                    ("extra_vols.npy", "extra_masks.npy", "extra_ids.npy"))
    SPECS = {"stage_a": "c6-64slice", "stage_b": "c7-96slice-384", "stage_k": "k4-32slice-256"}

    def __init__(self, root: Path, layout: str = "auto", extra: tuple = ()):
        """`extra`: more roots of the SAME layout (e.g. an external corpus beside the Stage A shards)."""
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
            self.slot_meta = [dict(s) for s in c6[0][2]["slots"]]
            self.img = int(c6[0][2].get("img", 336))
            self.layout = f"{layout} span {c6[0][2]['depth_range']} img {self.img} canon {bool(c6[0][2].get('canon'))}"
            parts = [(d / "vols.npy", d / "masks.npy", d / "ids.npy") for _, d, _ in c6]
            for r in extra:
                sub = Corpus(Path(r), layout)
                if (sub.layout, sub.slot_sizes) != (self.layout, self.slot_sizes):
                    raise SystemExit(f"--extra-corpus {r}: {sub.layout} {sub.slot_sizes} does not match "
                                     f"{self.layout} {self.slot_sizes}")
                parts += sub.parts
        else:
            if extra:
                raise SystemExit("--extra-corpus needs a stage_a / stage_b layout")
            self.slot_sizes = SLOT_SIZES
            self.layout = "raptor44"
            parts = []
            for v, m, i in self.RAPTOR_PARTS:
                if v not in found:
                    raise SystemExit(f"corpus part {v} not found under {root} -- attach "
                                     f"dreaddevelopment/knee-raptor-corpus and knee-raptor-corpus-ext")
                parts.append((found[v], found[m], found[i]))
        self.parts = parts
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
        self.slot_sizes, self.layout = SLOT_SIZES, "synthetic raptor44 canon True"
        self.slot_meta = [{"name": f"s{i}", "plane": p} for i, p in
                          enumerate(("Sagittal", "Sagittal", "Coronal", "Coronal", "Axial"))]
        self.ids = np.asarray(uids)
        self.row = {u: k for k, u in enumerate(uids)}
        self.vol = rng.integers(0, 255, (len(uids), 44, 336, 336), dtype=np.uint8)
        self.masks = np.ones((len(uids), 44), np.uint8)
        self.masks[::3, 34:36] = 0
        self.masks[1::4, 36:44] = 0

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
    """Evenly spaced within each slot, each slot keeping its share of `n`, the full count taken from the slot sizes."""
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
    """Each window's slot index and its relative depth inside that slot (0 = first, 1 = last)."""
    starts = np.cumsum((0,) + tuple(slot_sizes))[:-1]
    c = np.asarray(centres)
    slot = np.searchsorted(starts, c, side="right") - 1
    n = np.asarray(slot_sizes)[slot]
    pos = np.clip((c - starts[slot] - 1) / np.maximum(n - 3, 1), 0.0, 1.0)
    return slot.astype(np.int64), pos.astype(np.float32)


class Windows(torch.utils.data.Dataset):
    """One study -> (uint8 windows [K,3,H,W], slot [K], depth [K], targets [12])."""

    def __init__(self, corpus, uids: list[str], y: np.ndarray, k: int | None, seed: int = 0, thin: int = 0,
                 mirror: float = 0.0):
        self.c, self.uids, self.y, self.k, self.seed, self.thin = corpus, uids, y, k, seed, thin
        self.rng = random.Random(seed)
        self.mirror = mirror
        self.sagittal = slot_is_sagittal(corpus) if mirror else None

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        vol, mask = self.c.study(self.uids[i])
        cs = window_centres(mask, self.c.slot_sizes)
        if not cs:
            cs = [len(mask) // 2]
        keep = getattr(self.c, "slot_keep", None)
        if keep is not None:
            s_of, _ = window_meta(cs, self.c.slot_sizes)
            k0 = min(keep)
            mid = int(np.cumsum((0,) + tuple(self.c.slot_sizes))[k0] + self.c.slot_sizes[k0] // 2)
            cs = [c for c, s in zip(cs, s_of) if s in keep] or [mid]
        elif self.k is None and self.thin:
            cs = thin_windows(cs, self.c.slot_sizes, self.thin)
        if self.k is not None:
            cs = self.rng.sample(cs, self.k) if len(cs) >= self.k else \
                [self.rng.choice(cs) for _ in range(self.k)]
        w = np.stack([vol[c - 1:c + 2] for c in cs])
        slot, pos = window_meta(cs, self.c.slot_sizes)
        w, slot, pos, y = torch.from_numpy(np.ascontiguousarray(w)), torch.from_numpy(slot), torch.from_numpy(pos), self.y[i]
        if self.mirror and self.rng.random() < self.mirror:
            w, pos = mirror_windows(w, slot, pos, self.sagittal)
            y = mirror_targets(y)
        return w, slot, pos, torch.from_numpy(y)


def with_gold(tr: list[str], ytr: np.ndarray, g_ids: list[str], yg: np.ndarray, repeats: int):
    """The training list with each graded study appended `repeats` times; the sampler shuffles them in."""
    return tr + [u for u in g_ids for _ in range(repeats)], np.concatenate([ytr, np.repeat(yg, repeats, axis=0)])


def soup(init: dict, trained: dict, w: float) -> dict:
    """w * init + (1 - w) * trained, tensor by tensor; integer buffers keep the trained values."""
    return {k: (w * init[k].to(v.device, torch.float64) + (1 - w) * v.double()).to(v.dtype)
            if v.is_floating_point() else v for k, v in trained.items()}


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


def build_encoder(arch: str, weights_dir: Path, res: int | None = None, drop_path: float = 0.0) -> nn.Module:
    import timm
    spec = dict(ARCHS[arch])
    spec["res"] = res or spec["res"]
    kw = dict(pretrained=arch not in UNTRAINED, num_classes=0, global_pool=spec["pool"])
    if drop_path:
        kw["drop_path_rate"] = drop_path
    return timm.create_model(spec["timm"], **kw)


class WindowMIL(nn.Module):
    """Encoder + per-finding attention over windows."""

    def __init__(self, encoder: nn.Module, res: int, n: int = len(LABELS), drop: float = 0.2,
                 n_slots: int = 0, spatial: bool = False):
        super().__init__()
        self.encoder, self.res, self.n_slots, self.spatial = encoder, res, n_slots, spatial
        self.affine = 0.0
        self.zoom = 0.0
        self.enc_chunk = 0
        self.channels_last = False
        f = encoder.num_features
        self.norm = nn.LayerNorm(f)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.w = nn.Parameter(torch.zeros(n, f))
        self.b = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.w, std=0.02)
        if n_slots:
            self.slot_emb = nn.Embedding(n_slots, f)
            self.depth_proj = nn.Linear(8, f)
            for t in (self.slot_emb.weight, self.depth_proj.weight, self.depth_proj.bias):
                nn.init.zeros_(t)
        if spatial:
            if not (hasattr(encoder, "forward_features") and hasattr(getattr(encoder, "head", None), "norm")):
                raise SystemExit("--spatial-attn needs a conv encoder with a pooled head (convnext_*)")
            self.sq = nn.Parameter(torch.zeros(n, f))
            self.sq_norm = nn.LayerNorm(f)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def where(self, slot: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        ang = pos.float()[..., None] * (torch.arange(1, 5, device=pos.device, dtype=torch.float32) * torch.pi)
        return self.slot_emb(slot) + self.depth_proj(torch.cat([ang.sin(), ang.cos()], -1))

    def prep(self, x: torch.Tensor, gain: torch.Tensor | None) -> torch.Tensor:
        b, k = x.shape[:2]
        x = x.flatten(0, 1)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x = x.float().div_(255.0)
        if gain is not None:
            x = (x * gain.repeat_interleave(k).view(-1, 1, 1, 1)).clamp_(0, 1)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        if self.affine and self.training and gain is not None:
            n, dev, s = x.shape[0], x.device, self.affine
            u = lambda: torch.rand(n, device=dev) * 2 - 1                        # noqa: E731
            th, sc, tx, ty = u() * s * math.pi / 18, 1 + u() * 0.1 * s, u() * 0.05 * s, u() * 0.05 * s
            c, si = torch.cos(th) / sc, torch.sin(th) / sc
            m = torch.stack([torch.stack([c, -si, tx], 1), torch.stack([si, c, ty], 1)], 1)
            x = F.grid_sample(x, F.affine_grid(m, list(x.shape), align_corners=False), mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        return (x - self.mean) / self.std

    def encode(self, xp: torch.Tensor) -> torch.Tensor:
        """Windows -> one feature each (N, F); with spatial attention, one per finding (N, n, F)."""
        if not self.spatial:
            return self.encoder(xp)
        t = self.encoder.forward_features(xp).flatten(2).transpose(1, 2).float()
        a = torch.softmax(self.sq_norm(t) @ self.sq.t() / t.shape[-1] ** 0.5, dim=1)
        p = torch.einsum("npc,npf->ncf", a, t)
        return self.encoder.head.norm(p.reshape(-1, p.shape[-1], 1, 1)).flatten(1).view(p.shape)

    def forward(self, x, mask=None, gain=None, slot=None, pos=None):
        if self.zoom:
            h_ = x.shape[-1]
            c = int(round(h_ * self.zoom))
            o = (h_ - c) // 2
            z = F.interpolate(x[..., o:o + c, o:o + c].flatten(0, 1).float(), size=(h_, h_), mode="bilinear",
                              align_corners=False).view(*x.shape[:2], x.shape[2], h_, h_)
            x = torch.cat([x.float(), z], 1)
            mask = None if mask is None else torch.cat([mask, mask], 1)
            slot = None if slot is None else torch.cat([slot, slot], 1)
            pos = None if pos is None else torch.cat([pos, pos], 1)
        b, k = x.shape[:2]
        xp = self.prep(x, gain)
        if self.enc_chunk and not self.training:
            f = torch.cat([self.encode(c) for c in xp.split(self.enc_chunk)])
        else:
            f = self.encode(xp)
        h = self.norm(f.view(b, k, *f.shape[1:]).float())
        if self.n_slots:
            if slot is None or pos is None:
                raise ValueError("this model was built with slot/depth embeddings: pass slot and pos")
            w = self.where(slot, pos)
            h = h + (w[:, :, None] if self.spatial else w)
        a = self.att(h)
        if self.spatial:
            a = torch.diagonal(a, dim1=2, dim2=3)
        if mask is not None:
            a = a.masked_fill(~mask[..., None], float("-inf"))
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bknf->bnf" if self.spatial else "bkn,bkf->bnf", a, h)
        return (pooled * self.w).sum(-1) + self.b


def load_split(path: Path) -> list[str]:
    return pd.read_csv(path).StudyInstanceUID.astype(str).drop_duplicates().tolist()


def autocast(dev, amp: bool):
    """fp16 on CUDA (a T4 has no bf16), bf16 on Apple's MPS (no GradScaler needed), nothing on CPU."""
    if amp:
        return torch.autocast("cuda", dtype=torch.float16)
    if str(dev).startswith("mps"):
        return torch.autocast("mps", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def predict(model, ds, dev, amp, bs, sagittal: torch.Tensor | None = None) -> np.ndarray:
    """(N, 12) probabilities; with `sagittal` (mirror averaging) (N, 2, 12): the plain pass, then the mirrored pass in
    the MIRRORED knee's columns -- `mirror_average` makes one prediction of the two."""
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=ds_workers(),
                                     collate_fn=pad_collate, pin_memory=str(dev).startswith("cuda"))
    out = []
    with torch.compiler.set_stance("force_eager"):
        for x, m, sl, ps, _ in dl:
            x, m, sl, ps = (t.to(dev, non_blocking=True) for t in (x, m, sl, ps))
            with autocast(dev, amp):
                p = torch.sigmoid(model(x, m, slot=sl, pos=ps).float())
                if sagittal is not None:
                    xm, pm = mirror_windows(x, sl, ps, sagittal)
                    p = torch.stack([p, torch.sigmoid(model(xm, m, slot=sl, pos=pm).float())], 1)
            out.append(p.cpu().numpy())
    shape = (0, len(LABELS)) if sagittal is None else (0, 2, len(LABELS))
    return np.concatenate(out) if out else np.zeros(shape, np.float32)


def ds_workers() -> int:
    return int(os.environ.get("TW_WORKERS", "4"))


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


def predict_sharded(model, corpus, uids, dev, amp, bs, world, rank, thin: int = 0,
                    sagittal: torch.Tensor | None = None) -> np.ndarray:
    """Rank r scores uids[r::world]; every rank gets all of them back, in `uids` order."""
    mine = uids[rank::world]
    p = predict(model, Windows(corpus, mine, np.zeros((len(mine), 12), np.float32), k=None, thin=thin),
                dev, amp, bs, sagittal)
    if world == 1:
        return p
    import torch.distributed as dist
    parts = [None] * world
    dist.all_gather_object(parts, (mine, p))
    got = {u: row for us, ps in parts for u, row in zip(us, ps)}
    return np.stack([got[u] for u in uids]) if uids else p


def param_groups(model: WindowMIL, bb_lr: float, head_lr: float, llrd: float) -> list[dict]:
    """The encoder at bb_lr and everything else at head_lr."""
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=sorted(ARCHS), required=True)
    ap.add_argument("--corpus", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--layout", choices=["auto", "raptor44", "stage_a", "stage_b", "stage_k"], default="auto")
    ap.add_argument("--synthetic", type=int, default=0, help="N random studies instead of a corpus")
    ap.add_argument("--train", type=Path, help="csv with StudyInstanceUID")
    ap.add_argument("--val", type=Path, help="csv with StudyInstanceUID -- scored, never selected on")
    ap.add_argument("--gold", type=Path, help="train.csv; its fully-labelled rows are scored, never used")
    ap.add_argument("--gold-train", type=Path, help="train.csv, for an all-data fine-tune from --init: its graded rows "
                    "join training at their grades, each drawn --gold-repeats times an epoch; --val may then lie inside "
                    "training (its oof_all.csv is in-sample, for the submission gate only)")
    ap.add_argument("--gold-repeats", type=int, default=38)
    ap.add_argument("--soup-init", type=float, default=0.0, help="after training, weights = w x --init's + (1 - w) x "
                    "the trained ones, before scoring and saving")
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
    ap.add_argument("--res", type=int, default=0, help="encoder input size; 0 = the arch's own")
    ap.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth rate")
    ap.add_argument("--init", type=Path, help="start from these weights instead of the backbone's pretraining: our "
                    "last.pt, or a public checkpoint dict {'model': ...} with backbone.* / clsW / clsb keys (the CC0 "
                    "0.928 CoAtNet). A bare filename is searched for under --corpus. With --epochs 0: evaluate only")
    ap.add_argument("--slots", default="", help="comma-separated slot names to use, e.g. cor_1,cor_2 (a coronal-only "
                    "teacher); train AND eval")
    ap.add_argument("--allow-missing-labels", action="store_true",
                    help="train on studies with some targets missing (NaN): those targets carry no loss")
    ap.add_argument("--aug-affine", type=float, default=0.0,
                    help="training-only per-window affine, strength 1 = rotate 10 deg / zoom 10%% / shift 5%%; no flips")
    ap.add_argument("--llrd", type=float, default=1.0, help="layer-wise LR decay per ViT block; 1 = off")
    ap.add_argument("--pos-emb", action="store_true", help="slot + depth embedding per window")
    ap.add_argument("--fit-budget", action="store_true",
                    help="time real training steps first, then cut --epochs to what fits the budget")
    ap.add_argument("--eval-reserve-h", type=float, default=0.0, help="with --fit-budget: hours kept for eval")
    ap.add_argument("--ckpt-blocks", type=int, default=-1,
                    help="gradient-checkpoint only the first N encoder blocks; -1 = all (the default), 0 = none")
    ap.add_argument("--mps", action="store_true",
                    help="no CUDA: train on Apple's GPU under bf16 autocast instead of the CPU")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the encoder in place, for training steps only (eval runs eager)")
    ap.add_argument("--extra-corpus", type=Path, nargs="*", default=[],
                    help="more roots of the same layout, joined to --corpus (e.g. NYU clinical parts); ids must not repeat")
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="each window ALSO as its centre crop, this fraction of the field (0.6714 = the middle 94 mm of 140), "
                         "resized back; both views in one bag, train and eval (2x the windows). 0 = off")
    ap.add_argument("--enc-chunk", type=int, default=0,
                    help="evaluation only: encode this many windows per forward (a large encoder on ~80 windows); 0 = all")
    ap.add_argument("--only-weak4", action="store_true", help="train on the weak four only (WEAK4); K's recipe")
    ap.add_argument("--spatial-attn", action="store_true",
                    help="per-finding attention over the positions of each window's feature map, in place of the global "
                         "average; starts as the average (convnext_* only)")
    ap.add_argument("--conf-weight", choices=["fork", "none"], default="fork",
                    help="fork: loss x (0.25 + 0.75|2y-1|), built for a table where 0.5 means 'not stated'; none: plain "
                         "BCE, for graded tables whose mid levels are readings (and for masked targets)")
    ap.add_argument("--eval-windows", type=int, default=0,
                    help="score an evenly spaced ~N windows per study (thin_windows); 0 = every window")
    ap.add_argument("--mirror", type=float, default=0.0,
                    help="training: this share of draws is the knee mirrored medial <-> lateral, with the Medial / Lateral "
                         "targets swapped and MCL unknown (canonical caches only; needs --allow-missing-labels)")
    ap.add_argument("--mirror-tta", action="store_true",
                    help="every prediction is the mean of the plain and the mirrored pass (MCL: the plain pass); "
                         "oof_all.csv holds that mean, oof_plain.csv the plain pass. Stored in last_meta.json, so the "
                         "submission path does the same")
    a = ap.parse_args()
    if not 0.0 <= a.mirror <= 1.0 or (a.mirror and not a.allow_missing_labels):
        raise SystemExit("--mirror is a probability in [0, 1] and needs --allow-missing-labels (MCL is unknown on a mirrored knee)")

    world, rank, local = dist_setup()
    if a.bs % (a.micro * world):
        raise SystemExit(f"--bs {a.bs} must be a multiple of --micro {a.micro} x {world} GPU(s)")
    t_start = time.time()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = (f"cuda:{local}" if torch.cuda.is_available()
           else "mps" if a.mps and torch.backends.mps.is_available() else "cpu")
    amp = dev.startswith("cuda")
    if amp:
        torch.backends.cudnn.benchmark = True
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=True)
    log(f"arch {a.arch} {ARCHS[a.arch]} | device {dev}"
        + (f" {torch.cuda.get_device_name(local)}" if amp else "") + f" | amp fp16 {amp} | world {world}")

    gold_ids: list[str] = []
    if (a.gold_train or a.soup_init) and not a.init:
        raise SystemExit("--gold-train and --soup-init fine-tune --init's weights: pass --init")
    if a.gold_train and a.gold:
        raise SystemExit("--gold scores the graded studies, --gold-train trains on them: not both")
    if not 0.0 <= a.soup_init < 1.0 or a.gold_repeats < 1:
        raise SystemExit("--soup-init in [0, 1), --gold-repeats at least 1")
    if a.synthetic:
        uids = [f"syn{i:04d}" for i in range(a.synthetic)]
        corpus = SyntheticCorpus(uids)
        rng = np.random.default_rng(1)
        tgt = pd.DataFrame(rng.choice([0.0, 0.5, 1.0], (len(uids), 12), p=[.6, .2, .2]),
                           index=uids, columns=LABELS)
        tr, va = uids[: a.synthetic * 2 // 3], uids[a.synthetic * 2 // 3:]
    else:
        corpus = Corpus(a.corpus, a.layout, tuple(a.extra_corpus))
        if a.slots:
            names = [m_.get("name") for m_ in getattr(corpus, "slot_meta", [])]
            bad = [n_ for n_ in a.slots.split(",") if n_ not in names]
            if bad:
                raise SystemExit(f"--slots {bad} not in this corpus's slots {names}")
            corpus.slot_keep = {names.index(n_) for n_ in a.slots.split(",")}
            log(f"slots: only {a.slots} (indices {sorted(corpus.slot_keep)}), train and eval")
        tgt = pd.read_csv(a.labels).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        tr, va = load_split(a.train), load_split(a.val)
        if a.gold:
            g = pd.read_csv(a.gold).dropna(subset=LABELS)
            gold_ids = [u for u in g.StudyInstanceUID.astype(str) if u in corpus.row]
            gtab = g.set_index("StudyInstanceUID")
        if set(tr) & set(gold_ids) or (a.epochs > 0 and set(tr) & set(va) and not a.gold_train):
            raise SystemExit("train overlaps val or gold -- refusing")
    miss_tr = [u for u in tr if u not in corpus.row]
    miss_va = [u for u in va if u not in corpus.row]
    tr = [u for u in tr if u in corpus.row and (tgt.loc[u, LABELS].notna().any() if a.allow_missing_labels
                                                 else tgt.loc[u, LABELS].notna().all())]
    va = [u for u in va if u in corpus.row]
    if a.limit_train:
        tr = tr[: a.limit_train]
    if a.limit_val:
        va = va[: a.limit_val]
    log(f"train {len(tr)} (missing from corpus {len(miss_tr)}) | val {len(va)} "
        f"(missing {len(miss_va)}) | gold {len(gold_ids)}")
    ytr = tgt.loc[tr, LABELS].to_numpy(np.float32, copy=True)
    if a.only_weak4:
        if not a.allow_missing_labels:
            raise SystemExit("--only-weak4 needs --allow-missing-labels (the other findings become unknown)")
        ytr[:, [i for i, f in enumerate(LABELS) if f not in WEAK4]] = np.nan
        log(f"only-weak4: training targets kept for {', '.join(WEAK4)}; the other eight carry no loss")
    nwin = [len(thin_windows(window_centres(corpus.study(u)[1], corpus.slot_sizes), corpus.slot_sizes,
                             a.eval_windows)) for u in va]
    log(f"val windows per study: median {int(np.median(nwin))}, min {min(nwin)}, max {max(nwin)}"
        + (f" (thinned to ~{a.eval_windows})" if a.eval_windows else ""))
    if a.pos_emb and not str(getattr(corpus, "layout", "")).startswith(("stage_b", "stage_k")):
        log("⚠️ --pos-emb on a non-canonical corpus: a slot's depth means lateral on one knee, medial on another")

    res = a.res or ARCHS[a.arch]["res"]
    enc = build_encoder(a.arch, a.weights_dir, res, a.drop_path) if rank == 0 else None
    barrier(world)
    if rank != 0:
        enc = build_encoder(a.arch, a.weights_dir, res, a.drop_path)
    model = WindowMIL(enc, res, n_slots=len(corpus.slot_sizes) if a.pos_emb else 0, spatial=a.spatial_attn).to(dev)
    if a.spatial_attn:
        log("spatial attention: a query per finding over each window's feature map (zero-initialised = the average)")
    if a.init:
        import hashlib
        src = a.init if a.init.exists() else next(iter(sorted(a.corpus.rglob(a.init.name))), None)
        if src is None:
            raise SystemExit(f"--init {a.init}: not found (nor under {a.corpus})")
        sd = torch.load(src, map_location="cpu", weights_only=False)
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        ren = {"clsW": "w", "clsb": "b"}
        sd = {("encoder." + k[len("backbone."):]) if k.startswith("backbone.") else ren.get(k, k): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        zero_init = {k for k in missing if k.startswith(("slot_emb.", "depth_proj.", "sq"))}
        if unexpected or set(missing) - zero_init:
            raise SystemExit(f"--init {src}: missing {sorted(set(missing) - zero_init)[:8]}, unexpected {unexpected[:8]}")
        a.init_sha256 = hashlib.sha256(src.read_bytes()).hexdigest()
        init_sd = ({k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                   if a.soup_init else None)
        log(f"init from {src} (sha256 {a.init_sha256[:16]}): {len(sd)} tensors, strict but for "
            f"{len(zero_init)} zero-initialised slot/depth weights")
    if a.aug_affine:
        model.affine = a.aug_affine
        log(f"augmentation: per-window affine at strength {a.aug_affine} (training only), plus the intensity gain")
    if a.enc_chunk:
        model.enc_chunk = a.enc_chunk
        log(f"evaluation encodes {a.enc_chunk} windows per forward")
    if a.zoom:
        model.zoom = a.zoom
        log(f"zoom: every window also as its centre {a.zoom:.4f} of the field, both views in one bag (train and eval)")
    if (ARCHS[a.arch].get("ckpt", True) and hasattr(model.encoder, "set_grad_checkpointing")
            and (amp or os.environ.get("TW_FORCE_CKPT"))):
        import timm.layers as _tl
        if hasattr(_tl, "set_reentrant_ckpt"):
            _tl.set_reentrant_ckpt(False)
        elif world > 1:
            raise SystemExit("this timm cannot switch off re-entrant checkpointing; DDP would break")
        n_blk = len(model.encoder.blocks) if hasattr(model.encoder, "blocks") else None
        if a.ckpt_blocks < 0:
            model.encoder.set_grad_checkpointing(True)
        elif n_blk is None:
            raise SystemExit(f"--ckpt-blocks {a.ckpt_blocks}: {a.arch}'s encoder has no .blocks; use -1 (all)")
        else:
            import torch.utils.checkpoint as _cp
            model.encoder.set_grad_checkpointing(False)
            for blk in list(model.encoder.blocks)[: a.ckpt_blocks]:
                blk.forward = (lambda x, _f=blk.forward, **kw: _cp.checkpoint(_f, x, use_reentrant=False, **kw))
        log(f"gradient checkpointing: {'all' if a.ckpt_blocks < 0 else a.ckpt_blocks} of "
            f"{n_blk if n_blk is not None else 'the'} encoder blocks")
    if amp and ARCHS[a.arch]["timm"].startswith(("convnext", "resnet")):
        model.encoder.to(memory_format=torch.channels_last)
        model.channels_last = True
        log("channels_last encoder and inputs")
    if a.compile:
        model.encoder.compile()
        log("encoder compiled (torch.compile, in place)")
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable {n_trn / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    log(f"encoder {n_enc / 1e6:.1f}M params, num_features {model.encoder.num_features} | res {res} | "
        f"drop_path {a.drop_path} | llrd {a.llrd} | pos_emb {a.pos_emb}")

    prev = (np.nan_to_num(np.nanmean(np.where(np.isfinite(ytr), ytr, np.nan), 0), nan=0.5)
            if a.allow_missing_labels else ytr.mean(0))
    if a.mirror:
        prev = (1 - a.mirror) * prev + a.mirror * prev[MIRROR_PERM]
        log(f"mirror: {a.mirror:.2f} of training draws mirrored (Medial <-> Lateral targets swapped, MCL masked)")
    prev = np.clip(prev, 0.03, 0.7)
    tta = slot_is_sagittal(corpus).to(dev) if a.mirror_tta else None
    if a.mirror_tta:
        log("mirror averaging: every prediction is the mean of the plain and the mirrored pass (MCL: plain only)")
    pos_w = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    groups = param_groups(model, a.bb_lr, a.head_lr, a.llrd)
    opt = torch.optim.AdamW(groups, weight_decay=a.wd, fused=amp)
    if a.llrd < 1.0:
        log(f"layer-wise LR: {len(groups) - 1} encoder groups, {groups[0]['lr']:.2e} (embeddings) .. "
            f"{groups[-2]['lr']:.2e} (top) | head {groups[-1]['lr']:.2e}")
    n_gold_train = 0
    if a.gold_train:
        g = pd.read_csv(a.gold_train).dropna(subset=LABELS).drop_duplicates("StudyInstanceUID")
        g = g.set_index(g.StudyInstanceUID.astype(str))
        g_ids = [u for u in g.index if u in corpus.row]
        if set(g_ids) & set(tr):
            raise SystemExit("--gold-train: graded studies are already in the training list")
        reps = a.gold_repeats
        if a.limit_train:
            g_ids, reps = g_ids[: max(1, a.limit_train // 4)], 2
        n_tr = len(tr)
        tr, ytr = with_gold(tr, ytr, g_ids, g.loc[g_ids, LABELS].to_numpy(np.float32), reps)
        n_gold_train = len(g_ids)
        log(f"gold-train: {len(g_ids)} graded studies at their grades ({len(g) - len(g_ids)} not in the corpus), "
            f"{reps} draws each -> {len(tr)} draws an epoch ({(len(tr) - n_tr) / len(tr):.1%} graded); the loss's "
            f"pos_weight stays the report studies'")
    tr_ds = Windows(corpus, tr, ytr, k=a.k, seed=a.seed + 1000 * rank, mirror=a.mirror)
    sampler = (torch.utils.data.distributed.DistributedSampler(tr_ds, num_replicas=world, rank=rank,
                                                                shuffle=True, seed=a.seed, drop_last=True)
               if world > 1 else None)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=a.micro, shuffle=sampler is None, sampler=sampler,
                                        drop_last=True, num_workers=ds_workers(), worker_init_fn=seed_worker,
                                        persistent_workers=ds_workers() > 0, pin_memory=amp)
    if world > 1:
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
            with autocast(dev, amp):
                logits = net(x, gain=gain, slot=sl, pos=ps)
            if a.allow_missing_labels:
                known = torch.isfinite(y)
                y0 = torch.where(known, y, torch.zeros_like(y))
                w = ((0.25 + 0.75 * (2 * y0 - 1).abs()) if a.conf_weight == "fork" else 1.0) * known
                loss = ((F.binary_cross_entropy_with_logits(logits.float(), y0, pos_weight=pos_w, reduction="none")
                         * w).sum() / known.sum().clamp_min(1))
            else:
                w = ((0.25 + 0.75 * (2 * y - 1).abs()) if a.conf_weight == "fork" else 1.0)
                loss = (F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_w,
                                                           reduction="none") * w).mean()
            scaler.scale(loss / accum).backward()
        return loss

    def to_dev(batch):
        return [t.to(dev, non_blocking=True) for t in batch]

    epochs, probe = a.epochs, None
    if a.fit_budget and a.epochs > 0:
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

    hist, last3, step_times = [], [], []
    waits, loss_trace, t_end, stopped_early = [], [], None, None
    for ep in range(epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        net.train()
        t_ep, n_opt, stop = time.time(), 0, False
        tot, recent, n_recent = torch.zeros((), device=dev), torch.zeros((), device=dev), 0
        gnorm = torch.full((), float("nan"), device=dev)
        t_win, n_win = time.time(), 0
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(tr_dl):
            if n_opt >= steps_per_epoch:
                break
            t0 = time.time()
            if t_end is not None:
                waits.append(t0 - t_end)
            x, sl, ps, y = to_dev(batch)
            last_micro = (it + 1) % accum == 0
            loss = micro_step(x, sl, ps, y, sync=last_micro).detach()
            tot += loss
            recent += loss
            n_recent += 1
            if len(loss_trace) < 60:
                loss_trace.append(round(float(loss), 5))
            if last_micro:
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                n_opt += 1
                if n_opt % 20 == 0 or n_opt >= steps_per_epoch:
                    stop = all_any(time.time() - t_start > a.time_budget_h * 3600, world, dev)
            n_win += 1
            t_end = time.time()
            if it % 50 == 0:
                mean_recent = float(recent) / n_recent
                if not math.isfinite(mean_recent):
                    raise SystemExit("non-finite loss")
                step_times.append((time.time() - t_win) / n_win)
                log(f"  ep{ep} it{it} loss {float(loss):.4f} (mean of last {n_recent}: {mean_recent:.4f}) | "
                    f"grad-norm {float(gnorm):.3f} | scale {scaler.get_scale() if amp else 1:.0f} | "
                    f"{step_times[-1]:.2f} s per {a.micro}-study micro-step")
                recent.zero_()
                n_recent, t_win, n_win = 0, time.time(), 0
            if stop:
                stopped_early = f"time budget at epoch {ep} iter {it}"
                break
        hist.append({"epoch": ep, "loss": float(tot) / max(it + 1, 1), "sec": time.time() - t_ep,
                     "opt_steps": n_opt})
        if not math.isfinite(hist[-1]["loss"]):
            raise SystemExit("non-finite loss")
        log(f"epoch {ep}: loss {hist[-1]['loss']:.4f}  {hist[-1]['sec'] / 60:.1f} min")
        if ep >= epochs - 3:
            last3.append({k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
        if stopped_early:
            log(f"⛔ stopped early: {stopped_early} -- the schedule did not finish")
            break

    if world > 1:
        import torch.distributed as dist
        chk = torch.tensor([sum(float(p.detach().double().sum()) for p in model.parameters() if p.requires_grad)],
                           device=dev, dtype=torch.float64)
        both = [torch.zeros_like(chk) for _ in range(world)]
        dist.all_gather(both, chk)
        vals = [float(t.item()) for t in both]
        log(f"DDP weight checksum per rank: {vals}")
        if max(vals) - min(vals) > 1e-6 * max(1.0, abs(vals[0])):
            raise SystemExit(f"ranks ended with different weights: {vals}")
    if a.soup_init:
        model.load_state_dict(soup(init_sd, model.state_dict(), a.soup_init), strict=True)
        log(f"soup: {a.soup_init} x the --init weights + {1 - a.soup_init} x the trained ones")
    peak = torch.cuda.max_memory_allocated(dev) / 1e9 if amp else None
    per_fwd = float(np.median(step_times)) if step_times else None
    bench = {"arch": a.arch, "device": torch.cuda.get_device_name(local) if amp else str(dev), "world": world,
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

    def score(uids, out_dir, name):
        """Predict `uids`; rank 0 writes <name>.csv."""
        raw = predict_sharded(model, corpus, uids, dev, amp, a.eval_bs, world, rank, a.eval_windows, tta)
        p = raw if tta is None else mirror_average(raw[:, 0], raw[:, 1])
        if rank == 0:
            pd.DataFrame(p, columns=LABELS).assign(StudyInstanceUID=uids).to_csv(out_dir / f"{name}.csv", index=False)
            if tta is not None:
                plain = {"oof_all": "oof_plain", "oof_gold": "oof_gold_plain"}[name]
                pd.DataFrame(raw[:, 0], columns=LABELS).assign(StudyInstanceUID=uids).to_csv(out_dir / f"{plain}.csv", index=False)
        return p

    t_ev = time.time()
    p_last = score(va, a.out, "oof_all")
    bench["eval_sec_per_study"] = (time.time() - t_ev) * world / max(len(va), 1)
    pg = score(gold_ids, a.out, "oof_gold") if gold_ids else None
    if a.save_weights and rank == 0:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   a.out / "last.pt")
        (a.out / "last_meta.json").write_text(json.dumps({
            "arch": a.arch, "res": res, "pool": ARCHS[a.arch]["pool"],
            "timm": ARCHS[a.arch]["timm"],
            "layout": corpus.layout, "slot_sizes": list(corpus.slot_sizes), "img": getattr(corpus, "img", 336),
            "pos_emb": bool(a.pos_emb), "n_slots": model.n_slots, "llrd": a.llrd, "drop_path": a.drop_path,
            "zoom": a.zoom, "spatial": bool(getattr(model, "spatial", False)),
            "mirror": a.mirror, "mirror_tta": bool(a.mirror_tta), "mean": MEAN, "std": STD}, indent=1, default=str))
    if len(last3) >= 2:
        avg = {k: (sum(sd[k].double() for sd in last3) / len(last3)).to(last3[0][k].dtype)
               if last3[0][k].is_floating_point() else last3[0][k] for k in last3[0]}
        model.load_state_dict(avg, strict=True)
        d3 = a.out / "avg3"
        if rank == 0:
            d3.mkdir(exist_ok=True)
        score(va, d3, "oof_all")
        if gold_ids:
            score(gold_ids, d3, "oof_gold")
        if rank == 0:
            if a.save_weights:
                torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                           d3 / "last.pt")
                (d3 / "last_meta.json").write_text((a.out / "last_meta.json").read_text())

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
                   "aug_affine": a.aug_affine, "zoom": a.zoom, "spatial_attn": a.spatial_attn, "conf_weight": a.conf_weight,
                   "only_weak4": a.only_weak4,
                   "mirror": a.mirror, "mirror_tta": bool(a.mirror_tta), "init": str(a.init) if a.init else None, "slots": a.slots or None,
                   "allow_missing_labels": a.allow_missing_labels,
                   "gold_train": n_gold_train, "gold_repeats": a.gold_repeats if a.gold_train else None,
                   "soup_init": a.soup_init or None,
                   "init_sha256": getattr(a, "init_sha256", None),
                   "labels_sha256": __import__("hashlib").sha256(Path(a.labels).read_bytes()).hexdigest()
                   if a.labels else None,
                   "primary": "last epoch", "secondary": "mean of last-3 epoch weights (fixed, not selected)",
                   "gold_macro_report_only": gold_auc, "hist": hist, "bench": bench,
                   "wall_min": (time.time() - t_start) / 60}
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
        (a.out / "bench.json").write_text(json.dumps(bench, indent=2))
        log(f"BENCH {json.dumps(bench)}")
        log(f"gold-58 macro {gold_auc} -- reported only, never selected on")
        log(f"done in {summary['wall_min']:.1f} min -> {a.out}")
    if world > 1:
        import torch.distributed as dist
        barrier(world)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
