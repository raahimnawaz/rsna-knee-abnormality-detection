"""`mattiaangeli/knee-mri-fold-weights` — the F6 arm 3 architecture, transcribed from its kernel.

    python fusion/dinov3_model.py --check      # build all 5, load strict, print the cfg

WHAT THIS IS. Five folds of **DINOv3 ViT-S/16 @336** fine-tuned on this competition, with
token-level **slot-type** conditioning and an `xcodex` readout. Published as a CC-licensed Kaggle
Dataset; **the encoder weights are embedded in the checkpoint**, so DINOv3's gated licence is not
a blocker for inference and nothing is downloaded (`timm` is built `pretrained=False`).

PROVENANCE. Everything here is copied from `mattiaangeli/bend-the-knee-to-dinov3-ensembled`
(pulled 2026-08-13, cell 26). `PLAN.md` §9h spec'd it off the weights; this file is the
transcription that spec asked for. **The classes below are the kernel's, not a reimplementation** —
that was a deliberate instruction (§9h), because the failure mode here is a model that loads
strict, runs, and is quietly wrong.

THE THREE THINGS THAT ARE EASY TO GET WRONG, and why they are written the way they are:

1. **`stem: 'native'` means the 16 slices are CHANNELS, not 16 forward passes.**
   `patch_embed.proj.weight` is **(384, 16, 16, 16)** — `in_chans=n_slice`. Read the weights.
2. **The slot token is inserted between the prefix tokens and the patches, and RoPE has to be
   told.** `ViTSlotToken.__init__` bumps `vit.num_prefix_tokens` **and every block's
   `attn.num_prefix_tokens`** by one, because `EvaAttention.forward` applies rope to
   `q[:, :, npt:, :]` and the rope table only covers the 441 patches. Without the bump the rope
   table is one token out of phase and the model is silently wrong. §9h called this the
   highest-risk part and it is.
3. **`Net.forward` keeps `[CLS] ++ [slot_tok] ++ patches`** — `f[:, :1]` then `f[:, orig:]`. So
   `tok[:, 0]` is CLS (what `base` pools) and `tok[:, 1:]` is **the slot token plus the patches**
   (what the delta attends over). The slot token IS in the delta's KV.

WHAT THIS FILE DOES NOT PROVE. Like `ft_b_model.py`, this arm ships **no fingerprint and no OOF**,
so `--check` proves only that the tensors load strictly into the architecture declared here.
Strict loading catches a wrong `img_size`, a wrong `in_chans`, a wrong pool or head width. It does
**not** prove our pixels are their pixels — see `dinov3_pixels.py` for that, and §9h's
pre-registered gate for what would count as passing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
D = PROJ / "data"
CKPT = D / "external" / "dinov3_vits16_folds"

N_SLOT_TYPES, MASK_IDX = 6, 0
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# The pixel-path constants, kept here so the two files cannot drift. NOTE these are NOT
# pilkwang's: the band is (0.12, 0.88) against pilkwang's (0.2, 0.8), and the slot list is
# (plane, fat-suppression) pairs in a DIFFERENT ORDER from pilkwang's six recovered slots.
CROP_MM, SIZE, SLICE_BAND, N_SLICE = 130.0, 336, (0.12, 0.88), 16
SLOTS = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1),
         ("Coronal", 0), ("Axial", 1), ("Axial", 0)]


# --------------------------------------------------------------------------------------------
# The encoder wrapper. Copied from the kernel; see note 2 in the docstring.
# --------------------------------------------------------------------------------------------
class ViTSlotToken(nn.Module):
    """Insert a learned slot-type token after the prefix tokens, keeping RoPE in phase."""

    def __init__(self, vit, n_cat, dim=None):
        super().__init__()
        self.vit = vit
        d = dim or vit.embed_dim
        self.tok = nn.Embedding(n_cat + 1, d, padding_idx=MASK_IDX)
        self.num_features = vit.num_features
        self._orig_prefix = getattr(vit, "num_prefix_tokens", 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for blk in vit.blocks:
            a = getattr(blk, "attn", None)
            if a is not None and hasattr(a, "num_prefix_tokens"):
                a.num_prefix_tokens = a.num_prefix_tokens + 1

    @staticmethod
    def _maybe(mod, x):
        return x if mod is None else mod(x)

    def forward_features(self, x, cat):
        v = self.vit
        x = v.patch_embed(x)
        pos = v._pos_embed(x)
        rope = None
        if isinstance(pos, tuple):
            x, rope = pos
        else:
            x = pos
        x = self._maybe(getattr(v, "patch_drop", None), x)
        x = self._maybe(getattr(v, "norm_pre", None), x)
        npt = self._orig_prefix
        tok = self.tok(cat).unsqueeze(1)
        x = torch.cat([x[:, :npt], tok, x[:, npt:]], dim=1)
        if rope is not None:
            if getattr(v, "rope_mixed", False):
                for i, blk in enumerate(v.blocks):
                    x = blk(x, rope=rope[i])
            else:
                for blk in v.blocks:
                    x = blk(x, rope=rope)
        else:
            x = v.blocks(x)
        return v.norm(x)

    def forward_head(self, x, pre_logits=True):
        return self.vit.forward_head(x, pre_logits=pre_logits)


# --------------------------------------------------------------------------------------------
# The readout. `pool='xcodex'` -> CodexResidualPool. Copied from the kernel.
# --------------------------------------------------------------------------------------------
def _seg_mean_max(v, sidx, B):
    """Per-study mean ++ max over the study's series. (T,D) -> (B, 2D)."""
    Dm = v.shape[1]
    cnt = torch.zeros(B, device=v.device, dtype=v.dtype).index_add_(
        0, sidx, torch.ones(v.shape[0], device=v.device, dtype=v.dtype))
    mean = torch.zeros(B, Dm, device=v.device, dtype=v.dtype).index_add_(0, sidx, v)
    mean = mean / cnt.clamp(min=1).unsqueeze(1)
    mx = torch.full((B, Dm), -10000.0, device=v.device, dtype=v.dtype)
    mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, Dm), v, reduce="amax", include_self=True)
    return torch.cat([mean, mx], 1)


def _pad_kv(x, sidx, B, norm):
    """Ragged (T,P,D) series-per-study -> dense (B, S*P, D) + key-padding mask."""
    T, P, Dm = x.shape
    cnt = torch.bincount(sidx, minlength=B)
    S = int(cnt.max().item())
    starts = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(T, device=x.device) - starts[sidx]
    pad = x.new_zeros(B, S, P, Dm)
    pad[sidx, pos] = x
    keep = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    keep[sidx, pos] = True
    return norm(pad.reshape(B, S * P, Dm)), ~keep.repeat_interleave(P, dim=1)


class _GatedDelta(nn.Module):
    def __init__(self, d, n_labels, n_heads, dropout):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.d_norm = nn.LayerNorm(d)
        self.dw = nn.Parameter(torch.randn(n_labels, d) * (1.0 / d ** 0.5))
        self.db = nn.Parameter(torch.zeros(n_labels))
        self.gate = nn.Parameter(torch.zeros(n_labels))

    def delta(self, pat, sidx, B, return_attn):
        kv, kpm = _pad_kv(pat, sidx, B, self.kv_norm)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, kv, kv, key_padding_mask=kpm,
                           need_weights=return_attn, average_attn_weights=True)
        return (self.d_norm(att) * self.dw).sum(-1) + self.db, w


class CodexResidualPool(_GatedDelta):
    """`base` on the CLS token, plus a gated label-query delta over slot-token ++ patches."""

    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout),
                                  nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 0], sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return base + self.gate * d_, w


class Readout(nn.Module):
    def __init__(self, pool, d, n_labels=12, pe=64):
        super().__init__()
        if pool != "xcodex":
            raise ValueError(f"only 'xcodex' is transcribed; checkpoint asks for {pool!r}")
        self.pool_kind, self.k = pool, n_labels
        self.pres_emb = nn.Embedding(N_SLOT_TYPES + 1, pe, padding_idx=0)
        self.pool = CodexResidualPool(d, n_labels, pe=pe)
        self.drop = nn.Dropout(0.2)

    def forward(self, f, slot, sidx, B, return_attn=False):
        pe = self.pres_emb(slot)
        pres = torch.zeros(B, pe.shape[1], device=f.device, dtype=f.dtype).index_add_(0, sidx, pe)
        return self.pool(f, slot, sidx, B, pres)[0]


class Net(nn.Module):
    """`stem='native'` and `n_meta=0` remove DepthCompress, SlotDepthMixer and meta_mlp;
    `cond='token'` removes `slot_emb`. Those branches are asserted away, not transcribed."""

    def __init__(self, enc, cond, n_meta=0, pool="mean_max", stem="native", n_slice=16):
        super().__init__()
        if stem != "native" or n_meta != 0 or cond != "token":
            raise ValueError(f"untranscribed config: {stem=} {n_meta=} {cond=}")
        self.enc, self.cond = enc, cond
        self.tokens = pool in ("xattn", "xres", "clsadd", "xcodex")
        self.readout = Readout(pool, enc.num_features)

    def forward(self, im, slot, smeta, sidx, B, vm=None):
        f = self.enc.forward_features(im, slot)
        inner = getattr(self.enc, "vit", self.enc)
        orig = getattr(self.enc, "_orig_prefix", getattr(inner, "num_prefix_tokens", 1))
        f = torch.cat([f[:, :1], f[:, orig:]], 1)      # CLS ++ slot_tok ++ patches
        return self.readout(f, slot, sidx, B)


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_from_cfg(cfg: dict) -> Net:
    stem = cfg.get("stem", "native")
    in_ch = 3 if stem == "compress" else cfg.get("n_slice", 16)
    kw = {"img_size": cfg["img"]} if "vit_" in cfg["backbone"] else {}
    enc = timm.create_model(cfg["backbone"], pretrained=False, num_classes=0, in_chans=in_ch, **kw)
    if cfg["cond"] == "token":
        enc = ViTSlotToken(enc, N_SLOT_TYPES)
    return Net(enc, cfg["cond"], cfg.get("n_meta", 0), cfg["pool"],
               stem=stem, n_slice=cfg.get("n_slice", 16))


def load_fold(fold: int, dev: torch.device | None = None) -> tuple[Net, dict]:
    """One fold, loaded with the kernel's own acceptance rule: `enc.*` may be missing (timm
    rebuilds buffers), nothing may be unexpected, and no non-`enc.` key may be missing."""
    dev = dev or pick_device()
    p = CKPT / f"m_f{fold}.pt"
    if not p.exists():
        raise SystemExit(
            f"missing {p} -- kaggle datasets download mattiaangeli/knee-mri-fold-weights --unzip")
    z = torch.load(p, map_location="cpu", weights_only=False)
    m = build_from_cfg(z["cfg"])
    missing, unexpected = m.load_state_dict(z["state_dict"], strict=False)
    bad = [k for k in missing if not k.startswith("enc.")]
    if bad:
        raise SystemExit(f"fold {fold}: missing non-encoder keys {bad[:5]}")
    if unexpected:
        raise SystemExit(f"fold {fold}: unexpected keys {list(unexpected)[:5]}")
    if missing:
        raise SystemExit(f"fold {fold}: encoder keys missing, load is NOT strict: {missing[:5]}")
    return m.to(dev).eval(), z["cfg"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        ap.print_help()
        return 0
    dev = torch.device("cpu")          # the checks are tiny; MPS buys nothing and costs transfers
    print(f"device {dev} · timm {timm.__version__} (their kernel pinned 1.0.22)\n")

    torch.manual_seed(0)
    im, sidx, sm = torch.rand(2, N_SLICE, SIZE, SIZE), torch.tensor([0, 0]), torch.zeros(2, 0)
    outs = []
    for f in range(5):
        m, cfg = load_fold(f, dev)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        with torch.no_grad():
            outs.append(m(im, torch.tensor([1, 3]), sm, sidx, 1)[0])
        print(f"  fold {f}: strict OK · {n:.1f}M params · {cfg['backbone']} @ {cfg['img']} · "
              f"pool={cfg['pool']} cond={cfg['cond']} n_slice={cfg['n_slice']}")
        if f == 0:
            with torch.no_grad():
                d = (m(im, torch.tensor([1, 3]), sm, sidx, 1)
                     - m(im, torch.tensor([2, 5]), sm, sidx, 1)).abs().max().item()
            cond_delta = d
        del m

    spread = torch.stack(outs).std(0).mean().item()
    print("\nAll five load strictly (every enc.* tensor consumed, nothing unexpected).")
    print(f"  slot conditioning is LIVE      : max|Δlogit| {cond_delta:.4f} on a slot-id swap "
          f"(0.0 would mean `enc.tok` is dead weight)")
    print(f"  the folds are DISTINCT         : mean per-label std {spread:.4f} across the five "
          f"(0.0 would mean duplicated weights)")
    if cond_delta < 1e-6 or spread < 1e-6:
        print("\n  ⛔ one of the two checks collapsed — do not score this arm.")
        return 1
    print("\nNOTE: this proves the ARCHITECTURE, not the pixels. See PLAN.md §9h's four")
    print("corrections before writing dinov3_pixels.py — the slot table is NOT pilkwang's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
