"""`sadamtorres/rsna-ft-b` — the F6 arm 1 architecture, transcribed from its inference kernel.

    python fusion/ft_b_model.py --check      # build all 5, load strict, print their OOF

WHAT THIS IS. Five independently fine-tuned DINOv2 **ViT-B/14 @336** models with an attention-pool
head, published as CC-licensed Kaggle Datasets and claiming **LB 0.883 solo** (§3l-1). Each
checkpoint carries `backbone`, `head`, and the author's own `oof_macro`. `timm` is built
`pretrained=False`, so nothing is downloaded and no gated licence is touched — the checkpoint IS
the backbone.

**Why this arm and not the other two (§9e).** It is self-contained; it is the strongest claim; and
its `oof_macro` **0.7222** lands on the same scale as our own 0.7229 baseline (§2j), which is the
first time another team's local number has been directly legible in this project.

THE ONE STRUCTURAL DIFFERENCE FROM `pilkwang_model.py`. That file could verify itself — the fork
ships a `fingerprint()` per member and we matched 20/20 at 7e-06. **There is no fingerprint here
and no shipped OOF**, so `--check` can only prove the tensors load strictly into the architecture
this file declares. Strict loading is a real check (it catches a wrong `img_size`, a wrong feature
mode, a wrong head width) but it is **not** proof that our pixels are their pixels. That question
belongs to `ft_b_pixels.py` and is settled by the gate recorded there, not here.
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
CKPT = D / "external" / "ft_b_dinov2_vitb14_336"

BACKBONE = "vit_base_patch14_reg4_dinov2"
IMG = 336
K = 32                 # slices sampled per series; the head's `pos` buffer is (1,1,K,1)
FEAT = "both"          # cls ++ patch-mean, so d_in = 2 * 768
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


class Head(nn.Module):
    """Attention pool over slices, mean-within-plane over series, MLP to 12 logits. Theirs."""

    def __init__(self, d_in: int, d: int = 256, k: int = K, n_cls: int = 12, p_drop: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.plane_emb = nn.Embedding(3, d)
        self.register_buffer("pos", torch.linspace(0, 1, k).view(1, 1, k, 1))
        self.pos_proj = nn.Linear(1, d)
        self.att = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.mlp = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(p_drop),
                                 nn.Linear(3 * d, d), nn.GELU(),
                                 nn.Dropout(p_drop), nn.Linear(d, n_cls))

    def forward(self, x: torch.Tensor, pl: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x (B, S, K, d_in) -- S series per study, K slices per series
        h = self.proj(x) + self.pos_proj(self.pos) + self.plane_emb(pl)[:, :, None]
        a = self.att(h).softmax(dim=2)
        sv = (a * h).sum(dim=2)                       # one vector per series
        outs = []
        for p in range(3):                            # mean within each plane, then concat
            sel = (pl == p) & mask
            n = sel.sum(1, keepdim=True).clamp(min=1)
            outs.append((sv * sel[..., None]).sum(1) / n)
        return self.mlp(torch.cat(outs, dim=1))


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_fold(fold: int, dev: torch.device | None = None) -> tuple[nn.Module, Head, float]:
    """One fold. Strict load on both halves -- a silently-partial load is the failure this
    catches, and it is the only self-check this arm ships."""
    dev = dev or pick_device()
    p = CKPT / f"ft_f{fold}_best.pt"
    if not p.exists():
        raise SystemExit(f"missing {p} -- kaggle datasets download sadamtorres/rsna-ft-b --unzip")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    bb = timm.create_model(BACKBONE, pretrained=False, num_classes=0, img_size=IMG)
    bb.load_state_dict(ck["backbone"], strict=True)
    hd = Head(bb.num_features * (2 if FEAT == "both" else 1))
    hd.load_state_dict(ck["head"], strict=True)
    return bb.to(dev).eval(), hd.to(dev).eval(), float(ck.get("oof_macro", float("nan")))


@torch.no_grad()
def embed(bb: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """(K,3,336,336) -> (K, 1536). CLS ++ patch-mean, exactly as `FEAT='both'` does."""
    ft = bb.forward_features(x)
    n = bb.num_prefix_tokens                          # 1 CLS + 4 registers
    return torch.cat([ft[:, 0], ft[:, n:].mean(1)], dim=1).float()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        ap.print_help()
        return 0
    dev = pick_device()
    print(f"device {dev} · timm {timm.__version__} · {BACKBONE} @ {IMG}\n")
    for f in range(5):
        bb, hd, oof = load_fold(f, dev)
        e = embed(bb, torch.zeros(2, 3, IMG, IMG, device=dev))
        n_bb = sum(p.numel() for p in bb.parameters()) / 1e6
        n_hd = sum(p.numel() for p in hd.parameters()) / 1e6
        print(f"  fold {f}: strict OK · backbone {n_bb:.1f}M + head {n_hd:.2f}M · "
              f"embed {tuple(e.shape)} · their oof_macro {oof:.4f}")
        del bb, hd
    print("\nAll five load strictly. NOTE: this proves the architecture, NOT the pixels --")
    print("there is no fingerprint and no shipped OOF here. See ft_b_pixels.py for that gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
