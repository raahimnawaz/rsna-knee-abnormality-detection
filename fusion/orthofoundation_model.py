"""Load OrthoFoundation-L weights into timm's `vit_large_patch16_dinov3`; the key remap is derived from the weights.

    python fusion/orthofoundation_model.py    # checks the sha256, the strict load and a forward pass
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
CKPT = PROJ / "data" / "_orthofoundation" / "OrthoFoudation-L.pth"
TIMM_NAME = "vit_large_patch16_dinov3"
SHA256 = "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"


def remap(sd: dict) -> tuple[dict, torch.Tensor]:
    """Their key names -> timm's. Returns (state_dict, rope_periods)."""
    out, periods = {}, None
    for k, v in sd.items():
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        if k == "rope_embed.periods":
            periods = v
            continue
        if k == "storage_tokens":
            out["reg_token"] = v
            continue
        if k == "mask_token":                       # SSL-only, unused at inference
            continue
        if k.endswith(".attn.qkv.bias") or k.endswith(".attn.qkv.bias_mask"):
            if v.abs().sum() != 0:                  # verified zero; refuse if that ever changes
                raise SystemExit(f"⛔ {k} is NOT zero -- the qkvb variant is required, not this one")
            continue
        k = k.replace(".ls1.gamma", ".gamma_1").replace(".ls2.gamma", ".gamma_2")
        out[k] = v
    return out, periods


def build(ckpt: Path = CKPT, img: int = 336, **timm_kw):
    """`timm_kw` passes through to create_model (e.g. global_pool="token"); the load stays strict."""
    import timm
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)   # untrusted pickle
    sd, periods = remap(raw)
    m = timm.create_model(TIMM_NAME, pretrained=False, num_classes=0, img_size=img, **timm_kw)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"⛔ non-strict load: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    if periods is None:
        raise SystemExit("⛔ checkpoint carried no rope_embed.periods")
    with torch.no_grad():                            # non-persistent buffer: absent from the state dict
        m.rope.periods.copy_(periods.to(m.rope.periods.dtype))
    return m, periods


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=336)
    a = ap.parse_args()

    import hashlib
    h = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    print(f"sha256 {h}")
    if h != SHA256:
        print(f"⛔ SHA MISMATCH, expected {SHA256}")
        return 1
    print("  matches the recorded checkpoint")

    m, periods = build(img=a.img)
    assert torch.allclose(m.rope.periods.float(), periods.float(), atol=0), "rope periods not applied"
    print(f"loaded {TIMM_NAME} STRICT | num_features {m.num_features} "
          f"| params {sum(p.numel() for p in m.parameters()):,}")
    print(f"rope periods = THEIRS (bf16-rounded), first 4 {[round(float(x),6) for x in periods[:4]]}")

    m.eval()
    with torch.no_grad():
        f = m(torch.zeros(1, 3, a.img, a.img))
    print(f"forward at {a.img}px -> {tuple(f.shape)}  ({a.img}/16 = {a.img//16} -> {(a.img//16)**2} patches)")
    print("\n⚠️ This proves the ARCHITECTURE and the conditioning, NOT that the features are good.")
    print("   Strength on the out-of-fold set is the gate, never the 58 gold studies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
