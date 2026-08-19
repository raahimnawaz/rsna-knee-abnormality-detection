"""OrthoFoundation-L -> timm `vit_large_patch16_dinov3`, with the remap read off the WEIGHTS.

    .venv/bin/python fusion/orthofoundation_model.py            # verifies sha, strict load, rope, forward

PLAN.md §C-5. The checkpoint is `data/_orthofoundation/OrthoFoudation-L.pth` (their spelling),
1,213,056,638 bytes, sha256 385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9.
Licence is the authors' 2026-08-18 email, NOT a LICENSE file -- there is none in their repo.

WHAT IT IS, READ FROM THE TENSORS RATHER THAN THE README: cls_token (1,1,1024), storage_tokens
(1,4,1024), patch_embed.proj.weight (1024,3,16,16), rope_embed.periods (16,), 24 blocks,
303,227,920 params. That is DINOv3 ViT-L/16 with 4 registers and RoPE, taking **3-channel** input.

⛔ THE TRAP, AND IT IS §9h's EXACTLY -- A WRONG TABLE RUNS PERFECTLY AND SCORES WRONGLY.
timm holds `rope.periods` as a **non-persistent buffer**, so `load_state_dict` NEVER sets it and
silently keeps timm's own values. And they do not match:

    theirs   1.0, 1.335938, 1.78125,  2.375,    3.15625,  4.21875,  5.625,    7.5
    timm     1.0, 1.333521, 1.778279, 2.371374, 3.162278, 4.216965, 5.623413, 7.498942

Theirs is the **bfloat16-rounded** form of the same geometric series -- they pretrained in bf16 --
and max|diff| is 8.0e-02. The weights were adapted to the periods the network actually SAW, so
theirs are the correct ones and this file copies them in explicitly. A strict load would have
reported success either way. `--check` asserts they match afterwards.

WHY THE QKV BIAS IS DROPPED AND WHY THAT IS SAFE, ALSO MEASURED: `attn.qkv.bias` is exactly zero
in **all 24 blocks**, and `attn.qkv.bias_mask` is all zeros too. So the `_qkvb` timm variant is the
wrong target and the plain one is right; dropping those tensors loses nothing. `mask_token` is
SSL-only and unused at inference.

⚠️ 303 M params against the port's current `vit_small_patch14_reg4_dinov2` at ~22 M. `SlotNet`
runs B*K images through the encoder, so this is ~14x the encoder cost per tile and it has to be
priced against §3e's 9 h submission budget BEFORE it is proposed for the scored path. Patch 16
divides 336 exactly (21x21 = 441 patches), so the existing tile size needs no change.
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


def build(ckpt: Path = CKPT, img: int = 336):
    import timm
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)   # untrusted pickle
    sd, periods = remap(raw)
    m = timm.create_model(TIMM_NAME, pretrained=False, num_classes=0, img_size=img)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"⛔ non-strict load: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    if periods is None:
        raise SystemExit("⛔ checkpoint carried no rope_embed.periods")
    with torch.no_grad():                            # THE non-persistent buffer, see docstring
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
    print("   Strength on the 4,407-study OOF via §9e's rule is the gate. Never gold-58 (§3b).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
