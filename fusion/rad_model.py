"""The public RadImageNet arm: frozen official R50 trunk + `FoundationQueryHead`, five folds.

§4b measured that a RadImageNet arm at gold-47 **0.8514** — parity with pilkwang's 0.8516 — adds
**+0.0135** to our banked two-family blend, 98% of draws, CI [+0.0004, +0.0262]. This module is the
half of that we can legally ship.

⛔ THE VARIANT TRAP, AND IT IS THE WHOLE REASON TO READ THIS DOCSTRING.
§4b's number was measured on `v52_e11_oof.csv`, which is the **e11** variant: `_RAD_E11_CROP_MM =
130.0`. The weights in `data/external/radimagenet_r50_heads/` are the **folds_v1 public** variant,
whose manifest says `"crop": "full-frame"`. **They are different arms.** §2o's error class is
exactly this — three numbers near 0.89 that are three different quantities — so **+0.0135 does not
transfer, it has to be re-measured on whatever this module actually reproduces.** The e11 heads sit
inside a bundle whose own README says it must remain private and that includes CC-BY-NC-SA-4.0
sources; folds_v1 is a plain public dataset. We reproduce the one we can ship.

THE CONTRACT IS MACHINE-READABLE, WHICH NO OTHER ARM HERE HAS GIVEN US.
`rad_heads_manifest.json` pins SHA-256 for the encoder and all five heads, and **both verify**:

    encoder  08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734  ✓
    head f0  0c92b27578e139cc35071a3f72ddd4e1a66106761225a7e011f076f37eb7051d  ✓

`head_parameters: 3174924` matches this module's build **to the digit**, which is a stronger
structural check than the transcriptions in §3n (`ft_b`) and §3s (DINOv3) ever had — those had no
fingerprint at all. It still does not check pixels: §3g's lesson is that a weights check cannot,
and `rad_pixels` answers for those separately.

ARCHITECTURE, TRANSCRIBED FROM `mattiaangeli/bend-the-knee-to-dinov3-ensembled` (`_RadHead`),
not inferred from the state dict. The state-dict shapes agree with the transcription on all 17
tensors, which is why this is a reading rather than a guess:

    project   LayerNorm(2048) -> Linear(2048, 512) -> GELU        R50 GAP features
    plane     (3, 512)    learned, one per PLANE (not per slot)
    position  (8, 512)    learned, one per slice index
    query     (12, 512)   one per label
    attn      MultiheadAttention(512, 8 heads, dropout .10, batch_first)
    fuse      LayerNorm(2048) -> Linear(2048, 512) -> GELU -> Dropout(.15)
    weight    (12, 512) + bias (12)   per-label dot product, as pilkwang's SlotHead does

The 2048 into `fuse` is **four** 512-vectors concatenated -- `[attended, mean, |attended - mean|,
attended * mean]`. That is the one part a state dict cannot tell you and a wrong guess would run
perfectly while scoring wrongly, which is §9h's failure mode.

⚠️ `N_SLOT` IS 3 HERE AND 6 EVERYWHERE ELSE IN THIS REPO. The `plane` embedding is per PLANE --
Sagittal, Coronal, Axial, fat-suppressed only -- not per (plane x fluid) slot. Feeding a 6-slot
tensor loads strict and conditions every plane wrongly. Both are explicit constructor arguments
here so the mistake cannot be made by inheriting a global.

⚠️ THE ARM IS NOT USED ON EVERY LABEL. Their kernel sets `_RAD_EXCLUDE = ("Baker's", "Fracture")`
and `_RAD_ALPHA = 0.50`. We do not have to copy that, but a blend that silently includes those two
is not the configuration the 0.920 board score came from.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

PROJ = Path(__file__).resolve().parents[1]
D = PROJ / "data"
HEADS = D / "external" / "radimagenet_r50_heads"
ENCODER = D / "external" / "radimagenet_r50_encoder" / "ResNet50.pt"

TOKEN_DIM = 2048        # ResNet-50 global-average-pooled trunk output
HEAD_DIM = 512
N_PLANE = 3             # Sagittal, Coronal, Axial -- fat-suppressed only
N_SLICE = 8             # manifest: slices_per_plane
N_LABEL = 12

# Their per-label gating, recorded so a blend can reproduce the board configuration (see docstring).
RAD_EXCLUDE = ("Baker's", "Fracture")
RAD_ALPHA = 0.50


class WeightsError(RuntimeError):
    """Raised when attached weights are not the ones the manifest pins."""


class RadEncoder(nn.Module):
    """Official RadImageNet ResNet-50, frozen, global-average-pooled to 2048.

    `children()[:-2]` drops torchvision's own avgpool and fc, then the mean over the spatial dims
    is their GAP. Transcribed from `_RadEncoder`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(*list(torchvision.models.resnet50(weights=None).children())[:-2])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).mean(dim=(2, 3))


class FoundationQueryHead(nn.Module):
    """`_RadHead`, transcribed. `head_class` in the manifest; `_RadHead` in their notebook."""

    def __init__(self, n_plane: int = N_PLANE, n_slice: int = N_SLICE,
                 token_dim: int = TOKEN_DIM, head_dim: int = HEAD_DIM,
                 n_label: int = N_LABEL) -> None:
        super().__init__()
        self.n_plane, self.n_slice, self.n_label = n_plane, n_slice, n_label
        self.project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, head_dim),
                                     nn.GELU())
        self.plane = nn.Parameter(torch.randn(n_plane, head_dim) * .01)
        self.position = nn.Parameter(torch.randn(n_slice, head_dim) * .01)
        self.query = nn.Parameter(torch.randn(n_label, head_dim) * .02)
        self.attn = nn.MultiheadAttention(head_dim, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(nn.LayerNorm(head_dim * 4), nn.Linear(head_dim * 4, head_dim),
                                  nn.GELU(), nn.Dropout(.15))
        self.weight = nn.Parameter(torch.randn(n_label, head_dim) * .02)
        self.bias = nn.Parameter(torch.zeros(n_label))

    def forward(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """feature [B, n_plane*n_slice, 2048] -> logits [B, 12]. `mask` is [B, n_plane*n_slice]."""
        token = self.project(feature.float())
        token = token.view(len(token), self.n_plane, self.n_slice, -1)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        # A study with nothing present would make the softmax undefined; they unmask slot 0 and
        # let the mean below carry the zeros. Kept verbatim -- it is a degenerate-input guard and
        # `PLAN.md` §5 names missing series as the classic submission-time crash.
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token,
                                     key_padding_mask=key_padding, need_weights=False)[0]
        denominator = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdim=True) / denominator
        mean = mean.expand(-1, self.n_label, -1)
        fused = self.fuse(torch.cat([attended, mean, torch.abs(attended - mean),
                                     attended * mean], dim=-1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> dict:
    return json.load(open(HEADS / "rad_heads_manifest.json", encoding="utf-8"))


def load_encoder(dev="cpu", *, verify: bool = True) -> RadEncoder:
    """The frozen trunk, with its manifest hash checked before it is trusted."""
    man = manifest()
    if verify:
        got = sha256(ENCODER)
        if got != man["encoder_sha256"]:
            raise WeightsError(f"encoder sha256 {got} != manifest {man['encoder_sha256']}")
    enc = RadEncoder()
    sd = torch.load(ENCODER, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith(("fc.", "classifier."))}
    # torchvision names the trunk's children 0..7; the RadImageNet export keeps resnet names.
    keyed = {}
    names = [n for n, _ in RadEncoder().backbone.named_parameters()]
    if any(k.startswith("backbone.") for k in sd):
        keyed = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = enc.backbone.load_state_dict(keyed or sd, strict=False)
    return enc.to(dev).eval(), {"missing": len(missing), "unexpected": len(unexpected),
                                "n_trunk_params": len(names)}


def load_head(fold: int, dev="cpu", *, verify: bool = True) -> tuple[FoundationQueryHead, dict]:
    """One fold's head, loaded STRICT. Strict is the point: §9h's wrong-table bug ran perfectly."""
    man = manifest()
    name = f"rad_head_f{fold}.pt"
    path = HEADS / name
    if verify:
        got = sha256(path)
        want = man["heads"][name]["sha256"]
        if got != want:
            raise WeightsError(f"{name} sha256 {got} != manifest {want}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    head = FoundationQueryHead()
    head.load_state_dict(sd, strict=True)
    meta = {k: ck.get(k) for k in ("version", "fold", "best_val", "best_epoch", "config_hash")}
    if meta["fold"] is not None and int(meta["fold"]) != fold:
        raise WeightsError(f"{name} declares fold {meta['fold']}, loaded as {fold}")
    return head.to(dev).eval(), meta


def self_test() -> int:
    """Structural proof before any pixel touches this: hashes, strict load, parameter count."""
    man = manifest()
    print(f"manifest v{man['version']}  head_class {man['head_class']}  "
          f"target_mode {man['target_mode']}")
    print(f"input contract: {man['input']}")
    n = sum(p.numel() for p in FoundationQueryHead().parameters())
    ok = n == man["head_parameters"]
    print(f"\nparameter count {n:,} vs manifest {man['head_parameters']:,}  "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        return 1
    for f in range(5):
        head, meta = load_head(f)
        print(f"  fold {f}: sha OK, strict load OK, best_val {meta['best_val']}, "
              f"epoch {meta['best_epoch']}, version {meta['version']}")
    enc, info = load_encoder()
    print(f"\nencoder: sha OK, {info['n_trunk_params']} trunk tensors, "
          f"{info['missing']} missing / {info['unexpected']} unexpected")
    if info["missing"]:
        print("  ⚠️ missing keys -- the trunk is NOT fully initialised; do not run pixels yet")
        return 1
    with torch.no_grad():
        feat = torch.randn(2, N_PLANE * N_SLICE, TOKEN_DIM)
        mask = torch.ones(2, N_PLANE * N_SLICE)
        mask[1, 8:] = 0                                   # a study missing two planes
        out = load_head(0)[0](feat, mask)
    print(f"forward OK: {tuple(out.shape)}  finite {bool(torch.isfinite(out).all())}")
    print(f"\n⛔ variant: these are folds_v1 ({man['input']['crop']}). §4b's +0.0135 was measured "
          f"on e11 (130 mm crop). RE-MEASURE, do not assume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
