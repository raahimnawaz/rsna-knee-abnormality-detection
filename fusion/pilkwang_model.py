"""The fork's architecture, transcribed, so its 20 members can be run off the platform.

WHY A TRANSCRIPTION AND NOT AN IMPORT. `pilkwang/rsna-knee-baseline-v1` is a notebook. Its scored
path (`infer_from_package`) assumes the Kaggle mount: a competition root, a DICOM tree, an attached
DINOv2 dataset, a 9-hour budget it measures itself against. None of that exists here and none of it
is the model. What *is* the model is `SlotHead`, `Model` and `build_model`, and those are copied
below unchanged -- same tensor ops in the same order, so the same weights compute the same map.

THE FINGERPRINT IS WHY THIS IS SAFE, and it is the whole reason to do the model before the pixels.
Each checkpoint ships `fingerprint`: the member's output on a bag of random bytes generated from a
seed. It answers one question and answers it exactly -- *is the object we just built the object
these weights were fitted to?* -- and it answers it **without any image being involved** (see
`IMPROVEMENTS.md` §3g, which is the correction that opened this whole route). So a transcription
error and a pixel error, which would otherwise arrive as one indistinguishable bad number, are
separated: run this first, and anything that goes wrong afterwards is pixels.

    python fusion/pilkwang_model.py --check          # all 20 members, ~2 min, no images

Tolerance is theirs: `FINGERPRINT_TOL = 2e-3`, sitting between GPU-numerics disagreement (~1e-5)
and the order-one move that any real difference in the forward path produces. We run float32
throughout for the same reason they do -- autocast would make the value depend on which device
happened to compute it, and the point of the number is that it does not.

Their `fingerprint()` builds its input on the DEVICE it is checking, via `.to(dev)` after a
CPU-side `torch.randint` with a seeded generator. That ordering is preserved exactly: generating on
the device instead would draw from a different RNG stream and fail every member.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
WEIGHTS = PROJ / "data" / "external" / "pilkwang_weights"

# --- verbatim from the fork ------------------------------------------------------------------- #

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
           "Fracture"]

SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
SLOTS = SLOTS_RECOVERED
N_SLOT = len(SLOTS)

POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}
SLOT_PRIOR_STRENGTH = 0.55
SLOT_PRIOR_TABLE: dict[str, tuple[int, ...]] = {}

SEED = 2026
FINGERPRINT_TOL = 2e-3
GROUP = 3


class WeightsError(RuntimeError):
    """Raised when attached weights cannot be trusted to be the ones that were fitted."""


class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study."""

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and n_out == len(TARGETS):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer("slot_prior", p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class Model(nn.Module):
    """Encoder plus head. The bag is flattened for the encoder and folded back for the head."""

    def __init__(self, backbone, dim, pool="cls_mean", prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        import torch.nn.functional as F
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)


def build_model(unfreeze_last, source="facebook/dinov2-small", variant="small",
                pool="cls_mean", prior=False, quiet=False):
    """Load the encoder and open the last `unfreeze_last` blocks.

    `source` defaults to the Hub id rather than the fork's attached-dataset lookup, which has no
    meaning off the platform. The member configs name `facebook/dinov2-small` as `backbone`, so
    this is the same object by the same name; the fingerprint is what proves it.
    """
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(str(source))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    if not quiet:
        print(f"  backbone: {n_layer} blocks, last {unfreeze_last} trainable, dim {dim}")
    return Model(bb, dim, pool=pool, prior=prior)


def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    """The model's output on a fixed synthetic bag. No image is involved -- see §3g."""
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size),
                         generator=g, dtype=torch.uint8).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out


def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=""):
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise WeightsError(f"{tag}fingerprint shape {got.shape} != stored {exp.shape}")
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(f"{tag}fingerprint differs by {d:.4g} (tolerance {tol:g})")
    return d


def load_member(m, path=WEIGHTS, dev=None, quiet=True):
    """Build one member and verify it against its stored fingerprint. Returns (model, dev, d)."""
    dev = dev or pick_device()
    cfg = m["config"]
    ck = torch.load(Path(path) / m["file"], map_location="cpu", weights_only=False)
    model = build_model(int(cfg["unfreeze_last"]), variant=cfg["variant"],
                        pool=cfg.get("pool", "cls_mean"),
                        prior=bool(cfg.get("prior", False)), quiet=quiet).to(dev)
    model.load_state_dict(ck["model"])
    d = check_fingerprint(model, dev, int(cfg["img"]), ck["fingerprint"], tag=f"{m['id']}: ")
    return model, dev, d


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def manifest(path=WEIGHTS):
    return json.loads((Path(path) / "manifest.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fingerprint every member")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    man = manifest()
    mem = man["members"][:a.limit] if a.limit else man["members"]
    dev = torch.device(a.device) if a.device else pick_device()
    print(f"{len(mem)} member(s), device {dev}, tolerance {FINGERPRINT_TOL:g}\n")

    ok = bad = 0
    worst = 0.0
    for m in mem:
        try:
            model, _, d = load_member(m, dev=dev)
            worst = max(worst, d)
            ok += 1
            print(f"  {m['id']}  fold {m['fold']}  holdout {m['holdout']:.4f}  "
                  f"fingerprint matches within {d:.2g}")
            del model
        except WeightsError as e:
            bad += 1
            print(f"  {m['id']}  FAILED: {e}")
        except Exception as e:            # noqa: BLE001 - report and continue
            bad += 1
            print(f"  {m['id']}  ERROR: {type(e).__name__}: {e}")

    print(f"\n{ok} matched, {bad} failed. Worst deviation {worst:.3g} "
          f"(tolerance {FINGERPRINT_TOL:g}).")
    if bad == 0 and ok:
        print("The architecture transcription is EXACT. Anything wrong from here is pixels.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
