"""Qwen3.5 VLM as a study reader (experimental): each window is one image in a single sequence, and each finding is read at its own name's position."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# What the model is asked to write, in the order of LABELS. Plain words it has seen in reports.
NAMES = ["ACL", "MCL", "medial meniscus", "lateral meniscus", "medial osteoarthritis",
         "lateral osteoarthritis", "patellofemoral osteoarthritis", "effusion", "synovitis",
         "Baker's cyst", "bone contusion", "fracture"]
HF_ID = {"qwen35vl": "Qwen/Qwen3.5-2B"}
HF_REVISION = {"qwen35vl": "15852e8c16360a2fea060d615a32b45270f8a8fc"}
QWEN_MEAN, QWEN_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
PATCH, MERGE, TEMPORAL = 16, 2, 2


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * B(A(dropout(x))). Base frozen; B starts at zero, so at step 0 the
    model is exactly the pretrained one."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.05):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        return out + (F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scale).to(out.dtype)


def add_lora(root: nn.Module, r: int, alpha: float) -> int:
    """Wrap every nn.Linear under `root` (the language model's decoder layers). Returns the count."""
    n = 0
    for name, mod in list(root.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cname, LoRALinear(child, r, alpha))
                n += 1
    return n


# ----------------------------------------------------------------------------- pixels
def patchify(img: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """[N,3,H,W] normalised -> ([N*gh*gw, 3*2*16*16], gh, gw), Qwen2VLImageProcessor's layout."""
    n, c, h, w = img.shape
    gh, gw = h // PATCH, w // PATCH
    p = img.reshape(n, c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    p = p.permute(0, 2, 5, 3, 6, 1, 4, 7)                       # n, gh/m, gw/m, m, m, c, P, P
    p = p.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, TEMPORAL, -1, -1)
    return p.reshape(n * gh * gw, c * TEMPORAL * PATCH * PATCH), gh, gw


def test_patchify(res: int = 64) -> None:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    x = torch.randn(3, 3, res, res)
    ref, gh, gw = Qwen2VLImageProcessor.patchify(None, x, PATCH, MERGE, TEMPORAL)
    got, gh2, gw2 = patchify(x)
    assert (gh, gw) == (gh2, gw2) and torch.equal(ref.reshape(-1, ref.shape[-1]), got), "patchify != processor"


# ----------------------------------------------------------------------------- the reader
class QwenStudy(nn.Module):
    def __init__(self, qwen: nn.Module, ids: dict, res: int, slot_names: list[str], n: int = len(LABELS)):
        super().__init__()
        if res % (PATCH * MERGE):
            raise ValueError(f"res {res} must be a multiple of {PATCH * MERGE}")
        self.qwen, self.res, self.ids, self.slot_names = qwen, res, ids, slot_names
        self.encoder = qwen                     # train_windows logs encoder size through this name
        self.n_slots = 0                        # slot/depth enter as text headers and order, not embeddings
        hid = qwen.config.text_config.hidden_size
        self.encoder.num_features = hid
        self.tokens_per_image = (res // PATCH) ** 2 // (MERGE * MERGE)
        self.norm = nn.LayerNorm(hid)
        self.w = nn.Parameter(torch.zeros(n, hid))
        self.b = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.w, std=0.02)
        self.register_buffer("mean", torch.tensor(QWEN_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(QWEN_STD).view(1, 3, 1, 1), persistent=False)

    # -- the token sequence for one study, from its windows' slots (already in anatomical order)
    def sequence(self, slots: list[int]) -> tuple[list[int], list[int], list[int]]:
        I = self.ids
        seq, mm = list(I["prefix"]), [0] * len(I["prefix"])
        cur = None
        for s in slots:
            if s != cur:
                h = I["slot"][s]
                seq += h
                mm += [0] * len(h)
                cur = s
            seq += [I["vision_start"]] + [I["image"]] * self.tokens_per_image + [I["vision_end"]]
            mm += [0] + [1] * self.tokens_per_image + [0]
        base = len(seq)
        seq += I["suffix"]
        mm += [0] * len(I["suffix"])
        return seq, mm, [base + j for j in I["label_pos"]]

    def prep(self, x: torch.Tensor, gain: torch.Tensor | None) -> torch.Tensor:
        x = x.float().div_(255.0)
        if gain is not None:
            x = (x * gain).clamp_(0, 1)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def one(self, x: torch.Tensor, slot: torch.Tensor, pos: torch.Tensor, gain) -> torch.Tensor:
        order = sorted(range(len(slot)), key=lambda i: (int(slot[i]), float(pos[i])))
        o = torch.tensor(order, device=x.device)
        x, sl = x[o], [int(slot[i]) for i in order]
        pix, gh, gw = patchify(self.prep(x, gain))
        seq, mm, lp = self.sequence(sl)
        dev = x.device
        out = self.qwen(input_ids=torch.tensor([seq], device=dev),
                        mm_token_type_ids=torch.tensor([mm], device=dev, dtype=torch.int32),
                        pixel_values=pix,
                        image_grid_thw=torch.tensor([[1, gh, gw]] * len(x), device=dev),
                        use_cache=False)
        h = out.last_hidden_state[0, lp].float()                 # (12, hid)
        return (self.norm(h) * self.w).sum(-1) + self.b

    def forward(self, x, mask=None, gain=None, slot=None, pos=None):
        b, k = x.shape[:2]
        if slot is None or pos is None:
            raise ValueError("QwenStudy orders windows by slot and depth: pass slot and pos")
        outs = []
        for j in range(b):
            keep = mask[j] if mask is not None else torch.ones(k, dtype=torch.bool, device=x.device)
            idx = keep.nonzero().flatten()
            g = gain[j] if gain is not None else None
            outs.append(self.one(x[j, idx], slot[j, idx], pos[j, idx], g))
        return torch.stack(outs)

    # -- optimiser groups: vision at bb_lr, LoRA at lora_lr, the head at head_lr
    def param_groups(self, bb_lr: float, head_lr: float, lora_lr: float) -> list[dict]:
        vis, lora, head = [], [], []
        for n_, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if ".visual." in f".{n_}":
                vis.append(p)
            elif "lora_" in n_:
                lora.append(p)
            else:
                head.append(p)
        return [{"params": vis, "lr": bb_lr}, {"params": lora, "lr": lora_lr}, {"params": head, "lr": head_lr}]


# ----------------------------------------------------------------------------- building
def slot_headers(slot_meta: list[dict] | None, n_slots: int) -> list[str]:
    if not slot_meta:
        return [f"\nSeries {i + 1}:" for i in range(n_slots)]
    return [f"\n{m.get('plane', 'Series')}, {'fluid-sensitive' if m.get('fluid') else 'other'}:" for m in slot_meta]


def token_table(tok, headers: list[str]) -> dict:
    enc = lambda s: tok.encode(s, add_special_tokens=False)                    # noqa: E731
    special = {k: tok.convert_tokens_to_ids(t) for k, t in
               [("vision_start", "<|vision_start|>"), ("vision_end", "<|vision_end|>"), ("image", "<|image_pad|>")]}
    suffix, label_pos = [], []
    head = enc("\nWhich findings are present?<|im_end|>\n<|im_start|>assistant\n")
    suffix += head
    for j, name in enumerate(NAMES):
        suffix += enc(name if j == 0 else ", " + name)
        label_pos.append(len(suffix) - 1)
    return {"prefix": enc("<|im_start|>user\nKnee MRI, one study."), "slot": [enc(h) for h in headers],
            "suffix": suffix, "label_pos": label_pos, **special}


def tiny_config():
    from transformers import Qwen3_5Config
    return Qwen3_5Config(
        text_config=dict(hidden_size=64, num_hidden_layers=4, intermediate_size=128, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=32, linear_value_head_dim=32, vocab_size=248320,
                         layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
                         rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25,
                                              mrope_section=[3, 3, 2], mrope_interleaved=True),
                         tie_word_embeddings=True),
        vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=64,
                           patch_size=PATCH, spatial_merge_size=MERGE, temporal_patch_size=TEMPORAL,
                           num_position_embeddings=2304, deepstack_visual_indexes=[]),
        image_token_id=248056, video_token_id=248057, vision_start_token_id=248053, vision_end_token_id=248054,
        tie_word_embeddings=True)


def build(arch: str, weights_dir, res: int, slot_meta: list[dict] | None, n_slots: int,
          lora_r: int = 16, lora_alpha: float = 32.0, fp16_frozen: bool = False, grad_ckpt: bool = False,
          config: dict | None = None, tokenizer_dir=None) -> QwenStudy:
    """Qwen3.5 base (no LM head) + LoRA + the per-finding head. `config` rebuilds without a download
    (inference: the saved config + last.pt); otherwise qwen35vl fetches the checkpoint once."""
    from pathlib import Path
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5Model
    if arch == "qwentiny":
        cfg, src = tiny_config(), None
    elif config is not None:
        cfg, src = Qwen3_5Config.from_dict(config), None
    else:
        from huggingface_hub import snapshot_download
        src = snapshot_download(HF_ID[arch], revision=HF_REVISION[arch], cache_dir=str(Path(weights_dir) / "hf"),
                                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        cfg = Qwen3_5Config.from_pretrained(src)
    cfg._attn_implementation = "sdpa"
    cfg.vision_config._attn_implementation = "sdpa"
    cfg.text_config._attn_implementation = "sdpa"
    load_dtype = torch.float16 if fp16_frozen else torch.float32
    if src is not None:
        qwen = Qwen3_5Model.from_pretrained(src, config=cfg, dtype=load_dtype)
    else:
        qwen = Qwen3_5Model(cfg).to(load_dtype)
    qwen.visual.float()
    for name, prm in qwen.named_parameters():
        if name.endswith(("A_log", "dt_bias")):
            prm.data = prm.data.float()
    tok_src = tokenizer_dir or src or HF_ID.get(arch, "Qwen/Qwen3.5-2B")
    tok = AutoTokenizer.from_pretrained(tok_src)
    for p in qwen.language_model.parameters():
        p.requires_grad_(False)
    n_lora = add_lora(qwen.language_model.layers, lora_r, lora_alpha)
    if not fp16_frozen:
        qwen.float()
    if grad_ckpt:
        qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    headers = slot_headers(slot_meta, n_slots)
    m = QwenStudy(qwen, token_table(tok, headers), res, headers)
    m.n_lora = n_lora
    return m


@torch.no_grad()
def finite_check(model: QwenStudy, x: torch.Tensor, slot, pos, dtype=torch.float16) -> dict:
    """Run one study with every decoder layer's output recorded: the largest |activation| per layer,
    and whether anything went non-finite. fp16 tops out at 65,504."""
    peaks = []
    hooks = [l.register_forward_hook(lambda m_, i_, o_: peaks.append(float((o_[0] if isinstance(o_, tuple) else o_)
                                                                         .detach().abs().float().max())))
             for l in model.qwen.language_model.layers]
    try:
        dev = x.device.type
        with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
            y = model(x[None], slot=slot[None], pos=pos[None])
    finally:
        for h in hooks:
            h.remove()
    return {"layer_peak_abs": peaks, "max": max(peaks), "finite": bool(torch.isfinite(y).all())}
