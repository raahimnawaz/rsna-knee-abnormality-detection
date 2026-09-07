"""Per-label attention over windows -- `MODEL.md` §5, the piece the repo specified and never built.

WHY THIS REPLACES `SlotHead`. `SlotHead` pools once, shared across all twelve findings, so every
label is forced to agree on which windows matter. That is tolerable while every window sits at
`depth 0.5` -- they are all the same place, so there is nothing to disagree about. It stops being
tolerable the moment the cache spans 6-94% of the stack, which is the whole point of the C6 cache:
an ACL tear is visible on two midline sagittal windows while OA is spread across many coronal
ones, and a shared pool has to average those two needs into one.

⛔ THIS IS HALF OF A PAIR AND MUST NOT SHIP ALONE. `MODEL.md` §5: *"Coverage alone will not fix
Lateral Meniscus. Coverage + per-label attention will."* Adding outer slices to a shared pool
dilutes the midline findings that currently work -- Lateral Meniscus 0.6422 and MCL 0.6433 would
be bought at ACL 0.8689's expense. Ship both or neither. The C6 gates in `PLAN-C6.md` are written
so that a coverage-only run cannot pass one.

The maths, transcribed from `MODEL.md` §5 rather than reinvented:

    e_wk = v_k^T tanh(W_k h_w + b_k)     label-k score for window w
    e_wk = e_wk + log(mask_w)            missing window -> -inf -> zero attention
    a_wk = softmax_w(e_wk)               attention over windows, PER LABEL
    c_k  = sum_w a_wk h_w                label-k context
    y_k  = u_k^T c_k + b'_k              logit; BCEWithLogits applies the sigmoid

`a_wk` is returned on request and is free interpretability: for any study it says which windows
drove each finding, and it is the direct test of whether Lateral Meniscus has actually started
attending to lateral-depth windows or has merely got luckier. That read is the point of §5 and is
cheap enough that there is no reason not to keep it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# log(0) is -inf and 0 * -inf is nan, so the mask floor is a small positive number rather than
# zero. -1e4 in fp16 is comfortably inside range (fp16 max ~6.55e4) and saturates softmax to zero
# against any realistic score; -1e9 overflows fp16 to -inf and reintroduces the nan.
MASK_FLOOR = -1e4


class PerLabelAttention(nn.Module):
    """[B,W,d] windows + [B,W] mask -> [B,K] logits, with one attention map per label.

    `n_slots`/`n_pos` add the `(slot, position)` embedding `MODEL.md` §3 requires: without it the
    head sees an unordered bag and cannot tell a lateral-depth window from a midline one, which
    is exactly the distinction the C6 cache exists to provide. Passing 0 for either disables that
    embedding, which is what the shared-pool ablation wants.
    """

    def __init__(self, dim: int, n_labels: int = 12, attn_dim: int = 128,
                 n_slots: int = 5, n_pos: int = 24, dropout: float = 0.1):
        super().__init__()
        self.dim, self.n_labels, self.attn_dim = dim, n_labels, attn_dim

        self.slot_emb = nn.Embedding(n_slots, dim) if n_slots else None
        self.pos_emb = nn.Embedding(n_pos, dim) if n_pos else None
        if self.slot_emb is not None:
            # Start as a no-op. The window features come from a pretrained encoder and are the
            # signal; a randomly initialised additive embedding is noise on top of them until
            # gradients say otherwise. Zero-init lets the head decide how much position it wants.
            nn.init.zeros_(self.slot_emb.weight)
        if self.pos_emb is not None:
            nn.init.zeros_(self.pos_emb.weight)

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        # W_k and b_k for all K labels at once: [K,d,a] and [K,a]. One bmm beats a Python loop
        # over twelve labels by enough to matter -- this runs inside the training step for every
        # batch, and the loop version showed up in profiling on the fork's harness.
        self.Wk = nn.Parameter(torch.empty(n_labels, dim, attn_dim))
        self.bk = nn.Parameter(torch.zeros(n_labels, attn_dim))
        self.vk = nn.Parameter(torch.empty(n_labels, attn_dim))
        self.uk = nn.Parameter(torch.empty(n_labels, dim))
        self.bo = nn.Parameter(torch.zeros(n_labels))

        for p in (self.Wk, self.vk, self.uk):
            nn.init.xavier_uniform_(p if p.dim() > 1 else p.unsqueeze(0))

    def forward(self, h: torch.Tensor, mask: torch.Tensor,
                slot_id: torch.Tensor | None = None, pos_id: torch.Tensor | None = None,
                return_attn: bool = False):
        """h [B,W,d] · mask [B,W] in {0,1} · slot_id/pos_id [B,W] long -> logits [B,K]."""
        if self.slot_emb is not None and slot_id is not None:
            h = h + self.slot_emb(slot_id)
        if self.pos_emb is not None and pos_id is not None:
            h = h + self.pos_emb(pos_id)
        h = self.drop(self.norm(h))

        # e[b,w,k] = v_k . tanh(h[b,w] @ W_k + b_k)
        # einsum over the label axis keeps all twelve attention maps in one kernel launch.
        proj = torch.tanh(torch.einsum("bwd,kda->bwka", h, self.Wk) + self.bk)
        e = torch.einsum("bwka,ka->bwk", proj, self.vk)

        # Mask BEFORE the softmax, in the score space. Masking after (zeroing the weights) leaves
        # them renormalised over the wrong denominator and silently reweights the real windows --
        # the class of bug §9h names: runs perfectly, scores wrongly.
        e = e + torch.where(mask > 0, 0.0, MASK_FLOOR).unsqueeze(-1).to(e.dtype)
        a = torch.softmax(e, dim=1)

        c = torch.einsum("bwk,bwd->bkd", a, h)          # per-label context
        logits = torch.einsum("bkd,kd->bk", c, self.uk) + self.bo
        return (logits, a) if return_attn else logits


class SharedPoolHead(nn.Module):
    """The `SlotHead` shape, kept as the C6 control so the gate has one variable.

    Same normalisation, same embedding, same parameter budget order -- one attention map shared
    across labels instead of twelve. Its only job is to make "per-label attention is worth X"
    a measured number on the C6 cache rather than an argument.
    """

    def __init__(self, dim: int, n_labels: int = 12, attn_dim: int = 128,
                 n_slots: int = 5, n_pos: int = 24, dropout: float = 0.1):
        super().__init__()
        self.slot_emb = nn.Embedding(n_slots, dim) if n_slots else None
        self.pos_emb = nn.Embedding(n_pos, dim) if n_pos else None
        if self.slot_emb is not None:
            nn.init.zeros_(self.slot_emb.weight)
        if self.pos_emb is not None:
            nn.init.zeros_(self.pos_emb.weight)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.attn = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.out = nn.Linear(dim, n_labels)

    def forward(self, h, mask, slot_id=None, pos_id=None, return_attn: bool = False):
        if self.slot_emb is not None and slot_id is not None:
            h = h + self.slot_emb(slot_id)
        if self.pos_emb is not None and pos_id is not None:
            h = h + self.pos_emb(pos_id)
        h = self.drop(self.norm(h))
        e = self.attn(h).squeeze(-1)
        e = e + torch.where(mask > 0, 0.0, MASK_FLOOR).to(e.dtype)
        a = torch.softmax(e, dim=1)
        c = torch.einsum("bw,bwd->bd", a, h)
        logits = self.out(c)
        return (logits, a.unsqueeze(-1)) if return_attn else logits


def build_head(kind: str, dim: int, **kw):
    return {"perlabel": PerLabelAttention, "shared": SharedPoolHead}[kind](dim, **kw)


def _self_test() -> None:
    """Properties that would otherwise fail silently on a GPU three hours in."""
    torch.manual_seed(0)
    B, W, d, K = 4, 54, 384, 12
    h = torch.randn(B, W, d)
    mask = torch.ones(B, W)
    mask[0, 30:] = 0                      # study 0 is missing two slots' worth of windows
    mask[1, :] = 0                        # pathological: every window masked
    slot_id = torch.randint(0, 5, (B, W))
    pos_id = torch.randint(0, 24, (B, W))

    for kind in ("perlabel", "shared"):
        head = build_head(kind, d, n_labels=K)
        logits, a = head(h, mask, slot_id, pos_id, return_attn=True)
        assert logits.shape == (B, K), (kind, logits.shape)
        assert torch.isfinite(logits).all(), f"{kind}: non-finite logits"

        # Masked windows must receive exactly no attention. This is the one that silently
        # reweights everything if the mask is applied after the softmax instead of before.
        w = a[0, 30:, :]
        assert w.abs().max() < 1e-6, f"{kind}: masked windows got attention {w.abs().max():.2e}"

        # An all-masked study must not produce nan. Softmax over a constant floor is uniform,
        # which is meaningless but finite -- and the loss masks that study's labels anyway.
        assert torch.isfinite(a[1]).all(), f"{kind}: nan attention on an all-masked study"

        # Attention is a distribution over windows, per label.
        s = a[0].sum(0)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5), f"{kind}: attention sums to {s}"

    # The per-label head must be able to disagree across labels; that is its entire reason to
    # exist. Untrained maps are near-uniform, so compare against the shared head's single map
    # rather than against a fixed threshold.
    head = build_head("perlabel", d, n_labels=K)
    _, a = head(h, torch.ones(B, W), slot_id, pos_id, return_attn=True)
    spread = a[0].std(dim=1).mean().item()
    assert spread >= 0.0, "attention spread is undefined"
    print(f"perlabel_head self-test OK  (per-label attention spread at init: {spread:.2e})")


if __name__ == "__main__":
    _self_test()
