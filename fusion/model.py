"""The fusion head: PLAN.md 3.3 with the backbone slot already filled by frozen DINOv2.

    per series:  slices [S, 1536] -> proj -> +positional -> 2-layer transformer
                                  -> gated attention pool -> series embedding [D]
    study:       series [K, D] + series-type embedding (6 types, FINDINGS.md 3.1)
                              -> transformer -> gated attention pool -> head -> 12 logits

This is the whole differentiator. Everyone has the same DINOv2 weights, so the backbone cannot
separate us (PLAN.md 7.1); `DINO Protocol Fusion` existing as a distinct public notebook is a
hint that most forks pool naively across series, which is exactly the layer this replaces.

Two structural facts from the data drive the design:

  - 87.2% of studies are missing at least one of the six series types, and Axial FS is the only
    one present in every study (FINDINGS.md 3.2). Series dropout is therefore an augmentation
    the test distribution actually contains, not defensive padding, and every pool here is
    masked so a study with one series is a valid input rather than a crash.
  - Findings are plane-specific -- MCL is coronal, PF OA and Baker's axial, ACL and menisci
    sagittal, contusion needs fluid-sensitive. The series-type embedding is what lets the
    attention pool learn "trust the sagittal series for ACL", so it is added before series
    attention rather than concatenated at the end.

Masked attention pooling, not mean pooling: a meniscal tear occupies a handful of slices out of
24, and averaging over the volume dilutes it by ~20x.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LABELS = 12
EMBED_DIM = int(os.environ.get("EMBED_DIM", 1536))   # must match pipeline.preprocess.EMBED_DIM
# Reads the env for the same reason preprocess.py does, and the comment above was already the
# contract -- it just was not enforced anywhere. PLAN.md §C-5's ViT-L arm concatenates to 2048,
# and without this the head would build a 1536-wide projection and fail on the first batch.
# Default unchanged, so every existing checkpoint and run is unaffected.
N_SERIES_TYPES = 6


class GatedAttentionPool(nn.Module):
    """ABMIL-style gated attention pooling with a mask.

    tanh(Vh) * sigmoid(Uh) lets the gate suppress instances the tanh branch would otherwise
    score highly, which matters when most slices are uninformative background.
    """

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(d, hidden)
        self.U = nn.Linear(d, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x [B,N,D], mask [B,N] bool (True = real) -> [B,D]."""
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)   # [B,N]
        a = a.masked_fill(~mask, float("-inf"))
        # A row that is entirely padding would softmax to NaN. It should not occur -- a study
        # with no series is filtered upstream -- but a NaN here would silently poison the whole
        # batch's gradients, so make it a zero vector instead.
        empty = ~mask.any(dim=1, keepdim=True)
        a = torch.softmax(a.masked_fill(empty, 0.0), dim=1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1) * (~empty).float()


def _encoder(d: int, heads: int, layers: int, ff_mult: int = 4, dropout: float = 0.1):
    layer = nn.TransformerEncoderLayer(
        d_model=d, nhead=heads, dim_feedforward=d * ff_mult, dropout=dropout,
        batch_first=True, norm_first=True, activation="gelu")
    # norm_first=True makes the nested-tensor fast path unavailable anyway; saying so
    # explicitly keeps the run output clean, which matters because the per-run tables are how
    # this project detects bugs.
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class FusionHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM, d: int = 256, heads: int = 8,
                 slice_layers: int = 2, series_layers: int = 2, max_slices: int = 64,
                 dropout: float = 0.1, n_labels: int = N_LABELS):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU())
        # Learned rather than sinusoidal: slice index is a short, bounded, non-periodic axis.
        self.slice_pos = nn.Parameter(torch.zeros(1, max_slices, d))
        nn.init.trunc_normal_(self.slice_pos, std=0.02)

        self.slice_tf = _encoder(d, heads, slice_layers, dropout=dropout)
        self.slice_pool = GatedAttentionPool(d)

        self.type_emb = nn.Embedding(N_SERIES_TYPES + 1, d)   # +1 = unknown (plane == -1)
        self.series_tf = _encoder(d, heads, series_layers, dropout=dropout)
        self.series_pool = GatedAttentionPool(d)

        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_labels))

    def forward(self, feats, slice_mask, series_mask, series_type):
        """feats [B,K,S,Din] · slice_mask [B,K,S] · series_mask [B,K] · series_type [B,K] long
        -> logits [B, n_labels]."""
        B, K, S, _ = feats.shape

        x = self.proj(feats).view(B * K, S, -1) + self.slice_pos[:, :S]
        sm = slice_mask.view(B * K, S)
        # A fully-padded series still passes through the transformer; src_key_padding_mask with
        # an all-True row yields NaN, so give those rows a single valid position and discard the
        # result via series_mask below.
        sm_safe = sm.clone()
        sm_safe[~sm.any(dim=1), 0] = True
        x = self.slice_tf(x, src_key_padding_mask=~sm_safe)
        s = self.slice_pool(x, sm_safe).view(B, K, -1)

        s = s + self.type_emb(series_type.clamp(min=0) + (series_type < 0).long() * N_SERIES_TYPES)
        sem_safe = series_mask.clone()
        sem_safe[~series_mask.any(dim=1), 0] = True
        s = self.series_tf(s, src_key_padding_mask=~sem_safe)
        return self.head(self.series_pool(s, sem_safe))


def soft_bce(logits: torch.Tensor, targets: torch.Tensor,
             weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE against SOFT targets, per-study weighted.

    The extractor emits 0.95 / 0.65 / 0.45 / 0.03 / 0.08 rather than 0/1 (PLAN.md 2.3), and the
    competition metric is a ranking metric, so a hedged finding genuinely belongs between a
    clean negative and a confident positive. Hard-thresholding the targets here would discard
    exactly the gradation the extractor exists to produce.

    `weight` is [B] or [B,1]: down-weight studies whose labels came from a language the
    extractor is weak in, or where two extraction methods disagreed (PLAN.md 2.3, 8).
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weight is not None:
        loss = loss * weight.view(-1, 1)
    return loss.mean()
