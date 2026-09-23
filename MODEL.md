# The model

**Target: one model reaching ~0.92 gold-58 alone, before any ensembling.**
Reference point: the public frontier does exactly that — 0.9167 gold-58, 0.924 LB, **single model,
no ensemble, no TTA**. Our 20-member pilkwang ensemble reads 0.84. The gap is architecture, and
the three pieces of it are named below.

---

## 0. ⛔ The frontier's TRAINING recipe is public too — read 2026-09-23, code not description

This file copied the frontier's **input** (§2 is dreaddevelopment's slot table, verbatim). It never
copied its **training**, and that is public: `dreaddevelopment/knee-mri-training-the-twelve-finding-model`
(~670 lines, one script), inference `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`,
weights `dreaddevelopment/raptor-knee-widedense` / `raptor-knee-maxspan`. Their single model now
reads **LB 0.928** (maxspan: 64 slices over 2–98%) — above our entire 3-family blend (0.914).
Licences (checked 09-23 via `kaggle datasets metadata`): corpus parts 1–2, `widedense`, `maxspan`
and their soft labels `dreaddevelopment/rsna-knee-labels` are all **CC0-1.0**. The script's own
licence is not in the API; Kaggle's default for public notebooks is Apache 2.0 — confirm on its page.

| | ours (`train_port`, `runs_port_repro` 0.7597) | frontier (0.924 → 0.928 LB, one model) |
|---|---|---|
| input | 6 slots × **one** 3-slice window at depth 0.5 = 18 slices | 5 slots, **64 slices over 6–94%** (2–98% for 0.928) |
| windows / study | 1 per slot | train: **12 random** windows per study per step; eval: **every** window (42–62) |
| crop | 160 mm (runs_port) | 140 mm by `PixelSpacing`, stored 336, fed at 384 |
| backbone | DINOv2-S, last 6 blocks, lr 8e-6 | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, **fully fine-tuned**, lr 3e-5 / head 1e-3 |
| head | `SlotHead`, one shared pool | **per-finding attention MIL**: 12 softmax attentions over windows, per-finding linear |
| loss | BCE on `steven_v2` | BCE on LLM **soft** labels, `pos_weight = clip((1−p)/p, 1, 10)` |
| schedule | 30 ep, last epoch, one seed | 16 ep, OneCycle 15% warm-up, bs 8, clip 3.0, bf16, grad-ckpt |
| augmentation | rotation, scale, shift, intensity | **intensity ±10% only**; no flips (laterality is signal) |
| selection | last epoch | best epoch on gold-58 **+ top-3 weight average** — ⚠️ violates our §3 rule 1, so their 0.9167 gold read is selected; the LB number is not |

**Their own ablation trail** (inference-notebook comments + dataset cards, their numbers, LB):
CoAtNet alone **0.914** on a 3,200-study corpus; 3-arm blends of it 0.914–0.915 ("ensembling is worth
~+0.001 there"); retrained on all **4,349** studies → **0.924**; from a later 0.926, the same 44 slices spread
over a wider span → **0.917** (worse); **64** slices over that span → **0.928**. *Widening the span pays only if density
is kept.* Two levers moved their single model +0.014 LB: **more training studies** and **dense wide
coverage**. Neither is an architecture.

**What this means for C6.** Stage A's cache is their geometry already. Stage B's encoder
(DINOv2-small, partial unfreeze) and Stage C's ViT-L are **not** their recipe; the shortest path to
an arm of our own at public strength is their training script on our cache with our labels, and
Stage B/C become ablations *off that*. Their script also takes `--ckpt <ssl checkpoint>` and any timm
arch, which makes it the natural harness for a fair OrthoFoundation test (C-5 was frozen, 160 mm,
ImageNet-normalised — `DECISIONS.md`).

---

## 1. What is wrong with the current model, measured

| | ours (`slot_cache` + `SlotHead`) | frontier | evidence |
|---|---|---|---|
| slices per study | **18** | **64** | manifest vs their §"fixed input" |
| where in the stack | **`depth: 0.5`, the middle** | **6–94%, spread** | our manifest: every slot `depth 0.5` |
| crop | 160 mm → 0.476 mm/px | 140 mm mm-crop | §R, worth +0.027 measured |
| pooling | one shared pool over 6 slots | **per-label attention over ~42 windows** | their §"how the model reads" |
| gold-58 | 0.84 (20 members) | **0.9167 (one model)** | §U, their header |

**Slice coverage is the largest and it was never tested.** The lateral meniscus and the collateral
ligaments sit in the outer thirds of a sagittal stack. We sample the centre slice of each series
and nothing else. §U measured Lateral Meniscus **0.6422** and MCL **0.6433** — the two structures
furthest from `depth 0.5` — while ACL, a midline structure, reads 0.87.

⛔ **§U's "the model cannot localise the compartment" is superseded.** The lateral head is not
mislocalising; it is being shown slices that do not contain the lateral meniscus. Same symptom,
different cause, and this one has a cheap fix.

---

## 2. Input representation

**Five slots, 64 slices, fixed order, always the same shape.**

| slot | slices | series preference |
|---|--:|---|
| SAG-1 | 18 | sagittal, **prefer fluid-sensitive** |
| SAG-2 | 14 | sagittal, **prefer NOT fluid-sensitive** |
| COR-1 | 12 | coronal, prefer fluid-sensitive |
| COR-2 | 8 | coronal, second series |
| AX | 12 | axial |

Fluid-sensitive sequences show effusion, synovitis and acute injury; T1/PD show anatomy and
cartilage. The twelve findings split across both, so **mixing the preference across slots is
deliberate** — do not make all five fluid-sensitive.

**Sampling within a series: evenly across 6%–94% of the stack.** Not the middle. This is the single
most important line in this file.

**Missing slot → zeros + an explicit mask bit.** The model is told to skip it. Never substitute a
different plane; §9h's rule is that feeding a slot its neighbour runs perfectly and scores wrongly.

**Geometry:** crop a **140 mm** box about the image centre using DICOM `PixelSpacing`, then resize
to **336**. Crop in millimetres, never in pixels — a knee then occupies the same fraction of frame
at 0.3 and 0.5 mm/px, and the model is not asked to learn scale differences that carry no medical
information.

---

## 3. Windowing — 2.5D

Three **consecutive sampled** slices stack into the three channels of one image. The backbone sees a
little above and below the centre slice: most of the benefit of 3D at the cost of 2D.

⚠️ **Sampled, not neighbouring. Decided 2026-09-23.** A slot's slices are spread evenly over 6–94% of
the stack (`slots64.pick_depths`), so a window's channels are adjacent in the cache but **0–3 slices
apart in the stack**. They are true neighbours only when the stack is about 1.1× the slot's slice
count, 2–3 apart on a deep one, and repeated on a short one (`slots64.source_step`). This section
used to say "neighbouring", and that was never what the code did. The sampled form is kept on
purpose: it is the fork's convention (`pilkwang_pixels.band_indices`, 12 samples grouped in threes),
and that arm reads 0.8516. True (c−1, c, c+1) neighbours around each centre are the untested
alternative, at 3× the cache (~95 GB vs ~32 GB). Stage A records `source_step` per series so the
choice can be audited afterwards (AUDIT.md).

Stride 1 within a slot, dropping the two edge positions, gives
`(18−2)+(14−2)+(12−2)+(8−2)+(12−2) = 54` windows. The frontier scores 42, so it drops some; **start
at 54 and tune down if memory binds.** Windows carry a `(slot, position)` embedding so the head
knows where each came from.

---

## 4. Backbone

Interchangeable — the head below is what matters. Train the same head over three, blend later:

| | why |
|---|---|
| **CoAtNet-ish, 336** | what the frontier uses; conv stem gives locality a plain ViT lacks on fine structure |
| **DINOv2-small, 336** | known-good here, cheap, reproduces our existing path |
| **OrthoFoundation-L** | **the differentiator** — DINOv3-L continued-pretrained on 1.25 M knee images, in no public blend. Fine-tune, never frozen: 0.7710 was frozen |

Fine-tune the **last 6 blocks**, `lr_backbone 8e-6`, `lr_head 1e-3` — the settings the fork's own
20 members used.

---

## 5. The head — per-label attention `this is the part that matters most`

For a study with `W` windows and backbone features `h_w ∈ R^d`:

    for each label k in 1..12:
        e_wk = v_k^T tanh(W_k h_w + b_k)        # label-k score for window w
        e_wk = e_wk + log(mask_w)               # missing slots -> -inf
        a_wk = softmax_w(e_wk)                  # attention over windows, PER LABEL
        c_k  = sum_w a_wk h_w                   # label-k context
        y_k  = sigmoid(u_k^T c_k + b'_k)

**Twelve separate attention weight sets, one per finding.** An ACL tear may be visible on two
sagittal slices while osteoarthritis is spread across many coronal ones. A single shared pool
forces those to share one notion of which slices matter — which is precisely what `SlotHead` does
today, and precisely why the outer-slice findings fail even once the slices are present.

⚠️ **Coverage alone will not fix Lateral Meniscus. Coverage + per-label attention will.** Adding
outer slices to a shared pool dilutes the midline findings that currently work. Ship the two
together or neither.

`a_wk` is also free interpretability: for any study you can show which slices drove each finding —
and it is the direct test of whether Lateral Meniscus now attends to lateral-depth windows.

---

## 6. Loss

Soft-target BCE against the LLM probabilities. A hedged report becomes 0.8, not 1.0 — forcing
hedged sentences to hard labels throws away calibration the reports actually carry.

    L = mean_k  BCE(y_k, t_k)      t_k in [0,1], NOT thresholded

- **No label smoothing on top.** The targets are already soft; smoothing twice flattens signal.
- **Loss floor is ~0.204, not 0.** A run approaching 0 is memorising, not learning.
- **Do not reweight labels toward the "headroom" ones.** §T shows the model already beats its own
  labels on 5 of 12 — upweighting a label whose targets are the noisy part amplifies noise.
- Mask any label whose target is missing; never impute.

---

## 7. Training

| | |
|---|---|
| folds | 5, **site-grouped** (`folds_site.csv`) — plain folds leak ~0.024 (§2j) |
| epochs | 20–30, cosine, warmup 1 |
| batch | studies, not windows — every window of a study must share a forward pass or the attention is meaningless |
| precision | fp16 on Kaggle, fp32 on MPS |
| aug | flips **only if laterality is canonicalised first**; else in-plane rotation ±10°, intensity jitter. ⛔ No vertical flip — it moves the patella |
| slot dropout | 0.2, as now — makes missing-slot masking in-distribution |

**Validation is `score_oof.py` large-n, gold-58 for a sanity read only.** Never select on gold-58
(§3b: claims +0.0137, delivers −0.0034).

---

## 8. Inference

Score all 54 windows per study, fp16, **retry a failed study in fp32** so nothing is ever dropped
from the submission. No TTA in v1 — the frontier reaches 0.924 without it, and TTA is a late
+0.008 at best (§3p). Budget is 9 h and decode is 97.6% of it (§3e); the head is free.

---

## 9. Build order, and the gate on each

| # | change | expected | gate |
|---|---|---|--:|
| 1 | **slice coverage 6–94%, 64 slices** | the big one | Lateral Meniscus + MCL **+0.05 each**, large-n, paired |
| 2 | **per-label attention head** | the other big one | macro **+0.02** over shared pooling, same backbone, same cache |
| 3 | 140 mm mm-crop | +0.027 measured | already established (§W) — fold it in, do not re-test |
| 4 | OrthoFoundation-L backbone | unknown | parity with the DINOv2 arm, or park permanently |

**1 and 2 ship together** (see §5). 3 comes free with the rebuild. 4 is the only speculative item
and it is last.

**If 1+2 land, one model should reach ~0.90 gold-58 on its own** — which is what the frontier gets
and what no arm here has ever produced.
