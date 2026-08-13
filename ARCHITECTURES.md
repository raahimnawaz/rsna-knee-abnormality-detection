# How every arm actually works

One place for the mechanics of the five external models this repo consumes. Written 2026-08-13
late, after §3s opened DINOv3 and found its config advertising two features it does not really
use.

**Why this file exists separately from `PLAN.md` and `IMPROVEMENTS.md`.** Those record *decisions*
and *measurements*. This records *mechanism* — what each arm is, how it is loaded, what pixels it
expects, and which of its conventions will break silently if crossed with another's. Three of the
errors in this project's history were a fact true of one arm being applied to a different one.

> ## ⛔ THE CROSS-ARM TRAP TABLE — READ THIS BEFORE TOUCHING ANY PIXEL PATH
>
> Every row differs between arms, and **every one of them is silent** — wrong values, right
> shapes. Nothing raises.
>
> | | pilkwang | `ft_b` | DINOv3 | tonylica |
> |---|---|---|---|---|
> | resolution | 336 | 336 | 336 | **224** |
> | slices / series | 12 (+TTA windows) | **32** | **16, as CHANNELS** | 9 |
> | slice band | (0.20, 0.80) | full stack | **(0.12, 0.88)** | (0.20, 0.80) |
> | crop | 130 mm centred | **background trim, `max>8`** | 130 mm centred | 160 mm |
> | window | per-series | per-series 0.5/99.5 | **per-SLICE 1/99** | per-series |
> | normalisation | — | **ImageNet mean/std** | **none, `uint8/255`** | — |
> | canonicalise to | **LEFT** knee | **RIGHT** knee | **LEFT** knee | LEFT knee |
> | sagittal laterality | reverse channel order | reverse channel order | **untouched** | reverse |
> | slice ordering | geometric (signed normal) | geometric | **`InstanceNumber`** | geometric |
> | slot scheme | 6 **header-parsed** | plane only (3) | 6 **(plane × fluid-sens.)** | 6 header-parsed |
>
> **The two that have already cost time:** `ft_b` canonicalises onto a *right* knee where
> everything else uses a *left* one (§3n), and DINOv3's slot indices are **not** pilkwang's, so
> `data/slots_pilkwang.csv` must never be fed to it — `enc.tok.weight` is indexed by that order
> and a wrong table conditions every series wrongly while running perfectly (§9h).

---

## 1. `pilkwang` — the 20-member baseline we run

`data/external/pilkwang_weights`, CC0, 1.54 GB. **Our banked 0.899 is this plus per-target TTA
pooling** (§3d/§3e).

* **Architecture** — DINOv2 **small**, `SlotHead`, `pool='cls_mean'`. `fusion/pilkwang_model.py`.
* **The 20 members are 5 folds × 4 seeds of ONE config** (from its `manifest.json`) — so the
  ensemble we run has **zero architectural diversity**. This is the fact that motivated the whole
  F6 route.
* **It ships its own OOF** — `oof.npz`, 368 KB, all 4,407 studies, `gold_mask`=58. Imported by
  `fusion/import_pilkwang_oof.py`. **This single file made §2y, §3b and §3f possible in an hour.**
* **It ships a per-member fingerprint**, and we match **20/20 at 7e-06** (§3h). *No other arm
  ships one*, which is why every later arm's transcription is weaker evidence than this one's.
* **Slots are header-parsed and reconstructable** — `SLOTS_RECOVERED` = `SAG_FLUID_FS`,
  `COR_FLUID_FS`, `AX_FLUID_FS`, `SAG_FLUID_NOFS`, `COR_T1`, `SAG_T1`. Its `annotate()` recovers
  them from `SeriesDescription`/`SequenceName`/`ScanOptions`/`ScanningSequence`/`RepetitionTime`/
  `EchoTime`/`PixelSpacing`, **all seven of which are columns of
  `data/external/dicom_headers_zhukovoleksiy.parquet`**. Result: `data/slots_pilkwang.csv`.
* **Honest fold-resolved gold-47: 0.8516.** The all-20 read on the same studies is **0.9990** —
  an inflation of **+0.1474** (§3n). *Never compare an all-member number to this one.*
* `CROP_MM`/`SLICE_BAND` are **unguarded inference-time knobs** — the fingerprint checks the
  weights, not the pixels (§3g). `img_size` does trip it.

## 2. `sadamtorres/rsna-ft-b` — the strong diverse arm, in the blend

`data/external/ft_b_dinov2_vitb14_336`, 1.61 GB. **Claimed 0.883 solo.**

* **Architecture** — DINOv2 **ViT-B/14 @336** with 4 registers, `FEAT='both'` (CLS ++ patch-mean,
  so `d_in = 2×768`), attention-pool over K=32 slices then mean within plane, MLP to 12.
  `fusion/ft_b_model.py`.
* **Self-contained**: each checkpoint carries `backbone` *and* `head`, `timm` built
  `pretrained=False`, so nothing downloads and no gated licence is touched.
* **It ships its own `oof_macro` = 0.7222**, on the same scale as our 0.7229 baseline (§2j) — the
  only time another team's local number has been directly legible here.
* **Honest fold-resolved gold-47: 0.8522**, Spearman vs pilkwang **0.632**, **blend +0.0284**
  (§3o). This is the arm that proved F6.
* **Canonicalises onto a RIGHT knee** — the opposite of everything else here. Mixing it with
  `pilkwang_pixels.normalise_laterality` is silent.

## 3. `mattiaangeli` DINOv3 ViT-S/16 — transcribed, being scored

`data/external/dinov3_vits16_folds`, 439 MB, 5 folds, 23.5 M params each.
`fusion/dinov3_model.py` · audited in `IMPROVEMENTS.md` §3s · spec in `PLAN.md` §9h.

**Loads strict on timm 1.0.28** although the dataset ships a `timm-1.0.22` wheel.

* **`stem: 'native'` means the 16 slices are INPUT CHANNELS.** `patch_embed.proj.weight` is
  (384, **16**, 16, 16), so the encoder is built `in_chans=16` and one series is **one** forward
  pass. Not 16 passes. Read the weights, not the config.
* **The slot token is inserted between the prefix tokens and the patches, and RoPE must be told.**
  `ViTSlotToken.__init__` bumps `vit.num_prefix_tokens` **and every block's
  `attn.num_prefix_tokens`** by one, because `EvaAttention.forward` applies rope to
  `q[:, :, npt:, :]` against a 441-entry table. **Omitting the bump raises loudly** (442 vs 441) —
  §9h feared a silent failure and it is not one.
* **`Net.forward` keeps `[CLS] ++ [slot_token] ++ patches`.** The delta attends over the slot
  token *together with* the patches. Dropping it from the KV is shape-neutral and silent.
* **`n_sites: 109` is a DEAD config field** with no parameter behind it. This arm does **not**
  condition on site; `enc.tok.weight` (7, 384) is 6 slot types + padding.
* **⛔ Two of its headline features are nearly inert (§3s).** The `xcodex` cross-attention is gated
  to ~0.001 and never left its zero init — deleting it costs **+0.0003 macro** and saves 9.7% of
  forward. Slot conditioning enters at **2.1%** of a patch token's magnitude. **Stripped, the arm
  is: ViT-S/16 → CLS per series → segment mean ++ max ++ presence → LayerNorm → Linear.**
* **Its slot usage is anatomically correct** (§3s leave-one-out): axial-fluid → Baker's/synovitis/
  effusion, sagittal-fluid → ACL, sagittal-structural → PF OA, coronal-fluid → fracture/medial
  meniscus. Best evidence we have that it reads anatomy.
* **⚠️ Its slice order is `int(InstanceNumber)`, which is recoverable NOWHERE locally** — the
  NIfTIs have no patient frame and the headers parquet is one row per *series*. Reversal costs
  0.675 of a fold swap. **Local scoring therefore uses direction TTA; the submission path does
  not, because Kaggle has the DICOMs.** Local gold is a *lower bound*. (§9h)

## 4. `tonylica/rsna2026-models` — DROPPED, kept here so it is not re-tried

* **Loads STRICT into our existing `pilkwang_model.build_model(pool='cls_mean_focal', prior=True)`**
  — 233/234 keys matched, the only extra being `head.slot_prior`. Same six slots as pilkwang.
* Pixel path is `pilkwang_pixels` at **`img 224`, `crop_mm 160`, `n_slice 9`**.
* **⛔ Dropped (§3q): 0.788 against `ft_b`'s 0.852, and *more* correlated (0.704 vs 0.632).**
  Weaker **and** more redundant; the 3-arm blend was **−0.0089**. The public notebooks include it
  and never checked. **Free to run is not a reason to include.**

## 5. RadImageNet ResNet-50 — the frozen arm, and the fine-tune candidate

* **As published it is HEADS ONLY** (`radimagenet_r50_heads`, 58 MB, `FoundationQueryHead`,
  `best_val 0.8167`); the encoder is a separate dataset, now pulled to
  `data/external/radimagenet_r50_encoder/ResNet50.pt` (90 MB).
* **Verified**: `backbone.*` keys load **strict** into a torchvision `resnet50` trunk, 23.5 M
  params, 2048-dim features, forward at 336 → (2048, 11, 11).
* **It ships `fold_sha256` but NOT `folds_v1.csv`**, which is published nowhere — so its own fold
  split cannot be recovered and it has no honest OOF.
* **As an F6 arm it is expected to fail** — frozen encoder is §2e's weak configuration, and its
  `best_val` matches tonylica's profile. **Its value is as Workstream C**: the same weights
  *fine-tuned* are the one family nobody in this competition is running (§9f-C).

---

## The three rules that generalise across all of them

1. **Read the weights, not the config.** `n_sites: 109` had no parameter; `stem:'native'` silently
   redefined what a slice is; `patch_embed.proj.weight`'s second dimension was the whole story.
   Configs carry dead fields and inherited defaults.
2. **A strict load proves shapes, not behaviour.** Only pilkwang ships a fingerprint. For every
   other arm, add checks a strict load cannot do — `dinov3_model.py --check` verifies the slot
   embedding is wired to something and that the five folds are five *different* models.
3. **All-member gold reads are inflated, and the inflation is large.** +0.1474 measured on
   pilkwang. Any absolute AUC in this repo is meaningless without saying whether it is
   fold-resolved. Only pilkwang **0.8516** and `ft_b` **0.8522** are honest numbers on gold-47.
