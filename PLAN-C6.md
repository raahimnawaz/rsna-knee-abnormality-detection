# C6 — best single model, then the blend that actually wins

**Goal: the best model this project can build.** Not a prize, not an efficiency entry. That
changes what gets spent: the efficiency track rewarded a small fast encoder, and this plan spends
parameters and T4 hours instead.

Written 2026-09-07 after the local corpus was deleted. **Everything runs on Kaggle**, where the
competition DICOMs mount directly. Nothing is downloaded, and nothing is built from a
repackaging — D3 is why. (⛔ D3's reason was amended on 09-23, see `DECISIONS.md`. The DICOM route stands
because the corpus is gone, not because the NIfTI files cost 0.085.)

---

## 0. Two premises the record contradicts, corrected before planning on them

**"pilkwang is the best arm" is false.** Four arms sit at parity on the only honest instrument
(fold-resolved gold-47):

| arm | gold-47 | note |
|---|--:|---|
| `ft_b` | **0.8522** | strongest single |
| pilkwang | 0.8516 | ships its own OOF **and a per-member fingerprint** — that is its real value |
| tonylica's RadImageNet | 0.8514 | §4b |
| public RadImageNet folds_v1 | 0.8486 | §4c, χ² p=0.442 flat |

**The blend is the model.** 3-family blend **0.8932** against 0.8522 for the best single — the
single largest gain in the record, larger than any architecture delta. So the deliverable is not
"one great arm", it is *the best arm at the centre of a blend that has room for it.*

**Licences are not a blocker** (DECISIONS.md): competition §2.5.a explicitly carves out
incompatibly-licensed pretrained models. ⚠️ But **OrthoFoundation has no LICENSE at all** — the
only grant is the authors' email of 2026-08-18. **Preserve that email**; it is the whole claim.

---

## 1. What is being fixed, and the measured cost of each

| # | defect | cost | fixed in |
|---|---|--:|---|
| **D3** | ~~pixels are a third-party NIfTI repackaging~~ ⛔ amended 09-23: the files are fine; the port saw one centre window per series | **0.085** @ 14.1σ, cause unmeasured | `RUNBOOK.md` §2b, then Stage A |
| **§2e** | frozen encoder cannot adapt; everything lands near 0.70 | caps the project | Stage C |
| **D2** | 18 slices, every slot at `depth 0.5` | Lat.Men **0.6422**, MCL **0.6433** vs ACL **0.8689** | Stage A |
| **§R** | 160 mm crop → 0.476 mm/px | **+0.0268** @ 4.1σ | Stage A |
| **§5** | one shared pool across all 12 labels | never measured | Stage B |

You said the data is the number-one thing. The record agrees numerically: **D3 alone (0.085) is
larger than every architecture delta combined.** (⛔ Its cause was amended 09-23. Whether the input
explains it is `RUNBOOK.md` §2b's question.) Stages 0 and A are therefore the only stages
that cannot be skipped or reordered.

⛔ **Coverage and per-label attention ship together** (`MODEL.md` §5). Outer slices fed to a shared
pool dilute the midline findings that already work. Gate B is written so coverage-only cannot pass.

---

## Stage 0 — measure the data `CPU only, minutes, no GPU lottery`

**Nothing downstream may be decided before this runs.** Every item here is currently an
assumption, including two I was about to build on.

**0.1 Anisotropy — the fact that decides whether a 3D arm is even admissible.**
Per plane, over all series: median in-plane `PixelSpacing`, `SliceThickness`,
`SpacingBetweenSlices`, and the ratio. `FINDINGS.md` §6.4 confirms all four tags are present
200/200, so this is a header pass — **no pixels decoded.**

> **PRE-REGISTERED DECISION RULE.** Let *r* = median(through-plane ÷ in-plane).
> - **r ≥ 4** → isotropic 3D is **contraindicated**; see §2 below. Any 3D arm must use
>   anisotropic kernels, and Triad/MedicalNet-style isotropic encoders are excluded.
> - **r < 4** → an isotropic 3D arm is admissible and §2's exclusion is void.
>
> Fixing this *before* the number exists is the point. §3y and §3z were both pre-registered and
> both survived; §3t's verdict was not, and had to be retracted.

**0.2 Stack depth per plane.** Median/p10 slices per series. If the median sagittal stack is ~20,
then "18 slices across 6–94%" is nearly the whole stack and D2's fix is much cheaper than assumed
— and `slots64.py`'s repeat-padding behaviour becomes load-bearing rather than a corner case.

**0.3 Protocol composition.** Studies having each plane × fluid-sensitivity, and the resulting
five-slot fill rate. The `slots64.py` spec assumes two usable sagittals and two coronals per
study. **That is an assumption and it is untested.**

**0.4 Re-run the public-source survey.** `kaggle datasets list -s "rsna knee"`. The standing
caution is explicit: the 08-10 survey was 27 days stale by 09-06 and had missed **four** label
tables, one a different model family. Any arm list — including this plan's — expires.

**0.5 Re-pull the leaderboard.** "A leaderboard number older than a few days is not evidence."
Five beliefs about the outside world have already expired here; one lasted 24 hours.

**GATE 0:** 0.1–0.3 written into `DATA.md` as measured tables. **No AUC is produced or quoted at
this stage.**

---

## 2. The 3D arm — researched, and currently excluded on evidence

You asked me to research rather than invent one. I did, and the honest finding is that **the
strongest candidate fails on your data's physics, and a from-scratch 3D CNN repeats your own
diagnosed failure.**

**Triad** ([Nature/ScienceDirect 2026](https://www.sciencedirect.com/science/article/abs/pii/S1361841526000617)) is the leading open 3D-MRI foundation model — 129 K volumes, CC-BY-4.0,
+4.38% over Swin-B-scratch on classification. Two disqualifiers, both checked not assumed:
- **Anatomy: brain, breast and prostate only.** No musculoskeletal, no knee, and no MSK task in
  its downstream evaluation.
- **It requires isotropic 1 mm resampling to 256×256×128.** Clinical knee MRI is 2D fast spin
  echo, multi-plane. If Stage 0.1 returns r ≈ 10, isotropic resampling *interpolates ~10× data
  that was never acquired* along the slice axis. That is not a 3D reconstruction, it is
  invention — and it is upstream of the model, i.e. exactly the class of defect D3 is.

**A 3D CNN trained from scratch is the failure the record already paid for.** §3w-2 trained a
RadImageNet to **0.6924** while a **0.8514** one was a free download, and §4b's verdict was that
*our training* was refuted, not the architecture. 4,407 studies is not enough to train a 3D
encoder that beats a pretrained 2D one. [[diversity-is-not-the-constraint]] — screen on strength.

**If Stage 0.1 returns r ≥ 4** (which the clinical protocol makes likely), the principled 3D route
is not a video-model or an isotropic foundation model but **anisotropic kernels** — nnU-Net's
spacing-adaptive scheme: `(1,3,3)` convolutions and in-plane-only pooling until the axes equalise.
You already have nnU-Net wiring in `pipeline/nnunet_gate1.py`. **Deferred to Stage E, and only if
Stage D leaves a diversity gap that a parity-strength arm could fill.**

The field's convergence is corroborating rather than decisive: knee-MRI work from MRNet onward is
predominantly 2.5D, and reviews attribute that to exactly this anisotropy ([systematic review](https://link.springer.com/article/10.1007/s00330-024-11105-8), [KneeXNet-2.5D](https://www.nature.com/articles/s44401-026-00072-5)).

---

## Stage A — the cache `8 kernels, ~9 GPU-h`

`slots64.py` as specified, **rebuilt against Stage 0's measurements** — slot counts and the 6–94%
range are provisional until 0.2/0.3 land. From competition DICOMs, 140 mm mm-crop, 336 px, uint8,
64 slices. Slices cached, not windows (triples share 2 of 3 slices: 32 GB vs 95 GB, and Kaggle
caps kernel output at 20 GB). Geometry written into the manifest — §R happened because `box_mm`
was recorded nowhere.

**GATE A** — plumbing only, no AUC (§3b, [[never-select-on-a-tiny-set]]):
1. manifest records `fov_mm 140`, `depth_range`, and the resolved spacing.
2. Slot fill ≥ Stage 0.3's prediction, per slot. A slot that quietly stops filling reads as a
   coverage gain and is a coverage loss.
3. Render 20 studies at the 6% and 94% positions: real anatomy, not wrap or air.

---

## Stage B — do the data fixes pay? `2 kernels, ~14 GPU-h`

Encoder held at the fork's dinov2-small so **cache and head are the only variables.** Running an
encoder change first would confound a C6 gain with an encoder gain exactly as §3w-2 confounded
domain with architecture.

| arm | cache | head |
|---|---|---|
| **B0** | C6 | shared pool |
| **B1** | C6 | **per-label** |

Anchor `runs_port` = **0.7323 ± 0.0086**, fold 0, n=681, `score_oof.py`.

**GATE B:**
- **B0 ≥ 0.7591** — the anchor plus the crop's own +0.0268. B0 also gets coverage and correct
  pixels; failing to clear the crop alone refutes D2 on this data.
- **B1 − B0 > 0**, paired. First measurement of §5.
- **Lateral Meniscus and MCL must both rise** vs 0.6422 / 0.6433. Macro can rise while the two
  structures D2 named stay flat — that would mean the gain came from elsewhere and D2's mechanism
  is unproven.

⛔ **Training loss is not evidence.** Workstream C's loss fell monotonically through 36.7% of its
range and lost by 4.8σ. Only `score_oof.py` large-n decides.

**STOP** if B0 < 0.7591 — re-open D2 before spending T4 hours on a 303 M encoder.

---

## Stage C — the encoder, and knee pretraining `2 kernels, ~15 GPU-h`

Only after Gate B. `OrthoNet` already exists and unfreezes 6 blocks; it has never run.

| arm | encoder | differs by |
|---|---|---|
| **C0** | `vit_large_patch16_dinov3`, stock | — |
| **C1** | **OrthoFoundation-L** | **1,251,655 knee images. Nothing else.** |

`orthofoundation_model.py:1` maps OrthoFoundation-L onto timm's `vit_large_patch16_dinov3`, so C1
*is* C0 continued-pretrained. Same architecture, params, RoPE, head, cache, folds, schedule.
**This is the ablation §3w-2 could not give**, where domain and architecture moved together.

External corroboration, not evidence: knee-MRI-specific pretraining significantly outperformed
generic and general-medical pretrained weights on MRNet ([MedNet-FS, Sci Rep 2026](https://www.nature.com/articles/s41598-026-54489-x)).

### ⛔ C-PRE: resolve the normalisation from the checkpoint, or C1 is a false negative

`train_port.py:274-275` hardcodes ImageNet constants as buffers inside `OrthoNet`:

    self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])...)
    self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225])...)

and `fusion/orthofoundation_model.py` resolves **no** normalisation at all. Those constants are
correct for DINOv2. **They are an assumption for OrthoFoundation-L**, which was continued-
pretrained on 1.25 M greyscale knee MRIs and whose intensity distribution is not ImageNet's.

This is not hypothetical. A competitor hit exactly this swapping in RAD-DINO (ViT-B/14 finetuned
from dinov2-base on 882,775 chest radiographs, `image_mean` **0.5307**, `image_std` **0.2583**
vs ImageNet's 0.485/0.229). Their arm lost from epoch 1 and the gap widened every epoch, with
*lower* training loss — fitting worse, not overfitting. Their words: left to finish, the run
*"would have produced a real number, from a real 11-hour experiment, supporting a false
conclusion about the exact hypothesis we most wanted to test."*
<https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154>

**That is our hypothesis and our experiment.** An encoder fed inputs shifted from its pretraining
distribution underperforms silently — no error, no warning, a plausible curve pointing the wrong
way. It would read as "knee pretraining does not transfer" and be recorded as a null.

**Required before C1 runs:** read `image_mean`/`image_std` from OrthoFoundation's own checkpoint
or config; if it ships none, **measure** them from its pretraining corpus description or fall back
to greyscale stats — and record which, because a guessed constant is a confound either way.
Assert C0 resolves to byte-identical ImageNet constants, so §2's comparison stays clean.

**GATE C:** **C1 − C0 > 0**, paired, fold 0, `score_oof.py` large-n — **never gold-58** (§3b:
claims +0.0137, delivers −0.0034). ⚠️ **A null is a real result** and must be recorded as one; §4a
was the record reading a null as a refutation.

Bench the forward pass, do not price it by parameter count — predicted 14×, measured 6.33× (C-5).

---

## Stage D — the blend, which is where the model actually is `~6 GPU-h`

Best single is 0.8522; the 3-family blend is **0.8932**. Equal-weight rank-mean over families,
§9e's pre-registered rule, nothing fitted — §2z measured that re-fitting blend weights on
site-grouped folds is **−0.0000**.

Members: C6's winner, `ft_b`, public RadImageNet folds_v1, pilkwang. **Screen on strength first**
— every arm the record rejected was genuinely weak (0.6924–0.8025) and every rejection was
correct. Price the LB gain at ~half the gold gain (4c-4).

---

## ⛔ These prices are wrong — see `PREMORTEM.md` §1–2

Stage C's ~15 GPU-h comes from a benchmark measured at **6 tiles/study** (16.95 ms/tile,
101.71 ms/study) applied to a plan that uses **54 windows/study**. The corrected figure is
**50–160 GPU-h**, above the 30 h weekly quota and above the 12 h per-session cap. Gate B's bar and
the noise floor are also unsound. Re-price with a real bench before funding any GPU stage.

## Budget · 30 GPU-h/week

| stage | GPU-h | week |
|---|--:|--:|
| 0 measure | **0** (CPU) | 1 |
| A cache | ~9 | 1 |
| B0/B1 | ~14 | 2 |
| C0/C1 | ~15 | 3 |
| D blend | ~6 | 4 |

Fold 0 throughout; five folds only after a gate passes.

## Blockers

1. **`kaggle.json`** — `~/.kaggle/` holds only an `access_token`. `kaggle kernels push` cannot authenticate.
2. **Refresh `raahimnawaz/rsna-knee-code`** with `slots64.py` and `perlabel_head.py`, or kernels fail at import after the queue wait.
3. **Smoke every kernel at `--limit 20`.** Every cache build in this project's history died hours in on something a 20-study run would have caught.
