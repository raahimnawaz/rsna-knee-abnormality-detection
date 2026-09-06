# Decisions ledger

One line per settled question. **Check the tag before trusting a row.**

| tag | meaning |
|---|---|
| `CLOSED` | measured on a valid instrument. Do not re-derive. |
| `CONFOUNDED-160` | scored from a model trained on `slot_cache` tiles at 0.476 mm/px. **The verdict does not stand.** See §R below. |
| `OPEN` | live |

Evidence for every row is in `docs/archive/IMPROVEMENTS.md` and `docs/archive/PLAN.md` at the
section named. Those files are provenance — **not maintained, not read in full.**

---

## §R. THE RESOLUTION CONFOUND — read this before anything else `2026-09-06`

Every model this project trained on the `slot_cache` path was trained at the **wrong resolution**,
and it was one unset parameter.

`slot_cache.py`'s protocol slots carried `box_mm=None`, so `crop_box` returned the full 457 px
resample grid — **160 mm across 336 px = 0.476 mm/px**. The fork's `manifest.json` is unanimous
across all 20 members that average **0.8398**: `crop_mm=130.0` → **0.387 mm/px**. 23% finer, on
the six millimetre-scale labels this project measured itself losing.

`slot_cache.py:27` had already written down the cost: *"At 160 mm across 336 px a tear line is
about one pixel."*

**What was measured, with the instrument named on each row** (they are not directly comparable —
§9e's routing rule still binds):

| pixel path | crop | mm/px | arm | macro | instrument |
|---|--:|--:|---|--:|---|
| `slot_cache` tiles | 160 mm | 0.476 | `runs_cnn` RadImageNet R50 | 0.6924 | fold-0 OOF, n=691 |
| `slot_cache` tiles | 160 mm | 0.476 | `runs_port` dinov2-small | 0.7323 | fold-0 OOF, n=691 |
| `slot_cache` tiles | 160 mm | 0.476 | `runs_port_lr1` | 0.7358 | fold-0 OOF, n=691 |
| `slot_cache` tiles | 160 mm | 0.476 | `runs_of` OrthoFoundation-L | 0.7710 | large-n, n=4,349 |
| `dinov3_pixels.py` own path | **130 mm** | **0.387** | DINOv3 one model | **0.8025** | gold-47 |
| their DICOM path (shipped OOF) | 130 mm | 0.387 | pilkwang | 0.8469 | large-n, n=4,349 |
| their DICOM path (shipped OOF) | 130 mm | 0.387 | `ft_b` / RadImageNet | 0.8522 / 0.8514 | gold-47 / gold-47 |

**The one local arm that cropped at 130 mm — `dinov3_model.py:58`, `CROP_MM = 130.0` — is the one
local arm that escaped the band.** That is suggestive, not decisive: 0.8025 is a gold-47 number and
the band figures are not, so it is a different instrument and cannot be read as a clean rung.

**What this does and does not license.** It does **not** show any of these arms would reach 0.84;
no run at 130 mm has finished. It shows that *"five encoders, one band, therefore the encoder is
not the lever"* compared five models that shared a defect none of them was tested without. The
conclusion may still be true. It has not been measured.

⚠️ **On re-opening the arms below.** The stop rules were honoured and that discipline stands
(§3u, §3z-3) — a result is not renegotiated because it disappointed. This is the other case: the
**instrument** is now known to be miscalibrated, which is grounds to re-measure independently of
which way the number went. Do not let this become a habit of reopening losses.

---

## CONFOUNDED-160 — verdicts that rest on a 0.476 mm/px model

| # | claim as written | why it does not stand |
|---|---|---|
| §2y | our port earns no ensemble slot, −0.111 at 15.4σ | the port was 160 mm. Already partly reopened by §2y-2 |
| §2w step 4 | closed | reopened by §2y-2, 2026-08-19 |
| §3w-2 | Workstream C refuted at 4.8σ; *our training* is refuted | **both arms were 160 mm**, so the CNN-vs-ViT *paired* delta (+0.0399 ± 0.0084) survives as a relative result; the absolute verdict does not |
| §3y-3 | Stage 0 control 0.7358, Stage 1's bar is 0.7428 | the bar was set from a 160 mm control, so it is the wrong bar |
| §C-5 | OrthoFoundation-L fails its gate at 0.7710 | frozen features from a 160 mm cache. Best-matched encoder available; **re-measure before the arm stays parked** |
| §9e / `5d8f6fb` | five encoders, one band, so the encoder is not the lever | all five shared the defect |
| §3i-3 | our pixels are ≈34% of scrambled, hence the band | separately refuted on its own terms by §3i-6 (0.0165 of perturbation costs 0.0013) |

## CLOSED — valid instrument, do not re-derive

| route | verdict | where |
|---|---|---|
| crops as extra TTA windows (F2-cheap) | dead on both instruments; report −0.0031, gold-47 −0.0038, pre-registered | §3k, §3m |
| blending the fork's shipped `merge_gain` arm | +0.0007, 0/12 labels | §3h-2 |
| K16 from DICOM header rules | 56.9–60.8% vs ~50% chance | §2n, §2m |
| our rule extractor as a label source | last of six, 0.777 vs a free 0.893, 0/12 labels | §2f |
| per-label fusion of public label readers | near-duplicates (\|r\| 0.87–0.95); loses to best single | §2i |
| rank-mean `pilkwang` + `prvsiyan` | prvsiyan already contains pilkwang | §3c |
| re-fitting blend weights on site-grouped folds | −0.0000 | §2z |
| harmonising scanner/site away (ComBat-style) | −0.013 to −0.032 — case mix is real | §3f |
| selecting anything on gold-58 | claims +0.0137, delivers −0.0034 | §3b |
| post-hoc calibration / thresholds / priors | AUC is invariant to per-label monotone transforms | §3a |
| forking prvsiyan for its 0.906 | needs a private dataset; degrades to 0.899 | §3c |
| a C++ port for the efficiency track | 93 s per 0.001 AUC; decode is already native | PLAN §6.1 |
| slice **reversal** as a decode fault | null: +0.0007 at 1.5σ, 13/20 members | §3i-7 |
| the residual's *magnitude* explains the band | refuted: 0.0165 of perturbation costs 0.0013, band gap is 0.076 | §3i-6 |
| tensorised images at submission time | structurally impossible, not merely costly | PLAN §9g |

## Standing cautions — these were paid for

- **A submission costs ~2 h, not 74 s.** ~15 runs/week against a 30 h quota. Batch them. §3e
- **A leaderboard number older than a few days is not evidence.** Five beliefs about the outside
  world have expired here; one lasted 24 hours. Re-survey before any submission decision. §2x, §3l-1
- **Launch long runs under `caffeinate -i`.** ~2 of fold 0's 3.6 h were once spent asleep. §2v
- **Read a competitor's code, never its description.** Three claims about `pilkwang` came from its
  description; two were wrong.
- **Baseline is macro 0.7229 ± 0.0048**, site-grouped report-OOF over 2,612 studies. Anything from
  before 2026-08-10 is inflated by ~0.024 of site leakage. §2j
- **Price a future arm's LB gain at ~half its gold gain.** Gold +0.0135 and large-n +0.0102 at 8.8σ
  both delivered +0.006 shipped. §4c-4
- **Never price a forward pass by parameter count — `bench` it.** 303M/22M predicted 14×; measured
  6.33×, and the absolute cost was +1.9 min on a 9 h ceiling. §C-5
- **`data/tiles336` is `SAGITTAL_LR=0`; anatomical slabs need `SAGITTAL_LR=1`.**
  `assert_caches_compatible()` raises on any run consuming both.
- **RadImageNet's trunk is `CC-BY-NC-SA-4.0`** and is currently shipped knowingly. §4c-3
- **OrthoFoundation has no LICENSE file.** The only grant is the authors' email of 2026-08-18
  (Dingyu Wang / Dong Jiang) confirming competition use. **Preserve that email.** §C-5

## The failure mode this project repeats

A real measurement gets made, then used to answer a question it does not address.

- the training-loss curve, against a ranking gate — `[[loss-curve-is-not-evidence]]`
- the pixel residual's magnitude, against the band gap — §3i-6
- the memorisation probe, whose full dynamic range is 0.0102, against a 0.076 effect — §3i-7
- `crop_mm` read from `pilkwang_pixels.py`, a module `train_port.py` does not import — §2y-2
- and §R: seven runs sharing one defect, read as evidence about encoders

**Before believing a number, name the question it can and cannot answer.**

---

## §S. BOARD SURVEY `MEASURED 2026-09-06` — supersedes every earlier ceiling claim

Downloaded leaderboard, 3,239 teams. Every notebook below is confirmed against its author's own
LB row, per the standing rule.

| | 2026-08-18 | **2026-09-06** |
|---|--:|--:|
| teams | 1,904 | **3,239** |
| our 0.914 | rank 547 | **rank ~1374** (42nd pct) |
| rank 100 | 0.920 | **0.940** |
| rank 500 | 0.910 | **0.936** |
| top | — | **0.954** |

**0.914 lost 827 places without changing.** The board did not drift; it re-formed.

**The 0.936 plateau is one public notebook.** Eight unrelated authors sit at *exactly* 0.9360 —
`anhadmahajan06` (527), `kunaldesale2408` (512), `tamerlanomralinov` (390), `evgendvorkin` (564),
`ieshanmeghani` (640), `aadigupta7686` (709), `nishantkharga` (696), `anvithpothula` (752).
That is the free ceiling and **we are 0.022 under it — the fifth time (§3l-1, §4a).**

**The frontier is public too.** `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`
**0.9410, rank 92**, with its training notebook published 2026-09-06.
`mattiaangeli/bend-the-knee-to-the-dinosaurs-all-public` **0.9390** is an all-public blend of
pilkwang + prvsiyan + `raptor-*` + RadImageNet heads.

⚠️ **`pilkwang` — the fork this entire project is built on — is now rank 899 at 0.9350.**

### ✅ §R IS CONFIRMED EXTERNALLY, FROM THE FRONTIER NOTEBOOK'S OWN SOURCE

`knee-mri-twelve-findings-from-a-single-model.py:92-93`:

    IMG = 336
    CROP_MM = 140.0

and its line 53 states the mechanism outright — *"Every slice is cropped to a 140 mm box around
the centre of the image using the pixel spacing... so the model is not asked to learn scale
differences that carry no medical [meaning]."* The all-public blend carries `CROP_MM` of **130.0
and 140.0** at IMG 224/336.

**The entire public field crops in millimetres at 130–140 mm. Our training path was at 160 mm.
Nobody else is. §R was not a local hypothesis — it is the field's settled practice, and we are the
only ones outside it.**

### What this does to the plan

**The gap that matters is 0.004, not 0.026.** Running the free notebook reaches 0.936; the prize
line is 0.940. The OrthoFoundation route was priced against a 0.026 gap that does not exist.

1. **Get onto the public baseline first.** 0.914 → ~0.936 is available for the cost of a submission.
2. **0.936 is not a placing** — it ties several hundred teams and will shake out on the private LB.
   Beating the public blend needs an arm the field does not have.
3. **The only such asset we hold is OrthoFoundation-L** — knee-pretrained DINOv3-L, free, absent
   from every public blend surveyed, +1.9 min on a 9 h ceiling, and **never yet evaluated at a
   130–140 mm crop.** That is the differentiator, and §R is why its 0.7710 does not price it.

---

## §T. THE LABEL-NOISE CEILING, MEASURED `2026-09-06`

**The question:** is 0.97 reachable, or is it above what the available labels can teach?

**The instrument.** Competition ground truth is *image-read*, not report-derived — the official
overview states every study was labelled independently by **two subspecialty MSK radiologists with
a third adjudicating** (`REFERENCE.md` §2). The 58 gold studies in `train.csv` are that same
standard. So scoring `targets.csv` — the LLM report-extracted labels we actually train on — **as
if they were predictions** against gold-58 measures exactly what a model that perfectly learned its
training targets would score.

    macro 0.8873   bootstrap 95% CI [0.8505, 0.9195]   n=58, 12/12 labels scorable

| label | ceiling | | label | ceiling |
|---|--:|---|---|--:|
| ACL | 0.9933 | | PF OA | 0.9009 |
| MCL | 0.9637 | | Lateral Meniscus | 0.8621 |
| Medial Meniscus | 0.9543 | | Effusion | 0.8534 |
| Baker's | 0.9457 | | Contusion | 0.8306 |
| Medial OA | 0.9310 | | Lateral OA | 0.8298 |
| | | | Fracture | 0.7931 |
| | | | **Synovitis** | **0.7903** |

**The weak labels are weakest exactly where §resolution said we lose** — the meniscus/effusion/
synovitis group — and near-perfect on ACL. This is not uniform noise and it should not be modelled
as such.

### ⚠️ WHAT THIS DOES AND DOES NOT SAY

A model **can** beat its own training labels against clean truth — noise partly averages out. So
0.8873 is **not** a hard cap. It is the point at which further agreement with `targets.csv` stops
buying agreement with a radiologist, and every gain past it must come from denoising.

**Do not convert this to an LB number and act on it.** `score_gold.py` is explicit that the
gold→LB offset is the *smaller half* of the error: subset choice alone (gold-58 vs gold-47) swings
it +0.051 vs +0.039 on the same submission, and the F6 band missed because the gold read it was
applied to carried fold-recovery optimism. A naive `0.8873 + 0.039 ≈ 0.926` is **not** evidence
about the leaderboard, and the fact that it lands near the 0.936 plateau (§S) is **suggestive of
convergence, not proof of it** — the same error §3i-6 records, a real measurement aimed at a
question it cannot answer.

### THE HONEST READ ON A 0.97 TARGET

- The top of 3,239 teams is **0.954**, and ranks 1–100 span **0.014**.
- Our training labels reach **0.887** against a radiologist on the one image-read set we hold.
- So 0.97 is not "more of the same, harder". It requires **beating the report-extracted labels by a
  wide margin against image truth** — finding signal in pixels that the reports never carried.

That is a real research thesis and it names its own mechanism, which is more than "aim high" does.
It also says where to aim: **Synovitis 0.790, Fracture 0.793, Lateral OA 0.830, Contusion 0.831.**
Four labels carry most of the recoverable headroom, and macro-AUROC weights all twelve equally.

---

## §U. WHERE THE HEADROOM ACTUALLY IS: COMPARTMENT LOCALISATION `MEASURED 2026-09-06`

⛔ **This corrects §T's closing paragraph, which was mine and was wrong.** §T ranked labels by
*lowest ceiling* and named Synovitis / Fracture / Lateral OA / Contusion. That is the wrong
statistic. A low ceiling means the labels are noisy there, **not** that the model has room. The
right statistic is **ceiling − model**: how far below what the labels could teach the model sits.
Measured against pilkwang's shipped OOF on gold-58, bootstrapped 2,000 draws:

| label | ceiling | pilkwang | gap | 95% CI | |
|---|--:|--:|--:|---|---|
| **Lateral Meniscus** | 0.8621 | **0.6422** | **+0.222** | [+0.070, +0.385] | **HEADROOM** |
| **ACL** | 0.9933 | 0.8689 | **+0.123** | [+0.043, +0.224] | **HEADROOM** |
| **Lateral OA** | 0.8298 | 0.7079 | **+0.123** | [+0.012, +0.260] | **HEADROOM** |
| Medial Meniscus | 0.9543 | 0.8498 | +0.106 | [−0.005, +0.233] | |
| PF OA | 0.9009 | 0.8005 | +0.101 | [−0.019, +0.228] | |
| MCL | 0.9637 | 0.8844 | +0.078 | [−0.014, +0.196] | |
| Synovitis | 0.7903 | 0.7419 | +0.049 | [−0.076, +0.180] | |
| Baker's / Contusion / Medial OA | | | ~0 | spans 0 | at ceiling |
| Effusion | 0.8534 | 0.9429 | −0.089 | [−0.203, +0.007] | above its labels |
| Fracture | 0.7931 | 0.8944 | −0.100 | [−0.260, +0.036] | above its labels |

**Only three gaps clear zero at n=58, and one dominates.** Note also that the model **beats its own
training labels** on Fracture, Effusion, Contusion, Baker's and Medial OA — denoising is real and
already happening, so §T's "0.887 is not a hard cap" is confirmed by direct observation.

### THE MECHANISM: THE MODEL DETECTS THE FINDING AND NOT THE COMPARTMENT

Cross-scoring every prediction against every truth:

| pred \ truth | Medial Meniscus | Lateral Meniscus | | pred \ truth | Medial OA | Lateral OA |
|---|--:|--:|---|---|--:|--:|
| Medial Meniscus | **0.8498** | 0.5441 | | Medial OA | **0.9504** | 0.6925 |
| Lateral Meniscus | **0.6683** | 0.6422 | | Lateral OA | **0.8636** | 0.7079 |

**In both pairs the LATERAL head predicts MEDIAL truth better than it predicts its own.** Lateral
OA reads medial truth at 0.8636 against 0.7079 for its own compartment.

⛔ **It is NOT a flip.** Swapping the pair makes both worse (meniscus 0.746 → 0.606; OA 0.829 →
0.778), so §2n's reversed direction bit is not simply mis-signed. The failure is weaker and more
damaging: **the model detects that a finding is present and does not localise which compartment it
is in.** The lateral head is largely a shadow of the medial one plus a prevalence prior.

### WHY THIS IS THE BEST TARGET AVAILABLE

- **It is the largest measured gap in the project**, and it sits on labels macro-AUROC weights
  equally with the eight that are already near ceiling.
- **The field cannot be working on it.** Every public arm surveyed in §S is a general-purpose
  encoder blend, which lifts all twelve labels roughly uniformly. A compartment fix is targeted and
  is worth more per unit of effort than anything uniform.
- **The instruments already exist here.** §3z-4 flagged handedness as an unresolved confound with
  no cross-check; §C-4 already observed that `aagatti/nnunet_knee` segments medial and lateral
  meniscus separately and would give **a second independent instrument on exactly this axis, with
  no cache rebuild**. That was funded as a box calibrator. §U says its real value is here.

### ⚠️ THE RULE THIS MUST NOT BREAK

`n=58`, and the CIs are wide — Lateral Meniscus is somewhere in [+0.07, +0.39].
**This set generates the hypothesis. It must not be used to select anything**
(`[[never-select-on-a-tiny-set]]`, §3b: a choice made on 58 studies claimed +0.0137 and delivered
−0.0034). Any compartment fix is validated on the large-n instrument or on the leaderboard, with
its gate written down first.

---

## §V. THE COMPARTMENT EXPERIMENT — PRE-REGISTERED BEFORE ANY NUMBER `2026-09-06`

### 0. ⛔ FIRST: THERE IS NO CLEAN GATE, AND THAT IS THE HARDEST PART OF THIS DESIGN

pilkwang's per-label AUROC on the **large-n report instrument** (n=4,349, gold excluded) against
its reads on **gold-58 image truth**:

| | large-n (reports) | gold-58 (image) | difference |
|---|--:|--:|--:|
| Medial Meniscus | 0.8604 | 0.8498 | +0.011 |
| **Lateral Meniscus** | **0.7969** | **0.6422** | **+0.155** |
| Medial OA | 0.8779 | 0.9504 | −0.073 |
| **Lateral OA** | **0.8442** | **0.7079** | **+0.136** |

**The compartment asymmetry is far smaller against reports than against images.** Medial−lateral
is **0.064** on the report instrument and **0.208** on gold. The medial labels move little between
instruments; **both lateral labels read ~0.15 higher against reports.**

The most likely reading is that **the reports share the model's confusion** — an LLM extracting
"meniscal tear" from prose inherits whatever localisation the prose carries. If so, the large-n
instrument is **biased against this specific fix**: a model that genuinely learns to localise gets
credited only for the part the reports also localise. That is `[[measure-in-the-target-currency]]`
precisely — the precise, high-n instrument is the biased one here.

⚠️ And the alternative reading is live: **n=58 with 23 lateral-meniscus positives**, so part of the
0.155 is sampling. **Both readings are on the table and Stage A does not depend on which is true.**

### Stage A — the mechanism test. Label-free, so no instrument problem at all `RUN THIS FIRST`

§U's diagnosis is causal — *"the model detects the finding and does not localise the compartment."*
**Test it causally, with no labels and therefore no small-set and no currency problem.**

For each study, three forward passes through the fork's own weights on `tiles336_pilk`:
`FULL`, `MASK-LEFT` (left half of every tile zeroed), `MASK-RIGHT`.

Then per study compute the response of each head: `Δ_M = FULL − MASK` for Medial Meniscus,
`Δ_L` for Lateral Meniscus, on each occlusion side.

**⛔ The test needs no direction bit.** It asks only whether the two heads respond *differently*
to the two occlusions. Define, per study:

    S = (Δ_M[left] − Δ_M[right]) − (Δ_L[left] − Δ_L[right])

- **S ≈ 0 corpus-wide → the heads are not compartment-specific.** §U confirmed causally. Proceed.
- **|S| large and consistent → the heads DO localise**, and the gold-58 gap is something else —
  label noise, or lateral findings being genuinely harder. **§U's mechanism is refuted; stop and
  re-diagnose rather than building a fix for a problem that is not there.**

**PRE-REGISTERED BAR:** run on **all 3,599 cached studies, one member, paired bootstrap**. §U is
confirmed only if `mean|S|` is **under half** the mean absolute single-head response
`(|Δ_M| + |Δ_L|)/2`. Written before the run; not to be renegotiated after it.
⚠️ Zeroing is out-of-distribution for the backbone. Pair every arm against a **`MASK-NEITHER`
control that zeroes the same pixel count in a random location**, or the effect is measuring
"tile was tampered with" rather than "compartment was hidden" (§3y-3's resampling lesson).

### Stage B — nnU-Net as the direction instrument `ONLY IF A CONFIRMS`

`aagatti/nnunet_knee`, MIT, `fold_1` only. It segments medial and lateral meniscus **separately**,
which is a second independent read on the axis §3z-4 left with no cross-check at all.

- **Gate B1 (free):** their shipped `test_prediction.nii.gz` reproduces. Costs nothing.
- **Gate B2 (the real risk):** OAI-ZIB is **DESS sagittal**; this corpus is clinical
  multi-protocol. **Read agreement per plane × sequence, never pooled** — a pooled number that
  looks fine can be carried entirely by the sagittal fluid-sensitive slots
  (`[[aggregates-hide-their-composition]]`).
- **Gate B3:** nnU-Net-derived handedness vs §2n's K16 direction bit. §2n measured that bit at
  **56.9–60.8%** against ~50% chance. If nnU-Net agrees at ~57%, the two instruments corroborate
  and the bit is weak-but-real; if nnU-Net is decisive where the bit is not, **it replaces it**.

### Stage C — the fix, and what it must clear

Only after A and B. Whatever the fix (canonicalised handedness, compartment-specific crops,
mask-guided pooling), the gate is **pre-registered here, now**:

1. **Not gold-58.** It generated the hypothesis (§U). §3b: a choice made on these 58 claimed
   +0.0137 and delivered −0.0034. `[[never-select-on-a-tiny-set]]`
2. **Primary gate: large-n, per-label, on Lateral Meniscus and Lateral OA specifically** —
   paired against the same arm without the fix, same folds, same members. Macro is the wrong
   readout: eight of twelve labels are near ceiling and will dilute a real compartment gain to
   invisibility.
3. **Pre-register the attenuation.** Per §0 above, the report instrument may show roughly a third
   of the true image-truth effect. **The bar is therefore +0.02 on each lateral label at large-n,
   not the +0.12–0.22 gold suggests.** Setting the bar at the gold number would fail a working fix.
4. **Confirm in the deployment config before believing it** (`[[revalidate-in-deployment-config]]`)
   — an offline per-label A/B measures an estimator, not a shipped arm.
