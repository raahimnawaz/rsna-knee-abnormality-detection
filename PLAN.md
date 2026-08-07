# RSNA Knee Abnormality Detection — Plan

**Metric:** macro-averaged AUROC over 12 labels
**Timeline:** started Jul 30 2026 · entry/merger deadline **Oct 15** · final submission **Oct 22 2026**
→ **~10.5 weeks from today (Aug 7).**
**Prizes:** main track 10 places ($9k down to $5k); **efficiency track 3 places ($7k / $6k / $5k)**
**Field as of Aug 6:** 2,180 entrants, 164 participants, 158 teams, 437 submissions — very early.
**Constraints:** Kaggle notebook, ≤9h, no internet, `submission.csv`. Winners must open-source code
**and weights**, publish to the forum, and record a video.

---

> **⚠ Updated 2026-08-06 after phase-1 EDA — see `FINDINGS.md` for measured numbers.**
> The picture is far more extreme than the data description implied:
> **58 of 4,407 training studies carry labels (1.3%)**, reports span **9+ languages**
> (60% non-English), and **every non-English language has ≤8 gold-labelled studies**
> (French: zero). Sections 0 and 2 below are corrected; §7 timeline now front-loads two
> items that weren't in the original plan: **hand-labelling a validation set** and
> **external pretraining data**.

## 0. What the data description actually tells us

Read carefully, two sentences reframe the whole competition:

> "**Only a small subset of training studies carry per-condition labels.** We also provide the
> original text of the radiology report **from which you may wish to derive the labels for the
> remaining studies.**"

and

> "Report — the free-text radiology report. **May be in any of several languages**, depending on
> the reporting institution."

### 0.1 Confirmed: no reports at test time
`train.csv` has `Report`; `test.csv` has **only** `StudyInstanceUID`. So text is a
**training-time-only** signal. The image model must stand alone at inference. ✅

### 0.2 This is a weakly-supervised competition wearing a vision competition's clothes
The organizers are telling you outright that the labeled set is small and that **you are expected
to manufacture the rest of your training labels from reports.** That means:

> **The report→label extractor is the critical path, not a phase-4 enhancement.**
> Your effective training-set size — and therefore your ceiling — is set by how well you pseudo-label.

Two teams with identical vision stacks will be separated almost entirely by label quality and
label *quantity*. This is where the competition is won.

### 0.3 The reports are multilingual, which is the real difficulty
"Any of several languages" kills the standard playbook (English regex + NegEx, or a
PubMedBERT/CheXbert-style English clinical model). You need one of:

- **A multilingual instruction-tuned LLM run offline on your own machine** over ~all training
  reports, prompted to emit structured JSON per study. This is the strongest option and it is
  entirely training-side, so the no-internet rule doesn't apply.
- Language-ID → per-language rule sets. Brittle, but a useful cross-check on the LLM.
- A multilingual encoder (XLM-R / mBERT / multilingual-e5) fine-tuned on the small labeled subset,
  used to propagate labels to the unlabeled reports. Cheap, and a good *ensemble partner* to the
  LLM — disagreement between the two flags studies worth manual review.

Do at least two of these and reconcile. Radiology reports are templated, so agreement will be
high and the disagreements are exactly the informative cases.

### 0.4 Series metadata is handed to you — free
`train_series.csv` / `test_series.csv` provide **`Fluid_Sensitive`, `Fat_Suppression`,
`Anatomical_Plane`** for every series, at both train *and* test time. This deletes what would
normally be a week of DICOM-tag reverse-engineering. Use these columns directly as the series
router and as embedding inputs to the pooling layer. (Spot-check them against
`ImageOrientationPatient` on a sample, but expect them to be correct.)

### 0.5 Scale
819,640 files, 569.76 GB. At a median of 30 slices/series that's roughly **27,000 series**;
at ~4–6 series/study, roughly **5,000–7,000 training studies** (confirm from `train.csv` row
count). **Test set is ~1,300 studies** — small, which matters a lot for §7.

---

## 1. Data logistics — solve this before anything else

**You cannot host this dataset locally.** 570 GB zip + 570 GB extracted ≈ 1.14 TB; this machine
has 56.6 GB free on C: and 293 GB on D:. The browser "Download All" will fail. Plan:

| Need | Where |
|---|---|
| `train.csv` (reports + labels), `train_series.csv` | **Local.** Tiny. `kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv` |
| Report parsing / LLM extraction / label modeling | **Local.** Text only — no images needed. This is the critical path and it runs on your own GPU. |
| Sample of ~300–800 studies for pipeline development | **D: drive.** Script the Kaggle API over a stratified subset of study folders (~30–80 GB). |
| Full-scale image training | **Kaggle notebooks** (data pre-mounted at `/kaggle/input`, no download, free T4/P100 quota) **or rented cloud GPU with fast NVMe.** |
| Preprocessed cache (resampled uint8 volumes) | Build **once** as a Kaggle Dataset; it will be a small fraction of 570 GB and makes every training run cheap. |

The cached-preprocessing step is what makes the rest of the competition tractable — treat it as a
deliverable, not an implementation detail. Aim for fixed-size uint8 arrays at ~256–320 px,
16–32 slices per series, only the series you actually feed the model.

---

## 2. Critical path: reports → labels (weeks 1–3)

### 2.1 Extract structured attributes, not just the 12 bits
Run a multilingual LLM offline over every training report; emit JSON per study:

- The **12 target labels**, each with a confidence/certainty level.
- **Severity/grade**: meniscus grade 1/2/3, mild/moderate/severe OA, effusion small/moderate/large,
  MCL grade I–III, cartilage grading.
- **Morphology**: partial vs complete tear, horizontal/radial/bucket-handle, root tear, ramp lesion.
- **Sub-location**: anterior horn / body / posterior horn; ACL proximal/mid/distal.
- **Laterality of the knee** if the report states it (cross-check against DICOM — see §3.2).
- **Certainty**: definite / probable / possible / "cannot exclude" / negated.
- **Extra findings** outside the 12: chondral defect, plica, loose body, bursitis, tendinopathy,
  hardware, prior ACL reconstruction. These become auxiliary heads.

### 2.2 Build your own validation set — the provided one is unusable
**Measured:** only **58** studies carry gold labels, the rarest positive class among them is
MCL at **n=9**, and the set is clearly *enriched* (positive rates 16–60%) rather than sampled.
Worse, per-language gold coverage is: English 28, Spanish 8, Turkish 6, Croatian 4, Dutch 4,
Greek 3, Russian 3, German 2, **French 0**.

You cannot validate a 12-label multilingual extractor on that. So:

> **Hand-label a stratified 250–400 reports spanning all 9 languages.** Machine-translate to
> read them — you are labelling, not practising fluency. This is the single highest-leverage
> manual task in the competition and most teams will skip it.

Then: use all 58 gold studies for measurement (never fitting), plus your own set, and report
per-label **and per-language** agreement. A single weak language silently poisons ~10% of the
corpus and you would never see it in an aggregate number.

### 2.3 Turn extraction into training signal
- **Soft targets from certainty**: definite→0.98, probable→0.85, possible→0.65,
  "cannot exclude"→0.55, negated→0.02. The metric is a *ranking* metric, so hedged findings
  genuinely belong between clean and definite.
- **Confidence-weighted loss**: down-weight studies where the extractor was unsure or where the
  LLM and the encoder model disagreed.
- **Ordinal severity heads** (CORAL / cumulative logit) — using expected grade as the ranking
  score often beats the plain binary head, because it orders the positives correctly.
- **Auxiliary heads** for the extra findings, loss weight ~0.2–0.5. Denser gradient per study,
  and it forces the encoder to represent anatomy the 12 labels ignore.

### 2.4 Curriculum
Pretrain on the large pseudo-labeled set → fine-tune on the small gold-labeled set. Keep the gold
subset in a held-out fold for honest evaluation of the *whole* pipeline.

---

## 3. Imaging pipeline

### 3.1 Preprocessing
1. Group by series (folders already do this); sort slices by `ImagePositionPatient` projected onto
   the slice normal — **not** `InstanceNumber`.
2. **Transfer-syntax spread is a real cost**: uncompressed, JPEG Lossless, JPEG 2000, Implicit VR.
   JPEG 2000 decode is slow. Install `pylibjpeg` + `pylibjpeg-openjpeg` + GDCM and **benchmark
   decode time per syntax early** — this drives §7.
3. MRI has no HU standard → per-volume robust percentile normalization (clip 0.5/99.5 → [0,1]).
4. Resample in-plane to fixed mm/px; centre-crop/pad to a fixed FOV. The knee is protocol-centred,
   so a detector is overkill.
5. Fixed slice count per series (16–32).
6. Cache as uint8 `.npy`.

### 3.2 Laterality — still the sharpest trap
Four labels are side-specific (Medial/Lateral Meniscus, Medial/Lateral OA). "Medial" flips between
left and right knees, so a model fed raw mixed-handedness studies sees medial findings on both
sides of the image.

**Canonicalize every study to one handedness.** DICOMs are stripped to an allowlist of 86 tags —
**first check whether `(0020,0060) Laterality` and `BodyPartExamined` survived.** If not, derive
handedness from `ImagePositionPatient`/`ImageOrientationPatient` sign conventions, or train a small
left/right classifier on pixels. Audit visually on a sample either way.

> **TTA trap:** horizontal-flip TTA is *invalid* here unless you also swap the Medial↔Lateral
> output pairs. Either swap them or don't use hflip.

### 3.3 Architecture
**2.5D CNN + slice transformer + series attention** — the family that has won recent RSNA
volumetric competitions.

```
per series:
  slices → groups of 3–5 adjacent as channels
        → 2D backbone (ConvNeXt-tiny / EfficientNetV2-s / MaxViT-tiny), shared weights
        → per-slice features [S, D]
        → 2-layer transformer over slice axis (+ positional encoding)
        → attention-pool → series embedding [D]

study:
  series embeddings [K, D]
        + series-type embedding from (Anatomical_Plane, Fluid_Sensitive, Fat_Suppression)  ← given!
        → transformer / gated attention pool over series
        → head → 12 logits (+ auxiliary heads)
```

Findings are plane-specific, so feed multiple planes: MCL is coronal; PF OA and Baker's are axial;
ACL and menisci are sagittal; contusion needs **fat-suppressed fluid-sensitive** series. A
sagittal-only model caps out.

**Measured series structure** (see `FINDINGS.md` §3):
- `Fluid_Sensitive` and `Fat_Suppression` are **perfectly redundant** (10,361 at 0/0, 14,010 at
  1/1, zero off-diagonal). Collapse to one flag → **6 series types**, not 12.
- Median 5 series/study (range 3–14).
- Coverage: Axial FS **100%**, Sagittal nonFS 96.8%, Coronal FS 96.4%, Sagittal FS 94.2%,
  Coronal nonFS 77.3%, **Axial nonFS 19.4%**.
- **87.2% of studies are missing at least one of the six types.** Axial FS is the only series
  you can always count on.

Augmentation: affine (±10°, ±10% scale/translate), intensity/gamma jitter, slice-axis jitter,
coarse dropout, bias-field, Rician noise, and **series dropout — mandatory, not defensive**,
given that missing series is the norm (§5). No hflip (§3.2).

### 3.4 External data — likely decisive
Only **4,407** training studies, nearly all pseudo-labelled. That is small for a 12-label 3D
task, and the rules explicitly permit "freely & publicly available external data, including
pre-trained models." Candidates, in rough order of value:

- **OAI (Osteoarthritis Initiative)** — thousands of knee MRIs with expert OA gradings.
  Directly supervises Medial OA / Lateral OA / PF OA, three of the twelve.
- **MRNet (Stanford)** — 1,370 knee MRIs labelled for ACL tear, meniscal tear, abnormality.
- **fastMRI+** — lesion annotations layered over the fastMRI knee corpus.
- **KneeMRI (Rijeka)** — ACL injury grades; plausibly the same source as the Croatian reports.

Pretrain on these → fine-tune on the pseudo-labelled competition corpus. Check each licence
against the competition's external-data rules and post the datasets to the forum thread as
required.

### 3.5 Upgrades in priority order
1. Aux heads + soft labels from §2 — cheapest real gain.
2. Per-plane specialist models + a light stacker on OOF predictions.
3. A 3D branch (X3D-M, R(2+1)D-18, Video Swin-T) for ensemble diversity.
4. Cross-modal CLIP-style image↔report alignment pretraining, then fine-tune the image tower alone.
   Multilingual text encoder required. Stretch goal — only after the baseline is solid.
5. Label-correlation stacker on the 12 OOF logits (Effusion↔Synovitis, Medial OA↔Medial Meniscus,
   ACL↔Contusion). Usually +0.003–0.008 macro AUC for ~zero runtime.

---

## 4. Validation

- **MultilabelStratifiedGroupKFold**, 5 folds, grouped by patient. Stratify on the rarest labels.
- Keep gold-labeled and pseudo-labeled studies **identifiable** in every fold; always report
  metrics on the **gold** subset — pseudo-labels inherit the extractor's biases and will flatter you.
- Studies are from a "diverse international mix of imaging sites" → **site shift is the #1 CV↔LB
  divergence risk.** Monitor per-manufacturer/per-language performance if that's recoverable.
- **Prevalence differs across train / public LB / private LB** (stated explicitly). AUC is
  prevalence-insensitive *within* a label, so this is survivable — but it means public LB will be
  noisy and will not track private. **Trust CV.**
- Bootstrap CIs per label. With ~1,300 test studies and a rare label like Fracture, the private
  AUC for that label has a wide CI — some of the final ranking is luck. Don't chase fold noise.
- Final two submissions: (a) best CV ensemble, (b) most robust / smallest CV–LB gap.

---

## 5. Inference notebook

- Decode test DICOMs on the fly; **multiprocess** it (CPU-bound, will otherwise starve the GPU).
  Overlap decode with GPU inference via a prefetch queue.
- AMP fp16, `channels_last`, optional `torch.compile`.
- **Handle missing series.** Some test studies will lack a plane. Train with **series dropout** so
  the attention pool is robust, and explicitly unit-test single-series and degenerate inputs. This
  is the classic submission-time crash.
- Exact header:
  `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`
  Note `Baker's` contains an apostrophe and several columns contain spaces — quote handling matters.
- Guard against NaN, clip to [0,1], and wrap per-study inference in try/except emitting 0.5 on
  failure so one bad DICOM can't zero the submission.

---

## 6. Efficiency track

### 6.1 The formula, verified from the competition page

    Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400        [minimize]

`Benchmark` = `sample_submission.csv` score, `maxAUC` = best private-LB AUC.

Note `Benchmark − maxAUC` is **negative** (≈ 0.5 − 0.85 = −0.35), so the first term rewards higher
AUC. The score is not normalized and will come out negative; only the ordering matters. It is an
affine transform of the normalized form `(maxAUC − AUC)/(maxAUC − Benchmark) + t/32400`, so
**both rank identically and both give the same exchange rate.**

`sample_submission.csv` is all-0.5 → every prediction ties → **Benchmark = 0.500 exactly.**
Assuming `maxAUC ≈ 0.85`:

- 1 second of runtime → `1/32400` = **3.09e-5**
- 0.001 macro-AUC → `0.001/0.35` = **2.86e-3**

> **Exchange rate: 0.001 macro-AUC ≈ 93 seconds.** (0.01 AUC ≈ 15.4 min.)
> Robust to `maxAUC`: 108 s per 0.001 at 0.80, 81 s per 0.001 at 0.90.

### 6.2 The strategic consequence — accuracy dominates here
**The test set is only ~1,300 studies.** At 3 series × 16 slices that's ~62,000 slices — a lean
pipeline should finish in **5–15 minutes**, i.e. a runtime term of only 0.009–0.028. Even a heavy
30-minute ensemble only costs 0.056.

The entire runtime range you'd plausibly operate in (5 min → 45 min) is worth about **0.074
efficiency units ≈ 0.026 AUC**. Typical ensemble gains are ~0.01 AUC ≈ 0.029 units. So:

- Lean beats heavy, **but only by a modest margin** — this is not a "shrink the model to nothing"
  competition.
- **Do not sacrifice meaningful accuracy for speed.** Dropping from 0.845 to 0.800 costs 0.129
  units; you could never recover that by getting faster, since the whole runtime budget is ~1.0
  and realistic runtimes cost <0.06.
- With only 3 efficiency prizes and a far less contested track, a **strong** model with a
  well-optimized I/O path is the play.

Worked examples (`maxAUC` = 0.85; showing the normalized form for readability — same ranking):

| Submission | AUC | Runtime | Acc. term | Runtime term | **Eff (norm.)** |
|---|---|---|---|---|---|
| Full ensemble + TTA | 0.850 | 5.0 h | 0.000 | 0.556 | 0.556 |
| Ensemble, tuned I/O | 0.850 | 30 min | 0.000 | 0.056 | 0.056 |
| **Single model, tuned I/O** | **0.845** | **12 min** | 0.014 | 0.022 | **0.037** |
| Single model, trimmed hard | 0.835 | 6 min | 0.043 | 0.011 | 0.054 |
| Tiny/fast, accuracy sacrificed | 0.800 | 3 min | 0.143 | 0.006 | 0.149 |

### 6.3 What to optimize, ranked
1. **DICOM decode throughput — free.** Zero AUC cost, and likely the majority of runtime given the
   JPEG 2000 / JPEG Lossless mix. Multiprocess, decode only the slices you feed, never touch pixel
   data twice.
2. **Series pruning** using the provided metadata: keep sagittal + coronal + axial fluid-sensitive,
   drop localizers and non-FS duplicates.
3. **Slice subsampling** (16 vs 32) — measure the OOF cost first.
4. **fp16 + `channels_last`**, then ONNX Runtime / TensorRT. Zero AUC cost.
5. **Drop TTA before dropping models** — TTA typically buys ~0.002 for a 2–4× runtime multiplier.
   (And hflip TTA is invalid here anyway, §3.2.)
6. **Only then** shrink backbone / ensemble.

**Accept/reject rule:** take a change only if `Δt > 93000 × ΔAUC` seconds. Instrument the notebook
to log decode / preprocess / GPU / write time separately, then walk the list measuring
`(ΔAUC_oof, Δt)` for each step.

---

## 7. Schedule — 11 weeks

> **Revised 2026-08-07.** The original schedule deferred the first submission to weeks 4–6 and
> treated backbone choice as a week 7–9 scaling knob. Both assumed we would build a baseline from
> nothing. The public leaderboard reached 0.932 within 48 hours of the competition opening, on
> forks of shared DINOv2 notebooks (§7.1), so a baseline is now a fork and LB feedback is nearly
> free. The tracks below run in parallel rather than labels-then-vision, because the vision
> pipeline is what *evaluates* the label work.

| Weeks | Phase | Exit criterion |
|---|---|---|
| **1** | ~~Logistics + text EDA~~ **DONE 2026-08-06** — CSVs pulled, `FINDINGS.md` written | ✅ 58/4,407 gold; 9 languages; 6 series types |
| **2** | **Unblock.** Fork a public DINOv2 baseline → submit. Verify `(0020,0060) Laterality` survived the 86-tag allowlist (§3.2). **A/B our extractor against the public weak labels** on the 31 blind gold + 86 hand labels | A number on the board; laterality answered; **we know whether our labels are a moat** |
| **2–3** | **Feature cache — now the critical path (§7.2).** §3.1 preprocessing on Kaggle; benchmark decode per transfer syntax while in there. Frozen DINOv2 ViT-B/14 @518 → per-slice embeddings → publish as a Kaggle Dataset | ~2.4 GB of embeddings, downloadable. Local iteration unblocked **and the label track has a scoreboard again** |
| **3–4** | **Fusion head, trained locally.** §3.3 minus the backbone: slice transformer → attention pool → series-type embedding → series attention → 12 logits. Series dropout mandatory; no hflip unless Medial↔Lateral swap | Beats the public baseline on *our* held-out set, not the LB |
| **3–5** *(parallel)* | **Labels.** Finish the 217 remaining hand labels — now the priority item on this track, because it is the only thing that restores measurement (§7.2). Then the §2b-ii calibration and §2.11; fit the §1.3 soft-target constants. LLM extractor once the host is settled. ~~Fix §2.1–§2.7~~ — §2.1 is the report-only ceiling, not a bug, and §2.2 is done | Labels that beat the public ones **through a trained model**, not just against a report-derived reference |
| **5–7** | **External data** (§3.4). MRNet → ACL + meniscus; OAI → the three OA labels; fastMRI+ | Gain on the ~6 covered labels over pseudo-labels alone |
| **7–9** | **Scale.** Unfreeze the last N DINOv2 blocks now the architecture is settled. Per-plane specialists, correlation stacker (§3.5.5), ensemble | Ensemble beats best single by more than its CI |
| **9–10** | Efficiency variant + hardening: decode optimization, degenerate-input tests, full-size runtime profiling | Lean submission within ~0.01 AUC of best, ≤15 min |
| **11** | Final submission selection, open-source packaging, video | Two finals chosen on CV, both verified clean |

Entry deadline **Oct 15** — accept the rules well before then.

~~The load-bearing item is the **week-2 A/B**.~~ **Done 2026-08-07 and passed** — ours beats the
public weak labels on both references (0.777 / 0.749 vs 0.672 / 0.672 on gold), so the moat is real
and the LLM extractor stays justified. But the A/B answers a narrower question than it looks:
it compares label sources against a *report-derived* reference, and what decides the competition is
whether a **model** trained on our labels outscores one trained on `nekkon`'s.

**The load-bearing item is now the feature cache**, for the reason in §7.2: it is the only
instrument that can answer that question, and without it every remaining item on the label track is
unmeasurable. Two Kaggle sessions, and `kaggle_01` gates `kaggle_02` — if laterality did not
survive the 86-tag allowlist, four of the twelve labels are unreliable and that changes what the
cache is worth building against.

### 7.1 What the public leaderboard is actually doing — measured 2026-08-07

Top score 0.932; ranks 2–20 spread 0.900 → 0.811; **every submission dated 2026-08-06/07**. Nobody
built a weakly-supervised 12-label 3D pipeline in 48 hours — the field forked shared notebooks.
Six of the top 20 public notebooks are DINOv2-based (`DINOv2 at meniscus resolution`,
`DINOsaur V2`, `dinov2-ensemble`, `Public 4-fold DINOv2 v4`, `DINOv2 Base Physical Scale Soup`,
`DINO Protocol Fusion`). A frozen self-supervised ViT needs far less label signal than a CNN
trained from scratch, which is why 0.9 arrived so fast.

Consequences:

- **0.9 is table stakes, not a target.** The spread from 0.811 to 0.932 is people tuning one notebook.
- **`nekkon/weak-labels-for-all-12-knee-mri-findings` is public.** §5 called the extractor "the
  solution"; that is now partly commoditized. Ours has to beat the public set to be worth anything,
  which is measurable — hence the week-2 A/B.
- **DINOv2 replaces the §3.3 backbone slot and nothing else.** The slice transformer, attention
  pooling, series-type embedding, and series dropout all stand, and are likely *ahead* of the public
  forks — `DINO Protocol Fusion` existing as a separate notebook suggests most forks pool naively
  across series. Frictions: patch-14 fixes resolution to multiples of 14 (use 518, not 224 — a
  meniscal tear is small and 16×16 patches lose it), and the stem takes exactly 3 channels, so
  §3.3's "groups of 3–5 adjacent slices" becomes exactly 3.
- **Freeze the backbone and cache the embeddings.** ~4,400 studies × ~5 series × ~24 slices ≈ 530k
  slices; at the 1,536 dims embed() actually returns (CLS || patch-mean) in fp16, and 32 cached
  slices rather than 24, that is **2.4 GB** — earlier drafts said "under 1 GB" by computing at 768
  dims, corrected 2026-08-07. The fusion head then trains on the
  laptop in minutes per experiment. Differentiation cannot come from the backbone — everyone has the
  same weights — so it has to come from the fusion layer, the labels, and external data.

**Tested and rejected:** series metadata alone (counts and presence flags over the 6 types) predicts
nothing — macro AUC **0.471** on the 58 gold, 5-fold CV, below chance. There is no protocol-fingerprint
shortcut. Do not re-run this. See `eda_04_metadata_baseline.py`.

### 7.2 The extractor track has run out of instrument — measured 2026-08-07

The §2.2 compartment fix moved **~1,000 studies** off a flat 0.45 that could not contribute to a
ranking metric. Gold macro AUC moved **0.775 → 0.777**, against a bootstrap CI of **±0.038**. The
change is real and corpus-level evidence says so, but the 58 gold studies cannot see it, and
neither can the 83 hand labels at better than ±0.03.

That is not a one-off. It is the steady state for everything left in `IMPROVEMENTS.md` §2:

> **No remaining extractor change is measurable on the references we currently hold.** Each one
> is worth a few hundred to a thousand studies of label quality and a few thousandths of gold AUC.
> We can rank them by reading and by corpus statistics — which is how §2.2 and R10 were found and
> justified — but we cannot *score* them, and we cannot tell when to stop.

Three ways out, in order of cost:

1. **Finish the 217 hand labels.** Halves the CIs (§4.1 of `IMPROVEMENTS.md`) and is already on the
   schedule. Necessary, not sufficient — 303 items still leaves ±0.03 on a macro comparison.
2. **Measure labels through a trained model.** Train the §3.3 fusion head twice on identical
   splits, once on our pseudo-labels and once on `nekkon`'s, and compare *model* AUC. This is the
   claim that actually matters and the A/B in §7 does not test it: label-AUC → model-AUC is not
   monotone, and noise-robust training over 4,349 studies can absorb a 0.1 label-AUC gap entirely.
3. **LLM extractor as a second opinion** (§1.1). Its value is now as much *adjudication* —
   disagreement mining to target the next hand labels — as raw extraction quality.

**This re-ranks the feature cache.** It was scheduled as the thing that unblocks local vision
iteration. It is now also **the only instrument that can score the label work**, which is the
track we have most invested in. Path 2 is not reachable without it. Build it first.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| **Pseudo-label quality caps the whole solution** | Two independent extraction methods; validate per-language; confidence-weight the loss |
| **A weak language silently poisons a subset** | Per-language agreement metrics, not just overall |
| **Laterality tag stripped from the 86-tag allowlist** | Check first; fall back to geometry or a pixel-based L/R classifier |
| **Site/scanner/protocol shift** (explicitly international) | Site-aware CV, heavy intensity aug, per-site monitoring |
| **Prevalence shift train→public→private** (stated) | AUC is prevalence-insensitive within a label; don't calibrate to train prevalence; trust CV |
| **Rare labels + only ~1,300 test studies → wide CIs** | Bootstrap CIs; accept that some ranking is luck; don't overfit folds |
| **JPEG 2000 decode dominates runtime** | Benchmark per transfer syntax in week 1; `pylibjpeg-openjpeg` / GDCM |
| **Missing series in test studies** | Series-dropout augmentation + explicit degenerate-input tests |
| **570 GB won't fit locally** | Kaggle notebooks / cloud for pixels. ~~Local work is text-only~~ — **no longer true**: cached frozen DINOv2 embeddings are ~2.4 GB, so the fusion head trains locally (§7.1) |
| ~~**Public weak labels commoditize the extractor**~~ **retired 2026-08-07** | A/B run and passed — 0.777/0.749 vs 0.672/0.672 on gold, 0.864/0.862 vs 0.757 on hand labels. Ours wins on both references and both metrics |
| **Extractor gains are no longer measurable** (§7.2) — replaces the risk above | ~1,000 studies of label improvement moved gold macro by 0.002 against a ±0.038 CI. Finish the 217 hand labels; then score labels *through* a trained fusion head, not against a report-derived reference |
| **Everyone shares one backbone** | DINOv2 is universal, so it cannot differentiate. Push the edge into the fusion layer (§3.3), the labels (§2), and external data (§3.4) — the three places our work is already strongest |

---

## 9. Immediate next steps

> Rewritten 2026-08-07. The original five items — cancel the browser download, install the Kaggle
> CLI, characterise `train.csv`, benchmark decode, build the extractor — are done bar the decode
> benchmark, which now lives inside `kaggle_01`. Everything below needs a Kaggle session, which is
> the point: **there is nothing left on the local critical path that we can measure** (§7.2).

1. **Run `notebooks/kaggle_01_dicom_audit.py`** as a Kaggle Script with the competition dataset
   attached. ~30 min. Answers laterality (§3.2) and decode cost per transfer syntax (§6.3.1) in one
   pass. Download `dicom_audit.json` and record the result in `FINDINGS.md`.
   **If `(0020,0060) Laterality` did not survive the allowlist, stop and re-plan** — four of the
   twelve labels are side-specific and the fallback (geometry, or a pixel L/R classifier) has to be
   built before the cache is worth anything.
2. **Fork a public DINOv2 notebook and submit.** Still unmet from week 2, and it is ~an hour. Until
   a number exists on the board, no later change has a baseline to move against.
3. **Run `notebooks/kaggle_02_dinov2_cache.py`**, sharded, across as many sessions as it takes.
   Do the 224 pass first to get the pipeline honest, then re-run at 518. Publish
   `/kaggle/working/features` as a Kaggle Dataset.
4. **Then the two experiments that are currently impossible:** train the §3.3 fusion head, and
   train it twice on identical splits — our pseudo-labels vs `nekkon`'s — to find out whether the
   label moat survives contact with a model (§7.2 path 2).
5. In parallel, and only in parallel: the remaining 217 hand labels. They are what restores
   measurement on the label track itself.
