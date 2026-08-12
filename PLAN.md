# RSNA Knee Abnormality Detection — Plan

**Metric:** macro-averaged AUROC over 12 labels
**Timeline:** started Jul 30 2026 · entry/merger deadline **Oct 15** · final submission **Oct 22 2026**
→ **~10.5 weeks from today (Aug 8).**
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

> **CLOSED 2026-08-10 — this was the critical path and it is now a download.** §2.1 below calls
> for "a multilingual LLM offline over every training report". Four such tables are published
> as free Kaggle Datasets, the best scores **0.893** on gold-58, and the rule extractor built
> instead scores **0.777** and loses on 12/12 labels (`IMPROVEMENTS.md` §2f). Everything in §2
> is kept for provenance and because §2.1's *schema* — structured attributes, not twelve bits,
> with per-finding confidence — was right and is what the public readers emit. The build was
> the error, not the design. Do not spend further time on §2.1–§2.4.

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
2. ~~**Transfer-syntax spread is a real cost**: uncompressed, JPEG Lossless, JPEG 2000, Implicit
   VR. JPEG 2000 decode is slow.~~ **Measured 2026-08-07 and false.** The corpus is **200/200
   Explicit VR Little Endian** at **3.1 ms/slice** — one syntax, uncompressed, cheap. No
   `pylibjpeg`, no GDCM, no `dicomsdl`. See `FINDINGS.md` §6.
3. MRI has no HU standard → per-volume robust percentile normalization (clip 0.5/99.5 → [0,1]).
4. Resample in-plane to fixed mm/px; centre-crop/pad to a fixed FOV. The knee is protocol-centred,
   so a detector is overkill.
5. Fixed slice count per series (16–32).
6. Cache as uint8 `.npy`.

### 3.2 Laterality — still the sharpest trap
Four labels are side-specific (Medial/Lateral Meniscus, Medial/Lateral OA). "Medial" flips between
left and right knees, so a model fed raw mixed-handedness studies sees medial findings on both
sides of the image.

**Canonicalize every study to one handedness.** **Answered 2026-08-07** (`FINDINGS.md` §6.2):
`(0020,0060) Laterality` survives but is **empty on half the corpus** — 2,203 of 4,407 studies
carry a usable value. The geometry fallback works, but *not* at the obvious boundary: `x < 0`
agrees with the tag only 89.3%, while **`x < -62` agrees 97.7%** (cross-validated 97.32% ± 0.72%),
because the scanned knee sits at isocentre rather than the patient being centred. Tag first,
geometry second, source recorded per series. A pixel-based L/R classifier is no longer needed.

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

> **PROMOTED 2026-08-09.** This section was never carried into any phased route, despite its own
> heading. It is now **Phase 2** in §9. The reason it moved up: IMPROVEMENTS §2d shows the six
> failing labels are millimetre-scale structural findings, and **MRNet supervises three of those
> six directly** (ACL, meniscal tear) with real expert image reads rather than pseudo-labels.
> OAI covers the three OA labels, which are already our strongest — so MRNet is the higher-value
> of the two despite being the smaller corpus.

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

- ~~**MultilabelStratifiedGroupKFold**, 5 folds, grouped by patient.~~ **There is no patient to
  group by** — `PatientID` is present in every DICOM and is *unique per study* (4,407 IDs for
  4,407 studies), so it is de-identified per study. Plain multilabel-stratified 5-fold, ungrouped;
  `fusion/folds.py`. Stratify on the rarest labels.
- Keep gold-labeled and pseudo-labeled studies **identifiable** in every fold; always report
  metrics on the **gold** subset — pseudo-labels inherit the extractor's biases and will flatter you.
- Studies are from a "diverse international mix of imaging sites" → **site shift is the #1 CV↔LB
  divergence risk.** Monitor per-manufacturer/per-language performance if that's recoverable.
- **Prevalence differs across train / public LB / private LB** (stated explicitly). AUC is
  prevalence-insensitive *within* a label, so this is survivable — but it means public LB will be
  noisy and will not track private. **Trust CV.**
- **The public leaderboard is 30% of test; the final standing is the other 70%.** Confirmed
  2026-08-08. Two consequences, and the second is the one that costs people prizes:
  - On the ~1,300-study estimate above, the public LB is scored on **~390 studies**, and a label
    at Fracture-like prevalence contributes on the order of a hundred positives to it. A per-label
    AUC off that base has a CI wide enough to swallow most of the improvements we plan to make.
    **A public-LB move smaller than the gap between two of our CV folds is not evidence.**
  - The 30/70 split is *disjoint*, so every point of public-LB tuning is fitted to studies that
    contribute nothing to the final rank. Combined with the stated prevalence shift, public→private
    shake-up is the default expectation, not the tail risk. This is why §4's final-two-submission
    rule is (a) best CV, (b) smallest CV–LB gap — **neither of them is "best public LB".**
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

> **Would a C++ rewrite be worth it at, say, 0.96? No — and §6.1's own exchange rate is why.
> `ANSWERED 2026-08-10`** At 93 seconds per 0.001 AUC, a rewrite has to save *minutes* to be
> worth anything measurable. It cannot: the runtime is dominated by DICOM decode and GPU
> forward passes, and **both are already native code** — the decoder is C, the forward is CUDA.
> Python contributes orchestration only, plausibly under 5% of wall time, so a full port of a
> ~12-minute run saves well under a minute ≈ **under 0.001 AUC equivalent**, against weeks of
> work and a much harder §2.8 winner's-obligation deliverable (training code, inference code,
> weights, and a reproducible environment).
>
> The levers that *are* worth it are all above the language line and are already ranked in §6.3:
> slice count, backbone size, batching, and the decode backend — though note §3.1 measured the
> corpus as **200/200 Explicit VR Little Endian at 3.1 ms/slice**, so even the dicomsdl lever is
> gone. **The efficiency track rewards a strong model with a tuned I/O path, not a rewritten
> one**, and §6.2 below is the argument.

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
1. ~~**DICOM decode throughput — free.**~~ ~~**Demoted 2026-08-07.**~~ **Re-ranked 2026-08-08, and
   the demotion was half right.** The *decoder* is still irrelevant: there is no JPEG 2000, every
   file is Explicit VR Little Endian at 3.1 ms/slice, and `dicomsdl`/`pylibjpeg`/GDCM buy nothing.
   But "3.1 ms/slice ⇒ ~36 min single-threaded ⇒ GPU-bound" did not survive contact with the mount.
   That benchmark timed **decode of files already opened**, n=6, inside a 200-study audit. What the
   cache build actually pays is **per-file open latency** on a 570 GB network mount — ~19 ms each,
   ~700k of them, i.e. hours before a single kernel launches. The first 224 shard ran 9 h against a
   2.7 h estimate.

   **So the cache build is I/O-LATENCY-bound, not GPU-bound and not decode-bound.** The lever is
   concurrency of opens (oversubscribed workers, prefetch depth) and file *count* — which is what
   makes slice subsampling and series pruning pay twice. Resolution and slice count still dominate
   the GPU side of the submission notebook, where the test set is ~1,300 studies and the mount is
   read once. `kaggle_02`'s PROBE at 25/100/400 series prints the implied hours for the shard and
   is the instrument that settles this properly — the ~19 ms figure is an inference from failed
   runs, not a recorded measurement (see `FINDINGS.md` §6.1).
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

> **Method note, added 2026-08-10.** Characterise the field from its **code**, not its
> descriptions: `python -m kaggle kernels pull raahimnawaz/rsna-knee-lb-baseline-fork -p /tmp/fork -m`.
> Reading the description produced three wrong claims about the leading notebook, two of which
> shaped a route for a day — most importantly that its 0.891 was a training score, when
> `find_weights()` short-circuits it into inference from published weights.

> **RE-MEASURED 2026-08-09.** Top **0.940** over **908 teams** (was 0.932 / ~296). Our own first
> submission — an unmodified `pilkwang/rsna-knee-baseline-v1` fork — scored **0.891** at rank
> **230/908**. The conclusion below strengthens rather than changes: two days of field-wide
> tuning moved the top by 0.008, and the whole distance from mid-table to first is ~0.05 AUC.
> A public fork now *is* mid-table, so the fork is the floor to beat, not a milestone.

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

> **AMENDED 2026-08-10. The conclusion was right and the reason was wrong, and the wrong reason
> cost five days.** This section reads "no remaining extractor change is measurable, so stop
> tuning and go get a better instrument". The true statement is stronger and was available the
> whole time: **no remaining extractor change is worth measuring, because the component is
> 0.116 behind a free alternative** (`IMPROVEMENTS.md` §2f). "Out of instrument" framed the
> problem as *we cannot see our improvements* and led to building a better reference. "Behind
> the field" frames it as *there is nothing here to improve* and leads to a download. The
> instrument was never the binding thing on this track.

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
| ~~**JPEG 2000 decode dominates runtime**~~ **retired 2026-08-07** | Measured: 200/200 Explicit VR Little Endian, 3.1 ms/slice. The risk does not exist |
| ~~**GPU time is the cache-build constraint**~~ **corrected 2026-08-08** | It is not GPU time, it is **per-file open latency** on the mount (§6.3.1). ~700k opens at ~19 ms is hours before the GPU matters. Shard via `SHARD`/`N_SHARDS`; the build resumes, so a killed session loses one study; do the 224 pass first |
| **Kaggle assigns a P100 on roughly 4 of 5 draws, and `accelerator` in kernel-metadata.json does not override it** | Its PyTorch dropped Pascal, so a P100 session cannot run this at all — and `torch.cuda.is_available()` is True on one. `pick_device()` fails closed in the first seconds of `main()`, so a bad draw costs seconds and re-rolling is a viable strategy rather than a way to burn the weekly quota |
| **A silently-serial worker pool costs a whole session** | Three cache attempts died this way (21 h, then 9 h on the serial curve). The pool is spawn-context, workers are pinned to one thread, and the PROBE at 25/100/400 series warns inside minutes when the rate is near single-worker. `--self-test` now constructs the real pool and asserts it fans out |
| **Missing series in test studies** | Series-dropout augmentation + explicit degenerate-input tests |
| **570 GB won't fit locally** | Kaggle notebooks / cloud for pixels. ~~Local work is text-only~~ — **no longer true**: cached frozen DINOv2 embeddings are ~2.4 GB, so the fusion head trains locally (§7.1) |
| ~~**Public weak labels commoditize the extractor**~~ **retired 2026-08-07** | A/B run and passed — 0.777/0.749 vs 0.672/0.672 on gold, 0.864/0.862 vs 0.757 on hand labels. Ours wins on both references and both metrics |
| **Extractor gains are no longer measurable** (§7.2) — replaces the risk above | ~1,000 studies of label improvement moved gold macro by 0.002 against a ±0.038 CI. Finish the 217 hand labels; then score labels *through* a trained fusion head, not against a report-derived reference |
| **Everyone shares one backbone** | DINOv2 is universal, so it cannot differentiate. Push the edge into the fusion layer (§3.3), the labels (§2), and external data (§3.4) — the three places our work is already strongest |

---

## 9. Immediate next steps

> # COURSE CHANGE 2026-08-12 — THIS SUPERSEDES EVERYTHING BELOW, INCLUDING THE 08-11 BLOCK
>
> **`IMPROVEMENTS.md` §2w.** Our own pipeline is ~0.76 LB. **The fork we submitted on 2026-08-09
> scores 0.891 and is already banked.** We have spent Phase 0 rebuilding, in our own code, a thing
> that is free to download, and we are 0.13 below it. Rule 6 of the README already said *"the fork
> is the base, not a reference"*; it was never followed.
>
> **0.891 is the floor. Every number is a delta on top of it.**
>
> | # | step | cost |
> |---|---|---|
> | **E1** | Verify the live leaderboard and both lineages' real scores | minutes |
> | **E2** | Rank-mean `pilkwang` + `prvsiyan` — two published notebooks | 1 Kaggle run |
> | **E3** | Submit it (also retires the never-run inference path) | 1 of 5 daily |
> | **E4** | Make the port earn a slot: does *adding* it help? | 1 scoring run |
> | **E5** | Differentiators as deltas on the ensemble | ~1.6 h/fold |
> | **E6** | Efficiency track, co-primary | TBD |
>
> **Corrections to what this file and `REFERENCE.md` assert about the field**, surveyed live
> 2026-08-12: `0.899 let me cook` is `aadigupta7686`, not `prvsiyan`; and **no public
> `Yash Bishnoi` / B3 kernel exists** — that 0.903 is a writeup, so reproducing it is a training
> job and not a download. Every LB figure in this repo was read 08-10. Re-verify before use.
>
> **The compute budget doubles: a fold is ~1.6 h, not 3.6** (§2v — the laptop was asleep for ~2 h
> of the run that produced 3.6). So ~40 five-fold experiments remain, not ~20. **Always launch
> under `caffeinate -i`.**

> **COURSE CHANGE 2026-08-11 — superseded by the block above, kept for provenance.**
> `IMPROVEMENTS.md` **§2s** has the full argument; the operative consequences are:
>
> 1. **Step 5 is not a reproduction gate and cannot be one.** `pilkwang` publishes no local
>    number (`REFERENCE.md` 3.1 records `—`); its 0.891 is a leaderboard score from 20
>    rank-meaned published weights. There is nothing to land near. The fold-0 run started
>    2026-08-11 is **renamed the label-swap arm** and is read against a *predicted delta* — the
>    gate arm should land **0 to 0.021 below `runs_port`**, that being the gold-58 label-quality
>    gap (`steven_v2` 0.8873 vs `pilkwang_v2` 0.866). Read it asymmetrically: the `lixin`
>    confound handicaps the gate arm, so a negative delta is ambiguous and only a positive one
>    is clean.
> 2. **Step 5 depends on step 6, not the reverse.** The reproduction target lives on the
>    leaderboard because that is the only place `pilkwang`'s number exists. If "did we reproduce
>    the fork" is to be answered at all, it is answered by a submission. **Do not spend 18 h on
>    folds 1–4 to answer it locally — that run cannot answer it.**
> 3. **Before any A/B launches, state the reference and show it is neutral to both arms.**
>    Four measurements have now been lost to one error class (§2d, §2i, §2o, §2s): the
>    instrument entangled with the thing it measures. Every instance was caught after the run.
>    This is the pre-run check that would have caught all four, and it costs minutes.
> 4. **Phase 1 is priced wrong by ~5×.** Gold is the only instrument neutral to a label-source
>    change, and fold 0 holds **10 of the 58 gold studies** (9–10 per fold). So a label
>    experiment needs all five folds — ~18 h — and then reads at ±0.031 against differences of
>    ~0.021. The report-OOF instrument is valid **at fixed targets only** and cannot arbitrate
>    the severity-thresholded read, which is the best untested idea on the board. **Settle how
>    Phase 1 will be judged before building any of it.**
> 5. **Cheap thing to try first:** the rule extractor as a **referee**, not a label —
>    0.685 vs steven, 0.657 vs `pilkwang_v2`, against `lixin`'s lopsided 0.947 / 0.866. A
>    symmetric-but-noisy referee beats a sharp-but-lopsided one for a paired delta. It is a
>    scoring change, not a training run. §2s-g, and note the caveat there.
>
> **Amended again the same day, from six external concerns — `IMPROVEMENTS.md` §2t.**
>
> 6. **Submit NOW, beside step 5, and the reason is risk not measurement.**
>    `kaggle_03_submit.py` has never executed against a real test DICOM: the 9 h cap, no-internet,
>    weights-as-a-Dataset, degenerate series, a study missing a plane. All schedule risk, none of
>    it discoverable locally, none of it competing with the local instrument. And the CV↔LB
>    conversion is currently an interpolation between **two foreign anchors** while carrying the
>    claim that our gap is 0.19 rather than 0.22. §2s-e and §2t-1 reached "unblock 6" independently.
> 7. **THE COMPUTE BUDGET, which this plan never had (§2t-3).** 72 days to the deadline; 3.6 h
>    per fold measured; ~18 h per 5-fold experiment; ~40 h/week on the M5; 30 h/week Kaggle minus
>    a lottery that refuses ~4 draws in 5. **That is ~20 five-fold experiments for the entire
>    remaining project.** Phase 1 as written spends it several times over.
>    **RULE: an item that does not name its cost in folds is not a plan item.**
> 8. **The efficiency track is promoted to a live co-primary decision (§2t-4).** $18,000 over
>    three places against $5,000 for 10th, a far thinner field, and it rewards the measurement
>    discipline this project actually has. §6.2's "accuracy dominates" is true *of the formula*
>    and says nothing about which track to enter — those were conflated.
>    **Decide after the first submission:** LB below ~0.87 → efficiency primary; above → run both.
> 9. **The boring hypothesis has a test now (§2t-5).** "The top just ensembles more" is usually
>    right on Kaggle and was never argued against. It can be: **both anchors bracketing the gap
>    are already ensembles** — `pilkwang` 0.891 is a 20-member rank-mean, Yash B3 0.903 is a
>    5-fold mean — so the 0.04 sits *between ensembles*, where 5→20 members buys ~0.005–0.010.
>    **The distinguishing measurement is our own single-model LB point against `pilkwang`'s
>    20-member 0.891.** Large delta → boring wins, the answer is compute. Small → the label and
>    crop bets are the right ones. One more reason step 6 runs now.
>
> **Verification.** Every claim above is checkable, and the checks are cheap:
>
> | claim | check |
> |---|---|
> | `pilkwang` publishes no local number | `REFERENCE.md` 3.1 — the local column reads `—` |
> | public label tables are rules-clear | `REFERENCE.md` §1.2 clause **3.6.b**; `COMPETITION_RULES.txt` §6 |
> | MRNet is *unresolved*, blocked on the host | `REFERENCE.md` §1.3 — asked twice, only the LLM half answered |
> | the `lixin` referee is lopsided | `IMPROVEMENTS.md` §2i table: 0.947 steven vs 0.866 `pilkwang_v2` |
> | the label gap is 0.021 | §2f gold-58: `steven_v2` 0.8873, `pilkwang_v2` 0.866 |
> | gold cannot arbitrate on one fold | 10 of 58 gold studies in fold 0 (9–10 per fold) |
> | 3.6 h/fold, memory-bound | §2p, `data/_train_port_fold0.log` |
> | the paired σ is now reproducible | `for s in 1 2 3 4; do PYTHONHASHSEED=$s python fusion/score_oof.py fusion/runs_baseline fusion/runs_port \| grep PAIRED; done` → four identical lines |

> Rewritten 2026-08-08. Item 1 of the previous list (the DICOM audit) is **done** — `kaggle_01`
> plus `kaggle_01b` ran over all 4,407 studies and the results are in `FINDINGS.md` §6; laterality
> is answered and the geometry fallback is adopted.
>
> **Amended 2026-08-09, and the amendment is the important part.** The line that stood here —
> "everything below needs a Kaggle session, there is nothing left on the local critical path we
> can measure" — is **no longer true**, and it was load-bearing: it is why four sessions were
> spent re-rolling a GPU instead of asking whether the GPU was required. §9.1 has the measurements.
> The backbone runs on the M5 at 62.8 slices/s @224, and the corpus exists as per-series NIfTI at
> a size that fits on the laptop. **The local critical path is open again.**

**Where the cache build actually stands.** Five attempts, none finished, and none of them failed
at the modelling:

| attempt | outcome | cause | now |
|---|---|---|---|
| full corpus @518 | killed at 21 h | unsharded, no resume | `SHARD`/`N_SHARDS` + per-study resume |
| two relaunches | died ~1 h in | drew a P100; `is_available()` is True on it | `pick_device()` fails closed in the first seconds |
| 224 shard 0/4 | ran 9 h on the serial curve vs a 2.7 h estimate | pool forked a live CUDA context — **but see `IMPROVEMENTS.md` K7: a run that intended 224 and silently got 518 looks identical, and the evidence is gone** | spawn context, 8 pinned workers, PROBE at 25 series |
| 224 shard 0/4, retry | P100 draw, refused in the first seconds | the GPU lottery | nothing to fix — **this is the guard working.** First of the thirteen fixes validated in the wild |

**The sixth attempt would also have failed, and not on the GPU.** `IMG_SIZE=224` against a
518-native backbone raises on the first series (`IMPROVEMENTS.md` K14, fixed 2026-08-09). Twelve
of the fourteen fixes have still never touched a real DICOM. That is the whole of the current
risk, and §9.1 is the cheapest way to retire it.

> **SUPERSEDED 2026-08-09.** The cache exists — built locally from NIfTI (§9.1), not on Kaggle,
> and the fusion head has produced its first real number: **macro AUC 0.719** on 37 gold studies.
> The list below is replaced by the phased route in `README.md` ("Where this goes next"), which is
> now the operative plan. Summarised here so this section is not misleading:

> **REORDERED 2026-08-09 (later), by measured cost.** Two things moved. **Resolution outranks the
> ordering fixes**: the 224 cache resolves 0.71 mm/px because `imagenet_normalise` interpolates
> the correctly-built 457 px volume down to `IMG_SIZE`, and the per-label evidence says that,
> not the pseudo-labels, is the ceiling (IMPROVEMENTS §2d). **And the leaderboard step is done** —
> an unmodified `pilkwang` fork scored **0.891** on 2026-08-09, rank 230/908, top 0.940. What is
> still unmeasured is the CV↔LB mapping for *our* pipeline.

> **2026-08-10, later — the direction is settled and it is architectural.** `pilkwang`'s notebook
> was read directly rather than from its description: it **fine-tunes the last six encoder blocks**
> (`UNFREEZE_LAST = 6`, `LR_BACKBONE = 8e-6`). A frozen-embedding cache cannot do that at any
> resolution, under any head, with any labels — which is why resolution bought +0.013 and why
> every number here sits near 0.70. See IMPROVEMENTS §2e. It also uses DINOv2 **small** at **one**
> resolution, so we are behind on neither capacity nor resolution, only on trainability.
>
> The feasible direction, and the one real asymmetry: a study is six encoder inputs, so a full
> 10-epoch run is **~2 h on the M5** against Kaggle's 8 h budget, 30 h weekly quota and a GPU
> lottery that refuses four draws in five. Move the *fine-tuning* local. The pixel cache it needs
> is ~9 GB against 458 GB of NIfTI. K16's direction bit is back **on** the critical path — local
> pixels have no `ImagePositionPatient` — and is now justified by something that pays for it.

> **SUPERSEDED AGAIN 2026-08-10 (later). Both the Phase list below AND the "use the leaderboard
> as the instrument" route that replaced it are retired.** `extractor/bench_public_labels.py`
> re-pointed §7.2's decision gate at the four LLM-read label tables the field has published
> since week one: **ours 0.777, best public 0.893, 0/12 labels won, and adding ours to a
> rank-mean makes it worse.** So the critical path this document has assumed since §2 —
> *reports → labels* — is not ours to walk; it is a download. See `IMPROVEMENTS.md` §2f.
>
> Two corrections follow. **The extractor track is closed as a source of training targets**,
> which retires most of §2 and all of §7.2's premise. **And "the leaderboard is the instrument"
> was wrong** — five submissions a day against ~8 h runs and a 30 h quota cannot carry 20
> experiments, and the route that adopted it moved away from local measurement at exactly the
> moment local measurement found §2f. The replacement is a report-holdout OOF with ~880 studies
> (the 0.903 system's own OOF and cross-fitted gold-58 agree to 0.002), which is ~5× finer than
> the 37-study gold and free to run.
>
> **The operative plan is `README.md` "Where this goes next", rewritten 2026-08-10.**

> **SUPERSEDED 2026-08-10. Do not follow the Phase list below.** The resolution reordering above
> was tested and returned **+0.013** — inside the CI (IMPROVEMENTS §2d). The operative plan is
> now README "Where this goes next", which starts from a different diagnosis: **every macro this
> project has produced spans 0.695–0.719, a range of 0.024 against a ±0.038 CI, so no local
> experiment here has ever been resolvable.** §7.2 called this "out of instrument" for the
> extractor on 2026-08-07; the vision model was then scored on the same 37 gold studies and
> inherited it. The route that follows from that is: use the **leaderboard** as the instrument,
> build on the 0.891 fork rather than beside it, change one thing per submission, and decline
> anything the instrument cannot see — including the 22 h 518 rebuild and, conditionally, the
> K16/K18 direction export, which exists only because the NIfTI converter discarded
> `ImagePositionPatient` that the DICOM path carries for free.

**Phase 0 — one rebuild, at 518, carrying every known correction.**

1. **Settle resolution first, cheaply (~5 h).** `--limit` keeps all gold, so a 518 subset build
   compared against the *same* subset of the existing 224 cache — free — isolates resolution from
   corpus size before ~16 h is committed.
2. **Per-series slice direction.** `validate_nifti.py` check 4b, stratified, measures **33% of
   series stored back-to-front** — Sagittal 8/21 forward. The NIfTI affine has no direction
   cosines, so it must be exported from the DICOMs: extend `kaggle_01c` over all 24,371 series,
   CPU-only, no GPU lottery. ~~Cheap route is 2–3 header reads per series (~50k opens, ~20 min);
   full-header reads (~700k opens, 3.7 h) are the fallback.~~ ~~Note §6.2 records that the
   ~19 ms/open figure those estimates rest on is **itself unmeasured** — have the PROBE replace
   it.~~

   > **BOTH ROUTES ARE DEAD — RETRACTED 2026-08-10, MEASURED (`IMPROVEMENTS.md` §2n).** The two
   > costs above are the cost of *reproducing the converter's sort key*, and they silently
   > assume such a key exists in the headers. It does not. Three candidates tested over all
   > 24,371 series and scored against the 51 series the thumbnails settle: InstanceNumber
   > **56.9%**, sorted-filename **60.8%**, SliceLocation **56.9%**, against ~50% chance.
   >
   > **The 3.7 h fallback is dead too, and it would have bought nothing** — `inst` and `loc`
   > already return |rho| = 1.000, perfectly monotone in projection, so reading all 700k headers
   > exports the same signs as reading six. The open-latency figure this paragraph fretted about
   > was never the binding question.
   >
   > The route that works is to **measure the bit rather than infer it**:
   > `notebooks/kaggle_01e_direction_measure.py` ships the spatially first/middle/last thumbnail
   > per series and `pipeline/resolve_slice_direction.py --measured` reads direction off
   > directly, exactly as check 4b does for 51. Scoped to **sagittal only** (9,864 of 24,371
   > series) because medial/lateral is the slice axis only there — axial and coronal are already
   > served by `canonicalise`'s in-plane mirror. ~296k opens, ~20 min at the 122 opens/s
   > `kaggle_01d` actually measured.
   >
   > **The pattern to carry forward: this paragraph priced a route in two levels of detail and
   > never once asked whether the route could work.** A cost estimate is not a feasibility
   > check, and the cheapest experiment in the whole exercise — join three header rules against
   > 51 known answers — would have killed it on day one for free.
3. **Sagittal handedness (K18).** Medial/lateral is the slice axis for sagittal and nothing ever
   reversed it: 40.5% of series, 43.0% of studies left knees. Code is written and gated behind
   `SAGITTAL_LR_SLICE_FLIP`; it is an XOR against step 2's bit, so the two ship together.
4. **Finish the corpus.** 2,649 of 4,407 studies cached against 3,599 downloaded — free data.
5. **Rebuild once, at 518,** with 2–4 applied. Then **submit** as a dry run of `kaggle_03`, which
   has never executed against a real test DICOM, and take the CV↔LB mapping as the side effect.
   Then the train-vs-OOF diagnostic — under- and over-fitting want opposite fixes and one run
   separates them.

**Phase 1 — close the mechanical gap.** Rank-mean ensembling across resolutions then backbones;
the public baseline does both and we do neither. Measure the gold-in-training trade. Re-point the
§7.2 A/B at `pilkwang`'s extractor rather than `nekkon`'s CSV (README §5).

**Phase 2 — external data (§3.4), which no previous route included despite §3.4 calling it
"likely decisive".** MRNet and OAI supervise six of the twelve labels between them, and MRNet
covers three of the six that IMPROVEMENTS §2d shows are failing. Licence checks and the forum
posting requirement make it worth starting earlier than its position implies.

**Phase 3 — the differentiators.** Mask `absent` rather than re-target it (§1.3a). Finish the 217
hand labels — per IMPROVEMENTS §0 the only fix for the ±0.038 CI. Then §3.5's own first and last
items, which no route has carried: aux heads + soft labels, and the label-correlation stacker on
the 12 OOF logits. Attention+mean+max pooling and per-pathology query tokens last.

### 9.1 The cache may not need a Kaggle GPU at all — measured 2026-08-08

Four failed attempts have all failed on *Kaggle-specific* properties: the GPU lottery, the 9 h
cap, and above all the ~19 ms latency of opening one of ~700k small files on a network mount.
None of those are properties of the data. Two measurements say the whole train-side cache can be
built on the M5 instead.

**Throughput.** `vit_base_patch14_reg4_dinov2.lvd142m` on MPS, `dynamic_img_size=True`, fp32:

| resolution | tokens/slice | slices/s | 24,371 series @16 slices | @32 slices |
|---|---:|---:|---:|---:|
| 224 | 261 | **62.8** | ~1.7 h | ~3.5 h |
| 518 | 1,374 | **9.9** | ~10.9 h | ~21.9 h |

GPU time only, and therefore a floor — but there is no 9 h cap on a laptop, the run is already
resumable per study, and a bad draw is not a thing that can happen.

**Pixels that fit.** `davidadekanmi/rsna-knee-nifti-part1..8`, ~120 GB total against 719 GB free,
is the corpus as **one NIfTI per series** — `{StudyUID}_{SeriesUID}.nii`. Header of a sample:
512×512×22, int16, pixdim 0.33 × 0.33 × 3.4 mm, `sform_code=2` with a populated affine. That is
the DICOM pixel data repackaged, not a downsample: §3.1 asks for 518 from a 512 native source.

> **CORRECTION 2026-08-09 — two things above are wrong.** Both were read off a header without
> being tested, which is the same shape of mistake as §6.1's decode benchmark.
>
> **1. It is 12 parts and ~178 GB, not 8 and ~120.** Parts 9–12 were uploaded 2026-08-08 22:29,
> after this survey. Parts 1–8 are the 118 GB recorded. Still fits 719 GB free, but the Kaggle
> CLI does **not** delete each zip after `--unzip`, so the working peak is ~368 GB unless they
> are removed as the download proceeds.
>
> **2. "a populated affine" / "real affine" is wrong, and it is the expensive one.** Measured
> over all 1,393 series of part 1 (`pipeline/validate_nifti.py`): `sform_code` is indeed 2, and
> the affine is **diagonal spacing with zero translation and no rotation** — 0.0% of series
> carry either. The converter kept voxel spacing and discarded the patient coordinate system.
> `ImagePositionPatient` and `ImageOrientationPatient` are both gone, so from these files:
>
> - **the geometry laterality fallback cannot run at all.** It needs IPP[0]. That fallback is
>   what canonicalises the 2,204 studies with an empty `(0020,0060)` tag — half the corpus — and
>   without it Medial/Lateral Meniscus and Medial/Lateral OA are noise (§3.2). Four of twelve.
> - plane is not derivable either, and slice direction relative to the DICOM normal is unknowable.
>
> **This does not sink the corpus, because none of it has to come from the NIfTI.** `kaggle_01b`
> already resolved laterality for all 4,407 studies from the DICOM headers into
> `data/study_meta.csv`, and plane is in `train_series.csv`. `preprocess.study_laterality()`
> reads that table and `load_series_nifti()` takes laterality as an argument. Measured on part 1:
> **250/250 studies resolve — 48% by tag, 52% by geometry, 0% unresolved.** The dependency is
> that `kaggle_01b` was run over the whole corpus *first*; had it not been, this corpus would be
> unusable for 4 of the 12 labels and the reason would have been invisible.
>
> What is still open is **in-plane orientation and slice direction**, and with no rotation in the
> affine there is now no header that can settle either — the `kaggle_01c` thumbnails are the only
> instrument left. That is why it exports first/middle/last per series rather than just middle.

**And it deletes the actual bottleneck.** One file per series is **24,371 opens instead of
~700k**, off a local SSD rather than a network mount. The latency wall that burned K1 and K7
is not being optimised here — it is being removed.

Two other public copies were checked and **rejected**, both for the same reason:

- `barun2104/...-processed-3d-volumes` (7.4 GB) — `(20, 160, 160) uint8` per **study**. The series
  structure is gone, so §3.3's series attention has nothing to attend over, and with no spacing or
  laterality the §3.2 canonicalisation cannot run — that is 4 of the 12 labels.
- `aidenhopkins/rsna-knee-processed` (7.6 GB) — `(400, 6, 9, 224, 224) uint8` with a series mask,
  so structure survives, but 9 slices per series and undocumented windowing. Usable to *shake out*
  the fusion head, not to train the thing we submit.

**What this does not solve, and must not be waved through:**

- `nifti_train/` is **train only**. Test is processed on Kaggle by `kaggle_03` and always was.
- The third-party DICOM→NIfTI conversion is **not in `PREPROCESS_VERSION`**. It sits upstream of
  the fingerprint, which means the fingerprint cannot detect it. Slice order and the orientation
  convention behind that affine have to be validated against the DICOMs for a handful of studies
  before any of it is trusted — the geometry laterality fallback reads exactly that affine.
- `load_series()` takes DICOM paths. A NIfTI reader path is new code on the parity-critical file.

So the honest framing: this unblocks **every experiment** in item 4 and it does so tonight, on a
laptop, with no lottery. Whether it can also produce submission-grade features depends entirely on
the conversion validating against DICOM — which is a measurement, and one that needs a Kaggle
session to make. Item 1 is therefore not cancelled; it is demoted from blocker to check.
