# RSNA Knee Abnormality Detection

Twelve-label knee-MRI classification, macro-AUROC. Final submission **2026-10-22**.

## Read these in order

| doc | what it holds |
|---|---|
| **`PLAN.md`** | strategy, architecture, timeline, efficiency-track maths |
| **`FINDINGS.md`** | measured facts about the data — label coverage, languages, series structure |
| **`IMPROVEMENTS.md`** | **running friction log.** Open decisions, ranked weaknesses, resolved-bug provenance. Read before touching the extractor. |
| `labeling/README.md` | hand-labelling workflow |

## The four facts that shape everything

1. **58 of 4,407 training studies carry labels (1.3%).** The rest must be derived from the
   free-text radiology reports.
2. **Reports span 9 languages, 61% non-English** (English, Spanish, Turkish, Croatian, Greek,
   German, Bulgarian, Dutch, French). English-only clinical NLP is useless here.
3. **Gold labels are not a function of the report text** — they look like independent expert
   image reads. Careful full-text reading agrees with gold on 84.7% of labels, but ~2/3 of
   that gap is one-directional threshold error and is recoverable. See `IMPROVEMENTS.md` §2b.
4. **The leaderboard hit 0.932 within 48 hours of opening, on forks of public DINOv2
   notebooks** — and weak labels for all 12 findings are public too. So 0.9 is table stakes,
   the backbone cannot differentiate us, and "the extractor is the solution" is a claim to be
   earned rather than assumed. Measured in `PLAN.md` §7.1. As of 2026-08-09 the top is **0.940**
   over **908 teams**, and an unmodified public fork scores **0.891** at rank 230 — so the entire
   spread from mid-table to first is about **0.05 AUC**.

## Does our extractor actually beat the free one?

Yes, on both references. This is the week-2 decision gate (`PLAN.md` §7), and it is
reproducible with `python extractor/compare_methods.py`:

| label source | vs gold (n=58) | vs hand labels (n=83) |
|---|---|---|
| **rules (ours)** | **0.777 AUC / 0.749 bal-acc** | **0.864 / 0.862** |
| public (`nekkon`) | 0.672 / 0.672 | 0.757 / 0.757 |

Both metrics are reported because the public set is binary: with a single operating point its
AUC collapses to balanced accuracy, so an AUC-only table would flatter our graded targets for
free. The gap survives both metrics and both references.

**The two references are not independent, and that had to be checked.** The hand labels are
read from the *reports* — same modality the extractor works from, same person, and the
labelling UI highlights terms from the same `glossary.json` that drives the rules. So the
right question is whether switching reference gold→hand preferentially flatters *us*:

| source | vs gold | vs hand | lift |
|---|---:|---:|---:|
| rules | 0.775 | 0.864 | **+0.089** |
| public | 0.672 | 0.757 | **+0.085** |

Near-identical. Both sources are report-derived and both gain the same amount from a
report-derived reference, so the gap is not an artefact of shared vocabulary. *(Measured on
the pre-§2.2 extractor, which is the version the concern was raised against.)*

The table also re-prioritises `IMPROVEMENTS.md` by measurement. Hand-labelled reading scores
**0.838** on the gold studies it covers, so the headroom is ~0.06 — and it is concentrated:

- **Medial OA 0.720 → 0.957 by hand.** Compartment attribution (§2.2) was worth more than
  every other extractor fix combined; the first pass of it is now done and took Medial OA
  bal-acc 0.720 → 0.787.
- **Synovitis 0.621 rules / 0.639 public / 0.652 by hand.** Careful human reading barely beats
  a regex, so this is the report-only ceiling, not a bug — §2.1 is overweighted. Let the vision
  model learn it off the Effusion correlation instead.

## Layout

```
data/                  competition CSVs + derived label tables      (gitignored)
extractor/             report -> 12-label extraction, method A and method B
  rule_extractor.py      the rule extractor (clause-scoped, 9 languages). stdlib only
  run_extract.py         run over corpus + evaluate on gold. NB: works at import time
  llm_extract.py         method B via the Claude Batch API. --dry-run needs no key
  compare_methods.py     every label source scored head-to-head. the decision gate
  metrics.py             auc / bal_acc, shared. separate module so importing it is safe
  diagnose.py            bootstrap CIs + targeted failure diagnostics
  verify_claim.py        stress-tests the "labels aren't report-derived" claim
  calibrate_states.py    fits the soft-target ladder to P(gold=1|state). IMPROVEMENTS.md 1.3a.
                         The fit LOST (0.743 -> 0.699); kept as a measurement, not a setting
labeling/              hand-labelling workflow
  glossary.json          multilingual terms for 12 findings + negation/uncertainty/severity
  compartment_patch.py   adds _side_* / _compartment_struct. IMPROVEMENTS.md 2.2
  build_labeler.py       generates the offline labelling UI
  model_labels.csv       labels produced so far (86/303, all 30 blind gold done)
  item_id_map.csv        item_id -> StudyInstanceUID. the durable fingerprint, no report text
  rekey_labels.py        binds labels to StudyInstanceUID; --check catches sample drift
  eval_model_labels.py   vs gold and vs rule extractor
  error_direction.py     splits disagreement into calibration vs true divergence
pipeline/              shared by BOTH machines -- the one definition of preprocessing
  preprocess.py          DICOM *and NIfTI* -> model input. Fingerprinted; see "Preprocessing parity"
  build_cache_local.py   the feature cache, built on the M5 from NIfTI. 1,062 studies/h @224
  validate_nifti.py      5 checks that the NIfTI repackaging matches the DICOMs. All pass
fusion/                the differentiator (PLAN.md 3.3). Trains on the M5, MPS
  model.py               slice transformer -> attention pool -> series attention -> 12 logits
  dataset.py             cached features -> padded batches. `python fusion/dataset.py` self-tests
  folds.py               5-fold, grouped by patient-proxy. NB: there is no patient column
  train.py               training loop + pooled-OOF gold eval. --synthetic needs no cache
notebooks/             Kaggle-side only; the 570 GB of pixels never come local
  kaggle_01_dicom_audit.py    did (0020,0060) Laterality survive? + decode cost per syntax
  kaggle_01b_patients_laterality.py  header-only pass over all 4,407: PatientID + laterality
  kaggle_01c_series_geometry.py  per-series geometry + thumbnails so the NIfTI conversion can
                              be validated locally. CPU-only, no GPU lottery, ~5 min
  kaggle_02_dinov2_cache.py   frozen DINOv2 -> ~2.4 GB of per-slice features, shardable.
                              `--self-test` runs the whole scheduler locally, no DICOMs needed
  kaggle_03_submit.py         test DICOM -> submission.csv. No internet; weights from a Dataset
eda_01_labels.py       label coverage, prevalence, co-occurrence
eda_02_langs.py        language identification + series structure
eda_03_langid.py       lingua-based language detection (supersedes the heuristic)
eda_04_metadata_baseline.py   series metadata alone scores 0.471 on gold. do not retest
```

## Setup

```bash
python -m pip install -r requirements.txt
# auth: new-style KGAT_ token at ~/.kaggle/access_token (legacy kaggle.json no longer works)

# all four CSVs are needed -- eda_01 reads test.csv/test_series.csv too
for f in train.csv train_series.csv test.csv test_series.csv; do
  python -m kaggle competitions download -c rsna-knee-abnormality-detection -f "$f" -p data
done

python eda_01_labels.py            # writes data/lang_guess.csv   <- eda_03 needs this
python eda_03_langid.py            # writes data/lang_detected.csv
python extractor/run_extract.py    # writes data/pseudo_labels.csv
```

**Run `eda_01` before `eda_03`.** `eda_03_langid.py` reads `data/lang_guess.csv` to diff the
lingua result against the old heuristic, and `eda_01_labels.py` is the only thing that writes
it. Skipping straight to `eda_03` fails with `FileNotFoundError`.

**On a fresh clone, re-key the hand labels before trusting any evaluation** —
`labeling/model_labels.csv` joins on `item_id`, which is assigned *positionally* after a
shuffle (`sample_for_labeling.py:70-71`) and is not bound to `StudyInstanceUID`. If `lingua`
resolves to a different version, the language strata shift, the shuffle lands differently, and
those 86 labels silently describe the wrong studies. Hence the exact pin in `requirements.txt`.

```bash
python labeling/sample_for_labeling.py     # regenerates labeling_sample.csv
python labeling/rekey_labels.py --check    # fails loudly if the mapping drifted
```

Whole text pipeline runs in **6 s** on CPU (measured on an M5, 2026-08-07).

Then compare every label source, which needs no GPU either:

```bash
python extractor/compare_methods.py --fetch   # pulls the public weak labels into data/
python extractor/compare_methods.py           # the table above
```

## Kaggle-side work

The images are 570 GB and stay on Kaggle. The scripts run there in this order — the audits come
first, because if laterality were unrecoverable then four of the twelve labels would be
unreliable and that changes what the feature cache is worth building against. **Both audits are
done** (`FINDINGS.md` §6); the cache is not.

```
notebooks/kaggle_01_dicom_audit.py           -> /kaggle/working/dicom_audit.json        [done]
notebooks/kaggle_01b_patients_laterality.py  -> /kaggle/working/laterality_check.json   [done]
notebooks/kaggle_02_dinov2_cache.py          -> /kaggle/working/features  (publish as a Dataset)
```

**A bad Kaggle GPU draw is the single biggest way to lose a session here.** Kaggle assigns a
Tesla P100 on roughly four of five draws and `accelerator` in `kernel-metadata.json` does not
reliably override it; its PyTorch dropped Pascal, so a P100 cannot run this at all — while
`torch.cuda.is_available()` returns True on one. `pick_device()` checks compute capability
against the installed wheel's arch list in the first seconds of `main()` and refuses, so a bad
draw costs seconds and re-running is a viable strategy rather than a way to burn the weekly
quota. Run `python pipeline/preprocess.py` for the fingerprint; the guard prints the GPU it drew.

**Watch the PROBE line.** `kaggle_02` prints measured throughput at 25 / 100 / 400 series with
the implied hours for the shard, and warns when the rate is near single-worker. Three earlier
attempts burned 21 h, then two ~1 h sessions, then 9 h before anyone could tell the pool had
stopped parallelising. If that warning fires, kill the session.

The cache is the reason local work is **no longer text-only**. Frozen DINOv2 embeddings are
~2.4 GB for the whole corpus, so the §3.3 fusion head — slice transformer, attention pool,
series-type embedding, series attention — trains on a laptop in minutes per experiment. The
backbone is the same one everyone else has; the fusion layer, the labels, and external data are
where the edge has to come from.

`kaggle_02` shards via `SHARD` / `N_SHARDS` and skips finished studies on restart, because
decode is the bottleneck and this will not complete in one session.

## Training (the M5 is enough)

Two machines, one pipeline. Pixels never leave Kaggle; the laptop only ever sees vectors.

```
Kaggle   kaggle_01 audit  ->  kaggle_02 cache  ->  publish Dataset   [~2.4 GB]
M5       download  ->  fusion/folds.py  ->  fusion/train.py          [MPS, minutes/experiment]
Kaggle   upload fold*.pt  ->  kaggle_03_submit.py  ->  submission.csv
```

```bash
python fusion/dataset.py                              # self-test: shapes, masks, degenerate studies
python notebooks/kaggle_02_dinov2_cache.py --self-test # scheduler, resume, spawn pool. No DICOMs
python fusion/train.py --synthetic                    # whole loop on random features. No cache
python fusion/folds.py                                # writes data/folds.csv
python fusion/train.py --features data/features
python fusion/train.py --features data/features --limit 500   # fast iteration; keeps all gold
```

### Building the cache locally (the route that actually worked)

```bash
# 1. pixels: one NIfTI per series, ~178 GB zipped / ~386 GB extracted. Delete each zip as you go
python -m kaggle datasets download -d davidadekanmi/rsna-knee-nifti-part1 -p data/nifti --unzip

# 2. prove the repackaging matches the DICOMs BEFORE building anything (5 checks)
python pipeline/validate_nifti.py --geometry data/series_geometry.csv --thumbs data/series_thumbs.npz

# 3. build. IMG_SIZE feeds the fingerprint, so 224 and 518 caches cannot be confused
IMG_SIZE=224 caffeinate -dimsu python pipeline/build_cache_local.py --validated --out data/features_224
caffeinate -dimsu python pipeline/build_cache_local.py --validated --out data/features
```

Measured on a 16 GB M5: **1,062 studies/h at 224** (~4.2 h for the corpus), ~26 h at 518. The
`--validated` flag is a deliberate speed bump — without it the builder warns on every run that
checks 4/4b have not been re-earned for the current corpus revision.

**The NIfTI mirror is incomplete.** Parts 1–12 and 16–18 exist; 13–15 return 403. That is
**81.7% of studies** and **47 of 58 gold**. Re-check periodically — parts have appeared twice.

`kaggle_02 --self-test` is the one that guards a Kaggle session rather than a laptop run. It
drives the whole scheduling loop on synthetic series — prefetch window, per-study completion
accounting, the resume path, the throughput probe — and it constructs the **real spawn pool** and
asserts it fans out across workers, pins each to one thread, and does not re-search the mount on
re-import. Every one of those has cost a session at least once.

**`--synthetic` is not a toy.** It emits the exact shapes, dtypes, masks and edge cases the real
cache produces — single-series studies, unknown series types, minimum slice counts — so every
consumer is testable before a DICOM has been decoded. On random features it scores macro **0.518**
on the 58 gold studies. That is the point: chance is the correct answer, and anything meaningfully
above it would mean a leak.

**The M5 laptop is sufficient — no Studio, no 980 Ti.** The fusion head is 3.7M parameters over
cached vectors, so the binding constraint is RAM for the cache, not the accelerator. Measured on a
16 GB M5: **2.18 GB peak** with the full cache, the model and optimizer steps all live. The 980 Ti's
6 GB would fit too, but Maxwell has no tensor cores and PyTorch has deprecated sm_52, and it cannot
help with the one genuinely heavy step — the frozen backbone — because that runs on Kaggle either
way. Everything here is fp32 on purpose: 3.7M parameters gain nothing from mixed precision.

**Do not containerise the training.** Docker Desktop on macOS has no Metal passthrough — containers
run in a Linux VM and there is no Apple-silicon `--gpus`. A dockerised run trains on CPU. torch MPS
needs native macOS. Reproducibility comes from the pins in `requirements.txt` plus the preprocessing
fingerprint below.

### Preprocessing parity — the failure with no symptom

`pipeline/preprocess.py` is the single definition of DICOM → model input, imported by the cache
builder *and* the submission notebook. If those two ever disagree, the model is fed a distribution
it never trained on and simply scores badly — no traceback, no warning, and it would surface only
as an unexplained CV/LB gap. So every constant that changes a feature value is hashed into
`PREPROCESS_VERSION`, written into the cache manifest, and asserted by `kaggle_03` before it reads
a single study.

The manifest has to *travel with the weights* for that assert to mean anything: `kaggle_02` writes
`_shard*.json` beside the feature cache, `fusion/train.py` copies it to `manifest.json` beside
`fold*.pt`, and `kaggle_03` refuses to run at all if it cannot find one. A submission that cannot
prove which preprocessing built its heads is the exact failure this section is about, so it exits
rather than warning.

```bash
python pipeline/preprocess.py    # prints the manifest, including the fingerprint
```

Change `IMG_SIZE`, `TARGET_MM`, `FOV_MM`, `SLICES_PER_SERIES`, the model, or the canonical side, and
the fingerprint changes and the existing cache is invalid. That is the intended behaviour.

### What the folds can and cannot do

`fusion/folds.py` groups on a hash of the report text, because **`train.csv` has no patient
column** — the only same-patient signal available is that 150 studies share a report with another
(bilateral knees or follow-ups). It is a floor, not a solution: two studies on one patient with
genuinely different reports still leak and we cannot detect it.

Gold is scored **pooled out-of-fold, never per fold**. 58 gold studies over 5 folds is 8–16 each,
and MCL lands at zero positives in two of them.

## Where this stands, and what is wrong — 2026-08-09

The pipeline works end to end and produces **macro AUC 0.719** (37 gold studies, images only,
224px, 60% of the corpus, no tuning). Defects are listed **by cost**, and every claim here is
measured.

> The headline was `0.743` until 2026-08-09. That run's artefacts were destroyed by K17 and the
> `_nifti_axes` fix then deleted 43 mis-reformatted cache entries; `0.719` is the same
> configuration on the corrected cache. Two identical re-runs agree to three decimals on every
> label, so **training is deterministic** and A/B differences here are real. See
> `IMPROVEMENTS.md` §1.3a.

### 1. Resolution — the 224 cache throws away half the detail, and it is the biggest lever

`normalise_and_resample` resamples to `TARGET_MM` = 0.35 mm/px and centre-fits to
`round(FOV_MM/TARGET_MM)` = **457 px**, the full 160 mm knee. Then `imagenet_normalise`
interpolates that straight down to `IMG_SIZE`:

```
224   0.714 mm/px over the 160 mm FOV    261 tokens/slice   <-- the 0.35 resample is discarded
518   0.309 mm/px  ~= as designed      1,374 tokens/slice
```

**The labels this costs are exactly the ones failing.** Against the extractor's own per-label
AUC as a control, vision *beats* text on six labels (Medial OA +0.198, Effusion +0.120) and
collapses on six (Lateral Meniscus −0.332, Fracture −0.274, ACL −0.206). Spearman between the
two columns is **−0.17 (p=0.60)** — label quality does not predict vision performance, which
rules out the pseudo-labels as the ceiling. What the split tracks is the physical size of the
finding: centimetre-scale fluid and bone morphology succeed, millimetre-scale tear and fracture
lines fail. At 0.71 mm/px those lines are about one pixel. Full table and method in
`IMPROVEMENTS.md` §2d.

This outranks the two ordering defects below: they target the four medial/lateral labels,
resolution targets six including the two at chance. Since §2 and §3 force a rebuild anyway,
**rebuild at 518 rather than 224** and both land in one job.

### 2. Slice direction is mixed — this blocks a valid submission

`validate_nifti.py` check 4b, stratified across all six series types (n=51):

```
forward  66.7%      Axial     12/12 forward
reversed 33.3%      Coronal   14/18 forward
                    Sagittal   8/21 forward   <-- majority REVERSED
```

The NIfTI affine carries no direction cosines, so nothing in the file distinguishes the two, and
`load_series_nifti` does not flip. `load_series` — which `kaggle_03` uses at test time — always
sorts ascending by `ImagePositionPatient` projection. So roughly a third of cached series run
opposite to the test path, in the axis medial/lateral discrimination depends on, and
`PREPROCESS_VERSION` cannot see it because the conversion happens upstream of the fingerprint.

Not predictable from plane: a plane rule scores ~72%. It must be resolved per series.

**This depresses 0.719 rather than inflating it.** The first verdict said "100% forward" because
every thumbnail in that sample was `Axial_0` — see §8.

### 3. Sagittal handedness was never corrected — K18

`canonicalise()` mirrors left knees onto the right, but only **in-plane**, and only for axial and
coronal. For sagittal, medial/lateral is the **slice axis**, and `vol[:, :, ::-1]` is the only
reversal in the pipeline — nothing ever touched axis 0. The docstring's argument that a sagittal
flip "would mirror the knee front-to-back for no gain" is true of the in-plane axis and never
considered the other one.

`spatial_order` sorts ascending along the slice normal; for sagittal that normal is the patient's
left-right axis, on which medial is +x for a right knee and −x for a left one. So one sort gives
lateral→medial for one knee and medial→lateral for the other, and `FusionHead.slice_pos` is a
**learned per-index** embedding — slice 5 meant lateral in one study and medial in the next.

Exposure: sagittal is the largest plane at **9,864 of 24,371 series (40.5%)**, and **1,894 of
4,407 studies (43.0%)** are left knees. Four of twelve labels are medial/lateral pairs.

Code is written and **gated off** behind `SAGITTAL_LR_SLICE_FLIP` — one switch for both readers,
because a per-call decision would canonicalise the test set and not the training cache. It is an
XOR against §2's direction bit, so the two corrections compose; it cannot be turned on until that
bit exists. Off is a byte-level no-op and the fingerprint is unchanged.

### 4. The leaderboard anchor exists — and it is not ours

| | score | note |
|---|---:|---|
| LB top | **0.940** | 908 teams |
| ours, submitted 2026-08-09 | **0.891** | unmodified fork of `pilkwang/rsna-knee-baseline-v1`; rank **230/908** |
| our own pipeline | — | never submitted |

So Phase 1's first half is done and the reference implementation reproduces. **The whole spread
from rank 230 to rank 1 is ~0.05 AUC**, which sets the scale for everything below.

What is still unmeasured is the mapping from *our* CV to the LB. `0.719` is pooled-OOF on 37
**enriched** gold studies — §1.1 of `FINDINGS.md` records the 58 as "clearly curated to cover
every finding", positive rates 16–60% — while the public LB is ~390 natural-prevalence studies.
Ours is the harder instrument: on an enriched set the negatives for ACL are frequently knees
positive for meniscus or OA, so the model must separate pathology from pathology rather than from
healthy. AUC is insensitive to the prevalence *ratio* (`PLAN.md` §8) but not to how hard the
negatives are. **0.719 and 0.891 are not on the same scale.**

In its current state — 224 px, 60% of the corpus, a third of series back-to-front — our own
pipeline would very likely score *below* the 0.891 already banked. Submitting it is therefore not
a score move. It is still worth one run, because `kaggle_03_submit.py` has **never executed
against a real test DICOM** and every untested path in this repo has held defects. Treat the
first own-pipeline submission as a dry run of the inference path that returns the CV↔LB mapping
as a side effect.

### 5. The moat comparison is stale

`0.777 vs 0.672` was measured against `nekkon`'s published label CSV. The canonical public
notebook today (`pilkwang/rsna-knee-baseline-v1`, 251 votes) ships its own extractor: clause
scoped, multilingual, emitting `(score, confidence)` pairs, negation scoped by punctuation,
laterality by tag with a geometry fallback, slices sorted by IPP, `Anatomical_Plane` used, flip
augmentation refused for the same medial/lateral reason as ours — and it independently found the
Greek **MICRO SIGN U+00B5** issue that `FINDINGS.md` §2.2 treats as a distinguishing discovery.

The §7.2 A/B therefore answers a question the field has moved past. Re-point it at that
extractor — which we have now forked and run: **0.891**, see §4.

### 6. Mechanically behind

They ensemble DINOv2 **and** EfficientNet across 224/336 by **rank mean**, and train on the gold
studies at weight 3.0. We run one backbone, one resolution, no ensemble, and hold all 58 gold
out. Our backbone is already the same checkpoint theirs is — there is no "switch to DINOv2" step
remaining.

### 7. The corpus — 81.7% available, 60% actually cached

The NIfTI mirror has parts 1–12 and 16–18; **13–15 return 403**. That is 3,599/4,407 studies and
47/58 gold. Re-check periodically — parts appeared twice on 2026-08-09.

Separately from availability, the built cache currently holds **2,649 of 4,407 studies (60%)**,
which is why the pooled OOF covers 37 of the 58 gold rather than 47. Finishing the build against
what is already downloaded is free data before any new part appears.

### 8. Defects found by review, now closed

`kaggle_03` still carried a fourth hand-written copy of the embedding loop — on the *test* side,
under a docstring in `preprocess.embed` asserting that copies had been consolidated so train and
test could not drift. Migrated, and verified bit-identical. And `IMPROVEMENTS.md` §1.3 stated the
shrink picked `m = 20` when the cross-fitted folds actually run 20/50/20/50/50.

**K17 — a `--synthetic` smoke run overwrote the real result directory, and the guard passed it.**
`fusion/train.py --out` defaulted to `fusion/runs` regardless of `--synthetic`, so the smoke
command in its own docstring destroyed the 0.743 checkpoints, OOF and summary; the directory is
gitignored, so nothing survived. Worse, synthetic mode wrote **no** manifest, leaving the genuine
one in place to vouch for random-tensor weights — `assert_matches()` reads only
`preprocess_version` and would have passed a submission of pure noise. Now: `--out` resolves to
`fusion/runs_synthetic` under `--synthetic`, synthetic runs write a self-marking manifest, and
`assert_matches()` refuses it before the version check.

### 9. The failure mode this project keeps repeating

K13, K14, K15, K16, K17, K18 and the defects found on 2026-08-09 are one species: **a claim about the
data written as reasoning and never measured**, guarded by a self-test that shares the same
assumption. The clearest case is a docstring that justified picking the slice axis by voxel
spacing as "~10x separation, unambiguous"; measured across all 19,859 series the minimum ratio is
**1.005** and 84 series are under 1.5x, which silently reformatted 27 axial acquisitions into
sagittal ones.

Four rules follow, and they are cheap:

1. No claim about the data in a docstring without the measurement that produced it.
2. Every self-test must build the **real** backbone at least once. K14 and the `no_grad`
   regression both hid behind an injected `embed_fn`.
3. Any validation that samples must print its **coverage per stratum**. The Axial-only verdict
   passed three documents unchallenged because nothing printed the breakdown.
4. When a correction is axis-dependent, enumerate **every** axis before concluding. K18's
   docstring reasoned correctly about the in-plane axis and never mentioned the slice axis, so
   half the fix read as the whole of it for three months.

## Where this goes next

> **REWRITTEN 2026-08-10 after the resolution test returned +0.013.** This section has now been
> reordered three times in two days. That churn is itself the symptom, and §0 below is the
> diagnosis. The plan that follows is built to stop it, and its rules matter more than its steps.

### 0. The root cause: there is no working instrument

Every macro this project has produced, across every experiment:

```
0.695   0.699   0.708   0.719        range 0.024
```

against a macro CI of **±0.038**. **Every result is statistically indistinguishable from every
other.** The soft-target ladder "losing", resolution "winning", the calibrated arm, the full
cache — none of those conclusions are supported by the instrument that produced them.

`PLAN.md` §7.2 diagnosed this for the extractor on 2026-08-07 and called it "out of instrument".
The fix chosen was a vision model — scored on the **same 37 gold studies**, so it inherited the
same blindness rather than escaping it.

This is why README §9's failure pattern keeps recurring. When measurement cannot decide,
decisions get made by reasoning, and reasoning is exactly what K3–K5, K14, K16 and K18 were. The
pattern is not a discipline problem. It is what a blind instrument does to a project.

**Meanwhile there is a working instrument that has been used once.** The leaderboard: ~390+
natural-prevalence studies, several submissions a day, and 1,025 teams' worth of calibration.
One submission exists — an unmodified `pilkwang` fork at **0.891**.

### The rules (these bind the steps below)

1. **The leaderboard is the instrument.** Gold OOF is a smoke test, not evidence. Nothing is
   "better" until the LB says so.
2. **One change per submission.** Two changes in one submission measure nothing.
3. **Nothing below the instrument's resolution.** If an effect cannot be seen, it does not get
   worked on — regardless of how interesting the mechanism is.
4. **No infrastructure that is not on the critical path to a submission.** The NIfTI route cost
   K16 and K18 — both impossible on DICOMs, which carry `ImagePositionPatient` — to buy an
   effect measured at +0.013.
5. **The fork is the base, not a reference.** It scores 0.891 and its inference path demonstrably
   works. `kaggle_03_submit.py` has never executed against a real test DICOM.

### Phase 1 — get an instrument (this week)

1. **Re-submit the unmodified fork** to confirm reproducibility, and establish its CV on the
   *full 4,407* studies — not on 37 gold. That CV is the local proxy; its correlation with the LB
   is the thing being calibrated.
2. **Two or three submissions of deliberately varied strength** to fit CV↔LB. Without this
   mapping every local number remains unreadable.

### Phase 2 — port the differentiators, one per submission

Each is a single change against the 0.891 base, kept only if the LB moves:

- **(a) Our labels vs its extractor.** `pseudo_labels.csv` is a drop-in target swap. This is the
  §7.2 A/B finally run against the field rather than against `nekkon`'s CSV.
- **(b) Our fusion head vs its pooling.** `PLAN.md` §7.1 calls ours "likely ahead" — **untested,
  and the largest unmeasured claim in the project.** Test it or drop the assumption.
- **(c) The 86 hand labels** as a validation set the field does not have.

### Phase 3 — the levers that measured largest

Ranked by what has actually been measured, not by interest:

1. **Data.** 1,000 → 2,649 studies was worth **+0.024**; 224 → 518 was worth **+0.013**. Data is
   roughly twice resolution, and the corpus sits at 60% of 4,407. Finish it.
2. **External data** (`PLAN.md` §3.4, "likely decisive", omitted by every route until now). MRNet
   supervises three of the six weak labels with real expert reads; OAI covers the OA three.
3. **Gold at weight 3.0.** The fork trains on all 58; we hold all 58 out. Measure the trade
   rather than assuming the honest choice is the right one.
4. **Rank-mean ensembling** across resolutions then backbones.

### Explicitly not doing

- The full 518 rebuild (~22 h). Measured **+0.013**, inside the CI. Revisit only if the LB
  disagrees.
- K16/K18 and the per-series direction export — **only if the local cache survives Phase 2.** On
  the fork's DICOM path these defects cannot occur, so the fix may be moot.
- Any further extractor refinement below the CI. §2.2 was worth +0.002 on gold.


## Status — 2026-08-09

- [x] Data logistics, language ID, series structure
- [x] Rule extractor v1 — macro AUC **0.777** on gold, 95% CI [0.74, 0.82]
- [x] Compartment attribution (`IMPROVEMENTS.md` §2.2) — ~1,000 studies off the flat 0.45.
      Invisible in gold macro (±0.038 CI); justified on corpus evidence, per §0
- [x] Hand-labelling UI + all 30 blind gold studies labelled
- [x] **Our labels beat the public weak labels** on gold and on hand labels — the moat is real
- [x] Series-metadata shortcut tested and rejected (0.471, below chance)
- [x] **Soft-target ladder fitted against gold — and the fit LOST** (`IMPROVEMENTS.md` §1.3a).
      `absent` is 52% of the target matrix and sits at 0.08 against a measured 0.167; correcting
      that scored **0.743 → 0.699** (both arms pre-K17, on the pre-deletion cache whose default
      baseline is now 0.719; the sign and mechanism are unaffected — see `IMPROVEMENTS.md`
      §1.3a). How far each label's `absent` was raised predicts how much
      AUC it lost (corr **−0.776**), because the `absent` bucket is heterogeneous and re-targeting
      it to its mean teaches the mean instead of the discrimination. Better calibration, worse
      ranking. Kept as a measurement, not a setting
- [x] **Training code complete and tested end-to-end on synthetic features** — fusion head,
      dataset, folds, training loop, submission notebook. Everything that does not need pixels
      is done; training is blocked only on the cache
- [x] **Laterality answered** (`kaggle_01` + `kaggle_01b`, `FINDINGS.md` §6.2) — the tag survives
      on 50% of the corpus, and the geometry fallback agrees 97.7% at the `x < −62` boundary, not
      the obvious `x < 0` (89.3%). Tag first, geometry second, source recorded per series
- [x] **Cache-build harness hardened** (`IMPROVEMENTS.md` §6) — the GPU guard, the spawn pool and
      the preprocessing-parity assert all had faults that cost or would have cost a session.
      Fixed and self-tested locally; **not yet run against real DICOMs**
- [x] **The cache is built — locally, from NIfTI, not on Kaggle** (`PLAN.md` §9.1 + its
      correction). Five Kaggle attempts died on the GPU lottery, the 9 h cap and ~19 ms/open on a
      network mount; none are properties of the data. `pipeline/build_cache_local.py` builds it on
      the M5 at a measured **1,062 studies/h** at 224. The conversion was validated against the
      DICOMs first (`pipeline/validate_nifti.py`, 5 checks). In-plane layout is `as-is` at
      **median r = 1.0000**, best for 98% of series across all six types — identical pixels, so
      the repackaging is faithful. **Slice direction is not**: see "What is wrong" below
- [x] **First vision-model result: macro AUC 0.719** on 37 gold studies, images only, 224px, on
      60% of the corpus with default hyperparameters. A floor, not a ceiling — 518px, the rest of
      the corpus and any tuning at all are still unspent. Synovitis scores **0.685** despite being
      the extractor's *worst* label (0.607), which is the first evidence for §2.1's option (b)
- [x] **Training is deterministic** — two identical re-runs give 0.719 and 0.719, agreeing to
      three decimals on every label. Every A/B in these documents measures a real difference with
      no run-to-run floor beneath it; the ±0.038 CI remains the *sampling* limit on n=37
- [x] **The pseudo-labels are not the ceiling — measured** (`IMPROVEMENTS.md` §2d). Vision beats
      the extractor on six of twelve labels and Spearman between the two per-label columns is
      **−0.17 (p=0.60)**. The bottleneck is the image path: the 224 cache resolves **0.71 mm/px**
- [x] **Resolution tested and it is small** (2026-08-10). 1,000-study 518 cache, 3.88 h, trained
      against the same 1,000 studies at 224: **+0.013 macro**, inside the ±0.038 CI. Right where
      predicted — Lateral Meniscus +0.067, Fracture +0.051 — and too small to plan around. The
      same run measured 1,000 → 2,649 studies at **+0.024**, so **data is worth ~2× resolution**
- [x] **The real diagnosis: no working instrument.** All four macros this project has produced —
      0.695, 0.699, 0.708, 0.719 — span 0.024 against a ±0.038 CI, so none of its conclusions are
      supported by the measurement that produced them. The leaderboard is the instrument and has
      been used once. See "Where this goes next"
- [x] **First LB submission — 0.891**, an unmodified fork of `pilkwang/rsna-knee-baseline-v1`,
      2026-08-09. Rank **230/908**; LB top is **0.940**. Phase 1's first half is done and the
      reference implementation reproduces. Our *own* pipeline has still never been submitted, and
      in its current state would likely score below this — see "What is wrong" §4
- [ ] ~~**DINOv2 feature cache built and published (`kaggle_02`)**~~ — superseded by the local
      build above. Kaggle-side remains the path for the TEST set, which `kaggle_03` always did.
      The five failed attempts are tabulated in `PLAN.md` §9
- [ ] Hand-labelling — 86/303 done; the remaining 217 are the only validation set we will have
- [ ] **First own-pipeline submission** — a dry run of `kaggle_03`, which has never executed
      against a real test DICOM. Needs two Kaggle Datasets that do not exist yet
      (`rsna-knee-fusion` with `fold*.pt` + `manifest.json`, and `dinov2-weights` — there is no
      internet in the kernel). `rsna-knee-code` exists and needs updating to HEAD
- [ ] **The §7.2 A/B** — but against the *current* public extractor, not `nekkon`'s CSV. See
      "What is wrong" §5
- [ ] LLM extractor (method B) — host undecided (`IMPROVEMENTS.md` §1.1); ~$44 batched on the
      API, or ~30–90 min a pass on a 5090
- [ ] External data — MRNet, OAI, fastMRI+ cover ~6 of 12 labels with real expert image reads
