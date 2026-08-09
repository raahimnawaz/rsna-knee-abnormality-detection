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
   earned rather than assumed. Measured in `PLAN.md` §7.1.

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

## Status — 2026-08-08

- [x] Data logistics, language ID, series structure
- [x] Rule extractor v1 — macro AUC **0.777** on gold, 95% CI [0.74, 0.82]
- [x] Compartment attribution (`IMPROVEMENTS.md` §2.2) — ~1,000 studies off the flat 0.45.
      Invisible in gold macro (±0.038 CI); justified on corpus evidence, per §0
- [x] Hand-labelling UI + all 30 blind gold studies labelled
- [x] **Our labels beat the public weak labels** on gold and on hand labels — the moat is real
- [x] Series-metadata shortcut tested and rejected (0.471, below chance)
- [x] **Soft-target ladder fitted against gold — and the fit LOST** (`IMPROVEMENTS.md` §1.3a).
      `absent` is 52% of the target matrix and sits at 0.08 against a measured 0.167; correcting
      that scored **0.743 → 0.699**. How far each label's `absent` was raised predicts how much
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
      DICOMs first (`pipeline/validate_nifti.py`, 5 checks): in-plane layout `as-is` at
      **r = 1.0000** — identical pixels — and slice order 100% forward
- [x] **First vision-model result: macro AUC 0.743** on 37 gold studies, images only, 224px, on
      61% of the corpus with default hyperparameters. A floor, not a ceiling — 518px, the rest of
      the corpus and any tuning at all are still unspent. Synovitis scores **0.777** despite being
      the extractor's *worst* label (0.607), which is the first evidence for §2.1's option (b)
- [ ] ~~**DINOv2 feature cache built and published (`kaggle_02`)**~~ — superseded by the local
      build above. Kaggle-side remains the path for the TEST set, which `kaggle_03` always did Four attempts,
      none finished, none of them failing at the modelling: see `PLAN.md` §9 for the table. Next
      move is a 224 shard 0/4 proving run, judged on the PROBE line within minutes
- [ ] Hand-labelling — 86/303 done; the remaining 217 are the only validation set we will have
- [ ] **First LB submission** — fork a public DINOv2 baseline. Independent of the cache, ~an
      hour, and nothing else is measurable until it exists. It should stop waiting behind §9.1
- [ ] **The §7.2 A/B: fusion head trained twice on identical folds, our labels vs `nekkon`'s.**
      The only test of whether the label moat survives contact with a model
- [ ] LLM extractor (method B) — host undecided (`IMPROVEMENTS.md` §1.1); ~$44 batched on the
      API, or ~30–90 min a pass on a 5090
- [ ] External data — MRNet, OAI, fastMRI+ cover ~6 of 12 labels with real expert image reads
