# RSNA Knee Abnormality Detection

Code for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition: twelve findings per knee MRI study (ACL, MCL, medial / lateral meniscus, medial / lateral /
patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture), scored by macro AUROC over the twelve.
Finals 2026-10-22.

Three documents, three jobs:

| | read it for |
|---|---|
| [`PLAN.md`](PLAN.md) | **the current state**: scores, what is pending, what is decided, what is closed |
| this README | what the code is, how a submission is built, how to run it |
| [`docs/LOG.md`](docs/LOG.md) | the dated record of every run, each rule written before its result |

## The problem

- 4,407 training studies come with radiology reports; only 58 carry grades read from the images. The hidden test
  (~1,300 studies) is graded from the images, so the training labels are report-derived and noisy.
- A study is 3–14 MRI series in different planes, sequences and orientations, from many scanners.
- Inference runs offline on two T4 GPUs within 9 hours.

## How a submission is built

A submission is one Kaggle notebook in two stages:

1. **The public base.** heliosli's 0.9576 notebook: nartaa's Raptor (two CNNs, partly trained on OAI image grades)
   and heliosli's R6 + R7 (ConvNeXt models on knee-anchored views), blended per finding. 0.958 on the public board.
2. **Our members**, mixed into the base per finding at fixed weights:
   - **the teacher folds**: five folds of our window model on the Stage B slice stack (below);
   - **KDC** (10-18): the same network on knee-centred crops, trained on Lateral OA / PF OA labels decoupled from
     Medial OA. The first member of ours whose errors differ from the base's.

There are two finals: the **main final** (base + our members) and an **OAI-free** one (**D**: R6 + R7 + our five
consensus folds, no Raptor), in case models trained on OAI data are ruled out.

Every submission kernel checks the sha256 of each checkpoint and re-runs a few training studies from their DICOMs
against their saved predictions (the **gate**) before it predicts the test. Kernels are generated from committed
code, so every one names the commit it was built from.

## Our models

- **The Stage B stack** (`pipeline/`): each study becomes seven series slots (three sagittal, two coronal, two axial),
  96 slices across each series, a 140 mm field at 384 px, left knees mirrored to right. The same code runs at test
  time.
- **The window model** (`fusion/train_windows.py`): ConvNeXt-small over 3-slice windows of the stack, with a
  per-finding attention over windows that picks where each finding is read.
- **K crops** (`pipeline/kcache.py`): the knee joint and patella located by the SKM-TEA segmenter; four regions
  (lateral sagittal, lateral coronal, patellofemoral axial, anterior sagittal), 8 slices each at 256 px, read by the
  same model and trained on the four weak findings only. K on report labels tied the teacher; KDC is K on the
  decoupled labels.

## Labels

- `labeling/build_consensus_targets.py`: a consensus of several public LLM readers of the reports.
- The teacher targets (`targets_teacher_v*`): the consensus refined with the predictions of an OAI-trained CoAtNet.
- `labeling/build_decoupled_targets.py`: KDC's targets. Lateral OA and PF OA from the reports, masked so they no
  longer track the other compartments.

The tables live in `data/student/` (git-ignored), and their sha256 are pinned in the training kernels.

## How results are measured

| instrument | what it is | use it for |
|---|---|---|
| held-out folds, `fusion/score_oof.py` | ~850 studies a fold, scored against report labels | "is it broken"; comparing models trained on the same labels. **Biased against a model that learns to disagree with the reports** (KDC: 0.711 here, ~0.860 on the board) |
| gold-58, `fusion/score_gold.py` | the 58 training studies graded from images | direction only (11–27 positives a finding) |
| single-finding probe, `build_submit_kernel.py --probe` | our model on one finding, the other eleven at 0.5: score = (5.5 + AUROC) / 12 | the real target, one finding at a time (a shown score is ±0.006 of AUROC) |

## The repo

| path | what |
|---|---|
| **`fusion/`** | **the model, inference and scoring** |
| `train_windows.py` | the trainer; also imported at test time for the model |
| `infer_windows.py` | test-time inference straight from DICOMs (what a submission runs) |
| `score_oof.py`, `score_gold.py` | held-out scoring with paired comparisons; gold-58 |
| `folds.py` | the five folds, grouped by scanner site |
| `qwen_vlm.py`, `orthofoundation_model.py` | two closed model types still wired into the trainer; removed after the finals (PLAN) |
| `runs_cnxs_teach1_f*` | held-out predictions of the five teacher folds (main-final members) |
| `runs_cnxs_cons2_f*` | the five consensus folds (D's members) |
| `runs_kdc_f0` | KDC fold 0 |
| `runs_k_f0..2` | K folds 0–2 (closed: tie the teacher) |
| `runs_coatnet_oai_teacher_pa` | the OAI-trained CoAtNet teacher |
| `runs_c6_ortho`, `runs_port_repro` | scoring controls the training kernel embeds |
| **`pipeline/`** | **DICOM → model input** |
| `slots96.py`, `slots64.py`, `build_slots64.py` | the slice stack: Stage B (96 slices, used) on Stage A's base module, and its builder |
| `slots118.py` | Stage C (closed; still wired into the builders, removed after the finals) |
| `kcache.py` | K's knee-centred crops |
| `site_fingerprint.py`, `provenance.py` | scanner-site fingerprints for the folds; sha256 of external inputs |
| **`labeling/`** | the target tables (above) |
| **`extractor/metrics.py`** | AUROC |
| **`notebooks/`** | **Kaggle kernel builders and offline tests** |
| `build_backbone_ab_kernel.py` | a training kernel (Stage B or K) |
| `build_kcache_kernel.py` | CPU kernels that build the K crop cache |
| `build_submit_kernel.py` | our model as a submission, a stage, or a single-finding probe |
| `build_r6r7_kernel.py`, `build_public_blend_kernel.py`, `build_r6r7_bundle.py` | the public base with our stage; R6 + R7 in one decode |
| `build_finals_mounttest.py` | a free CPU check of a final's mounts, hashes and pins |
| `finals_wheel_pins.json` | the pinned wheels of the finals |
| `build_weights_sha_kernel.py` | sha256 of checkpoints on Kaggle |
| `offkaggle.py`, `submit_at_reset.py` | training off Kaggle; a timed submitter |
| `*_fixture.py` | offline tests (no GPU, no Kaggle) |
| **`provenance/`** | `inputs.json` (pinned inputs), `folds_site.csv` (the folds) |

Competition data, caches and weights are never committed (`data/` is git-ignored). Outside data with a data-use
agreement (OAI) never enters this repository.

## How to run

`K=.venv/bin/kaggle`, `PY=.venv/bin/python`, `C=rsna-knee-abnormality-detection`. Images live only on Kaggle.

```bash
# a training kernel: one fold of K / KDC (Stage B: --corpus stage_b --recipe v3)
$PY notebooks/build_backbone_ab_kernel.py --arch convnext_small --corpus stage_k --folds 0 --n-train 0 --recipe k \
    --time-budget-h 10 --targets data/student/targets_teacher_v3_dc.csv \
    --targets-from raahimnawaz/rsna-knee-student-targets --tag kdc --out <dir>
$K kernels push -p <dir>                 # read `fit-budget: … -> 16 epoch(s)` in its first minutes

# its results (small files only), scored against the teacher on the same studies
$K kernels output raahimnawaz/<kernel> -p fusion/runs_<name> --file-pattern '.*\.(json|csv|txt|log)$'
$PY fusion/score_oof.py fusion/runs_cnxs_teach1_f0/runs_convnext_small/fold0 fusion/runs_<name>/runs_convnext_small/fold0/avg3

# a submission or probe: hash the weights on Kaggle, build, push, read the log, then submit
$PY notebooks/build_weights_sha_kernel.py --kernels raahimnawaz/<training kernel> --out <dir> && $K kernels push -p <dir>
$PY notebooks/build_submit_kernel.py --ours-only --segmenter --probe "Lateral OA" \
    --weights-kernel raahimnawaz/<training kernel> --weights-sha <hex> --tag <t> --desc "<d>" --out <dir>
$PY -I notebooks/build_r6r7_kernel.py $PWD data/public_notebooks/heliosli-0-9576-16a9d953 <dir> --tag <t> \
    --desc "<d>" --main958 --wheel-pins notebooks/finals_wheel_pins.json
# + KDC as a second stage, on Lateral OA only (without these flags the build is byte-identical)
#   ... --k-weights-kernel raahimnawaz/<k1>,<k2>,<k3> --k-weights-sha <h1>,<h2>,<h3> --k-mix "Lateral OA=<share>"
$PY notebooks/build_finals_mounttest.py --final <dir> --out <mt dir>      # a free CPU check of mounts, hashes, pins
# in its log before submitting: `sha256 ok` · `GATE PASSED (n/n)` · 0 tracebacks
$K competitions submit $C -k raahimnawaz/<kernel> -v <n> -f submission.csv -m "<what, against what, and the rule>"

# offline tests
$PY notebooks/w4_fixture.py; $PY notebooks/public_blend_fixture.py; $PY notebooks/stage_a_fixture.py
$PY notebooks/mirror_fixture.py --out <dir>; $PY pipeline/kcache.py --check
$PY notebooks/canon_fixture.py --out <dir> && $PY notebooks/infer_parity_fixture.py --fixture <dir>
```

Kaggle's CLI can hang: wrap calls in `perl -e 'alarm 60; exec @ARGV' …` (macOS has no `timeout`). Kaggle runs at most
two GPU sessions at once and five submissions a day (reset 17:00 PDT).
