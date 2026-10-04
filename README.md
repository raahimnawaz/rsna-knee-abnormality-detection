# RSNA Knee Abnormality Detection

A training and inference pipeline for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
competition on Kaggle: twelve findings per knee MRI study, scored by macro AUROC.

The twelve findings are ACL, MCL, medial and lateral meniscus, medial / lateral / patellofemoral osteoarthritis,
effusion, synovitis, Baker's cyst, contusion and fracture.

## The problem

- **Weak labels.** 4,407 training studies come with radiology reports; only 58 carry radiologist grades read from the
  images. The hidden test set (about 1,300 studies) is graded from images. Training labels are therefore report-derived
  and noisy, and the 58 "gold" studies are the only local check on what the leaderboard measures.
- **Heterogeneous input.** A study is 3 to 14 MRI series (median 5) in different planes, sequences, resolutions and orientations,
  from many scanners.
- **Tight inference budget.** Submissions run offline on two T4 GPUs within 9 hours.

## Results

Public leaderboard, macro AUROC, as of 2026-10-03. The competition closes on 2026-10-22.

| Model | Public LB |
|---|--:|
| ViT-L, 64-slice input, public report labels | 0.901 |
| + denser, orientation-canonical input (96 slices at 384 px) | 0.915 |
| + better report-label table (64-slice input) | 0.916 |
| + both, with image-teacher targets | 0.927 |
| + trained on all 4,349 non-gold studies (**final single model**) | **0.929** |
| Strongest public single model (CoAtNet), for reference | 0.933 (0.928 with its author's own inference) |
| Strongest public ensemble, for reference | 0.943 |
| Public ensemble with our fine-tuned CoAtNet as its core | 0.943 |

The single model was trained end to end with this repository and scores close to the best public single model. The
best submission equals the public ensemble: replacing its core with our model did not improve it.

## Approach

**1. Fixed-slot slice cache** (`pipeline/`). Each study is reduced to a fixed tensor. Series are assigned to slots by
plane and sequence type (seven slots: three sagittal, two coronal, two axial). Slices are sampled evenly across the
central 2-98% of each stack, cropped to a 140 mm field of view in physical units, and resampled to 384 px. Left knees
are mirrored onto right knees and scanner sign conventions are undone, so "medial" is always on the same side.

**2. 2.5D windows with attention pooling** (`fusion/train_windows.py`). Three adjacent slices form one RGB window (up to
82 windows per study). A ViT-L encoder ([OrthoFoundation](https://arxiv.org/abs/2601.18250): DINOv3 continued-pretrained
on knee images) embeds each window. A per-finding attention head pools the windows, with learned slot and depth
embeddings that start at zero. Training draws 12 windows per study per step; inference scores every window.

**3. Report labels refined by image teachers** (`labeling/`, `fusion/local_teacher_folds.py`). Four ResNet34 fold
models are trained on the report labels. Each predicts only the studies it never saw. Their predictions are
rank-mapped onto the report label's own levels and averaged with it, giving the student's targets.

**4. Training on Kaggle's free GPUs** (`notebooks/build_backbone_ab_kernel.py`). Distributed training across two T4s
with fp16, gradient checkpointing on half the encoder blocks and `torch.compile`. One epoch over 4,349 studies takes
44 minutes.

**5. Submission** (`notebooks/build_submit_kernel.py`, `fusion/infer_windows.py`). Inference reads raw DICOMs with the
same code that built the training cache. The model can be submitted alone, rank-averaged into the public ensemble as
an extra family, or swapped in as the ensemble's core.

## Engineering practices

- **Kernels are built, not edited.** Every Kaggle notebook is generated from committed code. Source files are embedded
  and hash-checked at run time, and the build records the commit it came from.
- **Pinned inputs.** External tables and weights are held to SHA-256 records (`provenance/`); a changed upstream stops
  the run instead of silently moving the folds or targets.
- **Parity gate before every submission.** The submission kernel rebuilds eight training studies from raw DICOMs and
  must reproduce the model's recorded out-of-fold predictions (maximum difference under 0.03) before it writes a
  submission file.
- **Offline tests on synthetic DICOMs.** Phantoms with known answers check slice ordering, physical-unit cropping,
  anisotropic spacing and left/right canonicalisation, each with a control that proves the test can fail.
- **Paired evaluation.** Runs are compared on the same held-out studies with a paired bootstrap. Each comparison had
  its pass line written down before the result was read, and checkpoints were never selected on the gold studies.

## What was measured

| Change | Result | Kept |
|---|---|---|
| Denser, canonical input (96 slices at 384 px) | +0.014 LB | Yes |
| Better report-label table | +0.015 LB | Yes |
| Image-teacher targets | +0.026 on gold; at most +0.005 on LB | Yes, unproven |
| Training on all data instead of holding out a fold | +0.002 LB (noise) | Yes |
| Fine-tuned CoAtNet as the public ensemble's core | No change (0.943) | No |
| Zoomed crops | +0.016 on held-out data, mostly a resolution effect | No |
| External knees with pseudo-labels | +0.004 ± 0.005 | No |
| Fitted cross-finding stacker | −0.013 on gold | No |
| Per-site prior | −0.006 | No |
| Scanner harmonisation | −0.013 to −0.032 | No |
| Boosted trees on series metadata alone | 0.471 AUROC (chance) | No |

Two lessons stand out. Gains measured on 58 gold studies overstated leaderboard gains by a wide margin, while a
691-study held-out fold tracked the leaderboard closely (correlation 0.996 over four runs). And a ViT-L and a
CoAtNet trained on the same report labels landed within 0.005 of each other, which points at labels and training
recipe, not architecture, as the limit.

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | Slice-layout specs, the DICOM-to-cache builder, scanner fingerprints, input provenance |
| `fusion/` | Trainer, inference, encoder loaders, fold assignment, scoring |
| `fusion/runs_*/` | Run records: configuration, out-of-fold predictions, scores (no weights) |
| `labeling/` | Student-target construction from teacher predictions |
| `extractor/metrics.py` | AUROC and balanced accuracy |
| `notebooks/` | Kaggle kernel builders and offline test fixtures |
| `provenance/` | SHA-256 records and the site-grouped fold assignment |

## Running it

```bash
python -m pip install -r requirements.txt
```

Offline tests need no competition data:

```bash
python notebooks/canon_fixture.py --out /tmp/canon
python notebooks/infer_parity_fixture.py --fixture /tmp/canon
```

The pipeline itself runs on Kaggle, where the images are:

```bash
# 1. slice cache: one CPU kernel per shard
python notebooks/build_stage_a_kernel.py --stage b --shard 0 --out /tmp/cache
python -m kaggle kernels push -p /tmp/cache

# 2. training: one GPU kernel
python notebooks/build_backbone_ab_kernel.py --arch ortho --corpus stage_b --n-train 0 --epochs 11 \
    --targets <targets.csv> --targets-from <kaggle dataset> --tag <name> --all-data --out /tmp/train
python -m kaggle kernels push -p /tmp/train

# 3. submission: our model alone
python notebooks/build_submit_kernel.py --ours-only --weights-kernel <owner>/<training kernel> \
    --weights-sha <sha256 of last.pt> --tag <name> --out /tmp/sub
python -m kaggle kernels push -p /tmp/sub
```

Competition data is licensed by RSNA and Kaggle and is not included.

## Acknowledgements

- Public ensemble notebook by `mattiaangeli` (Apache 2.0) and CoAtNet weights and report-label table by
  `dreaddevelopment` (CC0), used as the reference ensemble and as the base label table.
- [OrthoFoundation](https://arxiv.org/abs/2601.18250) pretrained weights.
- Scanner metadata probe by `zhukovoleksiy`, used for site-grouped folds.
