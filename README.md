# RSNA Knee Abnormality Detection

Training and inference code for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition: twelve findings per knee MRI study (ACL, MCL, medial / lateral meniscus, medial / lateral /
patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture), scored by macro AUROC.

**Start at [`PLAN.md`](PLAN.md)**: the goal, where the score is lost, the current plan, what is closed, the map of
this repo and the commands. [`docs/LOG.md`](docs/LOG.md) is the dated record of every run.

## The problem in three lines

- 4,407 training studies come with radiology reports; only 58 carry grades read from the images. The hidden test
  (~1,300 studies) is graded from the images, so training labels are report-derived and noisy.
- A study is 3–14 MRI series in different planes, sequences and orientations, from many scanners.
- Inference runs offline on two T4 GPUs within 9 hours.

## The model

`pipeline/` reduces each study to a fixed stack: seven series slots (three sagittal, two coronal, two axial), 96
slices sampled across each series, a 140 mm field at 384 px, left knees mirrored to right. `fusion/train_windows.py`
trains a ConvNeXt-small on 3-slice windows of that stack, with a per-finding attention over windows that picks where
each finding is read. Labels are a consensus of public LLM report readers, refined with an image teacher.

Competition data, caches and weights are never committed (`data/` is git-ignored). Outside data with a data-use
agreement (OAI) never enters this repository.
