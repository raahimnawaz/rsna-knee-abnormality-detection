# The build plan — 2026-09-07 → 2026-10-22

**45 days. Three independent pipelines, each with its own gate, run in parallel.**

Read `DATA.md` first — it is the data plan, re-measured 09-06. `MODEL.md` is the architecture,
and per `DECISIONS.md` D4 it is where the score is: the labels we already have convert to ~0.938
and the model reproduces them at 0.849.

## The decision this rests on

**The local NIfTI path is abandoned.** §W measured it: identical config, folds and labels — our
pixels 0.7597, theirs 0.8453, 14.1σ. Five closed divergences did not move it. It is a third-party
repackaging covering 81.7% of the corpus. We stop paying for it and read DICOMs on Kaggle.

## Standing rules — these are what stop the circling

1. **Every stage ends in an artifact** — a published Dataset, a checkpoint, a submission. Not a
   finding.
2. **One gate per stage, written before the run. Fail → execute the written fallback.** Do not
   investigate a gate failure.
3. **Ship every week.** A submission every 7 days minimum, improvement or not. Never go 19 days
   blind again.
4. **Findings go in `DECISIONS.md` as one line.** If it does not change a stage here, it is not
   written down.
5. **L and P run at the same time.** L is CPU/API, P is Kaggle GPU, M is local. Never leave two
   idle — that is what 2026-09-06 cost.

---

## Stage 0 — bank the floor `Sep 7 · 1 day · 1 submission`

We sit at 0.914 / rank ~1374 while 0.936 is free.

Fork the public 0.936 notebook, run unmodified, submit the **kernel** not the file. Grep the log to
confirm every arm loaded — a refused arm silently ships the old path and burns a submission.

**GATE** LB ≥ 0.930. **FALLBACK** keep 0.914 banked and continue. Insurance, never the strategy.

---

## Stage L — labels `Sep 8–15 · CPU/API` — cheap, parallel, NOT the lever

⛔ **This stage lost its ⭐ on 2026-09-06.** `§3p` (restored, `DECISIONS.md` D4) measures our own
`targets.csv` at gold 0.8985 → LB ~0.938 against a 0.940 prize line, while the model reproduces
those same labels at only 0.8486. **The labels are not the binding constraint — we are further
from the labels we have than they are from truth.** Run L because it is CPU/API and costs Stage P
nothing, not because it is where the score is. The 0.05 is in `MODEL.md`.

`DATA.md` is the full spec and it was **rewritten on 2026-09-06 after an audit retracted two of
the three numbers this stage was built on** — reproduce them with
`python extractor/audit_label_stage.py`. The short version:

- **the 6.68:1 over-call is a binarisation threshold**, not a property of the labels. The same
  file reads 29.2:1 at cut 0.5 and 1.00:1 prevalence-matched, with macro-AUROC 0.8873 throughout.
  **The old "over-call < 2:1" gate was satisfiable by editing a constant. Deleted.**
- what is real, measured on the 30 studies that are both gold and hand-read: **graded output
  +0.086** (which the incumbent already banks) and **reading skill +0.055, 95% CI [+0.005,
  +0.111]**. That +0.055 is the honest headroom over `data/targets.csv`.
- the job is **six labels wide** — Effusion, Lateral OA, Contusion, Fracture, Lateral Meniscus —
  and **Synovitis is not an extraction problem at all** (mentioned in 12.4% of reports).

<!-- -->

    L0 fix glossary, re-export hand labels WITH their four states  (binarising costs 0.086)
    L1 finish the hand set: 217 of 303 remain -> the iteration instrument
    L2 graded extractor, criteria verbatim in the system block  -> data/severity_v1.jsonl
    L3 full pass for winners only, then the gate               -> data/targets_sev.csv
    L4 emit the `mentioned` mask   L5 version and freeze

**GATE — free, instant, no GPU, and PAIRED.** The marginal CI on gold-58 is 0.068 wide; pairing
against the incumbent on the same studies cuts the SD 4.1× and makes +0.008 the smallest
detectable delta.

    PRIMARY   paired macro-AUROC delta vs data/targets.csv, 4000 bootstrap resamples
              PASS  >= +0.010  AND  95% CI excludes 0
    GUARD     no single label falls more than 0.02 below the incumbent
    BUDGET    at most 3 candidates ever touch gold-58 -- iterate on the 303-study hand set

**Iterate on the hand set, not on gold-58.** §3b priced selection on 58 studies at *claims
+0.0137, delivers −0.0034*, and a large-n substitute was tested and rejected: the image model's
OOF ranks candidate label sets at ρ=0.429 (p=0.337) and puts its own training labels first.

**FALLBACK** three candidates spent without a pass → keep `data/targets.csv`, one line in
`DECISIONS.md`, move on. **This stage ends Sep 15 whether or not it has won.**

**Artifact:** `data/targets_sev.csv`, versioned, plus `mentioned` and the raw graded fields.

---

## Stage P — pixels `Sep 8–15 · ~8 h Kaggle GPU · RUNS IN PARALLEL WITH L`

`notebooks/kaggle_04_dicom_tiles.py`. **DICOMs, never NIfTI.** Geometry per `MODEL.md`: five slots,
64 slices sampled **6–94%** of each stack, 140 mm mm-crop via `PixelSpacing`, resize 336. Shard 8
ways; Kaggle caps concurrent GPU sessions at two. Publish as `rsna-knee-tiles140`.

⚠️ **Write every geometry constant into the manifest.** §R happened because `box_mm` was recorded
nowhere and silently defaulted.

**GATE** DINOv2-small, fold 0, the exact recipe that scored 0.7597 on NIfTI, same 691-study window:

    >= 0.83   pixel path fixed -> Stage M
    0.80-0.83 partial -> proceed anyway, do NOT chase the residue
    <  0.80   our DICOM decode is broken too

**FALLBACK** under 0.80: stop building arms, go straight to Stage B and compete as a blender. That
path is already worth ~0.94 and needs no pixels of our own.

---

## Stage M — model `Sep 16–Oct 5 · ~3 h Kaggle GPU per experiment`

`MODEL.md` is the spec. Build in this order; **1 and 2 ship together** — outer slices added to a
shared pool dilute the midline findings that currently work.

| # | change | gate |
|---|---|---|
| 1+2 | 64-slice coverage **and** per-label attention head | Lat. Meniscus and MCL **+0.05 each**; macro **+0.02** from the head |
| 3 | 140 mm crop | already measured at +0.027 (§W) — fold in, do not re-test |
| 4 | OrthoFoundation-L backbone, **fine-tuned not frozen** | parity with the DINOv2 arm, or **park permanently** |

Retrain on `targets_sev.csv` once Stage L clears. **If 1+2 land, one model should reach ~0.90
gold-58 alone** — which is what the frontier gets and no arm here has ever produced.

---

## Stage B — blend `Oct 1–12 · submit weekly`

Rank-mean over **families**, equal weight, nothing fitted (§9e). §2z: fitting weights is −0.0000.

| family | source | status |
|---|---|---|
| pilkwang, 20 members | public | held |
| RadImageNet R50 | public | held; NC licence carved out by rules §5a |
| **CoAtNet ×4** | `dreaddevelopment/raptor-*` | **public, unused, architecturally novel to us** |
| DINOv2 @ 140 mm | Stage P | ours |
| OrthoFoundation-L | Stage M-4 | ours, and unique |

**GATE per arm:** parity or better on the large-n instrument before it enters. Screen on
**strength** first — every arm below ~0.83 has lost. Price each arm's LB gain at ~half its gold gain.

---

## Stage 5 — buffer `Oct 13–22`

No new arms after Oct 12. Selection, kernel hardening, the ~2 h re-run margin, writeup. **Reserve
3 days for the submission itself failing.**

## Budget

| resource | available | planned |
|---|--:|--:|
| Kaggle GPU | ~195 h (30/wk) | ~30 h |
| submissions | ~97 (15/wk) | ~15 |
| M5 / MPS | unlimited | blending, scoring, Stage L |

**Resource-rich, time-poor.** Something should be running on Kaggle at all times.
