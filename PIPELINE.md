# The build plan — 2026-09-07 → 2026-10-22

**45 days. Three independent pipelines, each with its own gate, run in parallel.**

Read `DATA.md` first — it sets the ceiling. `MODEL.md` is the architecture.

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

## Stage L — labels `Sep 8–15 · CPU/API · minutes per iteration` ⭐ HIGHEST IMPACT

`DATA.md` is the full spec. The public labels over-call gold **6.68 : 1** because ten of twelve
criteria carry a severity or acuity cut-point and the extractors detect mentions.

    4,349 reports
      -> severity extraction, per-label schema  -> data/severity_v1.jsonl
      -> threshold mapping                      -> data/targets_sev.csv  (+ schema hash)

**GATE — free, instant, no GPU.** Score the labels *as predictions* against gold-58:

    macro-AUROC   current public 0.8873   TARGET >= 0.93
    over-call     current        6.68:1   TARGET <  2:1

Iterate the prompt until both clear. **This is the fastest loop in the project — use it.**

**FALLBACK** if it stalls under 0.91 after a week: keep the public targets, note the ceiling in
`DECISIONS.md` as one line, move on. Do not spend a second week here.

**Artifact:** `data/targets_sev.csv`, versioned, plus the raw graded fields.

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
