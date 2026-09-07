# Decisions ledger

One line per settled question. **Consult it; do not extend it.** A finding that does not change
`DATA.md`, `MODEL.md` or `PIPELINE.md` does not get written down at all.

Evidence lives in `docs/archive/IMPROVEMENTS.md` and `docs/archive/PLAN.md` at the § named. Those
files are provenance — not maintained, not read in full.

---

## The three that shape everything `MEASURED 2026-09-06`

| # | finding | number |
|---|---|---|
| **D1** | ~~**Labels over-call gold 6.68 : 1**~~ **⛔ AMENDED 2026-09-06 — the ratio was a binarisation threshold.** Same file: 29.2:1 at cut 0.5, 6.63:1 at 0.60, 1.00:1 prevalence-matched, **macro-AUROC 0.8873 throughout**. The over-call gate was deleted. What survives is measured on the 30 gold∩hand studies: **graded output +0.086** (the incumbent already has it) and **reading skill +0.055, CI [+0.005, +0.111]**. → `DATA.md`, `extractor/audit_label_stage.py` | +0.055 over `targets.csv`, not a ratio |
| **D2** | **We sample 18 slices, all at `depth 0.5`.** The frontier takes 64 across 6–94% of the stack. Lateral meniscus and collateral ligaments sit in the outer thirds. → `MODEL.md` | Lat. Meniscus 0.6422, MCL 0.6433, ACL (midline) 0.8689 |
| **D3** | **Our pixels are a third-party NIfTI repackaging and it costs 0.085.** Same config, folds, labels. → `PIPELINE.md` Stage P | ours 0.7597, theirs 0.8453, 14.1σ |

| **D4** | **The labels are not the binding constraint** — `§3p`, measured 08-13, **restored 09-06 after being dropped in that day's rewrite.** `steven_v2` (= our `targets.csv`) used as a predictor reads gold 0.8985 → LB ~0.938 at §3v's +0.039, against a prize line of 0.940. Our model reproduces **its own training labels** at 0.8486. **We are further from the labels we have than those labels are from truth.** → `DATA.md` §0 | teacher→gold 0.10, model→teacher 0.15 |

**What D1 and D4 together mean:** Stage L is cheap, parallel and raises a ceiling that will bind
later — it is **not** the highest-impact stage, and `PIPELINE.md`'s ⭐ was wrong. The 0.05 is in
`MODEL.md`. Run L because it costs a different machine, not because it is the lever.

## Board `MEASURED 2026-09-06 — expires in days; re-survey before any submission decision`

3,239 teams. Our 0.914 ranks **~1374**. Free public notebook **0.936** (8+ teams sit on it
exactly). Public frontier **0.941** — source published, single model, no ensemble, no TTA. Prize
line 0.940 ≈ rank 100. Top 0.954, and ranks 1–100 span only 0.014. `pilkwang`, the fork this repo
is built on, is now rank 899.

## CONFOUNDED-160 — verdicts measured on a 0.476 mm/px model

The crop confound was worth **+0.0268 (4.1σ)**, so each of these was measured ~0.027 too low. It
closes **23.8%** of the band gap; the rest is D3. **These are not `CLOSED` — re-measure before
citing.** The stop rules were honoured and that discipline stands: a miscalibrated *instrument* is
grounds to re-measure, a disappointing *result* is not.

| § | claim as written | why it does not stand |
|---|---|---|
| C-5 | OrthoFoundation-L fails its gate at 0.7710 | frozen features, 160 mm cache. **The one arm the field lacks — re-measure** |
| 2y / 2w-4 | our port earns no ensemble slot, −0.111 | the port was 160 mm |
| 3w-2 | Workstream C refuted; *our training* is refuted | both arms 160 mm — the CNN-vs-ViT paired delta survives, the absolute verdict does not |
| 3y-3 | Stage 1's bar is 0.7428 | bar set from a 160 mm control |
| 9e | five encoders, one band, so the encoder is not the lever | all five shared the defect |

## CLOSED — valid instrument, do not re-derive

| route | verdict | § |
|---|---|---|
| crops as extra TTA windows | dead on both instruments, pre-registered | 3k, 3m |
| blending the fork's `merge_gain` arm | +0.0007, 0/12 labels | 3h-2 |
| K16 from DICOM header rules | 56.9–60.8% vs ~50% chance | 2n, 2m |
| our rule extractor as a label source | last of six, 0.777 vs a free 0.893 | 2f |
| per-label fusion of public label readers | near-duplicates, loses to best single | 2i |
| rank-mean `pilkwang` + `prvsiyan` | prvsiyan already contains pilkwang | 3c |
| re-fitting blend weights on site-grouped folds | −0.0000 — **do not fit weights** | 2z |
| harmonising scanner/site away (ComBat) | −0.013 to −0.032; case mix is real | 3f |
| selecting anything on gold-58 | claims +0.0137, delivers −0.0034 | 3b |
| the over-call ratio as a label gate | a threshold, not a property — AUROC invariant across the whole sweep | audit §1 |
| "79.0% → 84.7% cell agreement" as the label headroom | same artifact; prevalence-matched already reads 84.8% | audit §1 |
| the image-model OOF as a large-n label gate | ρ=0.429 p=0.337 vs gold-58 order, and it ranks its own training labels first | audit §5 |
| exporting our own labels as bits | costs **0.086 macro**, measured. Grade, or do not bother | audit §3 |
| **weighting label sources by accuracy** — reverse-log, rank-discount or sharpened | **works as designed and still cannot win.** Equal rank-mean −0.0041 vs best single; reverse-log `−log(1−acc)` −0.0005; `acc^8` −0.0005. Sharpening the weights just converges the blend *to* the best member. Cause: the sources are one read repeated — steven_v4↔v2 **r=0.995**, LLM-to-LLM mean **0.819**. Weighting repairs the harm of including weak members; it cannot manufacture absent signal | `bench_external.py` |
| **building our own severity extractor from scratch** | the public frontier **already ships one** — `SEV_LOW`/`SEV_HIGH` in 9 languages, directional negation, OA inheritance, graded pathology, Synovitis backoff — and it scores **0.8617 vs our incumbent 0.8873** (paired −0.0255, P(better)=0.009). Its own `read_labels()` refuses to train on it, calling it *"the weaker labels"* | `bench_external.py` |
| a single severity prompt as the whole fix | six labels, six causes; Synovitis is mentioned in **12.4%** of reports and is an imputation problem | audit §2, §6 |
| post-hoc calibration / thresholds / priors | **AUROC is invariant to monotone transforms** | 3a |
| forking prvsiyan for its 0.906 | needs a private dataset | 3c |
| a C++ port for the efficiency track | 93 s per 0.001 AUC; decode is already native | PLAN 6.1 |
| slice **reversal** as a decode fault | null: +0.0007 at 1.5σ | 3i-7 |
| the residual's *magnitude* explains the band | 0.0165 of perturbation costs 0.0013 | 3i-6 |
| tensorised images at submission time | structurally impossible | PLAN 9g |
| "the model cannot localise the compartment" | **superseded by D2** — it is not being shown the slices | §U |

## Standing cautions — these were paid for

- **A submission costs ~2 h, not 74 s.** ~15/week against a 30 h quota. Batch them. `3e`
- **A leaderboard number older than a few days is not evidence.** Five beliefs about the outside
  world have expired here; one lasted 24 hours. `2x, 3l-1`
- **Launch long runs under `caffeinate -i`.** Also watch RAM: the tile cache is mmap'd, and at 9%
  free memory an epoch went 10.6 → 41.3 min on page eviction alone. `2v`
- **Read a competitor's code, never its description.** Three claims about `pilkwang` came from its
  description and two were wrong — and the frontier's *own header* is how D2 was found.
- **Baseline is macro 0.7229 ± 0.0048**, site-grouped. Anything before 2026-08-10 is inflated by
  ~0.024 of site leakage. `2j`
- **Price a future arm's LB gain at ~half its gold gain.** `4c-4`
- **Never price a forward pass by parameter count — `bench` it.** Predicted 14×, measured 6.33×. `C-5`
- **Licences:** RadImageNet is `CC-BY-NC-SA-4.0`, and competition rules §5a explicitly carve out
  incompatibly licensed pretrained models — **not a blocker**. OrthoFoundation has **no LICENSE at
  all**; its only grant is the authors' email of 2026-08-18. **Preserve that email.**
- **Write geometry into the manifest.** §R happened because `box_mm` was never recorded anywhere.
- **⛔ Two public label tables have GOLD BAKED IN** — `rayanbabur_soft` and `yunus_v5_merged`
  reproduce all 58 gold rows exactly, 0/1-valued there against 506 distinct values elsewhere.
  They score a fake **1.0000** on gold-58. `bench_external.py` refuses to report a score for a
  source that fails its leak check. **Never benchmark a downloaded table without that check.**
- **The public-source survey expires.** The 08-10 survey was 27 days stale on 09-06 and had
  missed **four** new label tables, one of them a different model family (Qwen3-8B). Re-run
  `kaggle datasets list -s "rsna knee"` before treating any source list as complete.

## The failure mode this project repeats

**A real measurement, aimed at a question it cannot answer.**

- the training-loss curve, against a ranking gate — `[[loss-curve-is-not-evidence]]`
- the pixel residual's magnitude, against the band gap — `3i-6`
- the memorisation probe, dynamic range 0.0102, against a 0.076 effect — `3i-7`
- `crop_mm` read from a module `train_port.py` does not import — `2y-2`
- seven runs sharing one defect, read as evidence about encoders — §R
- **lowest label *ceiling*, read as most model *headroom*** — corrected the same day, §T→§U
- **a compartment-attention architecture designed for what was a slice-sampling bug** — §U→D2

**Before believing a number, name the question it can and cannot answer.**
