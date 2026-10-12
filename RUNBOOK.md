# RUNBOOK — the current state, on one page

**Any session starts here.** Branch `labels-small-backbone` (off `main`, which stays the showcase tree). History up to
10-03 is at commit `9c23daf`: `git show 9c23daf:RUNBOOK.md`, `9c23daf:DATA.md`, `9c23daf:docs/REVIEW-2026-10-04.md`.

## Goal — 0.97 (owner, 2026-10-11)

Fix the labels, switch to a small backbone, train the best models of our own. **The public blend is not submitted
and not built on** (owner: "its useless"). Solo. Final 10-22 16:59 PDT; two finals.

## Scoreboard (pulled 10-11 ~15:40 PDT; re-pull before using, rule 8)

| | LB | note |
|---|--:|---|
| #1 | 0.966 | 20th 0.961; 100th 0.956; 5,788 teams |
| free public blend | 0.954 | 495 teams exactly there. `nartaa` CoAtNet (0.949 alone, 0.948 without its mirror pass) + `heliosli` ConvNeXt-small, 4 folds |
| ours, best | 0.943 | 10-03: the old public stack with our CoAtNet student as its core |
| ours alone, best | 0.929 | Ortho ViT-L, Stage B, teacher targets, all data (ref `56811647`) |
| stack + that Ortho at 1/3 | 0.941 | ref `56813945`, read 10-04: FAIL (rule ≥ 0.947) |

The public CoAtNet went 0.933 → 0.948 on the same architecture as our CoAtNet student. Its README names what
changed: several label sources blended (steven, pilkwang, JEV, its own read) with **masked** supervision, 2,399 OAI
knees, 96 slices at 384 px, weights averaged over three epochs. Its training script is our trainer's ancestor (same
optimiser, k = 12, per-finding attention) **without** our confidence weight.

## What this branch adds

- **Small backbones** (`train_windows.py --arch convnext_small | convnext_tiny`, 384 px, LayerNorm only). One
  ViT-L run was 8.3 h on both T4s; these are meant to make a fold of five cost what one run did.
- **`--conf-weight none`**: plain BCE. The old weight gave a 0.45 "below threshold" target a third of the loss.
- **Averaged weights are kept**: `avg3/last.pt` beside `last.pt` (they were scored, then dropped).
- **Fold CV in one kernel**: `build_backbone_ab_kernel.py --folds 0,1,2,3,4` → `runs_<arch>/fold<k>/`, the held-out
  predictions joined in `cv_last` / `cv_avg3` and scored; gold is in no training list (fold mean, report only).
- **Ensembles are servable**: `infer_windows.py --weights a.pt b.pt ...` builds each study once and averages;
  `build_submit_kernel.py --weights-sha sha,sha,...` gates every member on its own held-out predictions.
- **`build_submit_kernel.py --probe FINDING`**: one finding against eleven constant columns, so the public score is
  (5.5 + AUROC) / 12. One tick = 0.012 on that finding. This is the instrument for the weak four.

## Labels — `targets_consensus_v2` (`labeling/build_consensus_targets.py`)

- File `data/student/targets_consensus_v2.csv`, sha256 `a1ed4034f3728d5f…`; control `targets_consensus_v2_synbase.csv`
  (`78b9c9da…`: the same eleven columns, the base table's filled Synovitis). Both are in the private Kaggle dataset
  `raahimnawaz/rsna-knee-student-targets`.
- **Eleven findings:** seven readers of the same reports (steven_v2, pilkwang_v2, yunus, JEV FINAL2, nartaa's raw
  read, vmohitrao's graded states; base = `dread_soft`). Each reader's order is mapped onto `dread_soft`'s own levels
  and averaged; the target is half `dread_soft`, half that consensus. Nothing is fitted to gold.
- **Synovitis:** reports address it in ~14–22% of studies; `dread_soft` fills the other 3,751 cells from Effusion
  (0.35 / 0.76 / 0.82). Here a cell is supervised only where a reader read it (687) or where Effusion is read as
  absent (1,258, target 0.05). The other 2,404 cells are NaN and carry no loss (`--allow-missing-labels`). 45% is
  supervised; the public author's own mask supervises 49.5% and agrees with ours on 78% of cells.
- `lixin73`'s table is NOT a reader: `score_oof.py` scores against it.
- Public tables with gold inside (never use): `tasmeemreza`, `yehezkielhaganta`, `dhiyaneshbabu`,
  `lvthanhxun` fused, `gperdrizet`.

## Pending reads — each rule written before its result

1. **Is the small backbone a workhorse?** ConvNeXt-small, Stage B, `targets_consensus_v2`, fold 0, read on the 691
   (paired with `runs_c6_ortho` by `score_oof.py`). Reference points on the same 691: Ortho r1 0.8534, Ortho c7
   0.8410, Ortho c6 0.8298. **≥ 0.840 → it is the workhorse** and the five folds are trained; **< 0.825 → the recipe
   is wrong** (learning rate first), fix it on fold 0 before any more folds. Between: train the folds, and bench
   ConvNeXt-tiny against it. A screen, not a verdict: the labels differ from every reference run's.
   **✅ READ 10-11 (run complete 18:17 PDT): 0.8654 → the workhorse (rule: ≥ 0.840).** On the 681 scored studies of
   the 691: ConvNeXt-small 0.8654 (avg3 0.8653) against c6 0.8298, paired **+0.0356 ± 0.0039**; Ortho r1 0.8534 and
   c7 0.8410 on the same studies. 10 of 12 findings up on c6; Effusion down (0.824 vs 0.859: the reference is another
   reader's Effusion, and the c6 labels sit closer to it). 12/12 epochs, **7.8 min/epoch** on both T4s (Ortho: 36–44),
   104 min in all, peak 8.1 GB, 0 tracebacks. Gold-58, report only: 0.928 (Ortho r1 0.911, paired +0.0165 ± 0.0074;
   Synovitis 0.810, unchanged by the mask). Equal rank-mean with Ortho r1 on the 691: 0.8649, no gain (rule 4).
   The old fit LB = 1.091 × s691 − 0.004 would put this one model at ~0.940; it was fitted on other labels, so
   read 2 decides.
2. **The five-fold ensemble alone on the LB**, against Ortho all-data 0.929. Model against model on ~400 studies is
   ±0.005, so |Δ| < 0.010 is noise.
3. **Labels on the LB:** the same recipe on `dread_soft` against `targets_consensus_v2`, each alone. Consensus is
   kept if it is not below `dread_soft` by ≥ 0.005; it has *shown* a gain only at ≥ +0.010.
4. **The Synovitis mask**, by `--probe Synovitis`: masked against `_synbase`. Two ticks (0.024 on Synovitis) either
   way decides it; less is a tie and the mask stays (it removes a fill, it invents nothing).

## Running now (written 10-11 23:50 PDT; the Mac was on battery at 38% and sleeping between steps)

- **Fold 0 alone on the LB (read 2, first half):** commit-run kernel `raahimnawaz/rsna-knee-sub-a-ortho-cnxs-cons2-f0-nall`
  (the slug says "ortho" for every `--ours-only` build). Weights `runs_convnext_small/fold0/last.pt`, sha256
  `8fc45272b82839f4439f4f7a069d2b5754fc9191063a0cc2a3eae63a9c4e6e58` (avg3: `d08bc775d99864380267f468b505b7a2fb94cb5a058b2f47481e78797ddf8168`),
  read on Kaggle by the CPU kernel `raahimnawaz/rsna-knee-weights-sha` because the CLI download of the 99 MB file
  broke four times (BrokenPipe). **Submit only if its log shows `GATE PASSED` and `[sub A] ARMS:`**:
  `kaggle competitions submit rsna-knee-abnormality-detection -k raahimnawaz/rsna-knee-sub-a-ortho-cnxs-cons2-f0-nall -v 1 -f submission.csv -m "<what it is>"`.
  Compare with Ortho r1 0.927 (the same 3,502 training studies): |Δ| < 0.010 is noise.
- **Folds 1 and 2:** `raahimnawaz/rsna-knee-c7-train-convnext-small-cons2-f12-cv` (its own slug, so fold 0's output
  stays the latest version of its kernel). Same recipe, ~3.5 h. Folds 3 and 4 wait for the owner's word on the GPU
  budget (shared with casmi26).
- **Both PUSHED 10-11 23:50 PDT, built from `35a4c7a`, RUNNING at 23:51.** The submission commit run is v1 of its
  kernel; folds 1-2 are v1 of theirs, `--time-budget-h 5.0` (2.5 h a fold; 12 epochs took 94 min on fold 0).
  Fetch folds 1-2 small files with `--file-pattern '.*\.(txt|json|csv)$'`; read their shas by adding the kernel to
  `rsna-knee-weights-sha`'s `kernel_sources` and re-pushing it (CPU).

## GPU

30 h/week, reset Fri 17:00 PDT (10-16 is the last). **The quota is shared with casmi26**, which had a GPU kernel
running on 10-11. At most two sessions at once. The owner's quota page is the truth.

## Standing rules (the full list with what each cost: `9c23daf:RUNBOOK.md`)

1. Never select on gold-58 (macro SE 0.012). Report it only.
2. Write the rule before the run; a null is a null.
3. The 691 screens fixed-recipe changes; the LB judges labels. Keep the scoring reference out of the targets.
4. A weaker member at equal weight costs a blend (0.901→0.936, 0.916→0.939, 0.929→0.941 against 0.943).
5. Commit before building a kernel; push after each step. No Claude attribution in commits.
6. Leak-check any downloaded label table against gold before it trains anything.
7. A CPU smoke skips GPU-only branches; force them locally before an architecture's first GPU run.
8. A leaderboard number older than a few days is not evidence.
