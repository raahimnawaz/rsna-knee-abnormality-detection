# RUNBOOK — the current state, on one page

**Any session starts here.** Branch `labels-small-backbone` (off `main`, which stays the showcase tree). History up to
10-03 is at commit `9c23daf`: `git show 9c23daf:RUNBOOK.md`, `9c23daf:DATA.md`, `9c23daf:docs/REVIEW-2026-10-04.md`.

## Goal — 0.97 (owner, 2026-10-11)

Fix the labels, switch to a small backbone, train the best models of our own. **The public blend is not submitted
and not built on** (owner: "its useless"). Solo. Final 10-22 16:59 PDT; two finals.

## Scoreboard (pulled 10-12 11:26 PDT; re-pull before using, rule 8)

| | LB | note |
|---|--:|---|
| #1 | 0.966 | 10th 0.963; 20th 0.962; 100th 0.956; 5,849 teams |
| free public blend | 0.954 | 752 teams exactly there (ranks 216-967). `nartaa` CoAtNet (0.949 alone, 0.948 without its mirror pass) + `heliosli` ConvNeXt-small, 4 folds |
| **ours, best** | **0.945** | 10-12: FIVE ConvNeXt-small folds alone, no public model in it (ref `57112913`); rank 1,629 |
| ours, with the old public stack | 0.943 | 10-03: the old public stack with our CoAtNet student as its core |
| ours, one model, best | 0.938 | 10-12: ONE ConvNeXt-small, Stage B, `targets_consensus_v2`, fold 0 held out (ref `57098827`) |
| ours alone, before | 0.929 | Ortho ViT-L, Stage B, teacher targets, all data (ref `56811647`); r1 on 3,502 studies 0.927 |
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
   **✅ First half READ 10-12 (seen 09:19 PDT): fold 0 alone = 0.938** (ref `57098827`). Against Ortho r1 0.927 on the
   same 3,502 training studies: **+0.011, outside the ±0.010 noise band**, so a real gain by the rule, by one tick.
   It is one model against one model, and it changed backbone AND labels at once: read 3 separates them. The old
   691 fit forecast 0.940.
   **✅ Second half READ 10-12 (seen 11:25 PDT): the five folds alone = 0.945** (ref `57112913`). Against fold 0
   alone 0.938: **+0.007 → the ensemble's gain (rule: +0.003 or more)**. Against Ortho all-data 0.929: +0.016,
   outside the band. It is our best submission of any kind (the 0.943 was the public stack with our core) and
   sits 0.004 under the public CoAtNet alone (0.949: inside ±0.005, model against model) and 0.009 under the
   free blend.
3. **Labels on the LB:** the same recipe on `dread_soft` against `targets_consensus_v2`, each alone. Consensus is
   kept if it is not below `dread_soft` by ≥ 0.005; it has *shown* a gain only at ≥ +0.010.
4. **The Synovitis mask**, by `--probe Synovitis`: masked against `_synbase`. Two ticks (0.024 on Synovitis) either
   way decides it; less is a tie and the mask stays (it removes a fill, it invents nothing).

## The outside teacher (owner 10-12: "can we do OAI and the teacher") — rules written 10-12 ~09:50, before any result

- **OAI itself stays closed** (commit `94e7b54`, 09-28; `9c23daf:REFERENCE.md` §1.3): the NDA help desk answered the
  owner's email by restating the Data Use Certification (research use; sharing only with DUC-listed researchers;
  silent on publishing weights). DUC-gated data could not sit on Kaggle, our only GPU, and approval + download do not
  fit before 10-22. Not re-checked: whether the host has ruled on OAI since 09-26.
- **The teacher is how OAI's signal reaches us:** `nartaa`'s public CoAtNet (0.949 alone; its README: 2,399 OAI knees,
  masked supervision; weights "for research and educational use, including the RSNA Knee competition"), run over
  our training studies by `--predict-all`. It is NOT the earlier teachers: the four ResNet34 fold teachers learned
  from the same reports (+0.026 gold, ~0 LB) and the fastMRI+ teacher failed its gold gate (P 0.21). **Caveat:** it
  trained on all 4,349 training studies' report labels, so on them it partly repeats its own labels; what is new to
  us is its OAI supervision and its reading of the cells its own table masked (half of Synovitis).
- **5. Is it usable in OUR pipeline?** It runs on our Stage B cache (canonical orientation, our slots, 384 px whole
  image), not on its own preprocessing (its own slots, a 320 px centre crop, both sides as scanned). Gold-58, report
  only and optimistic for it (its epochs were picked on gold): **≥ 0.910 macro → use it; below → our input breaks
  it**, and the faithful route is its own notebook's code on the training DICOMs (~5 GPU-h) before anything trains on it.
- **6. The student, fold 0, on the LB.** The 0.938 model's recipe; targets = `targets_consensus_v2` with the teacher
  at half of every supervised cell and alone on the Synovitis cells the reports left unread
  (`build_consensus_targets.py --teacher ... --w-teacher 0.5`). (a) Alone against 0.938: a gain is shown at
  ≥ +0.010; it is kept unless ≥ 0.005 below. (b) `--probe Synovitis` against the 0.938 model's probe: two ticks decide.
- **Read 5 PUSHED 10-12 09:52 PDT:** `raahimnawaz/rsna-knee-c7-train-coatnet-oai-teacher-pa` v2, built from `138861b`
  (`--predict-all`, `--init raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, `--eval-bs 1 --enc-chunk 16`, both
  T4s, ~1 h). Its CPU smoke (v1, `c3f15b2`) loaded the checkpoint strict (486 tensors, sha `7e5315da…` = the
  author's manifest) and was then killed for memory in evaluation, 76 windows in one CPU forward: hence
  `--enc-chunk`. Outputs: `runs_coatnet/oof_all.csv` (all 4,349 non-gold studies) and `oof_gold.csv`. Then:
  `build_consensus_targets.py --teacher <oof_all.csv> --out data/student/targets_teacher_v1.csv`, upload to
  `raahimnawaz/rsna-knee-student-targets`, and the fold-0 student (read 6).
- **✅ Read 5 READ 10-12 (run complete ~12:30 PDT): gold-58 macro 0.920 → usable by the rule (≥ 0.910).** 62.8 min,
  strict load of `7e5315da…`, 0 tracebacks, `oof_all.csv` on all 4,349 (small files in
  `fusion/runs_coatnet_oai_teacher_pa/`). **It is not shown to read better than its student in OUR input:** gold,
  report only (SE 0.012 macro, 0.03–0.09 a finding), teacher / our consensus fold 0: macro 0.920 / 0.928; Lateral
  Meniscus 0.883 / 0.904, Lateral OA 0.770 / 0.824, PF OA 0.840 / 0.885, Synovitis 0.828 / 0.810. Its 0.949 on the
  LB is in its own preprocessing; ours may cost it, and gold cannot say (rule 1: nothing is selected here, read 6
  runs as written). 0.9230 against the reference on the 4,349 is its own training studies, not a held-out number.
  **If read 6 is a null, the faithful route (its notebook's code on the training DICOMs, ~5 GPU-h) is still
  untested, not closed.**
- **`targets_teacher_v1`** (`data/student/targets_teacher_v1.csv`, sha256
  `2980e8d6cdd0f3c539bc5a1c33be49a5e40e33b78ab10df2f1b7acaf35fef749`; in `raahimnawaz/rsna-knee-student-targets`,
  version of 10-12 12:37 PDT, four files): `--w-teacher 0.5`; no masked cell left (the teacher alone on the 2,404
  unread Synovitis cells); Spearman teacher against the consensus target 0.45 (Fracture) to 0.80 (Medial Meniscus).
- **Crops, for the record** (the owner had tried them): the millimetre FOV crop (+0.027, in every cache since 09-06);
  an nnU-Net anatomy crop (gate failed, closed 09-07, `b3efe95`); the zoom arm Z (691 +0.0159, but crop-specific
  +0.0070 at P 0.943 against the 336 px arm R → stopped 09-29, MCL the one crop-specific gain). Untried: an
  anatomy-anchored crop, and evaluating on the teacher's 320 px centre crop.

## Running now (written 10-12 ~09:35 PDT)

- **All five folds COMPLETE** (fold 0: `...-cons2-cv`; 1-2: `...-cons2-f12-cv`; 3-4: `...-cons2-f34-cv`): 12/12
  epochs each, no early stop, 7.5-8.6 min/epoch, 0 tracebacks. Held-out macro against the reference: fold 0 0.8667
  (847), folds 1-2 0.8658 (1,739), folds 3-4 0.8644 (1,763). Gold-58 per fold, report only: 0.928 / 0.927 / 0.921 /
  0.923 / 0.929. Small files in `fusion/runs_cnxs_cons2_{f0,f12,f34}/` (git-ignored).
- **`last.pt` sha256, read on Kaggle** (`build_weights_sha_kernel.py`, v2 of `rsna-knee-weights-sha`):
  - fold 0 `8fc45272b82839f4439f4f7a069d2b5754fc9191063a0cc2a3eae63a9c4e6e58`
  - fold 1 `a3113be6535d7d6e8b6c441c9000ae66d62ea2cf86408eeece58575a4205e00d`
  - fold 2 `d147aa3c816ad1ad2b23c8202b1f534112514c667660d4cf84b0e163c7ed08b3`
  - fold 3 `eb3bb1627d24a5cfa7118ee6721b6b154e970431223f5b9820dd5deca6c7af7d`
  - fold 4 `8b0768b0e78e80ecaa6cf8826102d44f8c9f79bc487863047da703aa364eecc8`
  (`avg3/last.pt`: `d08bc775…`, `0edc69f2…`, `54f9bf76…`, `7863c8a8…`, `f3f05439…`.)
- **✅ Read 2, second half: the five folds as one ensemble, alone. SUBMITTED 10-12 09:27 PDT, ref `57112913`,
  READ 0.945 (seen 11:25 PDT).** Kernel `raahimnawaz/rsna-knee-sub-a-ortho-cnxs-cons2-cv5-nall` v1, built from `81b8fd4`. Read in its
  commit log before the submit: 5 checkpoints `sha256 ok`; `GATE PASSED (1/5)` … `(5/5)`, each member rebuilding 3
  of its own held-out studies from DICOMs (max |dp| 0.00043 / 0.00030 / 0.00034 / 0.00036 / 0.00015);
  `[sub A] ARMS: convnext_small_stage_b x5 only, windows all`; 0 tracebacks. **Read:** against fold 0 alone 0.938;
  +0.003 or more is the ensemble's gain. 3 slots left in the UTC day (to 17:00 PDT).
- **Reads 3 and 4 on ONE model each, fold 0 (decided before the run, to spend 1.8 GPU-h instead of 9):** the same
  recipe on `dread_soft` (`8cec477f`), fold 0 held out, kernel `...-convnext-small-dread-cv`. It differs from the
  0.938 model in the targets only. **⏳ PUSHED 10-12 09:29 PDT**, v1, built from `023dac7`, 12 epochs, 3 h cap; due
  ~11:20. Then: its sha by `build_weights_sha_kernel.py`, `--ours-only` alone (read 3), and `--probe Synovitis` on
  it and on fold 0 of `...-cons2-cv` (sha `8fc45272…`) (read 4): three submissions.
  - Read 3: its LB alone against 0.938. The rule above stands: consensus is kept unless it is ≥ 0.005 BELOW
    `dread_soft`; it has shown a gain only at ≥ +0.010.
  - Read 4: `--probe Synovitis` on both fold-0 models (the consensus one has the mask, the `dread_soft` one has the
    Effusion fill). Two ticks (0.002 on the board = 0.024 on Synovitis) either way decides; less is a tie and the
    mask stays. The `_synbase` table is then not needed.
  - **✅ Run COMPLETE 10-12 ~11:30 PDT:** 12/12 epochs, no early stop, 118 min, 0 tracebacks, labels sha `8cec477f…`,
    same 3,502 / 847 split and seed as the consensus fold 0. `last.pt` sha256, read on Kaggle (v3 of
    `rsna-knee-weights-sha`): `cdedbae5509faa6093f3f97d83366f8d08037e0221308140b2ebe86edcb7f3f1` (`avg3`: `441f5432…`).
    Small files in `fusion/runs_cnxs_dread_f0/`.
  - **Held-out 847, report only (rule 3: the reference is one more report reader, and the consensus holds six of
    its kind, so this cannot judge labels; the LB does):** `dread_soft` 0.8618, consensus 0.8667, paired
    **+0.0049 ± 0.0021** for the consensus (2.3 sigma). It is two findings: Synovitis 0.815 → 0.845 (the mask) and
    Baker's 0.917 → 0.936; MCL 0.804 → 0.793; the other nine within 0.009. On the 681: 0.8607 against 0.8654.
    Gold-58, report only: 0.923 against 0.928.
  - **⏳ Read 3 SUBMITTED 10-12 12:35 PDT, ref `57117178`, PENDING** (owner: submit all three). Kernel
    `raahimnawaz/rsna-knee-sub-a-ortho-cnxs-dread-f0-nall` v1, built from `58b6dbb`. Read in its log before the
    submit: `sha256 ok`, `GATE PASSED (1/1)` (max |dp| 0.00032), `[sub A] ARMS: convnext_small_stage_b only, windows
    all`, 0 tracebacks. Against 0.938.
  - **⏳ Read 4, both commit runs PUSHED 10-12 12:35 PDT** from `58b6dbb`:
    `raahimnawaz/rsna-knee-sub-a-ortho-dread-f0-nall-p-synovitis` and `...-cons2-f0-nall-p-synovitis` (v1 each). A
    slug over 50 characters is refused (400): the tags are `dread-f0` and `cons2-f0`. After each gate: submit
    (2 slots left to 17:00 PDT). Score = (5.5 + AUROC) / 12.
- GPU used on 10-11/12 by this line: ~1.8 h (fold 0) + ~3.5 h + ~3.8 h (folds) + three short commit runs ≈ 9.4 h.

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
