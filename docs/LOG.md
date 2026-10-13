# LOG — every run and read on this branch, as written at the time

**Not the place to start: `RUNBOOK.md` is.** This is the record behind it: each rule as it was written before its
result, each kernel with the commit it was built from, each weights sha256, each gate line read before a submission,
each leaderboard ref. Entries are grouped by read, not strictly by time; a later ✅ / ⛔ line supersedes an earlier ⏳
one in the same block. Times are PDT. Append new blocks at the end, dated.

Before 10-11: `git show 9c23daf:RUNBOOK.md` and `9c23daf:docs/LOG.md`.

---

# 2026-10-11 and 10-12

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
   **✅ READ 10-12: `dread_soft` 0.933 (ref `57117178`), consensus 0.938 → consensus KEPT, +0.005, a gain not shown.**
4. **The Synovitis mask**, by `--probe Synovitis`: masked against `_synbase`. Two ticks (0.024 on Synovitis) either
   way decides it; less is a tie and the mask stays (it removes a fill, it invents nothing).
   **✅ READ 10-12: masked 0.532 (Synovitis 0.884, ref `57117247`), filled 0.531 (0.872, ref `57117246`) → a TIE,
   the mask stays.** Synovitis reads ~0.88 on the public test set, not gold's 0.81.

## The outside teacher (owner 10-12: "can we do OAI and the teacher") — rules written 10-12 ~09:50, before any result

- **OAI itself stays closed for the time left** (the host has since ruled conditionally: see the owner block of 10-12 below) (commit `94e7b54`, 09-28; `9c23daf:REFERENCE.md` §1.3): the NDA help desk answered the
  owner's email by restating the Data Use Certification (research use; sharing only with DUC-listed researchers;
  silent on publishing weights). DUC-gated data could not sit on Kaggle, our only GPU, and approval + download do not
  fit before 10-22.
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
- **⏳ Read 6 PUSHED 10-12 12:39 PDT:** `raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-cv` v1, built from
  `f87048d`; fold 0 held out, 12 epochs, 3 h cap, due ~14:40. The notebook differs from the 0.938 model's
  (`...-cons2-cv`, rebuilt and diffed) in the targets' name and sha only. Then: its sha by
  `build_weights_sha_kernel.py`; `--ours-only` alone against 0.938 and `--probe Synovitis` against ref `57117247`
  (two slots, after 17:00 PDT). Tag for the submit kernels: `teach1-f0`.
- **✅ Read 6, the run: COMPLETE 10-12 ~14:30 PDT**, 12/12 epochs, 111 min, 0 tracebacks, labels sha `2980e8d6…`,
  the 0.938 model's split and seed. `last.pt` sha256, read on Kaggle (v4 of `rsna-knee-weights-sha`):
  `1c58b2bcfe5adc0168fffba48f38735e5888ce86a3291db3ab6cb96155d93b20` (`avg3`: `aebc9273…`). Small files in
  `fusion/runs_cnxs_teach1_f0/`.
  **Held-out 847, report only (a label change: the LB judges, rule 3): 0.8804 against the 0.938 model's 0.8667,
  paired +0.0138 ± 0.0019 (7.3 sigma), 11 of 12 up** — MCL +0.029, Effusion +0.027, Lateral Meniscus +0.026,
  Contusion +0.026, Synovitis +0.023, PF OA +0.013, Lateral OA +0.010; Baker's −0.007. For scale: consensus over
  `dread_soft` was +0.0049 here and +0.005 on the LB. **The caveat that keeps this report-only:** the teacher trained
  on every training study's report labels (several of the same readers), so its targets may sit closer to the
  reference reader as well as closer to the images. Gold-58, report only: 0.929 against 0.928, paired +0.0018 ±
  0.0052 (PF OA 0.837 against 0.885, Synovitis 0.845 against 0.810).
- **Read 6, both commit runs COMPLETE 10-12 14:40 PDT; 6a SUBMITTED 17:00 (ref `57122696`, owner: "yes do 1-2"),
  6b waits for a slot.** Kernels `raahimnawaz/rsna-knee-sub-a-ortho-teach1-f0-nall`
  (6a, alone, against 0.938) and `...-teach1-f0-nall-p-synovitis` (6b, against the 0.884 probe), v1 each, built from
  `65e928d`. Read in both logs: `sha256 ok`, `GATE PASSED (1/1)` (max |dp| 0.00035), `[sub A] ARMS:
  convnext_small_stage_b only, windows all`, 0 tracebacks; the probe's `PROBE: only Synovitis is the model's`.
  Submit with `kaggle competitions submit rsna-knee-abnormality-detection -k <kernel> -v 1 -f submission.csv -m ...`.
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
  - **⏳ Read 4, both probes SUBMITTED 10-12 12:38 PDT, PENDING:** ref `57117246` = the `dread_soft` model
    (Synovitis filled), ref `57117247` = the consensus model (Synovitis masked). Kernels
    `raahimnawaz/rsna-knee-sub-a-ortho-dread-f0-nall-p-synovitis` and `...-cons2-f0-nall-p-synovitis`, v1 each, built
    from `58b6dbb` (a slug over 50 characters is refused with a 400: the tags are `dread-f0` and `cons2-f0`). Read in
    both logs: `sha256 ok`, `GATE PASSED (1/1)` (max |dp| 0.00032 / 0.00052), `PROBE: only Synovitis is the model's;
    the other 11 columns are 0.5`, 0 tracebacks. Score = (5.5 + AUROC) / 12. **0 slots left to 17:00 PDT.**
  - **✅ First half READ 10-12 (seen 12:56 PDT): the consensus (masked) model's probe = 0.532 → Synovitis AUROC
    0.884 on the public LB** (0.878–0.890 from the board's rounding), ref `57117247`. Gold says 0.810 and the
    held-out reference 0.845 for the same model: on the public test set Synovitis is not the 0.81 finding gold
    shows. The `dread_soft` probe (`57117246`) decides read 4: ≤ 0.530 or ≥ 0.534 is two ticks.
  - **✅ Read 4 READ 10-12 (seen 13:05 PDT): the `dread_soft` (filled) model's probe = 0.531 → Synovitis 0.872;
    masked 0.532 → 0.884. One tick for the mask = A TIE by the rule (two ticks decide) → the mask stays.** The
    `_synbase` table is not needed.
  - **✅ Read 3 READ 10-12 (seen 13:05 PDT): the `dread_soft` model alone = 0.933** (ref `57117178`) against the
    consensus model's 0.938: **consensus +0.005 → kept by the rule (not ≥ 0.005 below); a gain is NOT shown (needs
    ≥ +0.010).** Same sign and size as the held-out +0.0049. So of fold 0's +0.011 over Ortho r1 (0.927), about half
    sits with the labels and half with the backbone and plain BCE, neither half outside the noise alone.
- GPU used on 10-11/12 by this line: ~1.8 h (fold 0) + ~3.5 h + ~3.8 h (folds) + three short commit runs ≈ 9.4 h.

## Owner 10-12 ~15:55: "yes do 1-2" (tonight's slots; the two cheapest review arms; the student's folds if it confirms)

- **Tonight's five slots (reset 17:00 PDT):** read 6a (the student alone, kernel ready above) + four single-finding
  probes on the FIVE-FOLD model: Lateral OA, PF OA, Lateral Meniscus, MCL. Kernels
  `raahimnawaz/rsna-knee-sub-a-ortho-cv5-nall-p-{lateral-oa,pf-oa,lateral-meniscus,mcl}` v1, built from `cbfc5af`
  (tag `cv5`: the longer tag breaks the 50-character slug). Each scores (5.5 + AUROC) / 12; no rule, they replace
  gold's table (`docs/REVIEW-2026-10-12.md` §2.1). Read 6b (the student's Synovitis probe) waits for tomorrow's slots.
- **Dense coronal / axial arm (review §5 #1), rule written before the run:** Stage C cache (`pipeline/slots118.py`,
  spec `c8-118slice-384`: cor_1 16 → 26, ax_1 14 → 26, everything else Stage B), the 0.938 model's recipe and targets
  (`a1ed4034`), fold 0. Judged on the 847 held-out, paired against the 0.938 model's own held-out predictions
  (`score_oof.py`; fixed labels, so rule 3 allows it): **macro ≥ +0.004, OR the mean of Lateral OA, PF OA and MCL
  ≥ +0.015 → Stage C is the finals' cache; otherwise Stage B stays.** Cache kernels
  `raahimnawaz/rsna-knee-stage-c-s{0..7}of8`, built from `f7b5ec4`, pushed from 16:04 PDT (five CPU sessions at
  most at once). GATE C before any training: 8 manifests, one spec, canonical, 4,407 studies exactly once, 0 errors,
  slot fill equal to Stage B's (sag_2 0.973, cor_2 0.745, sag_3 0.218, ax_2 0.248). Then a CPU smoke, then the run
  with `--enc-chunk 64` (104 windows a study at evaluation).
  - **✅ Probes and read 6a SUBMITTED 10-12 17:00 PDT (00:00 UTC, all five slots), PENDING:** `57122696` the
    student alone (read 6a) · `57122701` Lateral OA · `57122722` PF OA · `57122728` Lateral Meniscus · `57122735`
    MCL. Read in every log before the submit: `sha256 ok`, `GATE PASSED` (1/1; 5/5 for the five-fold kernels, max
    |dp| ≤ 0.00043), the `PROBE: only <finding> is the model's` line, `[sub A] ARMS:`, 0 tracebacks; each probe's
    submission.csv has one non-constant column. AUROC = 12 × score − 5.5, ±0.006.
  - **✅ GATE C PASSED 10-12 16:35:** 8 manifests, one spec (`c8-118slice-384`), all canonical, one span (2–98%);
    4,407 studies, each once, equal to `train.csv`'s ids; 0 study errors, 0 decode failures; fill sag_2 0.973, cor_2
    0.745, sag_3 0.218, ax_2 0.248 (Stage B's to the third decimal); side R 2,295 / L 2,112; 7.6–9.7 min a shard.
    **CPU smoke clean** (v2 of `...-c8-train-convnext-small-smk-cv`, from `6227cc2`): 4,407 studies, slots (26, 20,
    26, 10, 26, 6, 4), fold 0 train 3,502 / val 847, 92–98 windows a study, 2 processes, 0 tracebacks. (v1 failed in
    22 s: pushed before shard 7's output was published — "have [0..6] of 8". Push a consumer only after every shard
    reads COMPLETE.)
  - **⏳ Dense arm PUSHED 10-12 17:01 PDT:** `raahimnawaz/rsna-knee-c8-train-convnext-small-cons2-cv` v1, built
    from `6227cc2`; its notebook differs from the 0.938 model's in `LAYOUT = 'stage_c'` and `--enc-chunk 64` only
    (rebuilt and diffed). 12 epochs, 3 h cap, due ~19:00. Read: `score_oof.py` paired against
    `fusion/runs_cnxs_cons2_f0/runs_convnext_small/fold0`, by the rule above.
  - **✅ The four probes READ 10-12 (seen 17:35 PDT), five-fold model:**

    | finding | LB probe | AUROC (± 0.006) | deficit, of 0.66 | gold-58 said |
    |---|--:|--:|--:|--:|
    | Synovitis (fold 0 alone, ref `57117247`) | 0.532 | 0.884 | 0.116 = 18% | 0.841 |
    | Lateral OA (`57122701`) | 0.533 | 0.896 | 0.104 = 16% | 0.822 |
    | PF OA (`57122722`) | 0.533 | 0.896 | 0.104 = 16% | 0.864 |
    | Lateral Meniscus (`57122728`) | 0.536 | 0.932 | 0.068 = 10% | 0.882 |
    | MCL (`57122735`) | 0.539 | 0.968 | 0.032 = 5% | 0.961 |
    | the other seven, by subtraction from 12 × 0.945 | | mean 0.966 | 0.24 = 36% | 0.946–0.986 |

    Half of the deficit is Synovitis + Lateral OA + PF OA (0.88–0.90). Lateral Meniscus (0.932) and MCL (0.968)
    are better than gold-58 said. Read 6b (the student's Synovitis probe, kernel ready) now matters more: Synovitis
    is the largest single term and the teacher is the one label source that could move it.
  - **⛔ Dense arm READ 10-12 19:37 PDT: FAIL by its rule → Stage B stays.** 847 held-out, paired against the 0.938
    model: macro 0.8697 against 0.8667, **+0.0030 ± 0.0023** (1.3 sigma; bar +0.004); Lateral OA 0.000, PF OA
    −0.009, MCL +0.009: **mean −0.0002 ± 0.0049** (bar +0.015). Lateral Meniscus +0.017, Contusion +0.017, ACL −0.009.
    Gold-58, report only: 0.921 against 0.928. 0 tracebacks, 123 min. **Confound, against the arm:** the fit-budget
    probe timed a cold cache (0.80 s a micro-step → 15.7 min/epoch projected; real epochs 8.8 min after a 22 min
    first one) and cut the run to **11 epochs of 12**. For the next run on a newly built cache: give the budget
    room (`--time-budget-h 4`) so the probe cannot cut an epoch. The Stage C shards and code stay; nothing uses them.
- **Affine arm (review §5 #2): HELD, not run.** Found while preparing it: the flag was tested 09-27 on ResNet34,
  −0.0019 ± 0.0048 (`75ab54c`). The owner approved it without that fact; ask before spending 1.8 GPU-h. `--k 24`
  (~3.3 h) is the alternative for the same hours.
- **✅ Read 6a READ 10-12 (seen 17:22 PDT): the student alone = 0.944** (ref `57122696`) against 0.938: **+0.006 →
  `targets_teacher_v1` is KEPT (not ≥ 0.005 below); a gain is NOT shown (the rule needs +0.010).** It is our best
  single model, one tick under the five-fold consensus ensemble (0.945), and outside the ±0.005 model-against-model
  band without reaching the bar. The held-out 847 had said +0.0138: about half of that was the targets moving
  toward the reference reader, as its caveat warned. Labels so far, same recipe and fold: `dread_soft` 0.933 →
  consensus 0.938 → teacher 0.944.
- **Folds 1–4 on `targets_teacher_v1`: NOT started.** The owner's go was conditional on ≥ 0.948; it read 0.944. Ask.
  (~7.5 GPU-h; at the consensus folds' +0.007 for five over one it would land near 0.951.)
- **OAI itself (owner's question, 10-12):** unchanged, closed for this competition (`94e7b54`): the NDA's Data Use
  Certification allows research use and sharing only with DUC-listed researchers, so the data cannot sit on Kaggle
  (our only GPU), a winner could not warrant the licence, and approval + download do not fit before 10-22. What we
  use is the public OAI-trained model as a teacher (read 6).
  **Host ruling, re-read 10-12 16:05 (discussion 741819, host reply ~09-21; our 09-26 re-read missed it):** OAI "does
  not violate" the rules IF it is accessible to all without an institutional review / sign-off or a legal team, and
  free; the host does not speak for OAI's own terms. So the rules side now turns on one fact we have NOT verified:
  whether the NDA request for OAI needs an institutional signing official (a web search says account-based, RAS /
  Login.gov plus the OAI terms; the 09-28 help-desk reply read as institutional). Unchanged whatever the answer: the
  images may not be re-shared, so they cannot go on Kaggle and training needs a GPU elsewhere; the access request is
  the owner's to make; download + MOAKS label mapping + a Stage B conversion is more than the ten days left.

---

# 2026-10-12, late evening — owner: "do all 3 but first if OAI helps lets do it"

Read as: the three open items of RUNBOOK §4 are approved, the OAI-derived one first. OAI's signal did help where we
could use it (the teacher's student, 0.944 against 0.938), so the order is: the four remaining folds on
`targets_teacher_v1`, then the spatial-attention experiment, and the student's Synovitis probe at the next slot reset
(17:00 PDT 10-13; none are left today).

- **Folds 1–4 on `targets_teacher_v1` (`2980e8d6…`), the 0.944 model's recipe.** Two kernels, two folds each:
  `raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-f12-cv` and `...-teach1-f34-cv`, `--time-budget-h 6.0`
  (3.0 a fold, as fold 0). ~3.7 GPU-h each.
  **Rule, before any result:** the five teacher folds as one ensemble, alone on the LB, against the five consensus
  folds' 0.945. Kept unless ≥ 0.005 below; a gain is shown at ≥ +0.010; at ≥ 0.945 it is the finals' candidate.
  The held-out macro per fold is report only (a label change).
- **✅ Folds 1–4 PUSHED 10-12 20:22 PDT**, v1 each, built from `47ff4ed`; each notebook differs from fold 0's in
  `FOLD_LIST` only (rebuilt and diffed). Due ~00:15.
- **Spatial attention (RUNBOOK §4.1), rule written before the run.** `train_windows.py --spatial-attn`: a query per
  finding over the 12 × 12 positions of each window's feature map, in place of the global average; then the existing
  per-finding attention over windows, each finding on its own token. Queries start at zero = the average.
  Checked on the Mac: with zero queries the output equals the plain model's (max |diff| 3e-7, with slot / depth
  embeddings and a padded bag); chunked evaluation equals unchunked; the queries receive gradient; a three-epoch run
  on the synthetic-DICOM fixture saves `sq` and `spatial: true`, and `infer_windows.py --check-against` rebuilds its
  held-out predictions from DICOMs (max |dp| 0.00002).
  **The arm:** the 0.938 model's recipe, labels (`a1ed4034`) and fold, plus `--spatial-attn`. Paired on the 847
  against the 0.938 model: **macro ≥ +0.004, OR the mean of Lateral OA and PF OA ≥ +0.015 → it goes into the finals;
  otherwise the global average stays.** (MCL is out of the target set: the probe read it at 0.968.)
- **Owner 10-12 ~20:46, two answers.** (1) "do it": the go for ONE submission, the five teacher folds as one ensemble
  read alone against 0.945 by the rule above. It covers that submission and nothing later. No slots are left today
  (all five went at 17:00), so it goes at the 17:00 PDT reset on 10-13 with the student's Synovitis probe, after the
  gate lines read clean. (2) The quota page: "it has 10.2 hrs" (the fold kernels were 25 min in; whether the page
  already counted them is not known). Against ~9.6 h for tonight's runs the attention run is NOT held; under 1 h is
  left until Friday's reset for anything else, casmi26 included.
- **Owner 10-12 ~20:57: "ballpark the OAI download, and training on an M5 Studio" — a local Mac as the GPU that is
  not Kaggle.** MEASURED on this MacBook Pro (base M5, 10 GPU cores, 16 GB; torch 2.14 MPS, fp16 autocast, random
  weights and input): ConvNeXt-small at 384 px trains at **17.7 img/s** (9.3 in fp32; 5.3 GB at 12 windows) and
  evaluates at 74.7 img/s. Kaggle's two T4s: 3,502 studies x 12 windows in 7.8 min = ~90 img/s, so this laptop is
  ~5x slower: one fold of the current recipe ~8 h here against 1.9 h. With 2,399 OAI knees added (6,748 studies):
  one all-data model, 12 epochs, ~15 h here; five folds ~61 h. A Studio is NOT measured: scaled by GPU cores, a
  Max-class chip (~4x) is ~4 h and ~15 h, an Ultra-class about half of that. **So a Studio removes the "Kaggle is
  our only GPU" blocker of RUNBOOK §3's OAI row; access, the download and the label mapping are what is left.**
  OAI size, MY ARITHMETIC and not an official figure (two web searches found no package sizes): the protocol is
  ~1,000 images a participant-visit at ~0.3 MB, and catalogs give 26.6 M images in the archive (4,796 participants)
  -> ~8 TB in all, ~1-1.5 TB for one visit of everyone, ~200-400 GB for the ~2,400 MOAKS-read knees if the archive
  lets a download be filtered by subject (not verified). This Mac has 716 GB free. The signing-official question is
  still unverified; today's search again describes an RAS login plus agreeing to OAI's terms.
- **Owner Qs 10-12 ~21:02 (costs and label ideas; nothing run, nothing approved).**
  (a) *Renting a 5090.* Web snapshots of mid-2026, not re-checked live: ~$0.20–0.40/h interruptible (Vast.ai),
  ~$0.65/h the on-demand median, ~$0.99/h RunPod's own datacentres. Speed NOT measured: from general benchmarks a
  5090 is ~4–6x Kaggle's T4 pair on this model, so a fold is ~20–30 min, five folds ~2 h, and the review's whole
  list plus finals (~27 Kaggle GPU-h) is ~5–7 h: **$10–30 covers everything**, and it lifts the weekly quota, which
  is the binding limit this week. The Stage B cache would have to be pulled from the Kaggle kernel outputs to the
  box. OAI images on a rented machine are a question of OAI's terms (someone else's hardware), not answered here.
  (b) *A large LLM (Kimi K2) to re-read the reports.* 4,407 reports, 4.84 M characters (median 977): ~1.6 M tokens
  of text plus a criteria prompt on each, **~$5–20** at ~$0.60 / $2.50 per million. Expected gain small: `0cbfb87`
  and `7b14436` still stand (every reader reads the same text, readers agree r 0.82–0.95, the best tables read gold
  ~0.90 against our image model's 0.93; the 10-04 review put a frontier read at +0.001–0.004), and the judge is the
  cost: a fold plus an LB slot, +-0.005. Seven readers in consensus were +0.005, not shown.
  (c) *Teacher–student from the reports* is the pipeline as built (report reader -> soft targets -> image model);
  a second round through an image teacher trained on the same reports was +0.026 gold, ~0 LB (closed, `9c23daf`);
  the outside OAI-trained teacher is the round that paid (+0.006).
  (d) *Bayesian handling of noisy labels:* `8dd4f82` stands. Soft graded targets, the seven-reader consensus and
  masked unread cells are the forms a ranking metric can use; a uniform noise correction changes no order, readers
  that share a text err together, and 58 image-read knees are too few to fit per-finding noise rates (rule 1). The
  one prior of this kind still on the list is Synovitis = own part + beta x Effusion (review §5 #7, in the finals).
- **Owner Q 10-12 ~21:07: "why did fastMRI fail as a teacher, can we be sure it was the model and not us?"** No.
  Re-read of `149cb0a` / `1766436` (09-28): what was tested is a ResNet34 at 224 px trained 10.5 min on the Mac on
  473 coronal-only knees, on FOUR findings (ACL, Effusion, Baker's, Contusion: all findings we now read at ~0.97),
  used as a relabelling teacher and judged on gold-58: report + NYU −0.0143 ± 0.0176, P 0.21. Four weaknesses are
  the test's, not the data's: (1) a model later measured −0.036 under the main one; (2) none of today's weak
  findings were in it (fastMRI+ has cartilage and meniscus boxes that the label file left blank); (3) a 58-knee
  judge with SE 0.018 cannot see the +0.003–0.010 that was expected, and rule 1 now forbids sizing anything on it;
  (4) a teacher role, where the public ConvNeXt uses the boxes as an auxiliary task. What the run DID show: it
  learned (hold-out Effusion 0.962) and part did not transfer (Baker's 0.718 → 0.525 on gold; a protocol no
  competition study uses). The 7,140 pseudo-labelled NYU knees were a separate null (+0.0035 ± 0.0051). **So the
  closed item is "a small fastMRI+ teacher on four findings", not fastMRI+ as data.** Still against it: the raw files
  were deleted 10-09 (corrected 10-12 ~21:13: a Stage A cache of the 556 knees is still on Kaggle, see the next
  entry), and the 10-04 review keeps NYU knees out of anything a final trains on (the fastMRI terms
  against the winner's licence obligation). My reading, not a new measurement.
- **Owner Qs 10-12 ~21:13: "can we just take their core model and put it in our code", "a better ConvNeXt", "bring
  back the fastMRI model". Answered, NOTHING approved, and RUNBOOK §1's "our own models only" stands until the owner
  says in words that it is changed.** Facts checked:
  - *The public OAI-trained CoAtNet* already loads strict in our trainer (`--arch coatnet --init`, read 5) and its
    README allows the competition. Three uses, none run: (A) as it is, a sixth member of our ensemble in our input
    (no training; gold there 0.920 against our 0.928); (B) in its own preprocessing beside ours (the Submission B
    path of `build_submit_kernel.py`, last used 10-03); (C) fine-tuned from its weights on our labels (done once
    with the pre-OAI public CoAtNet: 0.931 against the public 0.933, noise). My estimate for B: ~0.953–0.957, where
    the public blends already are.
  - *A bigger or newer ConvNeXt* has not been run on this recipe (only tiny and small are wired in; "bigger
    backbones" in §3 rests on ViT-L losing by 0.011). A base-size fold is ~2x the GPU time.
  - *fastMRI+:* the 09-28 teacher's weights are on this Mac (`fusion/runs_nyu_teacher/last.pt`, ResNet34, four
    findings) and the 556-knee cache is still a private Kaggle dataset (`raahimnawaz/rsna-knee-fastmri-plus-cache`,
    1.0 GB) but in Stage A format (64 slices), which `--extra-corpus` refuses beside Stage B. A Stage B rebuild
    needs the 88 GB of raw files again (deleted 10-09) and the converter and label script from `160f781` (removed
    from the tree by the showcase cleanup). Its labels cover ACL, Effusion, Contusion and cysts; the meniscus side
    mapping was never written; it has no synovitis label and is coronal only (nothing for PF OA).
- (The four owner entries above were first written with guessed times up to an hour late; corrected 21:21 to the
  commit clock: `968a95d` 20:59, `8cc8c36` 21:03, `905d624` 21:08, `19a0a45` 21:14.)
- **⚠️ DIRECTION CHANGED, owner 10-12 ~21:17: "the rule shouldnt be my own models i have always been for using
  whatever is avialable to us / and we finetune/pretrain the cnn".** RUNBOOK §1's "our own models only" is
  WITHDRAWN: public weights, models and data may be used and built on. Not asked, so not assumed: submitting the
  public blend as it stands (the owner's "its useless" was said of that). Asking before submissions and before GPU
  hours is unchanged.
  - **"finetune/pretrain the cnn", read as:** start from the public OAI-trained CoAtNet's weights and train it on
    our input and labels. The trainer already does this: `--init` maps the public head's names onto ours and lets
    the slot / depth embeddings and the spatial queries start at zero (`train_windows.py:641`).
    **The arm (NOT run; needs GPU hours):** `--arch coatnet --corpus stage_b --all-data`, `--init
    raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (sha `7e5315da…`, dataset
    `nartaa/rsna-knee-publication-swa-weights-20261007`), `targets_teacher_v1` (half of every cell is this model's
    own reading, which holds it to what OAI taught it while the input changes), 6 epochs, the CoAtNet package
    (bb-lr 9e-5, llrd 0.9, drop-path 0.1), `--conf-weight none`.
    **Cost:** the 10-02 CoAtNet student took 456 min for 7 epochs of 4,349 studies on ONE T4 = 65 min/epoch, so
    ~33 min/epoch on both and **~3.5 GPU-h** for 6: more than this week has left (~0.9 h after tonight's runs).
    Friday 17:00's reset, or a rented GPU.
    **Judge:** its init saw every training study, so no held-out fold can judge it; the LB does. Rule, before any
    result: alone against the SAME checkpoint as it is in our input (read (3) below). Less than +0.005 over that =
    the fine-tune did not pay for its hours; ≥ +0.005 = it replaces the as-is checkpoint as an ensemble member.
  - **Two reads that need no training, proposed for the 17:00 reset on 10-13 (NOT approved yet):** (3) the public
    checkpoint as it is, in our input, alone: what our input costs it against its 0.949, so how much room a
    fine-tune has; (4) the five teacher folds + it, against the five teacher folds alone: what it adds.
    `raahimnawaz/rsna-knee-c7-train-coatnet-oai-teacher-pa` already holds it re-saved in our format (`last.pt`,
    `last_meta.json`, `oof_all.csv`), so `build_submit_kernel.py --weights-kernel` can take it as a member. A
    submission kernel's own run is ~6 min of GPU (one on record, 10-03).

