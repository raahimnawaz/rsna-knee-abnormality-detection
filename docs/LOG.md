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

# 2026-10-12, late evening (20:22 – 21:57 PDT)

How to read this block: **A** is what was approved and started, **B** the owner's change of direction and the
fine-tune it asked for, **C** what an online pass found, **D** the swap it led to, **E** the owner's questions
answered from the record (reference; nothing in E was run).

## A. "do all 3 but first if OAI helps lets do it" — the teacher folds, spatial attention, the probe

Read as: the three open items of the RUNBOOK are approved, the OAI-derived one first. OAI's signal did help where we
could use it (the teacher's student, 0.944 against 0.938), so the order is: the four remaining folds on
`targets_teacher_v1`, then the spatial-attention experiment, and the student's Synovitis probe at the next slot reset
(17:00 PDT 10-13; none were left on 10-12).

- **Folds 1–4 on `targets_teacher_v1` (`2980e8d6…`), the 0.944 model's recipe.** Two kernels, two folds each:
  `raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-f12-cv` and `...-teach1-f34-cv`, `--time-budget-h 6.0`
  (3.0 a fold, as fold 0). ~3.7 GPU-h each.
  **Rule, before any result:** the five teacher folds as one ensemble, alone on the LB, against the five consensus
  folds' 0.945. Kept unless ≥ 0.005 below; a gain is shown at ≥ +0.010; at ≥ 0.945 it is the finals' candidate.
  The held-out macro per fold is report only (a label change).
- **✅ Folds 1–4 PUSHED 10-12 20:22 PDT**, v1 each, built from `47ff4ed`; each notebook differs from fold 0's in
  `FOLD_LIST` only (rebuilt and diffed). Due ~00:15.
- **Spatial attention (RUNBOOK §4), rule written before the run.** `train_windows.py --spatial-attn`: a query per
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

## B. 21:17 — the direction changes: use whatever is available; fine-tune the public model

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
    own reading, which holds it to what OAI taught it while the input changes), 6 epochs, `COATNET_PACKAGE`
    (bb-lr 3e-5 as the 10-02 CoAtNet student, drop-path 0.1, zero-initialised slot / depth embeddings, plain BCE;
    the ViT package it was first given is what smoke v1 refused).
    **Cost:** the 10-02 CoAtNet student took 456 min for 7 epochs of 4,349 studies on ONE T4 = 65 min/epoch, so
    ~33 min/epoch on both and **~3.5 GPU-h** for 6: more than this week has left (~0.9 h after tonight's runs).
    Friday 17:00's reset, or a rented GPU.
    **Judge:** its init saw every training study, so no held-out fold can judge it; the LB does. Rule, before any
    result: alone against the SAME checkpoint as it is in our input (read (3) below; RUNBOOK §4 read 3). Less than +0.005 over that =
    the fine-tune did not pay for its hours; ≥ +0.005 = it replaces the as-is checkpoint as an ensemble member.
  - **Two reads that need no training, proposed for the 17:00 reset on 10-13 (NOT approved yet):** (3) the public
    checkpoint as it is, in our input, alone: what our input costs it against its 0.949, so how much room a
    fine-tune has; (4) the five teacher folds + it, against the five teacher folds alone: what it adds.
    `raahimnawaz/rsna-knee-c7-train-coatnet-oai-teacher-pa` already holds it re-saved in our format (`last.pt`,
    `last_meta.json`, `oof_all.csv`), so `build_submit_kernel.py --weights-kernel` can take it as a member. A
    submission kernel's own run is ~6 min of GPU (one on record, 10-03).
  - **Fine-tune kernel, CPU smoke PUSHED 10-12 21:22:** `raahimnawaz/rsna-knee-c7-train-coatnet-oai-ft-smk-all` v1,
    built from `c359fc1` (`--smoke --arch coatnet --corpus stage_b --all-data --epochs 6 --targets
    data/student/targets_teacher_v1.csv --targets-from raahimnawaz/rsna-knee-student-targets --extra-datasets
    nartaa/rsna-knee-publication-swa-weights-20261007 --tag oai-ft-smk "--train-args=--init
    raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt --allow-missing-labels --conf-weight none --eval-bs 1
    --enc-chunk 16"`). The full run is the same line without `--smoke` and with `--tag oai-ft --time-budget-h 5.0`.
- **Owner 10-12 ~21:24: "well see how they do and how the ensemble them after we get the scores".** Read as: the
  public checkpoint ALONE in our input goes at the 17:00 reset on 10-13 with the two already approved (three slots);
  HOW to ensemble it with the teacher folds (which members, what weights) is decided after those scores, and that
  submission needs the owner's word then. Also asked: does the fine-tune have promise (my answer: modest; it should
  end near the checkpoint's own 0.949, + 0 to +0.005 from our input and labels; read (3) decides, by the rule
  above) and whether the plan stands (yes; the fine-tune waits for Friday's reset unless a GPU is rented).
- **⛔ Fine-tune smoke v1 ERROR 10-12 21:34 (5.6 min) — a wiring fault, found where it is free.** The checkpoint
  loaded in the training configuration (`init from …swa.pt (sha256 7e5315da…): 486 tensors, strict but for 3
  zero-initialised slot/depth weights`), then `--ckpt-blocks 12: coatnet's encoder has no .blocks; use -1 (all)`:
  on Stage B the builder gave every non-small arch the ViT package (`PACKAGE + SPEED`), and CoAtNet had only ever
  trained on Stage A, where no package is added. **Fix:** `COATNET_PACKAGE` in `build_backbone_ab_kernel.py`
  (bb-lr 3e-5, drop-path 0.1, `--pos-emb`, `--fit-budget`, `--conf-weight none`; every block checkpointed, not
  compiled). ConvNeXt-small's and Ortho's argument lists rebuilt and unchanged.
- **✅ Fine-tune smoke v2 COMPLETE 10-12 21:57 (20.0 min on CPU, 0 tracebacks), from `1b75b83`.** `init from
  …swa.pt (sha256 7e5315da…): 486 tensors, strict but for 3 zero-initialised slot/depth weights` · `gradient
  checkpointing: all of the encoder blocks` · `trainable 73.2M of 73.2M` · ALL DATA 4,349 with `targets_teacher_v1`
  (`2980e8d6…`) · one epoch, evaluation and the save ran · `SMOKE: the whole path ran`. Not reached on a CPU: fp16
  and the two-GPU step time, so the first GPU run's `fit-budget` line is the first real price (give it
  `--time-budget-h 5.0`). **The kernel is one push away; it waits for GPU hours** (Friday's reset, or a rented GPU):
  the smoke's build line without `--smoke`, `--tag oai-ft`.

## C. 21:45 — "use online resources to try and find a useable fix for our problem"

Read tonight: the leaderboard, the public notebooks and datasets (Kaggle CLI), four forum threads (745861 "Tricks",
746792 the 0.949 author's write-up, 743416 external data, 735304 "Best single-model score"). Everything below is what
other people report, unverified by us unless it says measured. Nothing is approved or run.

**Board (pulled 21:41):** #1 0.966; three teams 0.965; 0.963 at ~rank 7–11. A public notebook reads the medal lines
as gold to rank 21 (0.962), silver to 290, bronze to 581; the 0.954 tie group runs from rank 193 to 870; hidden test
~1,300 studies, public part ~390, one score ±0.007 (2 SD) in its simulation.

1. **Pseudo-labels from strong image models are what the top single models report, with no outside data** (735304):
   `azamat1ch` (5th) five-fold singles ConvNeXt-T 0.959, EfficientNetV2-S 0.956, CoAtNet 0.952; `colum2131` (40th)
   ConvNeXt-T 0.953 and DINOv2 ViT-S 0.951 "trained with pseudo labels … no external data"; `Raymond Yuen` (36th)
   0.938 → 0.949 on labels alone ("weak-label sources to train teacher models, then … pseudo-labels … a final model
   on them; different teachers seem to help different targets"; API cost under $5; cheap and expensive readers
   differ little); `CoolinLai` (21st) five-fold ResNet50 at 224 px 0.954; `Chris Deotte`: a pseudo-labelled single
   model matches the ensemble that labelled it.
   **Against our record:** RUNBOOK §3 closes "teacher rounds that learn from the same reports" on +0.026 gold, ~0 LB.
   That round's teachers were four ResNet34 at 224 px (a model 0.036 under its own student on the 691). A round from
   the five-fold ConvNeXt-small's out-of-fold predictions has NOT been run. `add_teacher` in
   `build_consensus_targets.py` already blends a teacher's order into the targets; the consensus folds' OOF is on
   disk and the teacher folds' arrives with tonight's kernels. **Proposed arm (not approved):** targets =
   `targets_teacher_v1` + our own five-fold OOF at half weight; fold 0, the 0.944 model's recipe, ~1.9 GPU-h; the LB
   judges it alone against 0.944 (kept unless ≥ 0.005 below, shown at ≥ +0.010).
2. **Model size is not the lever** (735304; 735154 "Scaling the encoder bought us nothing (+0.0011)"): the best
   reported singles are ConvNeXt-Tiny, ResNet50 and EfficientNetV2-S at 224 px. A bigger ConvNeXt is not proposed.
3. **OAI, from the 0.949 author (746792):** 2,399 BASELINE OAI knees, masked labels, supervising exactly **PF OA,
   Lateral OA and Synovitis** (our three weak findings); 0.942 → 0.945 with them, 0.948 with the 384 px full-field
   recipe, +0.001 from mirror averaging; a 12-month arm did not help; fewer windows at inference lost 0.006; the
   MOAKS mapping is not published (asked, unanswered). **Access (743416):** one participant reports individual
   approval at once through a Login.gov account, "Institutional Approval Required: No"; others report login and
   download errors; the host has still not said yes or no for OAI, and a 23rd-place participant warns against it.
   So the unverified fact of RUNBOOK §3's OAI note now has one report in favour; it is still the owner's login.
4. **The public stack is two models and a rank blend:** `nartaa`'s OAI-trained CoAtNet (0.949, Apache 2.0) and
   `heliosli`'s four-fold ConvNeXt-small (S6 / R7: knee-joint anchors, a side classifier; "Feel free to blend my
   model with yours!"), per-finding percentile ranks, 70/30 → 0.954, R7 50/50 → ≈0.957. A third reader
   (`goodpjw2008`, 0.929 alone) adds nothing at a flat 30% (0.948) and 0.000–0.001 with per-finding weights.
   Every one of them runs `nartaa`'s notebook unchanged as stage 1, in its own preprocessing. **For us:** because
   OAI supervised exactly our three weak findings, a per-finding rank blend with weights fixed beforehand (theirs
   heavier on those three) is the principled way to add it; that means its own notebook as stage 1 (our Submission
   B path), not the checkpoint in our input.
5. **Smaller label ideas, one report each:** train only on cells where the readers agree strongly (> 0.9 or < 0.1),
   dropping ~25% of cells ("improved CV and public LB", rank ~1,235); KneeCoT is ruled out by the host; KJD-25K
   (MICCAI 2026) and KneeXNet-2.5D could not be found by search.

## D. 21:50 — "do the swap": a pseudo-label round from our own five folds

- **Owner 10-12 ~21:50: "do the swap".** The last GPU session of this week goes to the pseudo-label student; the
  spatial-attention arm moves to Friday 17:00's reset (its kernel is built: `--tag cons2-sa`, one flag from the 0.938
  model's). Read as a go for the RUN; the student's submission was named as a separate ask and is not assumed.
  **The arm, written before the run:** `targets_teacher_v2` = `targets_teacher_v1` + the five teacher-label folds'
  held-out predictions (each study read by the fold model that did not train on it; `cv_last`, the weights that are
  submitted) at half of every cell, by `add_teacher` (the models' order on the target's own levels). Fold 0, the
  0.944 model's recipe and seed; the targets are the only change. **Rule:** alone on the LB against 0.944 (ref
  `57122696`); kept unless ≥ 0.005 below; a gain is shown at ≥ +0.010. The 847 held-out are report only: a label
  change, and folds 1–4's models trained on those 847 studies, so the targets carry a trace of them.
- **`build_consensus_targets.py --own-oof … --w-own` (21:49).** A second blend after `--teacher`; files must cover
  disjoint studies. Checks: with `--teacher` alone it rebuilds `targets_teacher_v1` byte for byte (sha256
  `2980e8d6…`); a dry run on the CONSENSUS folds' OOF (not the table that will train) covers all 4,349 studies, no
  cell masked, Spearman of the models against the target 0.64 (Fracture) to 0.88 (Medial Meniscus).
- **The public checkpoint alone in our input: submission kernel BUILT 10-12 21:52, not pushed (both GPU sessions are
  in use until ~00:15).** `raahimnawaz/rsna-knee-sub-a-ortho-oai-pa-nall`, built from `1dffa3f` with `--ours-only
  --weights-kernel raahimnawaz/rsna-knee-c7-train-coatnet-oai-teacher-pa --weights-sha
  f900846d3e821721cf571a9f0a477021bc0c2e0e71ddb5565dc5687752fbd0c1 --tag oai-pa`. That sha is the checkpoint as our
  trainer re-saved it at 0 epochs (146,463,855 bytes; v5 of `rsna-knee-weights-sha`), not the author's file
  (`7e5315da…`). Its gate checks the inference path against the predict-all run's own `oof_all.csv`.

## E. The owner's questions, answered from the record (nothing here was run)

- **Owner 10-12 ~20:57: "ballpark the OAI download, and training on an M5 Studio" — a local Mac as the GPU that is
  not Kaggle.** MEASURED on this MacBook Pro (base M5, 10 GPU cores, 16 GB; torch 2.14 MPS, fp16 autocast, random
  weights and input): ConvNeXt-small at 384 px trains at **17.7 img/s** (9.3 in fp32; 5.3 GB at 12 windows) and
  evaluates at 74.7 img/s. Kaggle's two T4s: 3,502 studies x 12 windows in 7.8 min = ~90 img/s, so this laptop is
  ~5x slower: one fold of the current recipe ~8 h here against 1.9 h. With 2,399 OAI knees added (6,748 studies):
  one all-data model, 12 epochs, ~15 h here; five folds ~61 h. A Studio is NOT measured: scaled by GPU cores, a
  Max-class chip (~4x) is ~4 h and ~15 h, an Ultra-class about half of that. **So a Studio removes the "Kaggle is
  our only GPU" blocker of RUNBOOK §3's OAI note; access, the download and the label mapping are what is left.**
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
  were deleted 10-09 (a Stage A cache of the 556 knees is still on Kaggle, see the next entry), and the 10-04 review keeps NYU knees out of anything a final trains on (the fastMRI terms
  against the winner's licence obligation). My reading, not a new measurement.
- **Owner Qs 10-12 ~21:13: "can we just take their core model and put it in our code", "a better ConvNeXt", "bring
  back the fastMRI model". Answered, nothing approved at the time (the direction changed four minutes later, §B).** Facts checked:
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
- **Three papers the owner sent (10-12 ~21:24), read: nothing to adopt.** PMC10741167 is peroneal-nerve injury on
  knee MRI (149 subjects, image-level split), not our findings. Sci Rep s41598-026-54489-x (MedNet-FS) is few-shot
  ACL on MRNet at AUC 0.76; its one relevant claim, knee-specific pre-training beats generic, is a small point for
  starting from the OAI-trained weights. The AJR report-classifier abstract (an SVM on knee MRI reports, F1 > 90%
  in-site, 77.6–90.2% across sites) agrees with our report ceiling of ~0.90: reading the text is not the limit.

---

# 2026-10-13, morning — the 00:15 steps ran at 08:02 (the Mac slept with the lid shut from ~22:00)

The fold watcher only fired its 5.5 h timeout on a dark wake (04:35); between 04:51 and 08:02 one push got through
in the seconds the Mac was awake. Nothing was lost on Kaggle: both fold kernels finished on their own.

- **✅ Folds 1–4 on `targets_teacher_v1` COMPLETE.** 12/12 epochs each, no early stop, 0 tracebacks, labels sha
  `2980e8d6…` in every summary; 105–112 min a fold. Held-out, report only (a label change): folds 1+2 macro 0.8822,
  folds 3+4 0.8795 (fold 0: 0.8804). Gold-58, report only: 0.929 / 0.931 / 0.930 by kernel, the five-fold mean
  0.931. Small files in `fusion/runs_cnxs_teach1_f12/` and `…_f34/`. `last.pt` sha256 (v6 of `rsna-knee-weights-sha`):
  - fold 1 `683857fad528ce5cc3518bb8e5be3314bfcfc89ff9d276a587c6c3a038aa5de6`
  - fold 2 `479cf64b1b5123f4684183e4f048be8b09fba8c34e4b03223aec3e3f97c88d75`
  - fold 3 `4921d0968d3616a90e3ec22948f40d60529f9f1ab3f4ed458b510da2a862269b`
  - fold 4 `c910af266a41af08042d48da83f2db6e7a43deeb7b6365531c5cab30b0fd4685`
  (`avg3/last.pt`: `090588c0…`, `ecc8c15d…`, `97c4b7a5…`, `8e7be5ac…`.)
- **⏳ The five teacher folds as one ensemble: submission kernel PUSHED 08:06**,
  `raahimnawaz/rsna-knee-sub-a-ortho-teach1-cv5-nall` v1, built from `3ab71b9` (`--tag teach1-cv5`, three weight
  kernels, five shas with fold 0's `1c58b2bc…`). Read its gate before 17:00. Approved for the reset (owner 10-12
  ~20:46).
- **✅ The public checkpoint alone in our input: kernel COMPLETE** (`…-sub-a-ortho-oai-pa-nall` v1, pushed 06:16,
  90 s): `1 checkpoint(s), sha256 ok` · `CHECK vs oof_all.csv: 8 studies rebuilt from DICOMs, max |dp| 0.00031` ·
  `GATE PASSED (1/1)` · `[sub A] ARMS: coatnet_stage_b only, windows all` · 0 tracebacks. Ready for the reset
  (approved 10-12 ~21:24). Its inference is 1.8–3.0 s a study on two T4s.
- **`targets_teacher_v2`** (`data/student/targets_teacher_v2.csv`, sha256
  `fa98f9434e525d4dea2baddfdea962ee94bd697d69663f4c7a91b2a4ec2eb915`): `--teacher` the CoAtNet's `oof_all.csv` at
  0.5, then `--own-oof` the three `cv_last/oof_all.csv` files of the teacher-label fold kernels at 0.5. The OOF
  covers the 4,349 studies exactly once, no NaN; no cell masked; every finding's mean is unchanged (the same levels,
  re-ordered); mean |change| per cell 0.016 (Fracture) to 0.053 (PF OA); Spearman of our models against the target
  0.69 (Fracture), 0.70 (Synovitis), 0.71 (Lateral Meniscus) to 0.89 (Medial Meniscus). In
  `raahimnawaz/rsna-knee-student-targets` since 08:03 (five files; the other four byte-identical).
- **⏳ The pseudo-label student PUSHED 08:04:** `raahimnawaz/rsna-knee-c7-train-convnext-small-teach2-cv` v1, built
  from `3ab71b9`; fold 0, 12 epochs, 3 h cap, due ~10:00. Its notebook differs from a same-commit build of the
  0.944 model's in the targets' name and sha only (16 lines, rebuilt and diffed). The trainer has gained
  `--spatial-attn` since the 0.944 run (`9c56c45`); the flag is off here. Then: its sha, `--ours-only` alone against
  0.944 by the rule of 10-12 ~21:50. **The submission needs the owner's word.**
- **GPU by our count since the owner's 10.2 h:** folds 1–4 ~7.5 h, the two submission kernels a few minutes; the
  student ~1.9 h leaves ~0.7 h until Friday 17:00.
- **Owner 10-13 ~08:08: "how did the runs go, sure submit".** The go for the pseudo-label student's submission, the
  one asked for at 08:07 and on 10-12 ~21:50. It covers that submission and nothing later, and it goes only after
  the student's kernel reads `sha256 ok`, `GATE PASSED (1/1)`, `[sub A] ARMS:` and 0 tracebacks. **So four slots at
  the 17:00 PDT reset on 10-13:** the five teacher folds (against 0.945), the student's Synovitis probe (against
  0.532), the public checkpoint alone in our input (against its own 0.949), the pseudo-label student alone (against
  0.944). One slot stays free for the ensemble with the public checkpoint, to be decided after the scores.
- **✅ The five teacher folds: submission kernel COMPLETE 10-13 08:12** (`…-sub-a-ortho-teach1-cv5-nall` v1, 201 s):
  `5 checkpoint(s), sha256 ok` · `GATE PASSED (1/5)` … `(5/5)`, each member rebuilding 3 of its own held-out studies
  from DICOMs (max |dp| 0.00021 / 0.00032 / 0.00030 / 0.00029 / 0.00073) · `[sub A] ARMS: convnext_small_stage_b x5
  only, windows all` · 0 tracebacks. Ready for the 17:00 reset.
- **✅ The pseudo-label student, the run: COMPLETE 10-13 ~09:55** (seen 10:20; the Mac had slept again). 12/12
  epochs, 105 min, 0 tracebacks, labels sha `fa98f943…`, the 0.944 model's split and seed. `last.pt` sha256 (v7 of
  `rsna-knee-weights-sha`): `4cc8c3d45263bd705b86ff0c8b98d391935bdd553a0076268cd5f2dabf41ae02`. Small files in
  `fusion/runs_cnxs_teach2_f0/`.
  **Held-out 847, report only (a label change, and the targets carry a trace of these studies): 0.8825 against the
  0.944 model's 0.8804, paired +0.0021 ± 0.0013 (1.7 sigma), 7 of 12 up** — Baker's +0.011, Lateral OA +0.008, ACL
  +0.007, Lateral Meniscus +0.007; Fracture −0.009. For scale: the teacher labels were +0.0138 here and +0.006 on
  the LB. Gold-58, report only: 0.934 against 0.929. A small move; the LB decides by the rule of 10-12 21:50.
- **✅ Its submission kernel COMPLETE 10:23** (`raahimnawaz/rsna-knee-sub-a-ortho-teach2-f0-nall` v1, built from
  `355b68a`, `--tag teach2-f0`, 103 s): `1 checkpoint(s), sha256 ok` · `CHECK vs oof_all.csv: 8 studies rebuilt from
  DICOMs, max |dp| 0.00073` · `GATE PASSED (1/1)` · `[sub A] ARMS: convnext_small_stage_b only, windows all` · 0
  tracebacks. **All four kernels for the 17:00 reset are gated.**
- **⏳ The 17:00 submitter is armed (10:25):** a background job in the session of 10-13 morning waits for 17:00:30
  PDT and submits the four gated kernels (v1 each), skipping any already submitted in the new UTC day; it runs when
  the Mac is awake. The messages are under RUNBOOK §4's table for a submit by hand.

## Audit, 10-13 ~10:36 — owner: "make sure everything has been tested correctly … not wasting compute on dead ends … know the true strength"

**The instrument, measured** (resampling our own held-out predictions against the reference reader; 1,200–3,000
draws with replacement; the test's AUROC is higher than the held-out's, so these are if anything high):
- one macro score on 390 studies (the public part): **SD 0.0079**; on 910 (the private part): 0.0052;
- the five teacher folds against the five consensus folds, paired (labels differ, recipe the same): **SD 0.0030** at
  390, 0.0020 at 910 (correlation of the two scores 0.92);
- one finding's AUROC at 390: SD 0.013 (Medial Meniscus) to 0.033 (Fracture); Synovitis 0.029, Lateral OA 0.022,
  PF OA 0.018;
- a Synovitis probe pair (teacher against consensus, fold-0 models): paired SD 0.010.
The 09-27 measurement on the older models said the same (one score ±0.008, two different models ±0.005).

**What that says about what is on record:**
- Sound: five folds over one (+0.007, an ensemble against its member); ConvNeXt-small over ViT-L (+0.011 LB and
  +0.012 paired held-out); the teacher labels (+0.006 single = 2 SD; tonight's five-fold read is the second read);
  the Synovitis mask as a tie (one tick = 1.2 SD); the dense-slice null (+0.0030 ± 0.0023 held-out).
- **Corrected:** the probe table's "± 0.006" was the readout's rounding, not the uncertainty. Each level carries
  0.02–0.03 of sampling noise. The weak three sit ~0.07 under the rest, so "these three are the weak ones" stands;
  their order, and "Lateral OA = PF OA", do not.
- Every absolute score (ours and the board's) is ± 0.008; the top ten are inside one SD of each other.

**Tonight's four reads, what each can and cannot show:**
1. five teacher folds against 0.945: readable from ~+0.006; a clean label-only comparison.
2. Synovitis probe: its two-tick rule is ~2.4 SD: sound.
3. the public checkpoint in our input: a drop of ~0.006–0.010 or more from 0.949 is readable.
4. **the pseudo-label student: its held-out gain (+0.0021 ± 0.0013, and optimistic) is under one SD of the LB's
   paired noise, so the read cannot confirm it.** The dose was also small (half weight, the order mapped onto the
   old levels, mean change 0.02–0.05 a cell): a null closes this dose, not pseudo-labels. Rule tightened before the
   result: five folds on `targets_teacher_v2` only at ≥ +0.006 (RUNBOOK §4).

**Compute not to spend:** the fine-tune (dominated by the public notebook as stage 1: the known 0.949, no GPU
hours; recommended to drop, the owner to confirm since the owner asked for it); five new folds unless an arm passes;
`--k 24` last (the 0.949 model trains at 12 windows too). **Worth its 1.9 h:** spatial attention (labels fixed, so
the 847 judge it at SE 0.002 against a bar of two SE). **Costs no GPU and has the largest expected gain:** the
per-finding rank blend with the public checkpoint in its own preprocessing (the owner decides after the scores).


## Second audit, 10-13 11:42–11:55 — owner: the same request again, plus "make sure we are using whatever is publicly available that can help us / we can learn from"

What was checked this time was run, not re-read. Public sources were pulled fresh (board 11:42, notebooks and their
attached datasets through Kaggle's public listing, eight forum threads); everything from them is what others report.

**1. The repo's numbers reproduce.**
- `score_oof.py` re-run on the stored predictions: teacher against consensus +0.0138 ± 0.0019; pseudo-label student
  against teacher +0.0021 ± 0.0013; dense against baseline +0.0030 ± 0.0023; consensus against `dread_soft` +0.0049 ±
  0.0021. All four as recorded.
- The noise table, resampled again from the five-fold OOF: one macro at 390 studies SD 0.0075 (0.0049 at 910); five
  teacher folds against five consensus folds, paired, SD 0.0030 (0.0020).
- Gold is in no OOF file (0 of 58). The four kernels for tonight read COMPLETE. `infer_windows.py` writes the CSV at
  full precision and our outputs sit in 0.06–0.99: ≤ 4 ties of 847 at six decimals (see 4. for why this was checked).

**2. What is NOT measured.**
- **Run-to-run noise.** No recipe has been run twice (`git log --all --grep seed`, `9c23daf`'s RUNBOOK: nothing).
  The 847's "paired SE ≈ 0.002" is study sampling only. A hint at the size: the 0.944 model and the pseudo-label
  student share seed, split and recipe, their targets differ by 0.02–0.05 a cell, and single findings moved −0.009
  (Fracture) to +0.011 (Baker's). So the dense arm's +0.0030 "fail" and a +0.004 "pass" are one reading, and an arm
  expected at +0.002–0.005 (`--k 24`, spatial attention) cannot be judged on one fold and one seed.
- **Pseudo-labels.** The 847 are scored against a report reader (`lixin73`), which counts a student that out-ranks
  the report from the image as wrong; the LB cannot see a label gain under +0.006. The student's +0.0021 is not a
  size for the lever. One round, half dose, order mapped onto the old levels (`add_teacher`), one fold: that is all
  that has been tried of what the top of the board names.
- **The 17:00 submitter.** Not running: no process, no scheduled task (`rsna-submit-b-curve-a` and
  `rsna-push-stage-b-dread` are old and disabled), no cron. The morning session's job died with it. The new UTC
  day's slots last until 17:00 on 10-14.

**3. True strength, against the outside.**
- Board, 5,938 teams: #1 0.966 (two), 0.965 ×3, 0.964 ×3, 0.963 rank 9–19, 0.962 rank 20–25, 0.957 rank 90–151,
  **0.956 rank 152–734 (583 teams)**, 0.954 rank 799–1,245 (447), 0.950 rank 1,345–1,593, 0.949 rank 1,594–1,687,
  **ours 0.945 at rank 1,792**. Medals by Kaggle's rule: gold ~21, silver ~297, bronze ~594: the last two inside the
  0.956 block, where rank follows the unrounded score.
- Public notebooks, 500 scored: 14 at 0.956 (all `nartaa` 384 px + `heliosli` R7 or S6 at full precision, 50/50),
  none higher; 29 at 0.954; 10 at 0.949 (`nartaa` alone). Without `nartaa`'s weights the best public notebook is
  0.946 (the old Raptor stack). R7's score alone is published nowhere.
- Forum (735304, 748011, 748111, 748295): rank 5 and 6 say competition data only; five-fold ConvNeXt-T 0.959; a
  single-fold 0.956; one "basically single" submission at 0.960; ConvNeXt-T 0.953 and DINOv2 ViT-S 0.951 "trained
  with pseudo labels … no external data"; single fold on LLM labels alone 0.947; "model is not the key. it is your
  labels"; a rank-29 team: five-fold OOF 0.937 against LLM labels, gold-58 0.938, LB 0.961.
- Ours on the same instruments (`scratch: audit_strength.py`, not kept): five consensus folds OOF 0.8636 against the
  held-out reader, 0.8997 against `dread_soft`, 0.9049 against their own targets, gold-58 0.9306, LB 0.945; five
  teacher folds 0.8796 / 0.9072 / 0.9146 (0.9676 against their own targets, which hold an image model), gold-58
  0.9313. About 0.015 under those pipelines wherever a number can be set beside theirs; other label tables, so
  indicative.
- **So the review's "0.97 needs OAI" is wrong** (corrected at §0.6 and §6 there): OAI is worth +0.003–0.005 to the
  one public model that has it.

**4. Public material: what we use, what we do not, what it teaches.**
- In use: `dread_soft` and six more report readers; `nartaa`'s OAI-trained CoAtNet as a teacher (`targets_teacher_v1`).
- **Not in use: the free 0.956 recipe itself**, 0.011 above our best: `nartaa` 384 px + R7's four folds, per-finding
  percentile ranks 50/50. **The 0.954 → 0.956 step was not a model:** the S6/R7 scores had been written at six
  decimals, which ties their small probabilities (Baker's: 43 distinct values in 58 studies); writing them at full
  precision is the gain (`goodpjw2008`'s notes, 10-13).
- **Not in use: `nartaa`'s second checkpoint** (224 px from a 0.8 centre crop, 0.945 alone, same dataset): at 35% of
  the CoAtNet slot it moved its author's rank 437 → 320 inside 0.954 and 233 → 156 inside 0.956, the displayed score
  unchanged.
- **What a weaker member does, measured by its own author:** `goodpjw2008`'s reader (0.929 alone) with `nartaa`
  0.950; on the 0.954 blend 0.954; "at full weight cost about 0.002" on the full-precision 0.956 base, so he runs it
  at a tenth. `prvsiyan`: 0.949 → 0.948 at a flat 30%. Our own 10-04: 0.941 against 0.943. Rule 4, three times over.
- **The fine-tune already exists in public:** `syouyatobita/rsna-knee-g231-nf1-weights` (10-11), `nartaa`'s checkpoint
  fine-tuned 6 epochs on all data at 320 px ("coat320_nartinit_ft6_gemini_s42_full"); no public notebook attaches it
  and no score is given. Nothing there argues for spending our 3.5 h on the same thing.
- **R7's trainer is public in full** (`compact_train.py`, 83 KB, in `jessky/rsna-knee-local-cv-cnx50` with the whole
  training tree, and embedded in the R7 notebook): our backbone with spatial attention, 176-window cover sampling,
  mirror augmentation, EMA, the state head. The recipe arms we queue one at a time are a published, scored whole.
- One more OAI-trained weight set, with no score and no notebook using it: `rauleg/rsna-knee-weights-dense-r288-attn-
  swa-oaij` (five ResNet34 folds on Kaggle + 1,660 OAI knees). New label tables (`hukumjailer` 5-source consensus,
  10-13; `shogomiyawaki`, 10-12): more readers of the same reports, not leak-checked, not proposed.
- An instrument we did not have: rank inside a tie block reads below 0.001 for any submission that becomes our best
  (`varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook`).
- **A rules risk:** thread 748276 (10-11, 15 votes, no host reply) argues OAI fails rule 2.6.b because NDA has barred
  China and five other countries since 04-2025. A participant's argument, not a ruling. It would take the free
  recipe, and our teacher labels and teacher folds with it; the five consensus folds are our only OAI-free model.

**5. Compute and slots, as recommended (nothing here is approved).**
- Tonight's four reads cost no GPU and stand. Read 1 is the one that decides something (the finals' candidate).
  Reads 2–4 are cheap information: none can buy GPU hours by itself (read 4 only at ≥ +0.006, which a true +0.002
  reaches about one time in ten).
- Drop the fine-tune (3.5 h) and `--k 24` (3.3 h). Hold spatial attention (1.9 h).
- The hours that are left (30 from Friday, shared with casmi26) are worth more on one full pseudo-label round as
  five folds, read five against five on the LB, than on recipe arms worth +0.004 on a model 0.011 under what is
  free. Its dose and teachers are not written yet.
- No training GPU and where the ranking is decided: the free recipe + the 224 px checkpoint + our five folds at a
  small fixed weight, told apart by rank inside the block. Needs the owner's word (RUNBOOK §4).

## Owner 10-13 ~12:00: "Use the public blend but put what lever we have that's better than what it has or what can improve it into it and submit that. Only submit those other 4 if they're actually worth anything"

Read as: the go for ONE submission, the free public blend with whatever of ours can add to it, and for its kernel's
commit run (minutes of GPU); and my judgment on the four reads approved earlier. It ends the "how to ensemble" question
left open on 10-12 21:24.

- **What we have that the blend lacks.** Nothing of ours is shown better than it (second audit, 3. and 4.). Two things
  can add: `nartaa`'s 224 px checkpoint (public; measured by its user as a rank gain inside the block) and our five
  folds as one more family. So the base is the public notebook that already carries the first,
  `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956` (sha256 `6406e92300011e48…`), run unchanged.
- **Which folds: the five on `targets_teacher_v1`** (stronger alone: +0.006 as one model on the LB, +0.016 OOF). The
  consensus folds are less like the public CoAtNet (within-class rank correlation 0.80–0.83 against 0.85–0.88, on
  gold-58 and on the 4,349, the public model in OUR input) and equal on gold-58 (0.931): a second variant if tonight's
  read 1 puts the teacher folds under them.
- **Weights, fixed before any run:** ours 0.25 on ACL, MCL, Medial Meniscus, Medial OA, Effusion, Baker's, Contusion,
  Fracture; 0.10 on Lateral Meniscus, Lateral OA, PF OA, Synovitis. From the 10-12 probes of our five folds (0.966–0.968
  on the first group, 0.88–0.93 on the second) against a blend whose macro is 0.956, and the correlation above: under
  a two-score normal model a member at par with ρ 0.85 adds ~+0.002 a finding at 0.25, one 0.006 behind ~+0.001, one
  0.03 behind ~0 at 0.10. Expected +0.001 macro (−0.001 to +0.002). Gold-58 was looked at for the correlation only;
  nothing was chosen on it.
- **`notebooks/build_public_blend_kernel.py`** (`6bc69f9`): the 13 public cells byte for byte, then our stage
  (`build_submit_kernel`'s OURS cell: sha256 of each `last.pt`, the gate on 15 held-out studies in the commit run,
  both GPUs) under a deadline of nine hours less 25 minutes from the notebook's own clock, then the blend. In the
  scored rerun a fault, a timeout, under an hour left, fewer than 90% of studies scored or a mean Spearman under 0.2
  with the public ranks all leave the public `submission.csv` as it is; in the commit run each stops the notebook. A
  study our reader cannot build keeps the public rank. `notebooks/public_blend_fixture.py`: 8 offline checks pass.
  **One thing the commit run has to show:** the public cells pin another image than our kernels' and install their
  own wheels, so our reader runs on other library versions (`OURS_ENV` line); the gate is what says it still
  reproduces the scored OOF.
- **⏳ Kernel PUSHED 12:12:** `raahimnawaz/rsna-knee-sub-pub956-teach1cv5` v1, built from `6bc69f9`.
- **The four reads.** Sent: 1 (the five teacher folds alone: the strength of the member that is now in the blend, and
  the finals' candidate) and 4 (the pseudo-label student: the one LB point on the lever proposed for Friday; only
  ≥ 0.950 or ≤ 0.940 would move that plan, about one outcome in six). **Not sent: 2** (the Synovitis probe: the
  teacher labels are kept either way) **and 3** (the public checkpoint in our input: the blend runs it in its own
  notebook, and the fine-tune it would have sized is dropped). Their kernels stay, gated.
- **Rule for the blend, written 12:15 before its commit run finished (RUNBOOK §4, read 5):** paired noise of such a
  pair ~0.0007 at 390 studies (resampled on a stand-in pair: consensus folds as the base, teacher folds as the member
  at these weights). ≥ 0.957 → our folds stay in; 0.956 → read the rank inside the block, nothing claimed; ≤ 0.955 →
  they come out.
- **✅ The blend's kernel COMPLETE 10-13 12:23** (`raahimnawaz/rsna-knee-sub-pub956-teach1cv5` v1, 11.5 min, from
  `6bc69f9`). The public cells, as published: `RAPTOR_STAGE_DONE` · R7 `OURS_STAGE_COMPLETE {"status": "PASS", …
  "scored": 3, "fallback_studies": 0}` · `R224_STAGE rc=0 ok=True` · `NOTEBOOK_COMPLETE`. Ours:
  `OURS_ENV torch=2.11.0+cu128 timm=1.0.25 pydicom=3.0.1` (another image than our kernels') · `5 checkpoint(s), sha256
  ok` · `GATE PASSED (1/5)` … `(5/5)`, max |dp| 0.00021 / 0.00032 / 0.00030 / 0.00029 / 0.00073, **the same five
  figures as in our own five-fold kernel: the reader is unchanged by the other image** · `OURS DONE: 3 test studies
  in 2.6 min` · `OURS_STAGE ok=True seconds 154 left_at_start_min 508` · `[public+ours] ARMS: the public blend as
  published + convnext_small_stage_b x5; w_ours …` · 0 tracebacks. `submission.csv`: the public file's columns and
  study order, finite in [0, 1]. Time, from the three test studies: the public cells 7.0 min (R7 325 s, most of it
  compiling), ours 2.6 min; on ~1,300 studies ours is ~1–1.5 h (4–8 s a study on two T4s), the public part is not
  known from this run.
- **⏳ The 17:00 submitter is armed (12:26)** in the session of 10-13 midday: read 5 (the blend), read 1 (the five
  teacher folds) and read 4 (the pseudo-label student), v1 each, at 17:00:40 PDT, skipping any already sent in the
  new UTC day. It needs the lid open.

## Owner 10-13 ~12:30: "Does it have our label quality. How can we maximize the gold labels and what we have of the reports. Could we make something that has the gold labels trained on multiple times with other slightly lower quality ones mixed in if that would work and only if it could help"

Answered from the record and from one new table. Nothing was run on a GPU and nothing is approved.

- **Does the blend have our labels?** Only through our five folds, at 10–25% of each finding's rank. The two public
  members were trained on their authors' labels: `nartaa`'s on three of the public readers we also use
  (`stevenleehans`, `pilkwang`, JEV) plus its own read and OAI's grades for three findings; R7's on its author's own
  LLM read with a weight per cell (`probe_targets_opusv11b_rr7_plr7_cdr.csv`, not public), report states as a second
  head and fastMRI+ boxes as an auxiliary task. **Ours are not shown to be better than theirs:** our two label steps
  read +0.005 and +0.006 on the LB (kept, neither shown), and the models trained on theirs score above ours.
- **The reports against the radiologists, on the 58 gold studies** (the three readers that carry gold rows, their
  mean value; the image model = our five teacher folds, which never trained on gold; report only):

  | finding | gold + | report AUROC | image AUROC | gold positives the readers call absent | gold negatives the readers call present |
  |---|--:|--:|--:|--:|--:|
  | ACL | 24 | 0.994 | 0.980 | 0 of 24 | 5 of 34 |
  | MCL | 9 | 0.982 | 0.955 | 0 of 9 | 3 of 49 |
  | Medial Meniscus | 26 | 0.945 | 0.972 | 1 of 26 | 4 of 32 |
  | Lateral Meniscus | 23 | 0.874 | 0.894 | 3 of 23 | 3 of 35 |
  | Medial OA | 15 | 0.926 | 0.983 | 1 of 15 | 4 of 43 |
  | Lateral OA | 11 | 0.872 | 0.822 | 2 of 11 | 4 of 47 |
  | PF OA | 21 | 0.905 | 0.852 | 4 of 21 | 0 of 37 |
  | Effusion | 35 | 0.859 | 0.981 | 0 of 35 | 2 of 23 |
  | Synovitis | 27 | 0.741 | 0.853 | 4 of 27 | 4 of 31 |
  | Baker's | 12 | 0.908 | 0.982 | 1 of 12 | 4 of 46 |
  | Contusion | 19 | 0.874 | 0.949 | 2 of 19 | **12 of 39** |
  | Fracture | 18 | 0.863 | 0.951 | 3 of 18 | 5 of 40 |
  | all | | 0.895 | 0.931 | **21 of 240 (9%)** | **50 of 456 (11%)** |

  So the report labels are wrong on about one cell in ten by the radiologists' reading, and the image model already
  reads gold better than the labels it was trained on (0.931 against 0.895): of the 21 positives the readers miss,
  it ranks 16 in its top half. Gold is enriched (Fracture 31% positive against 4% in our targets), so its prevalences
  say nothing about the test's. Per finding these are 2–12 studies: nothing is to be chosen on them (rule 1).
- **Training on gold, repeated, with report labels mixed in: it would not help in any way we could measure.**
  (1) 58 studies are 696 cells against 49,784 report-derived ones, 1.4%. What more data buys here is on record:
  fold → all data, +847 studies, was 0.927 → 0.929 on the LB (Ortho, 10-03). Fifty-eight studies, each counted as
  five for being clean, are ~+0.0005. (2) Repeating a study adds no information: a model fits 58 studies in a few
  epochs and they then stop moving it (the selection table of 08-12 is the same arithmetic). (3) The 847 resolve
  ~0.004 and the LB ~0.006 for a label change, so the result could not be read. (4) Neither public member trains on
  gold (R7's manifest: `"gold_in_training": 0`; `d571f4c` for `nartaa`'s line). (5) It ends that model's gold read.
  The history has it once, as "gold x3 = finals only" (`c962e52`), never run, its supposed source corrected in
  `d571f4c`. **Free version:** the 58 as ordinary rows in a final all-data fit; no GPU, no claim.
- **What the same idea looks like with enough data behind it: pseudo-labels.** "Better labels on top, slightly worse
  ones mixed in" is what a pseudo-label round builds, on all 4,349 studies and not on 58: the table above is the
  evidence that our out-of-fold predictions are closer to the radiologists than the reports are. It is the lever
  proposed for Friday's hours (RUNBOOK §4), and gold stays outside it as the one image-read check.
- **What is left in the reports:** as a label source, little: seven readers agree with each other at r 0.82–0.95
  and with gold at ~0.90, and a better reader was costed and not run (10-12, E). The one thing in the table that a
  reader could act on is a convention, not accuracy: Contusion, where the readers call 12 of 39 gold negatives
  present. Twelve studies; noted, not proposed.

## Owner 10-13 ~12:37: "Change of approach for the last 9 days: stop testing one change at a time" — the v3 recipe

The owner's message, in full: are the golds held out, and should we train on them and use the submission as the
test; then seven points. (1) ONE combined recipe "v3" on the ConvNeXt-small trainer: stronger pseudo-labels than
`targets_teacher_v2` (larger model share, from five-fold held-out predictions; the exact dose in the LOG before any
run, nothing chosen on gold-58), `--spatial-attn`, mirror augmentation with Medial ↔ Lateral targets swapped and MCL
masked on mirrored samples plus a mirrored pass averaged in at inference, `--k 24`, 16 epochs, submit the averaged
weights. (2) Start now on CPU: mirror code with unit tests, the label table, a CPU smoke of the full recipe. (3) GPU:
"I am getting [FILL IN]" (the blank was not filled in); prepare to train off Kaggle: pull the Stage B cache, and a
way to get off-Kaggle weights into a gated submission kernel. (4) **Standing go, no need to ask per run:** one fold
of v3 as a not-broken check, then the other four, then a second label round from v3's own held-out predictions if
hours remain; stop and tell the owner only if fold 0 is clearly worse than the current model. (5) **Standing go for
two submissions:** v3 five folds alone, and v3 inside the public blend in place of the current folds; ask before any
others. (6) Keep one OAI-free line alive for a final. (7) First, confirm the 17:00 submitter is running, re-arm if not.

This supersedes the "one arm at a time, each against a rule" queue of RUNBOOK §4 and the audit's recommendations to
drop `--k 24` and hold spatial attention: the owner has decided to run them together. What it costs is on record: if
v3 moves, nothing says which part moved it.

**7. The submitter was NOT running (checked 12:37–12:41).** No process (`ps`: no waiting shell or Python, no child of
any session), no scheduled task, no cron or launchd job; the last submission on the account is 10-12's. The 12:26
submitter was a session's background task and went with its session, as the morning's did. All three kernels read
COMPLETE. **Re-armed 12:42** with `notebooks/submit_at_reset.py` (`c24bd36`): it forks twice into its own process
session (pid 96147, parent 1), polls the wall clock so a Mac that slept sends on wake, lists the account's
submissions before every send and skips an item already dated on the new UTC day, and logs to `data/submitter.log`.
It fires at **17:02:00**, after 17:00:40, so if an older submitter is alive somewhere it goes first and this one
sends nothing. Tested without sending: the dry run lists tonight's three; against yesterday's slot day it skips the
two that were sent and keeps the one that was not. It does not survive a reboot.

**The golds.** Yes: the 58 graded studies are in no training list and no target table (the trainer and the table
builder both refuse them), and every run scores them as a report-only number. Training on them is not in v3: 58
studies are 1.3% of 4,407, worth ~+0.0005 by the fold → all-data step on record, which no instrument here can see;
and the gold read is the one number whose reference is a radiologist, so the only one not biased against
pseudo-labels (the 847 are scored against a report reader). "The submission as the test" is already how labels are
judged (rule 3). If the owner wants the 58 in anyway, it is a change to three guards and costs that read; not done.

**The dose, written 12:53 before the table is built. `targets_teacher_v3` = `targets_teacher_v1` with the five
teacher-label folds' held-out predictions at 0.75 of every cell** (`build_consensus_targets.py --own-oof
fusion/runs_cnxs_teach1_{f0,f12,f34}/…/cv_last/oof_all.csv --w-own 0.75`; the models' order on the target's own
levels, as in v2). The same predictions as v2, so the dose is the only change: reports 12.5%, the public CoAtNet
12.5%, our held-out models 75% (v2: 25 / 25 / 50). Why 0.75 and not 1.0: at 0.5 a cell where the report and the model
disagree lands in the middle; at 0.75 it lands on the model's side (report-positive but ranked low → ~0.26;
report-negative but ranked high → ~0.73), which is what makes it a pseudo-label; at 1.0 the reports are gone, and on
Lateral OA and PF OA the model is the weak reader (review §2.2), so it would train on its own errors there. Not
taken: a second model family in the held-out mean (the consensus folds are the weaker member, rule 4), per-finding
doses or teachers (nothing but gold could set them, rule 1). **Gold-58 was not consulted for any of this.**
**The OAI-free twin, same dose: `targets_consensus_v3` = `targets_consensus_v2` with the five CONSENSUS folds'
held-out predictions at 0.75** (no `--teacher`: nothing in it has seen the OAI-trained checkpoint).

**The recipe, v3, fixed before any run** (against the 0.944 model's: 12 epochs, k 12, global average, no mirror):

```bash
$PY notebooks/build_backbone_ab_kernel.py --arch convnext_small --corpus stage_b --folds 0 --n-train 0 --epochs 16 \
    --time-budget-h 7.0 --targets data/student/targets_teacher_v3.csv \
    --targets-from raahimnawaz/rsna-knee-student-targets --tag v3 \
    "--train-args=--allow-missing-labels --spatial-attn --mirror 0.5 --mirror-tta --k 24 --micro 1" --out <dir>
```

- `--mirror 0.5`: half of the training draws are the knee mirrored medial ↔ lateral (the public R7's augmentation,
  review §4.1). `--mirror-tta`: every prediction is the mean of the plain and the mirrored pass, MCL from the plain
  pass; `oof_all.csv` holds that mean (so the submission gate checks what is submitted), `oof_plain.csv` the plain
  pass. `last_meta.json` carries `mirror_tta`, and `infer_windows.py` does the same per checkpoint.
- `--k 24 --micro 1`: 24 windows a study, one study per forward per GPU. Two studies of 24 windows would be 48
  windows a forward, and 24 already peak at 8.1 GB on a T4. The batch stays 8 studies (1 × 2 GPUs × 4 accumulated).
- Submitted: `avg3/last.pt`, the mean of the last three epochs' weights, with `avg3/oof_all.csv` as its gate.
- **One fold per kernel:** ~16.4 min an epoch at k 24 (twice the 8.2 measured at k 12) × 16 = 4.4 h, plus ~0.35 h
  for four evaluation passes: **~4.9 h a fold on two T4s, ~24.5 h for five**, of a 30 h week shared with casmi26.
  A second label round is another ~24.5 h: it does not fit on Kaggle. Estimates, not measurements.

**"Clearly worse", fixed before fold 0 runs.** The check is for a broken recipe, not for a gain: the 847 are scored
against a report reader and under-credit pseudo-labels, and run-to-run noise has never been measured. Against the
current model's fold 0 (`runs_cnxs_teach1_f0`, 847 macro 0.8804, the 0.944 model) on the same studies, with
`score_oof.py`, v3's `avg3/oof_all.csv`: **stop and tell the owner if the macro is more than 0.008 below (4 paired
SE), or if any of MCL, Medial / Lateral Meniscus, Medial / Lateral OA is more than 0.03 below** (what a mirror or
swap fault would look like; single findings moved ≤ 0.011 between two like models), **or if it did not finish 16
epochs.** Otherwise folds 1–4 go without asking. Gold-58 is reported and decides nothing.

**Built 10-13 12:42–13:05, all on CPU; nothing has trained.**

- **The mirror** (`32fd2b6`). `train_windows.py --mirror P --mirror-tta`; `infer_windows.py` reads `mirror_tta` from
  `last_meta.json`. The canonical frame (`build_slots64.canon_ops`) puts the medial–lateral axis on the slice order
  of a sagittal slot and on the columns of a coronal or axial slice, so a mirrored knee is: a sagittal window's three
  slices in reverse order at depth 1 − pos; a coronal or axial slice flipped left-right at the same depth. Targets:
  Medial ↔ Lateral Meniscus, Medial ↔ Lateral OA; MCL becomes NaN (no loss). `pos_weight` uses the prevalence the
  loss sees (a medial column is its lateral twin's on the mirrored share). At test, MCL is the plain pass alone.
  **`notebooks/mirror_fixture.py`, 66 checks:** (1) two synthetic knees through the real DICOM builder, one with its
  anatomy reflected: `mirror_windows` of the first equals the second to ±1 grey level in all three planes, right and
  left knee, at the mirrored depth; without the mirror they differ by > 50, and a mirror that flips a sagittal slice
  in-plane, or keeps the unmirrored depth, fails; (2) the target swap; (3) on a random model the trainer's batched
  two-pass `predict` equals the submission path's arithmetic to 1e-5, and mirroring the input swaps the medial /
  lateral columns of the mean and nothing else; (4) `mirror=0` is the old sample (the existing canon and
  infer-parity fixtures still pass), `mirror=1` its mirror; (5) the whole flag set trains four epochs on a tiny conv
  encoder and writes `oof_all.csv` (the mean), `oof_plain.csv`, `avg3/`, and a meta with `mirror_tta`.
  **Also run under two-process DDP on this laptop** (the path Kaggle's two T4s take): equal weight checksums on both
  ranks, the sharded two-pass evaluation and `avg3` written.
- **The tables.** `targets_teacher_v3.csv` sha256 `6f786dda00286503c8a541fc254f3a648eca48809bb7c323a61552e71d39f36a`;
  `targets_consensus_v3.csv` `264015596475582aac0f1739b64a4f7128117201698ee2a4941bc9d5d1deafd5`; both 4,349 studies,
  no masked cell, in the private Kaggle dataset `raahimnawaz/rsna-knee-student-targets` (new versions 12:54 and
  13:00, with `folds_site_stage_b.csv` `b8d82239…`). Checks: the builder at `--w-own 0.5` rebuilds
  `targets_teacher_v2` byte for byte; v3 from the five git-tracked per-fold `oof_all.csv` files is the same sha as
  from the untracked `cv_last` files, so a fresh clone can rebuild it. **Size of the dose, measured on the tables
  (not on gold):** mean cell change against `targets_teacher_v1` 0.050 (v2: 0.033); cells crossing 0.5: 2,555 of
  52,188, 4.9% (v2: 3.4%). So v3 is 1.5× v2's dose. The OAI-free twin moves 0.080 a cell and 9.5% across 0.5 (its
  base held no model), and its 2,404 unread Synovitis cells are filled by the consensus folds' predictions alone.
- **✅ The CPU smoke of the full recipe on the real cache: COMPLETE 10-13 13:09** (pushed 12:55, 7.9 min of
  notebook time; `raahimnawaz/rsna-knee-c7-train-convnext-small-v3s-cv` v1 from `a85f8c8`: `--smoke`, the v3 flags,
  two epochs so the averaged weights are reached, `--k 2` as every smoke). Log: `targets_teacher_v3.csv … sha256
  6f786dda` found in the attached dataset · `corpus layout stage_b … canon True: 4407 studies` · `fold 0: train 3502
  | val 847` · two processes · `spatial attention:` · `mirror: 0.50 of training draws mirrored` · `mirror averaging:`
  · `fit-budget … -> 2 epoch(s)` · loss 1.2186 → 0.8247 · `DDP weight checksum per rank` equal · `SMOKE: the whole
  path ran` · **0 tracebacks**. Files: `oof_all.csv` and `oof_plain.csv` for the last epoch and for `avg3`; MCL
  identical in the two, the other findings moved by the second pass (up to 0.14 on two studies of an untrained
  model, which says only that the pass ran). **Not reached by a smoke:** GPU memory at 24 windows with spatial
  attention, fp16, the weights and `last_meta.json` (a smoke saves none; the local fixture covers the meta), and
  the time of an epoch. The first minutes of fold 0 show the first two and `fit-budget`'s epoch estimate.
- **Off Kaggle, and the way back** (`fea98f6`, `notebooks/offkaggle.py`). `train` runs one fold on a local copy of
  the cache with the kernel's own package and `V3_TRAIN_ARGS` (imported from the builder; `--recipe v3` there);
  `--dry` for fold 0 writes lists **identical, in order,** to the Kaggle fold-0 kernel's (3,502 / 847); it refuses a
  finished run that lost studies or epochs. `pack` makes one flat private dataset per fold (`last.pt`,
  `last_meta.json`, `oof_all.csv` side by side, which is what the submission kernel looks for), and prints the
  `--weights-dataset … --weights-sha …` line; `build_submit_kernel.py` and `build_public_blend_kernel.py` take
  `--weights-dataset`, with the same sha256 and gate as a kernel source. Tested offline only: two toy runs packed,
  a submission kernel built on them with the datasets attached. **Not tested: a real upload and its gate on Kaggle
  (needs a real fold), and `pull` (62 GB; nothing has been downloaded, and where to is the owner's answer).**
- **The OAI-free line.** Alive today as the five consensus folds (0.945). Its v3 is one table away from training
  and costs a second set of five folds; not started, no hours named for it.
- **`--time-budget-h 9.0` for a v3 fold, not the 7.0 written above (13:05, before any run; a budget, not a recipe
  change).** `--fit-budget` cuts epochs when 16 × (the measured epoch × 1.35) does not fit, and 7.0 h holds 16 epochs
  only up to ~19 min an epoch against an estimate of 16.4. At 9.0 h it holds to ~25 min, and 3 h of the session
  remain for evaluation. The real fold-0 kernel builds from `62c3d9d` with `--recipe v3` and carries the same flags
  as the smoke (checked in the built notebook); it is not pushed.
- **The cache is retrievable and its size is now measured, not estimated:** shard 0's manifest and ids fetched
  (`c7-96slice-384`, canonical, 551 studies → `vols.npy` 7.8 GB); eight shards ≈ 62.4 GB. Its files sit under
  `stage_b/` in each kernel's output.
- **Not done, and why:** no GPU run (the standing go names v3's folds, but no GPU is available: Kaggle has ~0.7 h
  until Friday, and the owner's GPU was left blank); no cache pull (destination unknown).

## Owner 10-13 ~14:15: "so why arent we using robot b in conjuntion with better answer keys" — R7's trainer with pseudo-labels

Nothing was run and nothing is approved. Read for this, all public and small: `manifest.json` of
`heliosli/rsna-knee-compact-r7cnxfsss` (R7's weights), and `README.md` and `code/compact_train.py` of
`jessky/rsna-knee-local-cv-cnx50` (the training tree). Sizes of a run are my estimates, not measurements.

- **Why not: it was never proposed.** The 11:55 audit noted the trainer is public and stopped there; the owner's v3
  names "the ConvNeXt-small trainer", ours, and ports three of R7's parts into it (spatial attention, mirror, more
  windows). Training R7's own recipe on better labels is the fuller version of the same idea and was not weighed.
- **What R7's run was (its manifest's `run_config`):** ConvNeXt-small at 384 px, crop 352; **176 windows a study**
  by cover sampling at a 3.5 mm physical step, batch 2, 16 epochs, 4 folds, EMA 0.999, wd 0.05, `compile`;
  `spatial_attn`, `mirror 0.5`, pair-ranking 0.5, OA extent 0.2, `syn_decomp`, a report-state head at 0.5
  (`state_targets_opusv11b_rr7.csv`), per-cell weights (`cell_weights_plr7_v7.csv`), and a fastMRI+ auxiliary task
  at 0.3 on 96 windows (`fm384_v1`). Its own pixel cache (`px384_v4` + a side-fix override, views v4, a side
  classifier, knee anchors). 4,344 pool studies, gold in none. Trained under `/root/autodl-tmp` (a rented machine);
  the trainer's comments name a 5090 and a PRO 6000, and say gradient checkpointing "keeps 208 windows/study on 32
  GB cards" at ~1.3× compute. **Inference runs with `flip_tta: False`** although it trained with mirror 0.5.
- **What is not public:** its targets, cell weights and state targets (the trainer wants its own schema: `is_gold`,
  `<finding>|prior` columns). `--state-targets`, `--extra` and the fastMRI+ task are optional arguments; `--folds-csv`
  takes a StudyInstanceUID,fold file such as ours. fastMRI+ was removed from this Mac on 10-09.
- **Cost against ours (estimate):** 176 windows × ~3,260 training studies is ~13.6× our k 12 epoch's windows, ~11×
  after the smaller crop: ~1.5 h an epoch on two T4s, ~25 h a fold, ~100 h for four, and 176 windows with gradients
  do not fit a 16 GB T4 (24 of ours peak at 8.1 GB). **It is not a Kaggle job.** On one rented 5090, roughly 6–8 h
  a fold, 25–33 h for four. Its cache has to be rebuilt from the DICOMs with its own code first.
- **Already public, unscored: a pseudo-labelled earlier R7.** `jessky`'s dataset holds five `cnxS6s_d50` folds, the
  S6 recipe trained on `labels_distill50` ("incorporates an external OOF teacher"), with `distill.py` in the tree.
  Its README calls it a provisional local-CV candidate and says second-order leakage is not ruled out. No public
  notebook attaches it (`kaggle kernels list --dataset`: none) and no score is given anywhere we have.
- **Never scored in public: R7 alone.** As trained it has no OAI in it (report-derived targets, fastMRI+ as a side
  task), so alone it would also be an OAI-free candidate, to be read against our 0.945.
- **So three reads cost no training:** R7's four folds alone; R7 with its mirror pass switched on, paired against
  it; `jessky`'s d50 folds (needs an inference notebook written from the tree's `compact_infer.py`). Each is a
  submission and the GPU minutes of a commit run: the owner's go.
- **Where it leaves v3:** with Kaggle's T4s only, R7's recipe cannot train and v3 is what fits. With a 32 GB card,
  R7's trainer on a pseudo-label table is the stronger use of the hours, and v3 becomes the fallback.

## Owner 10-13 ~14:40: the cost of R7 + pseudo-labels on a 5090; "we can do 1, but check it first"; "add oai to robot b"; "then do 3"

Read for this: `manifest.json`, `compact_infer.py`, `px_cache.py`, `distill.py`, `label_source_policy.py` of
`jessky/rsna-knee-local-cv-cnx50`, and the notebook `heliosli/0-957-latest-dataset-score-in-one-click` (source only).
Nothing was run on a GPU, nothing pushed, nothing submitted.

- **Check of read 1 (`jessky`'s d50 folds): NOT recommended as it stands.** (a) It is not R7's recipe with new
  labels: its manifest says 288 px, crop 256, 48 windows, 10 epochs, 5 folds (R7: 384 / 352, 176 windows, 16 epochs,
  spatial attention, mirror, the extra heads), so its score would not say what pseudo-labels add to R7. (b) The
  inference notebook that ran it is not public and the manifest omits what `compact_infer.py` needs (`--views`,
  whether a `--state-spec`); no held-out predictions are published, so there is nothing to gate a rebuilt path
  against: a wrong setting would score low and read as "pseudo-labels do not help". (c) The teacher behind
  `labels_distill50` is not named anywhere public ("an external OOF teacher"). Its code makes no network or shell
  calls (`compact_infer.py`, `px_cache.py`: numpy and torch only; the vendored `requests` is for its download
  scripts). My 14:15 note called it "an earlier R7 on pseudo-labels": that left out that it is the small recipe.
- **"Leakage", as its README means it:** its fold models' labels hold a teacher's predictions; if that teacher (or
  the model that taught it) had trained on the studies a fold holds out, the fold's held-out score is inflated. It
  flatters local validation only; the hidden test cannot leak. Ours has the same property (the public CoAtNet in
  `targets_teacher_v1` trained on every study), which is why label changes are judged on the LB.
- **Read 3 is one supported flag.** The R7 notebook's inference worker has `--flip-tta` ("every fold model also
  scores the mirrored windows", MCL from the plain pass); R7's manifest ships `flip_tta: False`. Its author built the
  flag on 10-11 and left it off on 10-12: no reason is given (no gain, or the time it doubles). The notebook writes
  R7's own predictions as `ours_sub.csv`, so R7 alone is that file published as the submission. **Read 3 needs a
  baseline:** R7 alone (read 2, not yet approved), or the blend's known 0.956.
- **"Add OAI to robot B".** OAI's images: not available to us in nine days (the owner's login, the files, and a
  grade mapping its one public user has not published). OAI's knowledge through the labels: yes, by keeping the
  OAI-trained CoAtNet in the table R7 trains on, as `targets_teacher_v1/v3` do. That R7 would no longer be OAI-free.
- **The 5090 run, one pass (all estimates; speed and memory on a 5090 never measured by us):** rebuild R7's pixel
  cache from the DICOMs ~3–6 h; R7's four public folds scored on their own held-out training studies as the teacher
  ~2 h; the table (reports + the OAI-trained CoAtNet + R7 held-out + ours held-out) on CPU; four folds × 16 epochs ×
  176 windows ~25–32 h with gradient checkpointing (a 32 GB card cannot hold the published batch without it);
  setup and a first-epoch check ~4–8 h. **~35–45 h of machine time.** Prices found today (snapshots from July–August
  2026): Vast.ai $0.20–1.34/h, median $0.53; RunPod $0.69–0.99/h. **≈ $20–45, plus a few dollars of disk; a failed
  late run doubles it.** Two days of wall clock after a day of porting.
- **"Do them on the GPU":** training and held-out checks can run there. A leaderboard read cannot: the scored
  rerun and its commit run are Kaggle's (~12 GPU-min a kernel; ~0.7 h left this week by our count).

## Owner 10-13 ~15:10: "Yes both checks"; "make sure those 3 submissions are actually worth it"; "if b is overall the best then we should add oai ones to improve it. Then we can change the code if we have to"

**The go:** two submissions and their commit runs (~25 Kaggle GPU-min of ~0.7 h): R7 alone, and R7 with its mirror
pass on. Both written here before either is built or run.

- **Kernels** (`notebooks/build_r7_read_kernel.py`): the public notebook our blend kernel runs (sha256
  `6406e923…`), cells unchanged, plus one cell that publishes R7's own predictions (`ours_sub.csv`, full precision,
  the file the blend cell reads) in place of the blend. `--flip` changes one line of the public R7 cell so the
  worker gets `--flip-tta`. The cell raises, and there is no score, if R7 left more than 2% of the studies unscored
  or the flag did not reach the worker (tested offline: seven cases). Slugs `rsna-knee-sub-r7-alone`,
  `rsna-knee-sub-r7-alone-flip`.
- **Read A, R7 alone. No pass or fail: a level, ± 0.008.** It decides three things. (1) "Is B the best": against
  the public CoAtNet's 0.949 and our five folds' 0.945. (2) The OAI-free final: at **≥ 0.950** R7 alone replaces the
  five consensus folds (0.945) as that candidate; under that the two are one reading and ours stays, being ours.
  (3) The baseline of the 5090 run: R7's recipe on a pseudo-label table has to read **≥ +0.006** over this to be
  shown.
- **Read B, R7 + mirror pass against read A.** The same four models on the same studies; only the second pass
  differs, so the pair is tighter than any we have measured (a label-only pair is 0.003; one displayed tick is
  0.001). **≥ +0.002 → the mirror pass goes on wherever R7 is used (the blend, the 5090 run's inference); −0.001 to
  +0.001 → no effect shown, it stays off (it doubles R7's GPU time: 8.6 s a study on a T4 in the commit run);
  ≤ −0.002 → off.** Expected 0 to +0.001: the CoAtNet's author measured +0.001 on his model, and R7's author built
  the flag and shipped it off.

**The three submissions armed for 17:02, looked at again against the plan as it now stands** (R7's recipe on a
5090; v3 the fallback). All three kernels are gated and cost no GPU; the day has five slots and the two reads above
take the other two, so none of the three displaces anything.

| read | what would change with its result | verdict |
|---|---|---|
| 5: the free blend + our five teacher folds | It is the only one that can become our best score (0.945 → ~0.956, rank ~1,800 → inside the 152–734 block) and a final candidate; and whether a 0.945 member adds anything at 10–25% | **worth it: the one that matters** |
| 1: the five teacher folds alone | Whether the OAI-trained CoAtNet's opinions in the labels help at five folds (one model: +0.006). That is the evidence for putting them in R7's label table on the 5090. Readable only at ≥ +0.006 | **worth it, moderately**: a null says little |
| 4: the pseudo-label student alone | Nothing in about five outcomes of six: one fold, half dose, our trainer; the 5090 run uses a stronger teacher either way. Only ≥ 0.950 or ≤ 0.940 would be news | **weak. Kept only because the slot is otherwise empty and the kernel is free**; first to go if a slot is wanted |

**OAI's images into R7 — re-opened; it was closed on the owner's word of 10-12 ("not pursued") and should have been
laid out again when the plan moved to R7.** What it takes, in order:
1. **The owner's account and signature** (NDA / OAI data-use terms; not something to be done for them). One forum
   participant reports individual approval at once; unverified. The terms' view of a rented machine is not known.
2. **The rules risk stands:** thread 748276 argues OAI is not equally accessible; no host reply.
3. The download: ~200–400 GB for the ~2,400 knees with semi-quantitative reads (my arithmetic of 10-12, unverified).
4. Labels: OAI's MOAKS reads → PF OA, Lateral OA, Synovitis (and possibly Medial OA, both menisci, Effusion). The one
   public user's mapping is unpublished; ours would be written from the MOAKS definitions.
5. Code: OAI knees through R7's cache builder (OAI has sagittal and coronal series, and no 2-D axial); R7's trainer
   already takes a weight per cell, so the nine findings OAI does not grade carry no loss.
6. Cost: ~1.55× the studies, so ~39–50 h of training in place of 25–32; ~55–65 h of machine in all, ~$30–65.
**Measured gain, by the one person who reports it: +0.003 to +0.005 on a single model** (the CoAtNet, 0.942 → 0.945
and on). On R7: unknown. **Order proposed:** read A first (is R7 the one to build on); the 5090 run on the
pseudo-label table needs no OAI access and starts when the GPU exists; OAI's knees join as soon as the account and
the files exist, as a second run or the first if they arrive in time.

**✅ Both R7 read kernels COMPLETE and gated, 10-13 15:25** (pushed 15:17, built from `a0fcd74`; ~7 min each, so
~0.25 h of Kaggle GPU for the two).
- `raahimnawaz/rsna-knee-sub-r7-alone` v1: `OURS_STAGE_COMPLETE {"status": "PASS", … "scored": 3,
  "fallback_studies": 0}` · `NOTEBOOK_COMPLETE` · `R7_READ flip_tta=False rows=3 unscored=0` · 0 tracebacks. Its 13
  public cells are byte-identical to the notebook's; `submission.csv` equals R7's own `ours_sub.csv` and differs
  from the blend the notebook wrote (kept as `public_blend_sub.csv`).
- `raahimnawaz/rsna-knee-sub-r7-alone-flip` v1: the same lines with `R7_READ flip_tta=True`; one public line differs
  (`if manifest.get('flip_tta'):` → `if True:`); both workers print `FLIP_TTA_READY {"perm": [0, 1, 3, 2, 5, 4, 6, …],
  "mcl_plain_pass_only": true}`; GPU time a study 9.3 s → 18.2 s; against the plain kernel on the three test studies
  MCL is identical and Medial / Lateral OA move by up to 0.10: the second pass ran and is swapped back.
- **What the commit run cannot show:** whether four folds × two passes finish inside the notebook's 8.3 h on ~1,300
  studies (about 3.3 h of GPU on two T4s by the per-study figure, plus decoding). A late run leaves studies
  unscored, and the last cell then raises: no score, a slot spent.
- **⏳ A second submitter is armed (15:26, pid 99722) for 17:06:00** with `submit_plan_2026-10-13-r7.json`: the two
  R7 reads first, then tonight's three, so it sends whatever the 17:02 submitter (pid 96147, still running) did
  not. Five slots, five submissions. Check after 17:10: `tail data/submitter.log` and `$K competitions submissions`.

## Owner 10-13 afternoon: "could we email people who do this irl … draft the emails in my gmail ill send"

Five drafts are in the owner's Gmail (nothing sent; the owner sends). Each states the setting in our own numbers only
(about 4,350 studies with report-derived labels, 58 radiologist-graded, about one answer in ten wrong against them,
Synovitis mentioned in under a quarter of reports; no report text, no images) and asks one question.
- Pranav Rajpurkar (CheXpert, MRNet): what to spend a week on; whether 58 graded studies can do more than evaluate.
  Address from Harvard's faculty page, filled in.
- Valentina Pedoia (OAI knee OA): which MOAKS sub-scores and cut-offs for PF OA, lateral OA, Synovitis; pitfalls.
- Akshay Chaudhari (knee MRI, SKM-TEA): label cleaning, image-graded outside data, or dense side supervision.
- Richard Kijowski (cartilage lesion detection): why the image model trails the report on PF and lateral OA.
- Curtis Northcutt (confident learning): whether it applies to correlated readers, 12 findings and 58 clean studies.
The last four have no recipient: no address could be confirmed from an official page (Kijowski's 2024 paper gives an
NYU address, and an undated profile has him at HSS since). Not written to: other competitors (Kaggle's rule on
private sharing; the forum is the place) and the host, where a public post asking for a ruling on OAI is still the
most useful message and is not drafted. A reply that changes the plan gets recorded here.

**Outreach, later the same afternoon (owner: "put thier emials in"; "also how about former simmilar competition
winners").** Four more drafts now carry a recipient, each from a place the person publishes it (no address is kept
in this repo): Pedoia and Chaudhari from the correspondence lines of their 2024–2026 papers (Europe PMC), Kijowski
from the correspondence line of a 2026 review (he now lists RadNet and HSS, not NYU), Northcutt from his own site.
Gmail's connector cannot edit a draft, so these are new drafts; the four earlier ones without a recipient are still
there for the owner to discard. Nothing sent.
- **Past winners: most of the well-known ones are IN this competition**, so a private message to them is the thing
  Kaggle's private-sharing rule is about. Today's leaderboard by member handle: Theo Viel (rank 11, 0.963), Nischay
  Dhankhar (34, 0.961; RSNA 2023's winning team), Ian Pan and Dieter (64, 0.959), Chris Deotte (154, 0.957), hengck23
  (295, 0.956), yu4u and tattaka, Guanshuo Xu, and the 2023 mammography winner. Not on it today: Qishen Ha (RSNA 2022
  cervical spine, 1st), Darragh Hanley (2022, 3rd), Yuji Ariyasu (2023 3rd; 2024 2nd's team), Harshit Sheoran (2023's
  winning team). One draft for those, with no recipient (they publish no address; LinkedIn or Kaggle is the channel)
  and a line asking anyone who is entered to ignore it. Their challenges had radiologist labels, so what they know
  best is the pipeline (find the joint, crop, classify), not label noise.

## 10-13 19:20: tonight's five are sent, two have scored; the public state re-checked (owner: "can we put the .966 in the public")

- **Sent:** all five by the 17:06 submitter (`data/submitter.log`: two connection resets and two failed listings
  retried, done 18:45). No slot is left until 10-14 17:00 PDT.
- **Five teacher folds alone: 0.949** (`57161107`) against the five consensus folds' 0.945: +0.004. Kept, not shown
  (bar +0.010); by read 1's rule the finals' candidate of our own family. Level with the public CoAtNet alone, so
  the 0.25 / 0.10 weights of read 5 were set for a weaker member than this one read as.
- **Pseudo-label student: 0.946** (`57162950`) against 0.944: +0.002, one SD of a label-only pair is 0.003. Kept;
  under the +0.006 that would train five folds on `targets_teacher_v2`, so they are not trained.
- **Pending at 19:20:** R7 alone (`57160504`), the free recipe + our folds (`57160505`), R7 + mirror pass
  (`57160736`).
- **Nothing above the free recipe is public** (pulled 19:11-19:15, notebook source only, nothing run). The board's
  top is three teams at 0.966 (CentralyDS, Sergey Bryansky, RB & Matheus & Tucker & Tony), three at 0.965, five at
  0.964; none has published weights or a notebook. Today's new public notebooks are the same recipe:
  `evgendvorkin/rsna-versia-7`, `degnonguidi/rsna-knee-abnormality-detection`, `medvax/rsna-knee-r7-source-trial`,
  `le4ves/rsna-knee-0-957-r7-blend` and `jingshuifu/rsna-knee-flip-tta` all attach R7's bundle, `nartaa`'s weights
  and `heliosli`'s encoder with `W_RAPTOR, W_OURS = 0.5, 0.5`; `lavinwins/rsna-gold-medal` is the 0.950 CoAtNet +
  `goodpjw2008` reader. `jingshuifu`'s forces `flip_tta` on (16:34, no score in its source): the same read as ours.
  `karttikjangid05`'s map and `varshithakolupuri`'s "what the top notebooks are made of" say the same in their own
  words: a handful of model datasets under every public notebook at 0.950 and above.

## 10-13 22:10: two more read; the free recipe + our five folds is our best, 0.956, rank 198 of 5,955

- **R7 alone: 0.949** (`57160504`). By read A's rule (under 0.950) the OAI-free final stays our five consensus folds
  (0.945); the two are one level within ± 0.008. Level with the public CoAtNet alone and with our five teacher folds
  (both 0.949). It is the baseline for R7's recipe on a pseudo-label table: shown at ≥ 0.955.
- **The free recipe + our five teacher folds: 0.956** (`57160505`). By read 5's rule: read the rank inside the block,
  nothing claimed. Board pulled 22:06: 5,955 teams; the 0.956 block is 724 teams, ranks 188-911; **ours 198**. The
  recipe's author (`varshithakolupuri`) 333, R7's author (`heliosli`) 199, `karttikjangid05` / `hitarthjain0` 319,
  `jessky` 327, `nartaa` 954 (0.955). So our folds do not cost the recipe, and it sits above its author's rank:
  under one displayed tick, not a claim (an author's best on the board is not known to be this exact recipe). **Our
  best submission and a final candidate**; it holds OAI-derived models (the CoAtNet; our teacher labels).
- **Medal lines by Kaggle's rule on 5,955:** gold to ~21 (0.962 at 21-27), silver to ~297, bronze to ~595. Rank 198
  is inside the public board's silver; the block spans silver and bronze, so the private part's order decides.
- **R7 + mirror pass (`57160736`): still pending at 22:08**, ~5 h after it was sent. It runs two passes of four folds
  and raises (no score) if it runs late.
- **Checked and closed: "the 384-trained CoAtNet is inferred at 320 px"** (`karttikjangid05`'s ensemble map). True
  and intended: the base notebook (`6406e923`) sets `INPUT_MODE = "crop"`, `INPUT_RES = 320` on the 384 px corpus,
  "must match training exactly": the model trained on 320 crops. Not a free lever.
- **22:27: R7 + mirror pass 0.949** (`57160736`) against R7 alone 0.949: by read B's rule (-0.001 to +0.001) no effect
  shown, **the pass stays off** (it doubles R7's GPU time). All five of tonight's are read.

## Owner 10-13 ~22:30: "casmi i will just not do dont consider it"; "stock that blend for later"; how to make an OAI account; "what is the strongest possible plan forward"

- **casmi26 is dropped:** the last Kaggle week (30 h from 10-16 17:00) is all this competition's. v3's five folds
  (~24.5 h) now fit beside nothing else.
- **The 0.956 blend (`57160505`) is kept as a final candidate.** Every submission stays selectable until 10-22 16:59;
  with none selected Kaggle takes the best public scores.
- **OAI access, read 22:40 on nda.nih.gov (the pages behind login were not seen):** sign in through NIH's Research
  Auth Service (eRA Commons, Login.gov or PIV/CAC) or create a Login.gov account; then the OAI permission group and
  its data-use terms (a third-party FAQ: My Account → Data Permissions → "Request Access to a Permission Group" →
  Osteoarthritis Initiative). Images are a separate, login-only page (`/oai/accessing_images`). The site carries a
  banner: "under review for potential modification in compliance with Administration directives". Whether the
  request asks for an institution's signing official is still not seen by us; the host's ruling (741819) makes that
  the rules question.

## 10-14 ~00:30: OAI access, read on nda.nih.gov with the owner signed in (their Login.gov + a new NDA account)

- **The rules question, answered from the source:** NDA's Data Permissions table lists the Osteoarthritis Initiative
  permission group with **"Institutional approval required: No"** and "Data Access Requests do not require
  Institutional sponsorship and will be approved immediately upon submission of a completed request." The host's
  condition (741819: free, no institutional review / sign-off) is met for us. Thread 748276's point (NDA's limits on
  some countries since 2025) is about equal access for others and still has no host reply.
- **The Data Use Certification, the clauses that bind this project** (one year; the owner certifies, not us):
  §3 no distribution or moving of the data to any third party or third-party system except DUC-listed collaborators;
  §8 security plan, explicitly "local machines or with a cloud service provider": accounts not shared, data never
  exposed to the internet, encryption / strong passwords, patched software; §9 delete from every machine when done
  or at expiry; §5 no re-identification and no publishing derived data that could re-identify (structural MRI);
  §11 an NDA Study for each publication, computational pipeline or other public disclosure of results; §12 the OAI
  acknowledgement and DOI in any presentation or publication (as `nartaa`'s and `jingshuifu`'s notebooks carry);
  §15 NDA may publish the owner's name and affiliation with a summary of the use.
- **What that means here:** the images stay off Kaggle; a rented GPU is a cloud machine under §8 (a datacenter tier,
  encrypted disk, no public ports, wiped at the end is my reading, not legal advice); trained weights are not
  subject-level data (the public OAI-trained weights exist on the same basis); the 10-22 public repo and any write-up
  need the acknowledgement and an NDA Study.
- State at the time of writing: the request form is open at its last step (one certification checkbox + Confirm),
  left for the owner to sign.

## 10-15 ~09:30: OAI access granted; MOAKS -> our twelve findings (`labeling/oai_moaks_targets.py`)

- **Access:** DAR **26933**, Osteoarthritis Initiative, approved 2026-10-15, expires 2026-10-15 + 1 year; the owner
  ticked and confirmed the DUC in their own Chrome (the certification is theirs; my attempt to tick it was blocked
  and was right to be). Downloaded through their Chrome: `MR Image Assessment_ASCII.zip`, 27 MB, from NDA's OAI
  Full Downloads page, into `~/oai_data/` (mode 700; `oai_data/` added to the home folder's local
  `.git/info/exclude`, since `~` is a remote-less git work tree). **Nothing OAI goes into this repo**; the script
  refuses a work tree with a remote and any non-ignored path.
- **The table:** `kMRI_SQ_MOAKS_BICL00.txt`, baseline BICL MOAKS reads: 3,017 reads of **2,400 knees** (2,030
  people; `nartaa` reports 2,399). Projects score different blocks: cartilage, bone marrow lesions and menisci on all;
  ACL, effusion-synovitis and Hoffa-synovitis on projects 22 + 65 (~1,660 knees); popliteal cyst on 22 alone (600).
- **The mapping, constants fixed from the two definitions before any model sees it** (REFERENCE.md §2 at
  `9c23daf`; OAI's MOAKS Descrip): OA per compartment = max over its subregions (full-thickness extent 1/2/3 ->
  0.60/0.92/0.97; any-depth size without full thickness 0/1/2/3 -> 0.02/0.08/0.30/0.45, since MOAKS does not grade
  partial depth); meniscus tear or maceration or root tear 0.93, "signal, not a tear" 0.08; ACL complete 0.97,
  partial 0.45, a repaired knee masked; Effusion 0/1/2/3 -> 0.02/0.10/0.90/0.98; Synovitis = max(Hoffa
  0.03/0.20/0.60/0.85, effusion-synovitis 0.02/0.10/0.45/0.70), the least certain row; Baker's yes 0.55 / no 0.03
  (presence, no size); Contusion 0.03 only where no lesion is seen anywhere, else masked; MCL and Fracture masked.
- **Output** `~/oai_data/derived/oai_moaks_targets_v1.csv` (local only), cells and share >= 0.5: ACL 1,654 / 3.4%;
  Medial Meniscus 2,399 / 41%; Lateral Meniscus 2,399 / 21%; Medial OA 2,399 / 21%; Lateral OA 2,399 / 19%; PF OA
  2,399 / 43%; Effusion 1,660 / 20%; Synovitis 1,659 / 14%; Baker's 600 / 42%; Contusion 444 / 0%.
- **Noise, measured on the 544 knees two projects read:** agreement at 0.5 Medial Meniscus 0.890, Lateral Meniscus
  0.939, Medial OA 0.912, Lateral OA 0.904, PF OA 0.910 (Spearman 0.77-0.88); Effusion 0.979 and Synovitis 0.957 on
  47 pairs. The same lab read both, so this bounds noise from below. Our report labels agree with gold ~0.85.
- **Open, for the training run:** which findings to supervise from OAI. The one measured use (`nartaa`, +0.003 to
  +0.005) supervised PF OA, Lateral OA and Synovitis only; the menisci map as cleanly. The images: the baseline
  visit's series for these 2,400 knees, downloaded to the rented GPU only.


## 10-15 ~10:05: the board moved, a free 0.957 is public; the RUNBOOK rewritten and the repo trimmed

- **Leaderboard, pulled 10:04 PDT (6,037 teams):** #1 0.967 (two teams), 0.966 ×4; gold to ~rank 22 (0.963),
  silver to ~301 (**0.957**), bronze to ~603 (0.956). A 0.957 block of 246 teams (ranks 136–381); the 0.956 block
  927 (382–1,308). **Ours 0.956 (`57160505`) is rank 408** (198 on 10-13). No submission of ours since 10-14 01:43
  UTC; today's five slots (to 17:00 PDT) unused.
- **What makes the free 0.957 (source read 10:05, not run by us):** `heliosli/this-is-new-version-0-9576-in-one-click`
  (10-14, title "0.9576"): stage 1 `nartaa`'s Raptor CoAtNet 384 px, stage 1b its 224 px checkpoint (0.35 of the
  Raptor half), stage 2 heliosli's **R6** bundle (`heliosli/rsna-knee-r6-frozen-fe055406`, four folds, its own
  Effusion / Baker's stackers) **with flip TTA forced on** (the R6 manifest predates the key), stage 3 per-finding
  ranks 50/50 (OA findings 0.5 too). `bliverrigpdoge/r7-raptor-knee-centred-convnext` (top by score): the old 0.956
  (R7 + Raptor) at 0.8 + the author's knee-centred ConvNeXt-small / EfficientNetV2-S crops (private weights) at 0.2.
  R7's own flip pass read 0 for us (`57160736`); this one is on R6, a different model, and is not separated from the
  224 px checkpoint in anything public.
- **So the base for a blend of ours is now the 0.9576 notebook.** `build_public_blend_kernel.py` pins the old slug
  (`PUBLIC`) and needs that changed first, plus a check that our stage still fits the 9 h beside R6's doubled pass.
- **Repo trimmed (owner: "make sure the repo is lean and easily understandable"):** RUNBOOK rewritten as a one-page
  current state (the 10-13 version: `git show c5ab90c:RUNBOOK.md`). Removed, nothing referencing them: the
  same-report CNN teachers (`fusion/local_teacher_folds.py`, `labeling/build_student_targets.py`; on `showcase`),
  `notebooks/core_alone_fixture.py` / `core_swap_fixture.py` and `fusion/runs_coatnet_raptor_check`, the ViT-L
  student and `dread` ortho run records, the spent `submit_plan_2026-10-13-r7.json`. Local-only, git-ignored legacy
  run folders (weights and logs, ~1.5 GB) moved out to `~/rsna-knee-legacy-runs/`. Kept although legacy, because
  the kernel builders embed them: `orthofoundation_model.py`, `qwen_vlm.py`, `window_curve.py`, `slots64/118`,
  `runs_c6_ortho`, `runs_port_repro`.

## Owner 10-15 ~10:15: "why would we train the knees when aren't there models that already have them trained, see how that guy got .957, consider an even smaller cnn in our blend, sure we can submit the .957 and ours, don't rent the gpus"

- **No rented GPU: track A (an R-family model retrained with OAI's knees) is closed.** OAI's images cannot go on
  Kaggle (DUC §3), so the MOAKS table (`~/oai_data/derived/oai_moaks_targets_v1.csv`) stays local and unused. The
  owner's point stands on the evidence: OAI's knees are already in the free recipe, through `nartaa`'s two CoAtNets
  (2,399 OAI knees in training), half of every public blend.
- **⛔ CORRECTED 11:00: the manifest read here was R7's, not R6's** (the dataset's version 2 holds R7 under R6's file
  names; see the next block). As first written: **R6, read from its manifest (`heliosli/rsna-knee-r6-frozen-fe055406`, tag `cnx_r7`, arm `E_lrprior`):** the R7
  family again: ConvNeXt-small 384 / crop 352, 176 windows, 16 epochs, 4 folds, first 16 blocks frozen, spatial
  attention, mirror 0.5, a fastMRI+ side task, report-derived targets; "No KneeCoT inputs or derivatives; gold never
  trained"; no OAI. So R6 is OAI-free, and with its flip pass it costs about what R7 with the flip pass did (the
  `57160736` kernel ran the whole public notebook with it and scored).
- **The go: one submission, the free 0.957 recipe + our five teacher folds.** Written 10:20, before the build:
  `notebooks/build_public_blend_kernel.py` now takes either checked public notebook (`PUBLIC` is a table of the two;
  the seams it checks all hold for the new one: one `NB_T0` cell, the competition mount, the last cell writes
  `submission.csv` and prints `NOTEBOOK_COMPLETE`, none of our names used). Base:
  `heliosli/this-is-new-version-0-9576-in-one-click` as pulled 10-15 10:05, run unchanged. Ours: the same five folds
  and the same weights as `57160505` (fixed 10-13: 0.25 on ACL, MCL, Medial Meniscus, Medial OA, Effusion, Baker's,
  Contusion, Fracture; 0.10 on Lateral Meniscus, Lateral OA, PF OA, Synovitis), nothing refitted.
  **Rule: ≥ 0.958 → our folds stay in; 0.957 → read our rank inside the 0.957 block against the recipe's author's,
  nothing claimed; ≤ 0.956 → our folds cost it and come out.** Caveat: if our stage runs out of the 9 h in the
  scored rerun the public file stands, and that reads as a 0.957 we cannot tell apart from "no effect".

## 10-15 ~10:50–11:10: the 0.957 kernel failed in the public R6 cell; heliosli overwrote R6 with R7; what is public at 0.957

- **`raahimnawaz/rsna-knee-sub-pub957-teach1cv5` v1: ERROR**, in the public notebook's own R6 cell, unchanged:
  `AssertionError: ('8c7b666e27dc0e8f', 'fe055406fad6a12f')` (manifest bundle against the notebook's `BUNDLE_ID`),
  after `RAPTOR_STAGE_DONE seconds 54` and `RAPTOR224_STAGE_DONE seconds 25`. Our stage never ran. Nothing submitted.
- **Why, from the dataset's own description:** `heliosli/rsna-knee-r6-frozen-fe055406` version 2 (10-14 18:12) is
  "R7 ConvNeXt-small inference bundle 8c7b666e27dc0e8f … under the same file names as version 1. Version 1 held R6
  fe055406fad6a12f (manifest.json sha256 6508ce69…)". The 0.9576 notebook ran at 12:21 against version 1; re-run now
  it fails for everyone. The manifest read at ~10:20 (tag `cnx_r7`) was version 2's, so it described R7.
- **R6 itself (version 1's manifest, read 11:00):** tag `cnx_r6fm_c352`, ConvNeXt-small 384 / crop 352, 176 windows,
  16 epochs, 4 folds, first 16 blocks frozen, targets `probe_targets_opusv11b_rr6_plr6_cdr.csv` ("report-derived
  targets"), built 10-11; no OAI named (whether its labels hold an OAI-trained teacher's opinions is not stated).
- **A kernel cannot pin a dataset version:** a CPU kernel (`raahimnawaz/rsna-knee-pintest-r6v1`, no GPU) with
  `dataset_sources: ["heliosli/rsna-knee-r6-frozen-fe055406/1"]` was accepted at push and mounted version 2
  (`bundle_id 8c7b666e…, tag cnx_r7`). So R6 needs our own private copy of version 1 under the same slug (the notebook
  finds a mount by slug); `build_public_blend_kernel.py --replace-dataset` swaps the source. `picond`'s "9576 R6 V1
  Fixed" does the same with a private `…-fe055406-v1`.
- **What is public at 0.957 (the code page sorted by score, 10:58; nothing public above 0.957):** heliosli's
  "old version 0.957" and `yuvrajlamba01`'s "Research Optimized" = Raptor 384 + 224 (0.35) + the R7 bundle
  `8c7b666e` (version 2), **no flip pass**, 50/50: runs as published today. heliosli's "0.9576" = the same with R6 +
  flip (claimed 0.9576, displayed 0.957). `bliverrigpdoge` = the old 0.956 at 0.8 + private crop models at 0.2.
  `leoprovorov` v2 = Raptor 384, Raptor 224 (plain + mirror), R7 and S6 (all data), private dataset. Below, at 0.956:
  `goodpjw2008`'s 2.5D ConvNeXt MIL reader (public weights `goodpjw2008/rsna-knee-2-5d-convnext-reader`, 0.929
  alone) at 1/10 on the old base. **Its author's own reads: at full weight or 1/4 it no longer adds to the 0.956
  base, "keep it at 0.01 to 0.05 or leave it out".** So a public small CNN exists and needs no building, and its
  author has measured that it does not help this base.
- **Our copy of R6 (version 1):** downloaded file by file (a whole-version download came back empty), all 8 files
  against the manifest's sha256 table, 0 mismatches, manifest sha256 `6508ce69…` as the dataset's description gives;
  uploaded private as `raahimnawaz/rsna-knee-r6-frozen-fe055406` (credit and source in its description). The CPU
  kernel on it: `MOUNT /kaggle/input/rsna-knee-r6-frozen-fe055406 bundle_id fe055406fad6a12f tag cnx_r6fm_c352`.
- **✅ `raahimnawaz/rsna-knee-sub-pub957-teach1cv5` v2 COMPLETE 11:54** (pushed 11:43, built from `cb9562a`, the
  public notebook sha256 `16a9d953…` with our R6 copy in place of heliosli's source). `RAPTOR_STAGE_DONE seconds 49`
  · `RAPTOR224_STAGE_DONE seconds 22` · `ASSETS_READY … /kaggle/input/rsna-knee-r6-frozen-fe055406 … flip_tta True`
  · R6 `OURS_STAGE_COMPLETE {"status": "PASS", … "scored": 3, "fallback_studies": 0, "seconds": 360.2, per study
  decode 38.2 s, GPU 18.7 s, 529 windows}` · `NOTEBOOK_COMPLETE` · ours: `5 checkpoint(s), sha256 ok` · `GATE PASSED
  (1/5)` … `(5/5)` · `OURS DONE: 3 test studies in 2.6 min` · `OURS_STAGE ok=True seconds 156 left_at_start_min 507`
  · `[public+ours] ARMS: …` · 0 tracebacks. `submission.csv`: 3 rows, 12 findings, in [0, 1], the public file's order.
- **Submitted 11:55, ref `57241351`.** Read by the rule written at 10:20 above. Pending.

## 10-15 afternoon: owner asks for a mixture-of-experts brainstorm; five independent read-only reviews

- **Brief:** one shared file (every number, the pipeline, constraints, closed routes; R6 + R7 listed as untested, not
  closed, after the owner asked whether it had been cut on evidence: it had not, it was an estimate). Five reviewers,
  read-only (no pushes, submits or edits): MSK radiologist, weak supervision / labels, medical-imaging modeling, Kaggle
  ensembling, inference systems.
- **Where they agree:** blending the public members is near its ceiling (~0.960; within-class ρ ≈ 0.83–0.89); our
  member at 10–25% caps any gain from improving it at ~+0.002 unless it reaches ~0.953 alone; R6 + R7 is worth one
  slot (four of five; systems: only through the worker's two-model path in one decode); never tune weights on the
  public board; realistic total +0.001 to +0.003 on the main final.
- **New, measured by the reviewers (report only, nothing selected):** teacher labels raised our folds' within-class
  correlation with Raptor 0.843 → 0.888 on gold-58 (Synovitis 0.846 → 0.955); our model ties Synovitis to Effusion
  at Spearman 0.91 against 0.70 in its own targets; cross-finding consistency rules (Synovitis + Effusion, Baker's +
  Effusion, OA compartments, Lateral Meniscus + Lateral OA, ACL + Contusion, …) are flat or negative on gold-58 and
  on the 4,349 held-out; R6's version-1 manifest has `mirror: None` (its flip pass shows it unmirrored-trained knees
  mirrored); R6/R7's fold split is not in the bundles; the claim "the hidden test has compressed DICOMs" is `goodpjw2008`'s, and
  their notebook installs decoders for it.
- **Checked by me on CPU (`raahimnawaz/rsna-knee-decodetest`, the public notebook's image, no decoders installed):**
  400 sampled training series and all 15 visible test series are Explicit VR Little Endian and decode 400 / 400 and
  15 / 15. The hidden test cannot be checked; our stage tolerates up to 10% unreadable studies.
- **Not yet verified:** `nartaa/rsna-knee-s6fall-weights` (an all-data S6 the modeling reviewer found; OAI status
  unknown); what `plr6` / `plr7` in R6 / R7's label files are (possibly pseudo-labels from an unnamed teacher).
- **The ranked plan** is RUNBOOK §3 ("Expert panel"). Nothing from it has been built or run.

## Owner 10-15 ~13:00: "which ones most aggressively target the 4 lowest findings … test the not considered until now and cut what should be cut … then A, B then D"

- **The "not considered until now" list, tested (CPU, report only, nothing selected):**
  - *Knee side* (the Stage B cache: DICOM tag, else the −62 mm geometry rule; per-study side from the eight shards'
    `slots.csv`): of 1,503 studies whose report names one side, **46 disagree (3.1%)**: 35 of 712 where the side came
    from geometry (4.9%), 11 of 791 from the tag (1.4%); the examples are plain ("MRI of right Knee …", cache L). **It
    costs nothing:** swapping Medial↔Lateral predictions on those studies lowers every held-out AUROC (teacher folds:
    Medial / Lateral Meniscus −0.0014 / −0.0012, Medial / Lateral OA −0.0003 / −0.0003; consensus folds alike), and on
    them the model ranks medial against lateral the report's way in 62–88% of discordant studies: it reads the
    compartments from anatomy, not from the image side. **Cut.**
  - *"Tricompartmental" reports:* 272 of 4,407. Lateral OA targets ≥ 0.5 in 50% of them (10.8% elsewhere), PF OA
    74%; the model is below 0.3 on a positive target in 1.5% of them against 3.0% elsewhere; gold has 4 such studies,
    Lateral OA positive in 2. No excess label noise there. **Cut.**
  - *v3 fold 0's check is contaminated* (its targets hold teacher folds 1–4's predictions, which trained on fold 0's
    847): true by construction. It stays a not-broken check only, as written.
  - *Run-to-run noise:* needs GPU hours this week does not have. **Cut.**
  - *`nartaa/rsna-knee-s6fall-weights`:* real. One all-data S6 ConvNeXt-small checkpoint, report-derived JEV-blend
    labels, "No OAI data was used for this model", **0.950 public alone** by its author (`nartaa/rsna-knee-s6fall-solo`;
    that notebook falls back to the OAI-trained CoAtNet only for a study the ConvNeXt cannot read). CC BY-NC 4.0.
  - *R6 / R7 label sources:* heliosli's tree (`jessky/rsna-knee-local-cv-cnx50`, `label_source_policy.py`):
    `own_reports_and_official_gold_only_v1`, "community targets are excluded"; the manifests say report-derived and
    gold never trained. **OAI-free as every stated source says; not proven.**
- **B, written 13:58 before any build or run.** heliosli's R6 + R7 in one decode through their worker's two-group
  path (`--models2`), in place of R6's flip pass, everything else as `57241351`: `notebooks/build_r6r7_bundle.py`
  makes bundle `0aa1bca3fbde76e6` (R6 bundle fe055406 + R7 bundle 8c7b666e, each checked against its manifest; same
  trainer md5, views, size/crop, anchors, state layout); z = (1 − w2) z_R6 + w2 z_R7 with w2 = 0.5, and 0 on Effusion
  and Baker's (R6's stackers; its Effusion stacker now declares R7 as a second line at weight 0, which leaves R6's
  Effusion as it was; its Baker's stacker reads group 1 only). Three public lines change: `FLIP_TTA` False, the
  `BUNDLE_ID`, the `C_SLUG`. Submitted with `--read`. **Rule: against `57241351` (same members but R6 with its flip
  pass): the same displayed score or higher → R6 + R7 is the ConvNeXt half from now on; one tick or more below → R6
  with its flip pass stays.** The pair differs only in that half, so it is tight (blend-weight pairs ±0.0007); one
  tick is the smallest readable step. Expected +0.0005 to +0.0015 (the panel), i.e. most likely the same tick.
- **D, written 14:01 before any build or run: the OAI-free final as a blend.** The 0.9576 notebook with its four
  Raptor cells dropped (the clock `NB_T0` kept in a cell of its own), Raptor's weights not mounted, and its blend cell
  replaced by the R-family's own predictions (`ours_sub.csv`); the R-family = B's bundle (R6 + R7, one decode, no flip);
  then our **five consensus folds** (`targets_consensus_v2`, no OAI-trained teacher; LB 0.945) by per-finding ranks at
  **0.35 on the eight findings and 0.20 on Lateral Meniscus, Lateral OA, PF OA, Synovitis**, fixed now. Basis: ours
  0.945 against R7 0.949 alone, within-class correlation with a public family ~0.84 (ours vs Raptor, gold-58), so the
  binormal optimum is ~0.35–0.40; lower on the four where the R-family's compartment views see what ours does not.
  Submitted with `--read`. **Rule: ≥ 0.950 → D is the OAI-free final in place of the five consensus folds (0.945);
  0.948–0.949 → kept only over R7 alone's 0.949 if strictly above it, else R7 alone is; < 0.948 → something is broken,
  look before using it.** Every member's stated sources name no OAI; R6 / R7's are statements, not proof.
- **B v1 (`raahimnawaz/rsna-knee-sub-r6r7-teach1cv5`, from `06dbe25`): ERROR, nothing submitted.** The bundle mounted,
  `FLUID_READY … "weights": [1.0, 0.0]` held, then both workers exited: `--state-stacker: one model group only (the
  json is fitted on one line's z and state read-out)`; every study fell back, and the public blend cell raised on it.
  ~5 GPU-min. My 13:58 reading ("its Baker's stacker reads group 1 only") was wrong: with two groups `score()` returns
  both groups' fold logits and `apply_state_stacker` averaged all eight. **Fix (`notebooks/build_r6r7_kernel.py`, from
  `46cc73b`):** two lines of heliosli's worker, the refusal removed and the stacker's z = `zs[:len(models)].mean(0)`
  (R6's four fold logits, as fitted), and the worker's `SOURCE_SHA` entry replaced by the patched file's sha256
  (`8401812c…` → `daed0256…`), every embedded source checked against `SOURCE_SHA` and compiled offline. With w2 = 0 on
  Baker's, Baker's is R6's own output again. B v2 pushed 14:13; D built from the same script.
- **✅ B v2 COMPLETE 14:26** (from `46cc73b`): `ASSETS_READY … rsna-knee-r6r7-dual … flip_tta False` · both workers
  `STATE_STACKER_READY {… "Baker's" …}` and `MODEL_READY {"models": 4, "models2": 4, "spatial_models": 8, …,
  "compile_requested": true, "trainer_md5": "8622009e…"}` · `WORKER_DONE … fallback 0` · R6+R7 `OURS_STAGE_COMPLETE
  {"status": "PASS", "scored": 3, "fallback_studies": 0, per study decode 38.6 s, GPU 20.4 s}` (R6 + flip: 18.7 s) ·
  `NOTEBOOK_COMPLETE` · ours `5 checkpoint(s), sha256 ok` · `GATE PASSED (5/5)` · `OURS_STAGE ok=True … left_at_start_min
  506` · `[public+ours] ARMS: …` · 0 tracebacks. **Submitted 14:26, ref `57244451`**, read by the 13:58 rule against
  `57241351`. Pending.
- **✅ D COMPLETE 14:38** (`raahimnawaz/rsna-knee-sub-oaifree-r6r7-cons5` v1, from `46cc73b`): no Raptor line and no
  Raptor file anywhere in the run · `ASSETS_READY … rsna-knee-r6r7-dual … flip_tta False` · R6+R7 `OURS_STAGE_COMPLETE
  {"status": "PASS", "scored": 3, "fallback_studies": 0}` · `NOTEBOOK_COMPLETE {"submission_from": "ours_sub.csv"}` ·
  ours (the five consensus folds) `5 checkpoint(s), sha256 ok` · `GATE PASSED (5/5)` · `OURS_STAGE ok=True …
  left_at_start_min 508` · `ARMS: … w_ours … 0.35 / … 0.20` · 0 tracebacks. **Submitted 14:38, ref `57244659`**, read
  by the 14:01 rule. Pending. (The ARMS line's "the public blend as published" is the template's wording; here the
  base is the R-family alone.)
- **A, what exists now:** private copies `raahimnawaz/rsna-knee-r6-frozen-fe055406` (R6 v1, mounted and checked on
  CPU), `raahimnawaz/rsna-knee-r6r7-dual` (checked on CPU, used by B and D), `raahimnawaz/rsna-knee-publication-swa-
  weights-20261007` (Raptor), `raahimnawaz/rsna-knee-s6fall-weights`, `raahimnawaz/rsna-orthofoundation-fold0-epoch7-
  20261004` (code; subfolders uploaded as zips, which the notebook reads; **its mount not yet tested**). The finals
  switch to them with `--replace-dataset`; `--read` exists for reads.

## 10-15 ~14:55: handoff state (owner: "save all this so the next session can pick up cleanly")

- **RUNBOOK §0 and §3 rewritten** as the current state: the three pending submissions and their rules in one table,
  the next steps in order, the panel's plan with each item's status, one consolidated cut list.
- **v3's time budget 9.0 → 10 h** (a budget, not a recipe change; the modeling review: at k 24 the cold-cache
  `--fit-budget` probe projects ~8.4 h for 16 epochs, a 6% margin under 9.0, and dropping to 15 epochs would trip
  fold 0's stop rule; 10 h still fits the 12 h session).
- **Saved off-repo:** the 0.9576 notebook the builders take as `--public-dir` (sha256 `16a9d953…`, as B and D were
  built) and three other pulled notebooks in `data/public_notebooks/` (git-ignored); the panel brief, the page source
  and the one-off audit scripts in `~/rsna-knee-work/2026-10-15/` (README there).
- **Still with the owner:** S6fall into D; the 4 stale remote branches. **Not done:** the mount test of our copy of
  heliosli's code dataset; the finals' builds.

## 10-15 ~15:00–16:30 (written 10-16): the weak four, measured; a plan for them cut by its own critique

- **Owner:** "attack our lowest score … train our own models when necessary (and not when there's a perfectly good one
  available) … we can't just rely on submitting the public notebook". A plan was proposed (SKM-TEA segmenter →
  cartilage measurements and a knee-centred compartment CNN; OAI images trained on the Mac for Synovitis); on the
  owner's request for a critical analysis three of its parts were **cut as repeats**: OAI images (Raptor already trains
  on 2,399 OAI knees for exactly PF OA, Lateral OA, Synovitis), the compartment CNN (R6 / R7 already read `views: v4`,
  knee-anchored compartment locals placed by the same SKM-TEA segmenter, both manifests; `bliverrigpdoge`'s crops
  +0.001), a fluid-volume feature (R6's fluid stacker). **Do not re-propose them.**
- **Measured on gold-58 (report only; scripts in `~/rsna-knee-work/2026-10-15/weak4/`):** our five teacher folds
  against the OAI-trained CoAtNet (`fusion/runs_coatnet_oai_teacher_pa`), AUROC and within-class Spearman ρ:

  | | ours | OAI CoAtNet | ρ |
  |---|--:|--:|--:|
  | Lateral OA (11 pos) | 0.822 | 0.770 | 0.94 |
  | PF OA (21) | 0.852 | 0.840 | 0.92 |
  | Synovitis (27) | 0.853 | 0.828 | 0.955 |
  | Lateral Meniscus (23) | 0.894 | 0.883 | 0.885 |

  OAI's image grades did not lift the one model that has them on these findings, and two families with different
  architecture, input and labels err on the same knees. **Blend arithmetic** (binormal, base 0.90 on a finding): a new
  member adds +0.013 at AUROC 0.85 / ρ 0.5, +0.003 at ρ 0.7, ~0 at ρ 0.9 (where every CNN on these labels sits).
- **"Traumatic cartilage loss counts as OA" (the criterion is ≥ ~1 cm of > 50 % loss, any cause): dead.** On gold
  with injured = ACL, Contusion or Fracture, injured Lateral OA positives ranked 0.71 against 0.95 for the rest; with
  injured = ACL or Fracture it is 0.84 against 0.81, and on the 4,349 held-out (consensus targets) injured positives
  rank 0.879 against 0.857. PF OA keeps a held-out gap of ~0.03 on 154 knees (≈ +0.0002 macro). Six knees made it.
- **Also closed:** Synovitis from Effusion (a perfect Effusion grade predicts gold Synovitis at 0.70; ours 0.85; gold
  Synovitis is 27 / 58, 22 of them with Effusion); a free gain in Raptor's inference (it reads all 96 slices, as
  trained); patient age (not in the headers) and repeat patients (one study per PatientID, 4,407 / 4,407).
- **Public resource checked:** `anatomy.pt` in `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` (112 MB) is a
  MONAI 2-D V-Net trained on SKM-TEA, four sigmoid channels `patellar_cartilage, femoral_cartilage, tibial_cartilage,
  meniscus` (`cloud_code/training/anatomy.py`), sagittal only, whole-series mean/std, 0.3125 mm, rows S→I and
  columns A→P (our canonical sagittal frame). Its author: "qDESS to clinical-protocol transfer unvalidated". In the
  blend it only places R6 / R7's views; **nothing in the blend measures cartilage or meniscus.**

## 10-16 ~10:00: the three reads, by their rules; the board moved

- **`57241351` = 0.957** (the 0.957 recipe + our five teacher folds). Rule: 0.957 → read our rank in the block against
  the recipe author's. The block is 319 teams at ranks 556–874, ours 825; heliosli has left it (now 0.958, rank 434),
  and ties sort by submission time, so the rank says nothing about our member. **Nothing claimed.** (Not read mode.)
- **B `57244451` = 0.957**, the same tick → **R6 + R7 in one decode is the ConvNeXt half from now on.** Read mode, so
  our member ran in it.
- **D `57244659` = 0.954** ≥ 0.950 → **D is the OAI-free final** in place of the five consensus folds (0.945).
- **Board (pulled 10-16 10:00 PDT, 6,098 teams):** #1 0.968; 0.963 at ranks 19–25 (gold ≈ rank 22); **0.958 is 433
  teams at ranks 123–555, so silver (top 5 %, ~304) and bronze (top 10 %, ~609) both sit inside 0.958**; ours 0.957,
  rank 825. New since 10-15: `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` (run 10-15 17:03) and dataset
  `heliosli/rsna-knee-r6-frozen-fe055406-w1`; most likely the free 0.958, **not yet read**.

## 10-16 ~10:30: the cartilage-measurement test — the rule, written before any build or run (owner's go)

- **What:** the SKM-TEA V-Net above on the fluid-sensitive sagittal series (our `sag_1` choice, our canonical frame and
  side) of a random 800 of the 4,349 non-gold training studies plus the 58 gold, on free Kaggle CPU kernels. Per study,
  fixed now: per compartment (lateral / medial tibiofemoral by the side of the joint-tissue centroid ± 4 mm; patellar;
  central femoral as trochlea) the volume, median and 10th-percentile thickness (distance transform at the skeleton),
  fraction thinner than 1 mm, connected components per slice, mean probability inside the mask and the uncertain-band
  fraction of femoral and tibial cartilage, meniscus and patellar cartilage; lateral / medial ratios; slice count,
  spacing, pixel size. One gradient-boosted model per finding on these alone, five folds by site
  (`folds_site_stage_b.csv`), targets `targets_consensus_v2` ≥ 0.5.
- **Rule:** the route continues (into the submission) only if, on Lateral OA **or** PF OA, the measurement model's
  held-out AUROC is **≥ 0.85 and** its within-class Spearman ρ with our five teacher folds' held-out predictions is
  **≤ 0.5**. Otherwise it is closed. Reported but not part of the rule: gold-58 AUROC (look only), CPU seconds a study
  (whether test-time segmentation fits the 9 h), the other two weak findings.
- **Either way the week then locks (owner):** v3 five folds from Fri 17:00 (standing go), the main final on the best
  free base with R6 + R7 as its ConvNeXt half plus our member, D as the OAI-free final. No new arms after this test.
- **The free 0.958, read (10-16 ~10:10, pulled into scratch):** `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters`
  is the 0.9576 notebook byte for byte in cells 2–5 and 7; cell 6 mounts `heliosli/rsna-knee-r6-frozen-fe055406-w1`
  (a new copy of R6 v1, same manifest sha256 `6508ce69…`); cell 8 changes the per-finding Raptor weight (R6 gets the
  rest): **Lateral OA 0.5 → 0.2, PF OA 0.5 → 0.6, Synovitis 0.5 → 0.8, Baker's 0.5 → 0.35** ("user decision 10-16",
  no basis stated; most likely chosen on the public board). Nothing new is trained. Lateral OA toward R6 agrees with
  gold-58, where the Raptor family is weakest there (0.770). For the main final this is the base to beat; whether its
  weights carry over to R6 + R7 as the ConvNeXt half is a read (one slot, owner's go), not a new arm.
- **Cartilage test, smoke (10-16 ~10:12, `raahimnawaz/rsna-knee-cartilage-smoke` v1, 4 gold studies, from `31a6eeb`):**
  `anatomy.pt sha256 ok` (R6's hash), 4 / 4 measured, 0 errors, **segmentation ~44 s a study on 4 CPUs** (decode ~1 s),
  so 1,300 test studies would need a GPU (est. 1–2 s a study on a T4). The outline images: 2 of 4 clean (femoral,
  tibial, patellar cartilage and meniscus on the right structures, in our canonical frame), **1 a large-field scan
  where the segmenter fired along the image edge** (meniscus 7,188 mm³), 1 a low-quality scan with fragments only.
  **Pre-run fix, no label read:** a 5 mm border band dropped and joint tissue kept within 45 mm (patellar cartilage
  70 mm) in-plane of the median of the confident joint-tissue pixels (`9a5041e`). The fragments are the segmenter's
  own limit on clinical scans: what the test measures, not fixed.
- **Smoke v2–v3 (10-16 ~10:25–10:45, 8 gold):** v2's gating centre was pulled to the edge by the edge firing itself;
  v3 takes the centre from the central 15–85 % of each slice (heliosli's window), and the outline images of 6 of 8
  knees are clean (femoral, tibial, patellar cartilage and meniscus on the right structures). The large-field scan has
  no confident tissue near the middle at all and fell through to "no gating": **now a failed segmentation**
  (`seg_ok 0`, no measurements; `a10bb4d`), the low-quality scan keeps its fragments. ~29–65 s a study on 4 CPUs.
  **Shards launched from `a10bb4d`: `raahimnawaz/rsna-knee-cartilage-s{0..5}of6`, 143 studies each (58 gold + 800).**
- **Code review of the test (10-16 ~11:00–12:00, owner: "fix everything"), every fix before any result is read:**
  1. *The scorer counted only knees the segmenter could read* (it dropped `seg_ok 0` and errors). A submission has to
     rank every knee, so **every sampled non-gold study now counts, a failed one with blank features**; the failure
     rate is printed; readable-only numbers are look only. On fake data a planted signal reads 0.735 on all studies
     against 0.761 readable-only: the size of the flattery removed. A missing study gives an INCOMPLETE verdict.
  2. *Duplicate studies across input files were resolved silently by file order* (a smoke folder could have supplied
     stale gold rows): **now a hard stop** naming the files. Pass the six shards' outputs only.
  3. *The vendored MONAI code from heliosli's dataset was imported unchecked* (the weights were checked): **pinned**,
     one sha256 over its 194 files, `1dcf1dab…` (dataset version of 10-04, downloaded and hashed 10-16), checked in
     `load_segmenter` before the import; a one-byte change is caught. The six running shards predate the check. **The
     blend kernels import the same vendored code through R6 / R7's worker unchecked: add the same check when the
     finals are built.**
  4. *Thin 3-D stacks* (9 of 159 sagittal series in the stage-0 sample have > 100 slices, up to 320; ~7 min a knee on
     CPU): `--slab-mm` averages them into slabs (heliosli's way); **off in this test** (it changes those knees'
     measurements, so the training rows would have to be re-measured the same way before it ships).
  5. *The medial / lateral split sits at the joint-tissue centre, not the notch:* kept (pre-registered; re-running the
     858 knees costs ~4 h), and **now checked on all 858**: the read prints the medial-minus-lateral extent around the
     split and the share of knees with the split 8 mm or more off the middle.
  6. *Tests:* `--check` now also covers the trochlea (exact distances), the patellar thickness, slab averaging, and
     `read_sagittal` on `canon_fixture`'s synthetic DICOM knees (left / right, side by tag / geometry, reversed storage,
     shuffled files): slices run lateral → medial, rows S → I, columns A → P in all four.
  The measurement functions and constants are byte-identical to `a10bb4d`, which the shards run (checked by AST).
- **✅ The cartilage-measurement test, read 10-16 14:25 by its rule: CLOSED.** Six CPU shards from `a10bb4d`, 858 / 858
  measured, `anatomy.pt sha256 ok` in each, 0 errors, **27 failed segmentations (3.1 %)**, segmentation 30.8 s a study
  (median, 4 CPUs). Scored with the review-fixed `fusion/cartilage_read.py` (every sampled study, failures blank):

  | | measurement model | rule | ρ with our folds | our folds, same studies | gold-58 (look) meas / ours |
  |---|--:|--:|--:|--:|--:|
  | Lateral OA (129 pos) | **0.654** | ≥ 0.85 | 0.22 | 0.861 | 0.673 / 0.822 |
  | PF OA (255) | **0.619** | ≥ 0.85 | 0.17 | 0.893 | 0.515 / 0.852 |
  | Lateral Meniscus (report) | 0.620 | | 0.21 | 0.924 | 0.647 / 0.894 |
  | Synovitis (report) | 0.471 | | 0.20 | 0.844 | 0.548 / 0.853 |
  | Medial OA (report) | 0.648 | | 0.28 | 0.897 | 0.632 / 0.983 |

  Its errors are independent of ours (ρ ≈ 0.2), its signal is not there. **Look only, after the verdict:** on the 500
  knees whose split sits within 8 mm of the middle (finding 5; 12 % are 8 mm or more off) it reads 0.648 / 0.595, so
  the split did not sink it; the best single measurements for Lateral OA are lateral-meniscus confidence and thickness
  (0.638), cartilage thickness carries next to nothing (patellar / trochlear thickness ~0.59 for PF OA); at a 0.90 base
  it would add +0.0009 to Lateral OA, **+0.00008 to the macro**. The SKM-TEA segmenter places cartilage on clinical
  scans but does not measure its loss. Measurements and verdict: `~/rsna-knee-work/2026-10-16/cartilage/`.
  **Closed, do not re-propose: cartilage / meniscus measurements from the SKM-TEA segmenter as a member.**
- **The week is locked (owner, 10-16):** no new arms. What is left: v3 five folds from the 17:00 GPU reset (standing
  go; §4 of the RUNBOOK); the main final on the free 0.958 base (whether R6 + R7 replaces its R6 + flip is one read,
  owner's go for the slot) plus our member (v3 if it passes its fold-0 check, else the five teacher folds); D as the
  OAI-free final; both on private dataset copies, without `--read`, with the vendored-code check of review finding 3.

## 10-16 ~15:00–18:30: repo cleaned; W4 built and set aside; K, the weak-four model, built — its rule before any run

- **Owner:** "remove everything from the repo that doesn't need to be there … delete the old branches … build something
  targeted at those weak points and train it"; then, on whether it attacks the right thing: "whatever attacks the
  weakest parts first, and only we do that" (no probes of public models).
- **Repo:** branches 7 → 2 (`labels-small-backbone`, `main`; the five old ones bundled in
  `~/rsna-knee-work/2026-10-16/old-branches-2026-10-16.bundle`, verified); `PLAN.md` (one page) replaces RUNBOOK.md and
  the 10-12 review; closed-experiment code and runs removed (`967df5b`). Still embedded by live builders, so kept until
  the finals build: `qwen_vlm.py`, `orthofoundation_model.py`, `window_curve.py`, `slots118.py`.
- **W4 (`--recipe w4` = v3 + loss ×3 on the four + a per-finding location bias), built and tested, NOT trained:** it
  learns from the same report labels as every model in the blend, and those share their mistakes on the four
  (ρ 0.89–0.955); it does nothing for Synovitis's label problem and puts no more pixels on the cartilage.
- **K, the weak-four model (`7204653`):** `pipeline/kcache.py` finds the knee joint centre J and the patella P with the
  SKM-TEA segmenter (9 sagittal slices, 0.3125 mm; weights and vendored MONAI sha-checked) and cuts four regions from
  the original series, each on a fixed grid in the canonical frame: **lateral sagittal** (6–42 mm lateral of J) and
  **lateral coronal** (the lateral half, 15 mm anterior to 20 mm posterior of J) for Lateral OA and Lateral Meniscus,
  **axial through the patella** for PF OA, **central sagittal around the front** (suprapatellar recess to Hoffa's fat
  pad) for Synovitis and PF OA; 8 slices each at 256 px (0.3125 mm; the front 0.375 mm). Trained at 384 px (the feature
  cell 6.7 mm against Stage B's 11.7 mm), on the weak four only (`--only-weak4`), spatial attention per finding,
  slot / depth embedding, all 24 windows each step, 16 epochs, `targets_teacher_v3`, folds by site. Tests: 24 geometry
  checks on synthetic DICOM knees (block positions within 6 px of the analytic ones; no medial block in the lateral
  regions; a left knee renders like a right one); a synthetic K cache trains end to end.
- **The same idea was closed 10-15 as redundant with R6 / R7 *in the public blend* (`bliverrigpdoge`'s knee-centred crops
  +0.001 there).** Reopened by the owner's direction: our own model aimed at the four, judged on them.
- **K's rules, written now:** (1) *fold 0 is a not-broken check:* against `runs_cnxs_teach1_f0` on the same held-out
  studies, stop and look if K's mean AUROC over the four is more than **0.03** below, or it ran fewer than 16 epochs;
  otherwise folds 1–4 go (GPU: the owner's go of 10-16 to train it). (2) *K counts on a finding only if* its single-
  finding LB probe beats our five teacher folds' probe on that finding (Synovitis 0.884, Lateral OA 0.896, PF OA 0.896,
  Lateral Meniscus 0.932) by **≥ +0.03**; probes need the owner's go for the slots.
- **K, build and checks (10-16 ~17:00–18:30):** smoke v1 (`7204653`, 10 train studies): every region filled in 10 / 10,
  ~12.5 s a study on 4 CPUs, the joint found in 9 / 10, the patella in 3 / 10; the montages showed the lateral
  compartment in both lateral regions and the patella over the trochlea where it was found, and pf_ax clipping the
  patella where the 35 mm default stood in. **Label-free fixes (`867c2c9`):** the default patella 40 mm anterior
  (heliosli's), patella detection over the whole field less a 5 % border, pf_ax 96 mm; smoke v2: patella found 4 / 10,
  found offsets 32–43 mm anterior, 21–50 mm superior; pf_ax shows patella and trochlea. **Shards launched 18:22 from
  `867c2c9`:** `raahimnawaz/rsna-knee-kcache-s{0..4}of5` (~881 studies each, ~3 h).
- **K at test time (`4bcb63c`):** the knee finder split into a CPU half (decode workers) and a GPU half, the same
  arithmetic in the same order: byte-identical crops on synthetic knees (`kcache --check`, 28 checks); with the real
  segmenter the training-cache path and the submission path give the same predictions (max |dp| 5.6e-17).
  `infer_windows` runs K in two passes; `build_submit_kernel --ours-only --segmenter [--probe <finding>]` builds K's
  submission or a single-finding probe. The OURS gate rebuilds training studies from DICOMs, so it checks K end to end.

## 10-16 ~21:00–: K's cache built; the training kernel's smoke; fold 0

- **The five cache shards (`867c2c9`), read 10-16 ~21:10–21:55:** 4,407 / 4,407 training studies (882 / 881 / 882 /
  881 / 881), **0 errors**, every shard `exited 0` (the segmenter's weights and the vendored MONAI tree are sha-checked
  in `load_segmenter` and stop the run on a mismatch), 156–215 min a shard on 4 CPUs (10.3–14.0 s a study, median).
  Region fill: lateral sagittal 100 %, lateral coronal 99.95 %, patellofemoral axial 99.9 %, anterior sagittal 100 %.
  **Anchors, all 4,407:** joint and patella found 2,281 (51.8 %); joint found, patella not 1,875 (42.5 %, P = J + the
  40 mm / 25 mm default); patella implausible 5; **no joint tissue 246 (5.6 %)**, J = the mean centre of the central
  sagittal slices. Not a stop by any rule; a K weakness to keep in view: pf_ax and ant_sag sit on the default offset
  in ~48 % of knees, and the lateral regions on a field-centre joint in ~6 %.
- **The training kernel's CPU smoke (`raahimnawaz/rsna-knee-k-train-convnext-small-k-smoke-cv`, from `90f975e`):** the
  five shards mount (4,407 studies, slots 8 / 8 / 8 / 8, canon), fold 0 = 3,502 train / 847 val (691 / 691 control
  studies in fold 0), only the weak four carry loss, two epochs, loss 1.3566 → 0.8206, DDP weight checksums equal on
  both ranks, `SMOKE: the whole path ran`, 0 tracebacks. (`grad-norm nan` at ep0 it0 is the value before the first
  optimizer step under accumulation, not a NaN gradient.)
- **K fold 0 launched 10-16 22:08 (`raahimnawaz/rsna-knee-k-train-convnext-small-k-cv`, from `1138775`, two T4s; the
  owner's go to train K):** `fit-budget: 0.49 s per 1-study micro-step -> 19.4 min/epoch; 9.99 h left -> 16 epoch(s)
  (cap 16)`; 437 steps an epoch (bs 8 = 1 × 2 GPU × 4), k 24, 384 px, fp16; epochs 0–1 measured 18.6 / 18.7 min, loss
  0.9243 → 0.8343. **Projected ~5.0 h of training + eval, done ~03:45 PDT 10-17; ~5.3 GPU-h a fold, ~26.5 h for five of
  the week's 30 h.** Its rule (written 10-16, above): stop and look if the mean AUROC over the four is > 0.03 below
  `runs_cnxs_teach1_f0` on the same studies, or < 16 epochs; otherwise folds 1–4.
- **Ops:** `kaggle kernels logs <kernel>` prints nothing while a run is live; `kaggle kernels logs -f <kernel>` streams
  the whole log so far (wrap it in `perl -e 'alarm 40; …'` and grep).
- **✅ K fold 0, read 10-17 ~07:15 by its rule: PASSES (not broken) → folds 1–4.** 16 / 16 epochs (18.4–18.7 min
  each, loss 0.9243 → 0.6685), 0 tracebacks, DDP checksums equal, 301 min, done 03:29 PDT. `score_oof.py` against
  `runs_cnxs_teach1_f0` on the same 847 held-out studies (sol56 reference, gold excluded), K = avg3:

  | | teacher f0 | K f0 |
  |---|--:|--:|
  | Lateral Meniscus | 0.911 | 0.907 |
  | Lateral OA | 0.827 | 0.826 |
  | PF OA | 0.867 | 0.862 |
  | Synovitis | 0.868 | 0.856 |
  | **mean of the four** | **0.8683** | **0.8628** (−0.0055; the stop line was 0.8383) |

  The other eight sit near 0.5 (no loss on them, as designed); K's gold-58 "macro" 0.672 is over all twelve and means
  nothing. **Look only, after the verdict** (`~/…/scratchpad/k_look.py`, same 847): within-class ρ K~teacher
  0.78–0.91 (negatives 0.79–0.83, positives 0.78–0.91), and a 50/50 rank average gains +0.007 / +0.007 / +0.004 /
  −0.001 (mean +0.0043). **On report-derived labels K learned what the teacher knows and makes most of the same
  mistakes**; whether its crops see an image-graded difference the proxy cannot is what the single-finding LB probe
  (rule 2, ≥ +0.03) decides. The prior for that bar is now weak.
- **Folds 1–4 (owner's go 10-17: "if it passes, launch folds 1–4"):** folds 1 and 2 pushed 07:20
  (`…-k-f1-cv`, `…-k-f2-cv`, from `335d1a6`); **Kaggle runs at most two GPU sessions at once** ("Maximum batch GPU
  session count of 2 reached" for 3 and 4), so a session script pushes fold 3, then 4, as slots free (~12:45), all
  five done ~18:30 PDT 10-17. GPU: ~5.4 h a fold → ~27 of the week's 30 h, ~3 h left for probes and finals' commit runs:
  **no room for a re-run**.
- **Owner 10-17 ~08:50: "Hold fold 3: … a fold-0-only single-finding probe of K on Lateral OA (one submission slot,
  go given for that one). Launch folds 3–4 only if it beats 0.896 by +0.03."** The fold 3–4 chain was stopped before
  it pushed anything; folds 1–2 keep running.
- **The Lateral OA probe's rule, written before it runs:** kernel `build_submit_kernel.py --ours-only --segmenter
  --probe "Lateral OA"` on K fold 0's `avg3/last.pt` (the checkpoint fold 0's rule scored), gated on its own
  held-out predictions. Before the submit its log must show `sha256 ok`, `GATE PASSED`, the PROBE line (only Lateral
  OA is the model's, the other eleven 0.5) and 0 tracebacks. The board shows three decimals: AUROC = 12 × score − 5.5,
  so a shown score covers ±0.006 of AUROC. The bar is our five teacher folds' Lateral OA probe 0.896 + 0.03 = 0.926
  ⇔ score ≥ 0.5355: **a shown 0.536 or higher → PASS: push folds 3 and 4; a shown 0.535 or lower (AUROC < 0.926) →
  FAIL: folds 3–4 are not launched**, and K stands at folds 0–2. Handicap known before the read: one fold against
  the teacher's five-fold ensemble (worth ~+0.003–0.007 on the consensus model's macro); the owner's bar stands as set.
