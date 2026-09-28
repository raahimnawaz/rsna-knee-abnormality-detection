# Pipeline Audit — rsna-knee-abnormality-detection

## Audit 2026-09-28 — the teacher → student path, before its first run

Audited at `5e4b6f0`. Covers what landed after the 09-26 audit: `--init`, `--slots`, `--allow-missing-labels`,
`--aug-affine`, `--compile` / `--ckpt-blocks`, the `resnet34` arch, the launcher's `--teacher-fold` and
`--predict-all`, `labeling/build_student_targets.py`, the public-CoAtNet check, and the serving path those feed.
It doesn't repeat what 09-23 and 09-26 fixed.

**Verdict:** the recorded numbers still mean what they say. The LB reads, the 691 and gold-58 are sound, and
the public CoAtNet's contamination (it trained on the 691 and on gold ×3) is already recorded and routed to "LB
only" (`RUNBOOK.md:255-257, 332`). The problems are all in the path that runs next. **Its next step can't run:
`--predict-all` refuses to start.** A CoAtNet student couldn't be submitted either. If the k-fold teachers come
back, the way their folds are pooled would inject a site-correlated shift into the student targets. None of this
produces a wrong recorded number. Two of the three would fail loudly, and one would quietly weaken the experiment.

### Pipeline map (the new path)

```
fastMRI+ h5 ─(converter: NOT BUILT)─► NYU teacher  train_windows --slots cor_1,cor_2 --allow-missing-labels
public CoAtNet ckpt ─ train_windows --init (strict; 486 tensors) ─► --epochs 0 eval      [691 + gold NOT held out]
launcher --teacher-fold K (1-4): train = pool − fold K, predict fold K ──────┐   (pool = fold ≠ 0, not gold)
launcher --predict-all: an outside teacher predicts pool ∪ 691 ──────────────┤
                                                                             ▼
build_student_targets.py: each teacher's ranks → base values on its studies; mean with base; uncovered = base
   └─ --gold: base vs combined on gold-58 (the "free gate")        [fits nothing; gold labels only score]
student: launcher --targets <table> [--init public CoAtNet] ─► LB
serve: build_submit_kernel ─► infer_windows (Ortho checkpoints only, see below) ─► submission.csv
```

### Findings

#### [MEDIUM] `--predict-all` always refuses to start — the next step on the RUNBOOK's order to run
**Where:** `notebooks/build_backbone_ab_kernel.py:275-279`; `fusion/train_windows.py:569-570`
**What:** predict-all sets `VAL = sorted(set(_pool) | set(VAL))` and then `TRAIN = _pool[:2]`, so both
training studies are also in VAL. `train_windows` then runs `if set(tr) & set(va) ...: raise SystemExit("train
overlaps val or gold -- refusing")`. That check comes before the model is built, so it fires whether or not
`--epochs 0` is passed. Reproduced on the committed partition (`provenance/folds_site.csv`): `2 train | 4193 val |
overlap: 2` → `train overlaps val or gold -- refusing`. It has never run: the CoAtNet check was built from
`b3cd3ba`, before `6c4bd3b` added the flag, and its `provenance.json` has no `predict_all` field.
**Impact:** "NYU teacher → `--predict-all` with `--init` on the competition knees" (`RUNBOOK.md:279-281`) dies
at startup. It costs a session and the queue wait, not a wrong number.
**Fix:** skip the train/val overlap check when `--epochs 0` (nothing trains; keep the train/gold check), or pass
`TRAIN = []`. Then make the launcher force `--epochs 0` and require `--init` under `--predict-all`, so a
forgotten flag can't train on two studies.

#### [MEDIUM] The submission path can serve only an Ortho checkpoint, but the target system's CoAtNet student needs it
**Where:** `fusion/infer_windows.py:119-128`; `notebooks/build_submit_kernel.py:109`;
`fusion/runs_coatnet_raptor_check/last_meta.json:19`
**What:** a CoAtNet checkpoint's `last_meta.json` records `rope_periods: null` (CoAtNet has no RoPE).
`build_model` then takes the "Stage A" branch and runs `m.encoder.rope.periods.copy_(...)`, which raises
`AttributeError` on a timm `MaxxVit`. The kernel also finds weights only in a directory named `runs_ortho`, while
a CoAtNet run writes `runs_coatnet/` (`build_backbone_ab_kernel.py`, `RUN = OUT / f"runs_{ARCH}"`). This is
separate from the ResNet `img_size` problem the RUNBOOK already lists (`:249`).
**Impact:** "Models (students), one per family: CoAtNet from the public 0.928 weights" (`RUNBOOK.md:233`) can't
reach a submission. It fails in the commit run's gate, so it costs a cycle, not a wrong LB.
**Fix:** touch RoPE only when the encoder has it (`getattr(m.encoder, "rope", None)`), and refuse if periods are
recorded but the encoder has no RoPE. Find `last.pt` by its sha256 under any `runs_*` directory, since the sha is
already the identity check. The existing `--check-against` gate then proves the CoAtNet path end to end.

#### [MEDIUM] Pooling k-fold teacher OOFs before ranking turns each fold model's calibration into a site-level target shift — *applies if the k-fold teachers return*
**Where:** `labeling/build_student_targets.py:43-48` (`read_table` concatenates the glob), `:75-81` (one ranking
over the pooled scores)
**What:** the four `--teacher-fold` models are different models. When their `oof_all.csv` files are
concatenated, a model that reads a finding systematically higher lifts every study in its fold up the pooled
ranking. Those studies' targets then move. Folds are site-grouped, so the shift lands on whole scanner sites,
which gives the student a site cue that doesn't exist at test time. **Measured on real OOFs:** two sibling runs
(`runs_r34v2_B`, `_C`: same labels and studies, one augmentation apart) were assigned to the two halves of the
691's sites, 200 random site splits. The half's mean pooled percentile moved by **0.03-0.10** depending only on
which model scored it (Fracture −0.100, ACL +0.065, Baker's +0.064, MCL +0.041). **Reproduced through
`combine`:** two folds with the same ranking quality and a +0.5 score offset in one fold moved the mean targets
by **−0.040 / +0.039**. Mapped per fold, the move is **−0.000 / −0.000**.
**Impact:** label noise correlated with site, of a size comparable to the ~+0.005 LB the student is priced at
(`RUNBOOK.md:312`). It could hide a real teacher gain and read as "teacher → student doesn't work." Today the
k-fold teacher is off the plan (the CNN failed its gate), so this applies only if the Mac CNN test revives it. The
NYU teacher is one model and is unaffected.
**Fix:** map each OOF file onto the base values of its own studies, then pool. That's the docstring's own rule
("within the studies that teacher covers") applied per fold model. Folds are label-stratified, so each fold keeps
its own base levels and real case mix. **Check once the OOFs exist:** per fold, the teacher's mean pooled
percentile against the base's. A gap of more than ~0.02 is this effect.

#### [MEDIUM] The "free gate" doesn't compute the statistic its pre-registered rule names
**Where:** `labeling/build_student_targets.py:94-95, 129-134` vs `RUNBOOK.md:299-301`
**What:** the rule is "PASS iff report + teacher beats report alone on the MACRO over the teacher's covered
findings, paired bootstrap P ≥ 0.95." `--gold` prints two point estimates of the **12-finding** macro. For a
teacher covering 4 findings the printed delta is diluted about 3×, and it carries no interval. The "+0.034 ±
0.008" in `RUNBOOK.md:204` came from outside the tool.
**Impact:** the gate that decides whether ~11 GPU-h go into a student has to be computed by hand each time. A
3×-diluted point estimate without a CI is the easy misreading.
**Fix:** restrict to the covered findings and run a paired bootstrap on sorted ids with a fixed seed (as
`score_oof.py` does). Print Δ, SE and P(Δ > 0).

#### [LOW]
- `read_table` keeps the **first** file's rows on duplicate ids (`build_student_targets.py:47`). A glob over the
  four teacher folds' `oof_gold.csv` (each fold predicts all 58) therefore gates on fold 1's teacher alone. If
  k-fold teachers are gated on gold, average the folds' per-file ranks instead.
- `--init <bare filename>` walks all of `/kaggle/input` with `rglob` (`train_windows.py:617`), DICOM trees
  included. That's the 6-minute walk 09-26 removed from the targets lookup. Skip `train_series` / `test_series`
  the same way.

### Verified clean
- **Teacher folds:** `TRAIN = pool − fold K`, `VAL = pool ∩ fold K`. The pool excludes fold 0 and gold, and folds
  are site-grouped, so a fold-K teacher never saw a study it labels. The 691 gets no k-fold teacher label, and
  gold predictions come from models that excluded gold.
- **Student:** trains only on the pool. `build_student_targets` fits nothing, keeps uncovered studies at the base
  value exactly, and uses gold labels only to score. The launcher still refuses NaN/out-of-range targets and
  keeps `targets.csv` = `steven_v2`, so the partition doesn't move.
- **Public CoAtNet:** its contamination is recorded where it's used. Anything started from it is judged on the
  LB only, and the NYU teacher is specified not to be it (`RUNBOOK.md:296-297`). `--init` is strict except for
  the zero-initialised slot/depth weights.
- **New training flags:** `--allow-missing-labels` gives the default loss exactly when every target is known,
  with prevalence from the training rows only. `--aug-affine` is training-only with no flips. `--compile` leaves
  eval eager and `last.pt` keys unchanged. `--slots` applies to train and eval alike.
- **Coronal-only teacher coverage:** `cor_1` fill is 1.000 (`DATA.md:404`), so no competition study reaches a
  coronal teacher as a blank window.
- **Serving an Ortho checkpoint:** the 09-27 commit runs passed the DICOM gate at max |Δp| 0.00079.

### Open questions
1. **Was `dread_soft` built or checked against gold-58?** It carries no gold rows, but a prompt or level
   mapping tuned on those 58 leaves no trace (`bench_external.py:143`). If it was, `dread`'s gold read (0.912)
   and the teacher → student ingredient (0.9269, `DATA.md` §12) are optimistic. The LB gave 44% of the gold gain.
2. **The fastMRI+ converter's slot contract.** For the NYU teacher's competition predictions to be in
   distribution, `CORPDFS_FBK` must map to `cor_1` (fluid) and `CORPD_FBK` to `cor_2`. It also needs the same
   per-series 2-98% window, 140 mm field and coronal left/right convention as Stage A. Knee side is not in the
   header. A per-slot intensity/geometry comparison against Stage A shards is the parity check, before training.
3. **Will gold-58 gate more than one teacher variant?** One pre-registered candidate is a test. Several, tried
   in turn, is selection on gold, which `DECISIONS.md` closed (claims +0.0137, delivers −0.0034).

---

## Audit 2026-09-26 — the live path, before the Qwen runs

Audited at `57693c3`; fixes landed in `b0e330b` and after. Covers the code that runs NOW: the Stage
A/B cache builder, `train_windows.py` (including the new `qwen35vl` arch), `score_oof.py`, the fold
split as the kernel builds it, `infer_windows.py`, and the new label path (`qwen_label.py`,
`bench_qwen_labels.py`). What 09-23 fixed is not repeated here.

**Verdict:** the recorded numbers still mean what they say. There's no leakage in the live path, and
the DICOM inference path is gated against the scored OOF (it reproduced to max |Δp| 0.00072 on
Kaggle on 09-26). The problems were in what the Qwen plan was about to run. Two would have wasted a
~10 h GPU session: a train/eval sequence-length skew in the VLM, and a ~4 GB allocation at the end
of training. A third made the plan's first GPU run impossible: the kernel could train only on
`steven_v2`. All three are fixed. What stays open is listed at the end.

### Pipeline map

```
DICOMs ─ build_slots64.build_study: per-SERIES 2–98% window (:261), 140 mm crop, per-study side (canon)
   └─► Stage A / B shards (Kaggle CPU)                                    [no cross-study statistic]
reports ─ public tables (steven_v2 = targets) │ qwen_label.py (local Qwen3.5-9B) → qwen_labels/<tag>/targets.csv
headers ─ site_fingerprint → folds.py --group-by site (gold pinned) ─ SPLIT
   └─► kernel PREP: val = the 691 fold-0 control studies; train = fold ≠ 0 AND not gold (build_backbone_ab_kernel.py:244)
train_windows.py: Windows (k random windows train / every or --eval-windows N eval) → WindowMIL | QwenStudy
   fitted: weights; pos_weight from TRAIN prevalence only (:586)                → oof_all, oof_gold, last.pt
score_oof.py: vs lixin gpt56 reference, gold excluded (:80, :159), arms on their intersection, paired
bench_qwen_labels.py: label tables vs gold-58 (two pre-registered candidates)
infer_windows.py: build_study + window_centres + WindowMIL, --check-against gate → build_submit_kernel (family rank-mean)
```

### Findings

#### [HIGH] The VLM would have trained on 12-image sequences and been scored on ~50 — *fixed*
**Where:** `fusion/train_windows.py` eval path (`predict_sharded`, `Windows`); `fusion/qwen_vlm.py`
**What:** `train_windows` trains on `--k 12` random windows per study and scores **every** window
(51 on the median Stage A study). That is the frontier's recipe, and it is harmless for `WindowMIL`,
whose attention pool is permutation-invariant and indifferent to count. `QwenStudy` feeds the windows
to a language model as one sequence. It would have trained on ~1,750-token sequences and been read
on ~7,800-token ones, with M-RoPE positions it never saw in training.
**Impact:** an unmeasured, one-directional loss on every validation and test read of the VLM arm. It
would have looked like "the VLM is weaker" rather than "the VLM was read out of distribution".
**Fix:** `--eval-windows N` scores an evenly spaced subset (`thin_windows`, identical to
`window_curve.thin` on Stage A: 0 mismatches on 3,000 random masks). The qwen build passes 12
(`build_backbone_ab_kernel.py:96`). Ortho's runs are unchanged (default 0 = every window).

#### [HIGH] The end-of-run DDP checksum would have allocated ~4 GB on the T4 — *fixed*
**Where:** `fusion/train_windows.py:717`
**What:** the checksum summed `p.double()` over **all** parameters. For the VLM that includes the
frozen 248k×2048 embedding: 508 M values in fp64, about 4 GB, allocated after a full session on a
card already holding the model, and before `oof_all.csv` is written.
**Impact:** a likely CUDA OOM at the last step of a ~10 h run, losing every prediction.
**Fix:** sum trainable parameters only, since frozen ones cannot drift between ranks. For `WindowMIL`
every parameter trains, so its behaviour is unchanged.

#### [HIGH] The training kernel could not train on any labels but `steven_v2` — *fixed*
**Where:** `notebooks/build_backbone_ab_kernel.py` PREP (targets built only from `llm_labels_v2.csv`)
**What:** the RUNBOOK's next GPU run is step 5 with only the targets changed. There was no way to
pass a table, and no record of which table a run trained on (`summary.json` lacked it).
**Fix:** `--targets <csv> --tag <t>` embeds the table, checks its hash in the kernel, refuses NaN or
out-of-range cells and duplicate studies, and gets its own slug. `summary.json` now carries
`labels_sha256`. Tested: `steven_v2` through this path hashes to `3c082a98`, which is step 5's
recorded `targets_sha256`, so a retrain differs from step 5 in the targets and nothing else.

#### [HIGH] `--targets` changed the fold partition — *caught by provenance on Kaggle, fixed 09-26 13:40*
**Where:** `notebooks/build_backbone_ab_kernel.py` PREP. The alternative table overwrote `data/targets.csv` before
`folds.build()` ran, and `folds.py` **stratifies on that file**.
**What:** the dread_soft retrain rebuilt a different site-grouped partition. That is a different fold 0, so the
"691" it would have been scored on was a different set of studies, and the one change the run was meant to test
(labels) came bundled with a second one (the split).
**Caught:** `provenance.check` refused `folds_site.csv` (recorded `780f939c`, got `f430236f`) 6 minutes in, before
any training. My 09-26 test had executed the targets block in isolation, so it could not see its interaction
with `folds.py`. That was a narrow test aimed at a cross-file question.
**Fix:** `targets.csv` stays `steven_v2` (the partition's definition). The run's labels go to `targets_run.csv`
and reach only the trainer and the pool filter. `provenance.json` records both hashes. Tested: the stratifier is
byte-identical to step 5's (`3c082a98`) and the trainer gets `dread_soft` (`8cec477f`). The same run also spent
those 6 minutes walking every DICOM for the label file; the walk now skips the image trees.

#### [MEDIUM] Qwen versions were unpinned — *fixed*
**Where:** the kernel's `pip install -U transformers`; `snapshot_download("Qwen/Qwen3.5-2B")`
**Fix:** `transformers==5.17.0` (the version the arch was tested against, installed only if the image
lacks `qwen3_5`), Qwen3.5-2B at revision `15852e8c` (`qwen_vlm.py:46`). The labeler manifest records
the MLX weights' revision and the `mlx_lm` version.

#### [MEDIUM] Nothing can serve the VLM yet — *open, by choice*
**Where:** `fusion/infer_windows.py:87-91` builds only a ViT-L `WindowMIL`.
**What:** a VLM checkpoint cannot reach a submission. It also needs `transformers` with `qwen3_5`
inside an internet-off notebook, which means a wheel dataset.
**Why open:** build it only if the arm earns a place (RUNBOOK, Qwen block). When built, reuse
`qwen_vlm.build(config=last_meta["qwen_config"])` (a rebuild from the saved config reproduces the
trained model exactly, checked locally) and keep the `--check-against` gate.

#### [LOW] Duplicate report texts cross folds — *checked, not leakage*
49 report texts (over 60 chars) repeat, and 11 of those groups span fold 0 and the training folds (25
fold-0 studies). They are templated normal reports (e.g. the Turkish "all normal" template) on
different scans. At most 3 groups share both text and series geometry, so there are ≤ 3 possible
duplicate scans, against 691 validation studies. One gold study shares a template with a fold-3
training study (both fold 3; gold is never trained on).

### Pass 4 — performance, from measured step times `2026-09-26`
Measured, not guessed: Stage A 2.69 s per 2-study micro-step (24 windows at 336 px) on ONE T4, peak 6.9 GB;
Stage B 3.52 s at 384 px on EACH of two T4s (DDP), peak 8.5 GB; eval 1.53 s/study (54 windows); the curve's pure
forward 1.28 s/study. **In place:** fp16 autocast + GradScaler, timm Eva's fused SDPA (`fused_attn=True`), uint8 to the
device with normalisation/resize there, DDP and sharded eval on Stage B, one inference process per T4.
- **[MEDIUM] Stage A-recipe runs use one of the two T4s** (`build_backbone_ab_kernel.py` launches NPROC 1 unless
  `stage_b`/qwen). Step 5 and the 09-26 dread retrain idle half the hardware for ~11 h — for dread by design (its only
  difference from step 5 must be the labels). Every later run: DDP on both (Stage B measured 55 min/epoch).
- **[MEDIUM] ~10 TFLOPS achieved per T4** — ViT-L/Eva ≈ 2×303 M×446 tokens per window forward; 24 windows × ~4×
  forward (backward + checkpoint recompute) in 2.69 s, and the forward-only curve lands at the same rate — about
  40–50% of what a T4 sustains on fp16 GEMMs. Suspect: Eva's unfused RoPE/norm elementwise work. `torch.compile`:
  unmeasured.
- **[MEDIUM] Gradient checkpointing is on for every block on CUDA** (`train_windows.py:571-579`): ~25–33% recompute
  while peak memory is 6.9 / 8.5 GB of 15. Partial checkpointing: unmeasured.
- ~~**[LOW-MEDIUM] Data wait is not timed separately.**~~ **Ruled out from numbers in hand:** step 5's epoch is
  1,751 micro-steps × 2.685 s = 78.4 min of compute against 78.7 min measured — data loading costs ~0. (The loop's
  `sec_per_micro_step` starts AFTER the batch arrives, so a data stall would show as a gap between the two; there is
  none.) `bench.json` now records the wait directly.
- **[efficiency track] The submission is decode-bound**: 1,322 × 3.8 s/study/worker over 4 workers ≈ 21 min of
  DICOM decode against ~14 min of T4 (RUNBOOK, the curve); `order_series` reads every header before the picked slices.
- **[LOW] No `pin_memory`** in the training loader — 8 MB uint8 batches, negligible.
- The VLM ran HF's PyTorch fallbacks for its linear-attention layers (`fla`/`causal_conv1d` absent) — part of its
  2.4 h/epoch.
**Bench read 09-26 20:58 (`RUNBOOK.md`, `fusion/runs_speed_bench/`): none adopted.** Partial checkpointing buys 5% (the recompute
estimate was high); no checkpointing OOMs; compile cuts the training step 17–25% with an identical loss but failed the eval clause
on a 12-study eval that likely timed warm-up. So of pass 4's list only **DDP on both T4s** is adopted. **v2 (09-27, eval eager, DDP on both T4s): `--compile --ckpt-blocks 12` ADOPTED, −28.6% per step, loss unchanged.**
**Next (as written before the bench):** DDP everywhere; then ONE ~0.5 GPU-h bench timing `torch.compile`, partial checkpointing and a local shard
copy separately, each adopted only if it cuts s/step ≥ 10% with an unchanged loss curve.

### Verified clean
- **Split:** the training pool excludes fold 0 and gold (`build_backbone_ab_kernel.py:244`), and
  `train_windows` refuses any overlap (`:527`). The Qwen labeler and C2's rank mapping use report
  text and other tables only, never gold labels.
- **Eval mechanics:** `predict` runs under `@torch.no_grad()` with `model.eval()` (`:372-374`), which
  disables LoRA and vision dropout. No epoch is selected, and gold-58 is reported only.
- **Scoring:** gold is excluded, arms are restricted to shared studies, ids are sorted, and the
  bootstrap is paired (`score_oof.py:80,159`).
- **Cache:** the only intensity statistic is per series (`build_slots64.py:261`). Laterality is per
  study. Nothing is fitted across studies.
- **Targets:** `steven_v2` has no duplicate or conflicting study rows.
- **Qwen labels:** they read only `Report`; a resume is refused if the prompt/VALUE/model hash
  differs; letter mass is 0.996–1.000 on the smoke. `bench_qwen_labels.py` refuses a partial table,
  runs both leak checks, and spends exactly two candidates.
- **VLM wiring:** patchify equals `Qwen2VLImageProcessor.patchify`; each finding is read at its own
  name's last token; a rebuild from saved config plus `last.pt` matches; the output is invariant
  to window order; gradients reach vision, LoRA and head, and nothing frozen gets one.

### Open questions
- **fp16 on a T4.** Qwen3.5 was trained in bf16, and a T4 has none. Per-layer peak activations on real
  weights (`qwen_vlm.finite_check`) and the Kaggle bench decide this before any long run.
- **The frontier's gold ×3.** `train_windows` has no per-study weights, and training on gold uses up
  the only instrument that can judge labels. If the label route goes there, hold out half of gold (a
  fixed 29/29 split) and read on the other half.

---

## Audit 2026-09-23

Audited **2026-09-23** at `2cb9782`. Covers the scored path (folds → training → both local
instruments → blend → submission kernel) and the C6 code the RUNBOOK says runs next (Stage 0,
`slots64.py`, `perlabel_head.py`, the OrthoFoundation loader). `fusion/archive/`, `extractor/`
internals beyond the leak check, and the EDA scripts were not audited.

**Verdict:** the recorded numbers look sound. I found no leakage, no split contamination and no
train/serve skew in the scored path. Gold is never trained on, nothing is selected on a
validation fold, and the RadImageNet serving contract matches its measurement path field by field.
**One recorded *conclusion* is wrong, though: D3** (first finding, added the same day). The problems
otherwise are in what runs next. The fold partition every gate bar was measured on can no longer be
rebuilt with certainty. It is now guarded by a hash record, and the sampling test checks fold 0.
**Every finding below is fixed, guarded or decided as of 09-23.** Each one says what was done and
how it was tested.

---

## Pipeline map

```
Kaggle DICOMs ──(kaggle_01* header audits)──────────────────────────────┐
   │                                                                     │
   ├─ local NIfTI repackaging (deleted 09-07) → slot_cache / *_pixels    │  D3 (amended)
   └─ C6 plan: slots64.py on Kaggle DICOMs         [NOT BUILT]           │
                                                                         ▼
public LLM label tables ── bench_public_labels.py --download ──► targets.csv (steven_v2)
                                                                     │
dicom_headers parquet ── site_fingerprint.py ──► site_fingerprint.csv│
                                                        │            │
                                                        ▼            ▼
                                     folds.py --group-by site  (stratified on targets+gold)
                                                        │
                                                        ▼  SPLIT: scanner-site groups, gold pinned
                           train_port.py  fold k: train = other folds MINUS gold; fixed epochs;
                                          last-epoch model → eval() + no_grad → oof_all.csv
                                                        │
                        ┌───────────────────────────────┴───────────────────────────┐
                        ▼                                                           ▼
     score_oof.py  vs lixin_gpt56, non-gold, n≈681/fold           score_gold.py  vs 58 image reads
     (paired bootstrap on shared studies)                          + GOLD_TO_LB 0.039 → LB estimate
                        │
                        ▼
     blend: equal-weight rank-mean over FAMILIES, nothing fitted   (build_f6_rad_kernel.py:309-339)
                        │
                        ▼  SERVE: notebooks/submissions/rsna-knee-f6-two-arm.ipynb (+ rad cell),
                           DICOM decode in-notebook, re-run by Kaggle on hidden test
```

**Fitted objects.** Only model weights. No global scaler or imputer. Intensity windowing is
per-series percentile (per-sample, so nothing crosses the split). Blend weights are not fitted.
Fold recovery (`fold_recover.py:58`) is label-free. `GOLD_TO_LB` is a constant estimated from
(gold, LB) pairs and is used only to read results, never to train.

**Standing hazard, already known.** The pixel reader exists twice, as local transcriptions
(`pilkwang_pixels`, `rad_pixels`) and inside the notebook, and the two must agree by hand.
`build_f6_rad_kernel.py`'s save/restore-and-assert of the module globals is the right defence.

---

## Findings

### [HIGH] D3's 0.085 is pinned on the wrong cause, through a false PASS in `contract_audit.py` — *amended 09-23*
**Where:** `fusion/contract_audit.py:85,88` (before the fix); `DECISIONS.md` D3; `fusion/runs_port_repro.log:3`;
`pipeline/slot_cache.py:130,291-293` vs `fusion/pilkwang_pixels.py:66-70`
**What:** D3 says the 0.0856 between our reproduction (0.7597) and the fork (0.8453) is "our pixels are
a third-party NIfTI repackaging". Two records contradict that:
- The fork's own weights, run on our NIfTI files through their reader, score **0.8457 / 0.8516**
  (report / gold-47) against **0.8453 / 0.8516** on their own pixels (`IMPROVEMENTS.md` §3m). The files
  cost nothing measurable.
- The reproduction trained on `tiles336_pilk`, shape `(3599, 6, 3, 336, 336)`: one 3-slice window per
  series at depth 0.5. The fork trains on 12 slices over 20–80%, with a random group of 3 per step.

`contract_audit.py` reported `slices` and `band` as PASS because it read them off `pilkwang_pixels`,
which `train_port` never imports. That is the same false-PASS error its own comment describes for
`crop_mm`. So Phase C said "what differs now is the pixels", and the difference in input was never
on the table. "Theirs" also differs in ways D3 calls "same": a report-hash holdout (not
site-grouped), a 4-seed mean, and the best epoch picked on that same holdout.
**Impact:** the 458 GB corpus was deleted on this attribution, Stage A is justified by it, and every
own-arm verdict that says "the encoder is not the lever" was measured on the centre-only input. The
0.0856 itself is real. What causes it is unmeasured.
**Fix (done in the same change):** `contract_audit.py` now reads both fields from `slot_cache`. D3 is
amended everywhere it recommends a route. `notebooks/build_sampling_test_kernel.py` isolates the
input as the single variable, with the decision rule pre-registered (`RUNBOOK.md` §2b).

### [HIGH] Stage 0 cannot find the competition data on Kaggle, and the local smoke test hides it — *fixed 09-23*
**Where:** `notebooks/build_stage0_kernel.py:78-89`, `notebooks/stage0_fixture.py:26`, `RUNBOOK.md:169`
**What:** `find_root()` scans only the **direct children** of `/kaggle/input`. This project already
measured that the competition mounts one level deeper, at
`/kaggle/input/competitions/rsna-knee-abnormality-detection` (`pipeline/preprocess.py:271`,
2026-08-07). The shipped submission notebook's `find_root` checks that path first. The fixture
builds its tree at `…/input/rsna-knee-abnormality-detection`, which is depth 1, so the RUNBOOK §2.1
smoke test passes locally for exactly the layout Kaggle doesn't use.
**Impact:** the next command in the RUNBOOK (`--limit 20` push) exits with
`no competition root under /kaggle/input`. It fails loudly, so this costs a cycle rather than a
wrong number. But RUNBOOK §2.5 maps that exact message to "competition not attached → fix
`competition_sources`", which sends the fix to the wrong place.
**Fix:** search two levels and require the marker file, as `preprocess.find_competition_root` does
(inline it, since Stage 0 deliberately imports nothing). Move the fixture root to
`input/competitions/rsna-knee-abnormality-detection` so the smoke test exercises the real depth,
and correct the §2.5 row.
**Done:** `find_root` searches two levels with a marker file and a name tie-break. `train_series/` is
the first image-tree candidate, and the fixture now uses the real layout. The new worker reproduces
every expected fixture number, and the old worker, run on the corrected fixture, exits with
`no competition root … saw ['some-other-dataset', 'competitions']`. The RUNBOOK §2.5 row is corrected.

### [HIGH] The fold partition, training targets and reference labels are unpinned, unhashed, and now exist only upstream — *guarded 09-23*
**Where:** `extractor/bench_public_labels.py:65-72`, `fusion/folds.py:73-76,145-153`, `README.md:102-122`
**What:** `download()` runs `kaggle datasets download <ref>` with **no version**. The site
fingerprints come from `kaggle kernels output zhukovoleksiy/rsna-metadata-probe`, which is also
the latest version. `folds.py` stratifies on `targets.csv` and groups on those fingerprints, so the
partition depends on both upstream artefacts. No fold file, no input hash and no dataset version is
committed. `git ls-files` has no `folds*.csv`, and the run manifests record neither the folds file
nor the label version. `data/` was deleted on 09-07, so the next rebuild pulls whatever is current
upstream.
**Impact:** one new upstream version silently changes the training targets, the neutral reference
(`lixin_gpt56`) and every fold assignment, with no error. Every bar in the RUNBOOK (0.7229 baseline,
0.7323/0.7358 controls, Gate B's 0.7591) was measured on a partition you can't then check you have
rebuilt. It is the "unseeded split" failure arriving through a download.
**Fix:** after the next rebuild, commit `data/folds.csv` and `data/folds_site.csv`. They hold only
UIDs and fold integers, and `fusion/runs_*/oof_all.csv` already commit UIDs. Write a small
`data_manifest.json` with the SHA-256 of every downloaded input, and pass `-v <version>` where the
Kaggle CLI allows it. Fold 0 can be cross-checked now: in a single-fold run,
`fusion/runs_port/oof_all.csv`'s study set is fold 0 minus unusable studies.
**Done:** `pipeline/provenance.py` holds every third-party input, and each fold file, to `provenance/inputs.json` by SHA-256. On first sight it records the hash; on a mismatch it exits and names both hashes; `--accept-new` moves the record. It is wired into `bench_public_labels --download`, `site_fingerprint.py` and `folds.py` (which also commits a copy of the partition under `provenance/`). `bench_public_labels.PINNED_VERSIONS` can fetch `owner/dataset/versions/N` when an old file has to be recovered. The sampling kernel checks its inputs against the same record before it decodes anything, and exports `provenance/` for you to commit. Tested: the first run records, a rerun matches, a changed parquet is refused, and the kernel stops before the decode on a planted mismatch. **Still open:** the record is empty until the first real rebuild, because the files that produced the 09-06 numbers were deleted with `data/`. The sampling test's fold-0 check is the only link back to them.

### [MEDIUM] `assign_series` gives `cor_2` the *third*-best coronal, not the second — *fixed 09-23*
**Where:** `pipeline/slots64.py:134-151` (`Slot("cor_2", …, rank=1)` at `:69`)
**What:** the candidate list already excludes the series `cor_1` took (`i not in used`), and then
`rank=1` skips one more. Reproduced:
```
coronals: A (30 sl, fluid), B (28 sl, non-fluid), C (12 sl, non-fluid)
→ cor_1 = A, cor_2 = C        # expected B
```
With exactly two coronals the `min(rank, len-1)` clamp hides the bug, which is why the self-test
passes.
**Impact:** every study with three or more coronal series fills `cor_2` with the shortest leftover
stack. That could be a scout or a short sequence, and `cor_2` is the slot RUNBOOK §2.4-C is already
unsure about. Stage 0's composition table gives the share of studies affected (`studies_with_ge2`
alone can't tell 2 from 3+; add a `≥3` column).
**Fix:** delete `rank` (the `used` set already means "next best"), or index into a sort that
doesn't exclude used series. Add a three-coronal case to `_self_test`.
**Done:** `rank` is removed (`used` already means "next best"). The three-coronal case is in
`_self_test`, which passes now and fails on the old code (it gave `cor_2` = the 12-slice series).
Stage 0 now also reports `studies_with_ge3` and `pct_with_3plus_*`, which say how many studies the
fix touches.

### [MEDIUM] C6 windows aren't "three neighbouring slices", and the storage argument assumes they are — *decided 09-23*
**Where:** `pipeline/slots64.py:23-27,78-113`; `MODEL.md` §3; `RUNBOOK.md:147`
**What:** windows are triples of consecutive **cache** slices, and cache slices are `pick_depths`
samples spread over 6–94% of the stack. Gaps between samples (in source slices) for an 18-slice
sagittal slot:

| stack depth | 12 | 18 | 24 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|
| gap between samples | 0–1 | 0–1 | 1–2 | 1–2 | 2–3 | 3 |

So a window's three channels range from duplicated slices on short stacks (gap 0) to about 6
slices apart on deep ones, and the spread varies per study. `MODEL.md` §3 describes neighbouring
slices. The 32 GB vs 95 GB storage argument ("consecutive windows share two of their three
slices") holds in the source stack only when stack depth ≈ n_slices / 0.88. RUNBOOK decision rule
B ("sagittal median ≥ 30 → keep the spec") is exactly where windows stop being neighbours.
**Impact:** unknown, and not known to be harmful. `pilkwang_pixels.band_indices` (`:84-85`) also
groups linspace samples rather than true neighbours, and that arm is 0.8516. But the spec, the
code and the cost argument disagree, and Stage A (~9 GPU-h) bakes one choice in.
**Fix:** decide on purpose before Stage A. Either rewrite §3 and the docstring to say "evenly
spaced triples", or cache true (c−1, c, c+1) neighbours around `n_windows` centres and re-price
the storage. Either way, record the per-study gap in the manifest.
**Done:** I chose the sampled form, the fork's convention, which is proven at 0.8516. `MODEL.md` §3 and the `slots64` docstring now describe what the code does. `slots64.source_step` gives the source-slice spacing, the manifest names it as a required per-series field for the Stage A builder, and it is in the self-test. True-neighbour windows remain the untested alternative, at 3× the cache.

### [MEDIUM] The gold leak check catches only verbatim copies, so gold-58 reads of public tables are upper bounds — *fixed 09-23*
**Where:** `extractor/bench_external.py:123-125`
**What:** `leaks_gold` fires only when a table matches gold in more than 99.9% of cells **and** is
0/1-valued. A table that blends gold in, softens it (0.05/0.95), or whose author tuned prompts or
thresholds against the 58 gold rows passes. The 58 gold rows are the only ground truth any
competitor has, so tuning on them is the likely form of leakage, not the unusual one.
**Impact:** D4 ("teacher reads gold 0.8985 → labels are not the binding constraint") and the closed
rows that compare label tables on gold-58 read third-party sources that may have been fitted to
those same 58 rows. That is §3b's selection effect (claims +0.0137, delivers −0.0034), done by
someone else. The size is unknown. The direction is optimistic.
**Fix:** add a distribution test to the leak check that compares gold rows with non-gold rows on
share of exact 0/1 values, number of distinct values and mean |p − 0.5|. The two known leaks would
have tripped it too ("0/1 there against 506 distinct values elsewhere"). Treat every
third-party gold-58 read as a ceiling, and note that in D4.
**Done:** `bench_external.gold_rows_differ` flags a table whose gold rows differ from its other rows (0/1 share +30 points, or mean |p−0.5| +0.15). Both benchmarks now print SUSPECT beside such a score and label every gold-58 read a ceiling. `bench_public_labels.py`, which scores `steven_v2` (D4's 0.8985) and had no leak check at all, now runs both checks. On synthetic data a 0.05/0.95-softened copy is flagged, while honest soft and honest all-binary tables are not. A tuned prompt remains undetectable by construction.

### [LOW] Both DataLoader workers draw identical augmentations — *fixed 09-23*
**Where:** `fusion/train_port.py:121,359-366`
**What:** `SlotDataset` creates `self.rng = np.random.default_rng(seed)` in `__init__`. With
`num_workers=2`, each worker gets a pickled copy in the same state. PyTorch reseeds the *global*
numpy RNG per worker, not Generator instances. The k-th sample on worker 0 and the k-th on worker 1
get the same gamma, gain and slot-dropout mask.
**Impact:** half the intended augmentation diversity. Probably small, but this file is the template
the C6 harness will copy.
**Fix:** create the generator lazily from `torch.utils.data.get_worker_info()` (seed + worker id),
or in a `worker_init_fn`.
**Done:** `train_port.seed_worker` seeds each worker from (seed, worker id), deterministically. `--legacy-worker-rng` reproduces the old shared stream, and the sampling kernel uses it so the input stays its only variable. Tested: legacy mode is byte-identical to the old code with 2 workers; the fix is deterministic across reruns; the workers now draw distinct streams.

### [LOW] `auc()` returns out-of-range values if a row with no label reaches it — *fixed 09-23*
**Where:** `extractor/metrics.py:17-22`; the one open door is `fusion/score_gold.py:95`
**What:** ranks are computed over **all** rows, including rows whose label is neither 0 nor 1.
`auc([1, 0, nan], [.9, .1, .5])` returns **2.0**. A NaN *prediction* returns NaN, which
`nanmean` then drops from the macro without warning. Every current caller filters incomplete rows
first, except `score_gold.gold_frame`, which keeps partially labelled rows
(`dropna(how="all")`). That is harmless only while all 58 gold rows are complete.
**Fix:** inside `auc`, keep only rows where y ∈ {0,1} and s is finite, and raise if the NaN count
isn't zero where the caller expects complete data.
**Done:** only rows labelled 0 or 1 are ranked, and a non-finite score warns. It is identical to the old function on complete labels (max diff 0 over 2,000 random draws with ties). All 23 callers pass complete 0/1 labels, so no recorded number moves.

---

## Operational (not pipeline bugs, but they gate the next step)

- ***Fixed 09-23:*** **The RUNBOOK's "only blocker" probably isn't one.** §1.1 says `kaggle.json` is "what the API
  needs". The pinned client (`kaggle` 2.2.4, via `kagglesdk.get_access_token_from_env`) reads
  `~/.kaggle/access_token` **first** and falls back to `kaggle.json` only after that. README:77 and
  `requirements.txt:17` already say this. Run `python -m kaggle competitions list -s "rsna knee"`
  with the existing token before minting a legacy key.
- **The calendar.** The final submission is **2026-10-22, 29 days out**. The PREMORTEM (09-07) put
  the plan at 6–8 weeks against 45 days, and nothing has been committed since. The RUNBOOK is 16
  days old, past its own "re-run §0.3" threshold.

---

## Verified clean

- **Gold never trains:** `train_port.py:353` removes `is_gold` from every training fold.
- **No selection on the validation fold:** fixed epoch count with the last-epoch model
  (`train_port.py:426-462`). No early stopping and no checkpoint picked by validation score.
- **Eval mechanics:** `model.eval()` and `torch.no_grad()` before OOF inference
  (`train_port.py:456-458`). The kernel's rad heads are `.eval()` under `@torch.no_grad()`
  (`build_f6_rad_kernel.py:189,195,198`).
- **Split:** scanner-site groups, integrity asserted (`folds.py:201-203`), deterministic (fixed
  seed, `sort=True` groupby, seeded tie-break). Patient grouping is not available: PatientIDs are
  unique per study (`folds.py:11-13`).
- **Large-n instrument:** excludes gold, scores paired arms on the intersection, bootstrap
  indices are deterministic (`score_oof.py:153`).
- **No fitted preprocessing crosses the split.** Windowing is per-series and there are no global
  statistics.
- **Blend:** equal-weight rank-mean over families, nothing fitted, per-arm fallback to pilkwang's
  rank for unreadable studies (`build_f6_rad_kernel.py:309-339`).
- **RadImageNet train/serve contract:** normalisation `x/127.5 − 1`, grey→3ch, 224 px, 8 slices,
  band 0.12–0.88, full frame and slot order are identical between the measurement path
  (`rad_arm.py:98-99`, `rad_pixels.py:61-67`) and the kernel (`build_f6_rad_kernel.py:89-91,249-250`),
  with SHA-256 and strict-load guards.
- **OrthoFoundation RoPE fix works.** I checked this because timm can cache RoPE embeddings at
  construction, which would make the post-hoc `periods.copy_` a no-op. In timm 1.0.28
  `vit_large_patch16_dinov3` sets `dynamic_img_size=True`, so `feat_shape=None`, nothing is cached,
  and `get_embed(shape=…)` recomputes from `periods` on every forward (`eva.py:690,879`;
  `pos_embed_sincos.py:1290-1300`).
- **`perlabel_head.py`:** masks in score space before the softmax, and all-masked rows stay
  finite.
- **No horizontal flip**, which would swap medial and lateral (`train_port.py:106-109`).
- **Fold recovery** uses no labels (`fold_recover.py:58-65`). Its upward bias from
  misassignment is documented where it is used.

## Open questions

1. Is `stevenleehans/…/llm_labels_v2.csv` still the version the recorded runs trained on? Nothing
   in the repo can answer this now. It decides whether the recorded bars still apply.
2. Did any public label table's author validate or tune against the 58 gold rows? That decides
   whether D4's 0.8985 is a ceiling or an estimate.
3. Stage 0's composition table: what share of studies have **three or more** coronal series? That
   is how many studies the `cor_2` bug affects.
