# RUNBOOK — read this first, then `PLAN-C6.md`

**Purpose: any session can pick this up cold and know the next command to type.** `PLAN-C6.md`
says *why*; this says *what to run, what to expect, and what to do with the answer.*

## ⛔ GOAL CHANGED 2026-09-24: THE GOAL IS TO WIN

Set by the owner on 09-24, superseding `PLAN-C6.md`'s "the best model, explicitly not a placing" (09-07).
Two ways to win, both verified on the live competition page 09-24:
- **Main leaderboard:** 10 prizes ($9,000 down to $5,000), ranked on the PRIVATE leaderboard. 09-23: top 0.958,
  10th 0.955, public notebooks 0.943, ours 0.914.
- **Efficiency track:** 3 prizes ($7,000 / $6,000 / $5,000). Score = (maxAUC − AUC)/(maxAUC − 0.5) +
  runtime_seconds/32400, lower wins; maxAUC is the best private score. GPU notebooks allowed. Only the
  **two selected finals** count, and they must beat the sample submission. At maxAUC ~0.96, **0.001 AUC ≈
  70 s of runtime**, so a fast ~0.93 single model beats a slow 0.958 ensemble. Ranks, not scores, show in
  the host's daily Efficiency Leaderboard notebook.
Deadlines: **entry and team merger Oct 15**, final submission **Oct 22** (23:59 UTC).

## ▶ NEXT SESSION STARTS HERE `handoff written 2026-09-25 ~02:30 PDT; updated 2026-09-26 00:40`

**⛔ 2026-09-26 — THE LEADERBOARD ANSWERED, AND IT OVERRULES THE 691.** Public LB: **Submission A (our Ortho alone) 0.901**;
**Submission B (ours at 1/3 in the public 0.943 stack) 0.936 — 0.007 BELOW the stack alone.** Both were verified to be the
scored model (gate: max |Δp| 0.00072), so this is the model, not the pipeline. What it means:
1. **The gap is the LABELS, and the 691 cannot see label effects.** ⚠️ *Corrected the same day:* the first version of this
   note said the 691 "ranks models wrongly" because `c6_ortho` beat "the frontier's CoAtNet recipe" by +0.025 there. That
   comparison was against OUR run of their recipe on OUR `steven_v2` targets (never submitted); the frontier's 0.914–0.928
   is THEIR model — gold-weighted soft labels (`dreaddevelopment/rsna-knee-labels`, gold ×3), all ~4,350 studies, epochs
   picked on gold. Architecture and cache comparisons at fixed targets remain valid on the 691 (`score_oof.py`'s own
   docstring: it "cannot arbitrate label SOURCES; an arm trained on a near-copy wins by construction" — `gpt56sol` and
   `steven_v2` agree at 0.98–1.00). What the 691 cannot do is tell us how far our TARGETS sit from the radiologist, and
   that is where the leaderboard says we lose: gold-58 reads ours 0.878 against the frontier's 0.917 (theirs gold-selected,
   so inflated) and the board's 0.950 single models credit labels. `[[measure-in-the-target-currency]]`.
2. **Blending a weaker member costs.** The equal-family rule gave a 0.901 model a third of the vote against families at
   ~0.94; `[[diversity-is-not-the-constraint]]` — screen on strength, and strength means the LB or gold, not the 691.
3. **Consequences for what is running:** Stage B keeps its pre-registered rule on the 691 — a fixed-target comparison,
   which the 691 can make — but a pass there buys at most a better model on the same labels: report its gold-58 beside
   it and check the LB before building on it. Do NOT select B
   as a final. The next lever is the LABELS (teacher → student, gold-weighted targets as the frontier used), judged on
   gold-58 and the LB. Efficiency: A at 0.901 in ~25 min is not a contender.

**⏩ QWEN — for the labels, and as a blend member `pre-registered 2026-09-26 09:40 PDT, before any result`.** The
owner's direction 09-26: use Qwen to help with the labels and as part of the blend, and make the data and pipeline strong.
*What it rests on, measured 09-26 on gold-58 (scratch bench, now `extractor/bench_qwen_labels.py`):* the public Qwen3-8B
tables are a DIFFERENT reader — Spearman with `steven_v2` **0.72** on non-gold rows (LLM-to-LLM 0.82, `DECISIONS.md`) —
and best alone at **0.8797** (`xhaitanya/qwen38b-labels-rsnaknee`, −0.0076 ± 0.011 vs `steven_v2`), far ahead on
**Fracture (0.884 vs 0.793) and Lateral OA (0.881 vs 0.830)**, behind on PF OA; its rank-mean with `steven_v2` reads
+0.0047 [−0.009, +0.019] — suggestive, and the best of 8 tables looked at, so partly selection. ⛔ **Two new public tables
carry gold** (`dhiyaneshbabu/rsna-knee-labelled-train`, `yehezkielhaganta/psuedo-labeling-knee`, 1.0000 on gold-58) —
never train on or bench them. `nartaa/rsna-knee-hpo-assets` holds gold out (not scorable); `lvthanhxun` v9 reads 0.754.

*Labels — `extractor/qwen_label.py`, tag `qwen35_9b_v1`, prompt hash `a37d356862e1d20a`.* Qwen3.5-9B (4-bit MLX) on this
Mac, the official criteria verbatim, one graded answer per finding read from the next-token distribution over six letters
(meets / probably / on the fence / present-but-below / absent / not mentioned) — no generation, 0.8 s per report, letter
mass 0.999 on the smoke. **THE RULE** (DATA.md §3's gate; the ONE change, made now: its per-label guard becomes a guard on
the frozen-six macro, because at n = 58 a true improvement breaches "no label below −0.02" by noise more often than not):
`python extractor/bench_qwen_labels.py`, exactly two candidates, once each — **C1** the Qwen table alone, **C2** its equal
rank-mean with `steven_v2` mapped onto `steven_v2`'s own values (trainable: keeps its marginal, changes only the order).
PASS = paired ≥ +0.010 with the 95% CI above 0, frozen-six ≥ −0.02; C2 preferred if both pass. The hand set and per-label
lines are diagnostics, not second chances.
- **PASS →** the next GPU run is step 5's model exactly (Ortho, Stage A, the same 3,502 studies, seed, epochs, recipe)
  with only `--labels` changed, submitted ALONE: judged against Sub A's **0.901** on the LB — same model, same pipeline,
  so the difference is the targets. Its 691 read is NOT evidence here (the 691 is scored against another LLM reading of
  the reports); its gold-58 is reported only.
- **FAIL →** `steven_v2` stays. The next label candidate is the frontier's own (`dread_soft` + gold ×3, MODEL.md §0), which
  tests "labels are the gap" without our reader.

*The blend member — a Qwen3.5-2B VLM (Apache-2.0), `fusion/qwen_vlm.py`.* The board's 54th place reports 0.950 from a
Qwen 3.5 2B VLM (vision encoder fine-tuned + LoRA on the language model, 384 px) — self-reported, a description not code,
so it is the least-evidenced piece of this plan. Built into `train_windows.py` so it inherits the corpus, the splits, the
eval and the scoring. Order: CPU wiring smoke → a ≤ 1 GPU-h bench on Kaggle (fp16 finiteness on a T4, s/study) → **ask
the owner** before the ~10 h run. Its rule is written here before that run, not after.

**✅ 2026-09-26 00:36 PDT — SUBMISSION B SUBMITTED** (ref **56573153**, kernel `rsna-knee-sub-b-ortho-blend` v1,
built from `c197d80`; hidden-test rerun pending — read it with `kaggle competitions submissions`). Its commit log
passed every gate: 8 fold-0 studies rebuilt from DICOMs on Kaggle reproduced `oof_all.csv` to **max |Δp| 0.00072**
(mean 0.00012); `[blend] ARMS: ... = 3 families`; the public publish cell's `COMPLETE`; no traceback; 3 rows. Our
arm ran on both T4s in 1.2 min for 3 studies; real DICOM decode there read **2.6–5.5 s/study/worker** (b ≈ 4, not
2.54, for A's rule). **The scheduled task did NOT do this:** it fired at 17:33 PDT and sat 7 h on its first Bash
permission prompt; it was stopped and the steps run by hand. The curve (`rsna-knee-window-curve` v1) and the Stage
B smoke v2 (`0bbd2d8`) were pushed at 00:30.

**✅ 2026-09-26 01:10 PDT — SUBMISSION A SUBMITTED** (ref **56573973**, `rsna-knee-sub-a-ortho-nall` v1, all 54 windows —
the curve's rule, below; built from `f9e7d9e`). Gate: 8 fold-0 studies reproduced to max |Δp| 0.00072; ARMS `ortho_stage_a
only`; 3 rows; no traceback. **Both B and A are pending** on the hidden-test rerun. **Stage B training** (v3) is running
since 01:01, expected ~12:00 PDT. Nothing else is queued; 3 submissions remained that day.

**What fires on its own, Fri 2026-09-25 17:20 PDT** (20 min after the weekly GPU quota resets, Sat 00:00 UTC):
the Claude desktop app's one-time scheduled task **`rsna-submit-b-curve-a`** (runs only while the app is open;
if closed, on next launch). In order, each step gated exactly as the blocks below say:
1. push **Submission B** (`rsna-knee-sub-b-ortho-blend`) and the **Stage B training run** (`rsna-knee-c7-train-ortho`,
   only if the Stage B block below says TRAINING SMOKE PASSED) — the two GPU sessions;
2. submit B if its commit log shows GATE PASSED and the 3-family ARMS line;
3. push the window curve, score it locally, pick A's window count by the pre-registered rule;
4. push and submit **Submission A** if its gate passes; 5. record everything here and in memory, commit, push.
**If it has not reported, read its run in the app's Scheduled sidebar, then do the remaining steps by hand
with the commands in the blocks below.** Everything it builds was re-tested end to end from `38459d1` on
09-25 (the submission, A and the curve kernels all pass their local fixture runs; an import-closure check now
refuses a kernel that would miss a module — that is how `slots96` was caught).

**Reading the results, next session:**
- **Leaderboard:** `.venv/bin/python -m kaggle competitions submissions rsna-knee-abnormality-detection`
  (hidden-test reruns take hours). B = our model at 1/3 in the public 0.943 stack; A = ours alone.
- **Stage B** (~11 h after its push, so ~Sat 05:00 PDT): `.venv/bin/python -m kaggle kernels output
  raahimnawaz/rsna-knee-c7-train-ortho -p /tmp/c7 --file-pattern '.*\.(csv|json|txt)$'`, then THE RULE:
  `.venv/bin/python fusion/score_oof.py fusion/runs_c6_ortho /tmp/c7/runs_ortho` (and `.../avg3`, secondary).
  Copy csv/json into `fusion/runs_c7_ortho/`, `git add -f`, record the verdict at the Stage B block. Also read
  its gold-58 line (reported only) and the per-rank checksum line. If it wins, submitting it needs
  `infer_windows.py` taught Stage B (spec c7 + `--canon` in `build_study`, 384 px, the slot/depth embedding
  from `last_meta.json`) — not built yet.
- **Next priority after Stage B: teacher → student pseudo-labels** (`DATA.md` §10b, RUNBOOK §0.3): the model
  reached label parity on gold-58 on 09-25, the board's best single models (0.950) got there through labels.
  ⚠️ judge it on gold-58 + the leaderboard, NOT the 691 (another LLM reading of the reports; it reads a move
  toward the images as a loss). Behind it: zoom windows for MCL/OA (`DATA.md` §10b), CoAtNet on Stage B as a
  second model, combined per label by the pre-registered winner-if-clear rule (`fusion/perlabel_select.py`).
- **Still the owner's calls:** cloud GPUs (compute is the constraint); a team merger before **Oct 15**.

**✅ Step 5 DONE 2026-09-24 11:20 — Stage A pays.** `rsna-knee-c6-train-ortho` (Ortho on our 64-slice
2–98% cache, 3,502 studies, 8 epochs, stopped at 99.5% of its schedule): **0.8298** on the 691 against
`ab_ortho`'s 0.8148, paired **+0.0149 ± 0.0031 (4.8σ, P 1.000), 11/12 labels up**; `avg3` 0.8304, +0.0154.
The pre-registered rule passes. **This is the model to ship**: its `runs_ortho/last.pt` sits in that
kernel's output (not downloaded). Record: `fusion/runs_c6_ortho/`. Gold-58, reported only: 0.878.
Eval cost at 54 windows: **1.53 s/study** on a T4 (the efficiency track's starting point).

The commands used to score it, for the next run of this kind:
```bash
.venv/bin/python -m kaggle kernels status raahimnawaz/rsna-knee-c6-train-ortho
.venv/bin/python -m kaggle kernels output raahimnawaz/rsna-knee-c6-train-ortho -p /tmp/c6o_out \
    --file-pattern '.*\.(csv|json|txt)$'                     # skip last.pt (~0.6 GB) until the submission needs it
.venv/bin/python fusion/score_oof.py fusion/runs_ab_ortho /tmp/c6o_out/runs_ortho          # THE RULE
.venv/bin/python fusion/score_oof.py fusion/runs_ab_ortho/avg3 /tmp/c6o_out/runs_ortho/avg3  # secondary
# then: copy the csv/json into fusion/runs_c6_ortho/ and `git add -f` it, like runs_ab_ortho
```

**Where things stand (all measured 09-23, all committed):** Stage 0 ✅ (`DATA.md` §9) · sampling test ✅
input is the main cause, +0.0441, ~half the gap (§2b) · Stage A cache ✅ built, GATE A passed (`DATA.md`
§10) · backbone A/B ✅ **ortho 0.8148 beats coatnet 0.8047, +0.0102 ± 0.0041, P 0.996** (§2c) · public
label tables ✅ closed, keep `steven_v2` (`DATA.md` §11) · C-PRE ✅ resolved (§4).

**⏸ 2026-09-24 15:00 — Submission B is BUILT and TESTED, and NOT RUN: the push hit "Maximum weekly GPU
quota of 30.00 hours reached."** Nothing was submitted. The quota resets **Saturday 00:00 UTC** (Fri 17:00
PDT). A one-time scheduled task in the Claude desktop app, `rsna-submit-b-curve-a`, fires Fri 2026-09-25
17:20 PDT and runs everything below, including A, submitting only on the gates named here. It runs only
while the app is open; if it has not reported, run this by hand, in this order:
```bash
S=/tmp/subb   # any scratch dir
# the public notebook is PINNED at the version built and tested against (sha256 2fc5a4206d76c13a), in
# gitignored data/; a fresh `kaggle kernels pull` would take whatever its author uploaded since
.venv/bin/python notebooks/build_submit_kernel.py --public-dir data/public_notebooks/bend-the-knee-2fc5a420 --out $S/subb
.venv/bin/python -m kaggle kernels push -p $S/subb                  # commit run: 3 visible studies, ~10-15 min
.venv/bin/python notebooks/build_window_curve_kernel.py --out $S/curve && .venv/bin/python -m kaggle kernels push -p $S/curve
# commit log must show "GATE PASSED" (8 fold-0 studies rebuilt from DICOMs reproduce oof_all.csv)
# and "[blend] ARMS: ... = 3 families". Only then:
.venv/bin/python -m kaggle competitions submit rsna-knee-abnormality-detection \
    -k raahimnawaz/rsna-knee-sub-b-ortho-blend -v <n> -f submission.csv -m "Sub B: ortho stage A 1/3 + public 0.943 stack"
```
What it is: `fusion/infer_windows.py` (built from `build_study`, `window_centres`, `WindowMIL` -- the
trainer's own parts) runs first in subprocesses, one per T4; then the public stack's 24 cells unmodified;
then our model joins as a **third family at 1/3** (the public stack is two families by its own
`family_mix`; §5.4's rule is an equal rank-mean over families, nothing fitted); then the public publish
cell. The weight is fixed by the rule, not tuned: nothing here can score the public stack out of fold.
Tested off Kaggle on the synthetic DICOM fixture (the gate passes on a match, exits 1 on a +0.05 offset;
the notebook's own cells run against a fake `/kaggle/input`). **The one untested risk is time:** the public
stack calls itself "comfortably in less than 9 hrs" on the hidden set and guards 8 h from its own start;
ours adds ~20-35 min in front. The commit run cannot price the hidden run.

**⏩ STAGE B — every gap in one training run `pre-registered 2026-09-25 01:40 PDT, before any result`.**
The owner asked to test all of `DATA.md` §10b's gaps and the model-side changes in the run that goes in
beside Submission B. One arm, `rsna-knee-c7-train-ortho`, against step 5's model:

| half | change | where |
|---|---|---|
| data | 96 slices (was 64), 384 px (was 336), left/right + scanner orientation canonical, a third sagittal and a second axial slot | `pipeline/slots96.py`, `build_slots64.py --spec c7 --canon`, 8 CPU shards `rsna-knee-stage-b-s{0..7}of8` |
| model | layer-wise LR decay **0.9 with a 9e-5 top** (amended 09-26, below), drop-path 0.1, a slot + depth embedding per window, both T4s (DDP), epochs fitted to a 10.3 h training budget (cap 14) | `train_windows.py` flags, `build_backbone_ab_kernel.py --corpus stage_b` |

Held fixed: Ortho, the 3,502 training studies, `steven_v2` targets, the 691 validation studies, k = 12,
8 studies per step, head lr 1e-3, and the encoder's TOTAL learning rate. **Amended 2026-09-26 00:30, before any
result:** decay 0.8 on step 5's flat 3e-5, as first written, cut the summed per-block LR 6× (7.2e-4 → 1.2e-4) and
left the patch embedding at 1.1e-7 (read off the smoke log) — that would test *less training*, not *layer-wise
decay*. Decay 0.9 with a 9e-5 top sums to 7.5e-4, step 5's budget, tilted toward the upper blocks. **The CPU smoke
of 09-25 was OOM-killed at its second step** (rank 1 SIGKILL; everything before it — the 4,407-study corpus, the
weights, the LR groups, the time probe, step 1 — ran): the smoke now checkpoints like the GPU run, DDP keeps
gradients in its buckets (`gradient_as_bucket_view`), and it scores 2 studies at k 2 to finish in minutes. **THE RULE: the package pays if `c7_ortho − c6_ortho > 0` paired on the
same 691, P(Δ>0) ≥ 0.95, primary = last epoch; `avg3/` secondary, not a second chance.** A pass says the
package works, not which half; a null or a loss ships nothing from it and says only that the package as a
whole did not help. Scored in-kernel (`score_oof.py runs_c6_ortho runs_ortho`) and again locally.
**Checked before any GPU hour:** Stage A bytes reproduced with the new builder; `canon_fixture.py` 17/17; on
the 20-study Kaggle smoke, the corrected right and left knees agree better as built than mirrored in all three
planes (coronal 0.966 vs 0.938, axial 0.953 vs 0.942, sagittal 0.997 vs 0.975); DDP on two CPU processes ends
with identical weights on both ranks. A Kaggle CPU smoke of the training kernel runs before the GPU push.
Cost: one ~11 h T4×2 session. Inference for this model needs `infer_windows.py` to learn Stage B's spec and
the embeddings — only if it wins.

**TRAINING SMOKE PASSED** — 2026-09-26 00:30–00:52 PDT, `rsna-knee-c7-train-ortho` v2 from `0bbd2d8` on Kaggle CPU: two
processes on the 4,407-study Stage B corpus, 26 LR groups (6.5e-6 embeddings .. 9e-5 top), the time probe, a full
epoch, **identical per-rank weight checksums**, sharded eval gathered into `oof_all.csv`, clean exit in 21.8 min.
**⏳ The full run was pushed 2026-09-26 01:01 PDT** (v3, built from `be9d9be`, T4×2, 10.3 h training budget, cap 14
epochs): expect it ~12:00 PDT. Read it with the commands at the top of this handoff; THE RULE is above.

**Submission A -- our model alone, at the window count the curve picks** (`build_submit_kernel.py
--ours-only --n-windows N`; the gate runs in its commit run only, since the scored rerun pays for every
second). **The rule for N, PRE-REGISTERED 2026-09-24 before the curve has a number:**

    est(n) = (AUC_54 - AUC_n) / 0.46  +  T(n) / 32400        pick argmin; within 0.001 of the min -> the larger n
    AUC_54 - AUC_n   paired point estimate on the 691, `score_oof.py curve/n54 curve/n<k>`
    T(n)             max(T_decode, T_gpu(n)); T_gpu(n) = 661 x forward_sec_per_study(n) (1,322 over two T4s)
    T_decode         1,322 x b / 4 decode workers; b = build s/study/worker from Sub B's commit log, else 2.54

**What the rule already implies, from numbers measured before the curve:** T_decode ≈ 1,322 × 2.54 / 4 ≈
**14 min**, and T_gpu(54) ≈ 661 × 1.53 s ≈ **17 min** (1.53 is the eval loop's per-study time, an upper
bound on the forward). So below ~40 windows the run is **decode-bound**, and the whole curve can buy at most
~3-4 min ≈ 0.003 AUC-equivalent. If that holds, **the next efficiency lever is the decode, not the model**:
`order_series` reads the header of every file in every series before `build_study` decodes the 64 it
keeps. Projected, not measured -- the curve's timings and Sub B's commit log will say.

**✅ MEASURED 2026-09-26 — the curve (`rsna-knee-window-curve` v1, from `0bbd2d8`) and the rule's answer: n = 54.**
The n = 54 pass reproduced the scored OOF (max |Δp| 0.0012, mean 0.0001). b = **3.8** s/study/worker (Sub B's gate
run, 8 studies, 4 workers) → T_decode = 1,322 × 3.8 / 4 = **1,256 s**.

| n | windows | Δ AUC vs n54 (paired, 681) | forward s/study (T4) | T_gpu = 661 × fwd | T(n) | est(n) |
|---|--:|---|--:|--:|--:|--:|
| **54** | 51.4 | 0 | 1.278 | 845 s | 1,256 s | **0.0388** |
| 36 | 35.3 | −0.0017 ± 0.0011 (1.5σ) | 0.788 | 521 s | 1,256 s | 0.0425 |
| 27 | 25.7 | −0.0056 ± 0.0015 (3.9σ) | 0.734 | 485 s | 1,256 s | 0.0510 |
| 18 | 16.1 | −0.0142 ± 0.0026 (5.4σ) | 0.363 | 240 s | 1,256 s | 0.0697 |
| 12 | 11.6 | −0.0483 ± 0.0044 (10.9σ) | 0.324 | 214 s | 1,256 s | 0.1438 |

**Decode-bound at every n, including 54**: on two T4s the model needs ~14 min, the DICOM read ~21 min. Fewer
windows cost AUC and buy no time, so the rule picks **n = 54** — Submission A runs every window. The efficiency lever
is the decode (`order_series` reads every header; 96-slice Stage B decodes 1.5× more), then the model.

⛔ **The "6 min inference on 2×T4" quoted for the public notebook in §0.3 was its COMMIT run on the 3
visible studies** (its log: 370 s). It is not an efficiency-track entry: at several hours on the hidden
set its runtime term alone is ~0.5-0.9. The efficiency track is a race between FAST models, and our
curve (`fusion/window_curve.py`, pushed with the submission above) is the first measurement for it.

⛔ **Every GPU kernel here has had two T4s and used one.** The public notebook asserts
`torch.cuda.device_count() == 2` on `machine_shape: NvidiaTeslaT4`, the shape every builder here requests;
`train_windows.py` runs on `cuda:0`. The submission and curve kernels use both. For training, the cheapest
use is a second seed on `cuda:1` in the same session (tuning plan item 3) -- CPU and shard I/O are the
unmeasured limits, so bench it first.

The traps found while writing it, kept because they apply to every inference kernel:
- Competition code notebook, **internet OFF**, T4, 9 h limit. Output `submission.csv` =
  `StudyInstanceUID` + the 12 label columns, probabilities (`data/sample_submission.csv`). The visible
  `test.csv` has 3 studies; the hidden test replaces it at scoring.
- Build each test stack with **`pipeline/build_slots64.build_study`** — the exact training code — over
  `test_series/<study>/<series>/` with `test_series.csv` (`Anatomical_Plane`, `Fluid_Sensitive`), span
  from the cache manifest (2–98%). Score **every** within-slot window (54 on a full study).
- Weights: attach `raahimnawaz/rsna-knee-c6-train-ortho` as a `kernel_source`; `runs_ortho/last.pt` is an
  fp16 state dict of `WindowMIL`. Rebuild with `timm.create_model("vit_large_patch16_dinov3",
  pretrained=False, num_classes=0, img_size=336, global_pool="token")` — no download needed.
- ⛔ **RoPE periods are NOT in last.pt** (a non-persistent buffer). After loading, set
  `m.rope.periods.copy_(m.rope.periods.bfloat16().float())` — Ortho's periods are exactly timm's rounded
  to bf16 (checked 09-24 on the checkpoint; 16 values `1.0, 1.335938, … 75.0`). Forgetting this runs
  cleanly and scores wrongly (§9h). Runs from `train_windows.py` after 09-24 also write `last_meta.json`
  with the periods, so later kernels read them instead. **Size, measured 09-24 on this checkpoint:** the
  RoPE embedding moves by up to 0.0079 and the output probabilities by at most **8.5e-5** (one window of
  noise input) -- real, and small for this model. `infer_windows.py` applies it and asserts it anyway.
- Normalisation: ImageNet mean/std, applied in `WindowMIL.prep` (not in last.pt; non-persistent).
- Then Submission A = our model alone (first LB number for a model trained here); Submission B = our
  model rank-meaned into the public 0.943 stack by family, nothing fitted (§5.4, §2z).

**Budget:** ~20.5 GPU-h used this week before step 5's ~9.4 h — the week is about spent. At most **2 GPU
sessions at once**. After the submissions, §0.4's Phase 2 order: retrain on all studies · second seed ·
coatnet on Stage A · dinov3l · 384 px.

**Step 5 ran long, and it does not void the read** (owner's live log, 09-24): **2.72 s per 2-study step on
Stage A vs 2.27 in the A/B**, so **76 min/epoch, not 67** — likely I/O across the four shard mounts. The
10.5 h budget stopped it at **epoch 7 iter 1,676 of ~1,751 = 99.5% of the OneCycle schedule** (~3,485 of
3,504 optimiser steps), where the LR is ~1e-4 of peak; `avg3` includes that epoch. Record it as "stopped at
99.5%", not 8/8. **Price Stage A runs at 76 min/epoch**: 3,502 studies → ≤ 7 epochs per session; all
~4,360 studies → ~95 min/epoch → ≤ 6. Speeding up the shard reads is worth doing before the next long run.

### ORTHO HAS NOT BEEN TUNED — the tuning plan `added 2026-09-24`

The A/B ran Ortho on the frontier's CoAtNet recipe **unchanged** (flat lr 3e-5 on all 24 blocks, no
layer-wise decay, no drop-path, intensity-only augmentation, 2.5D three-slice windows) — correct for a
fair A/B, not tuned for a 303 M ViT-L. Now that it won, tuning is the lever. Every accuracy experiment is
a full ~9–10 h T4 run and ~100 T4-h remain to Oct 22, so **few, high-prior changes, each pre-registered and
paired on the 691** — and the seed floor is unmeasured, so a +0.005 cannot yet be told from luck.

**In order:**
1. **FREE, no retraining — the efficiency curve.** ⏸ *Built 09-24 (`fusion/window_curve.py`,
   `notebooks/build_window_curve_kernel.py`, n = 54/36/27/18/12, both T4s, ~0.5 GPU-h), waiting on the
   quota with Submission B; score it locally with `score_oof.py curve/n54 curve/n<k>`.* With step 5's (or `ab_ortho`'s) saved weights, score
   the 691 at **fewer windows per study** (54 → 36 / 27 / 18, evenly spaced within slots) and log runtime
   per study. That picks the efficiency entry's operating point (0.001 AUC ≈ 70 s, banner above) at zero
   GPU training. Also try `torch.compile` + fused SDPA attention for free speed at identical output.
2. **Next GPU run — layer-wise LR decay + drop-path.** LLRD (e.g. decay 0.75–0.85 per block from the
   head down, so early knee features are kept) and `drop_path_rate` 0.1–0.2 — the standard ViT-L
   fine-tuning pair, strongest prior of any accuracy change. Needs a `--llrd` / `--drop-path` flag in
   `fusion/train_windows.py`. One run; rule: beats its baseline paired, P ≥ 0.95.
3. **Alongside it — a second seed of the current recipe.** Measures the seed-to-seed floor (PREMORTEM §4)
   so every later delta is readable, and gives a 2-member ensemble for free.
4. **If more compute is added** (cloud GPUs — an open question for the owner): **single-slice input**
   (each window = one slice as R=G=B, which is what Ortho was pretrained on, vs our three different slices
   in the channels); **light spatial augmentation** (small rotation/scale, NO flips — laterality is
   signal); **lower resolution 336 → 224** (441 → 196 tokens, ~2.3× faster, efficiency); **distil Ortho
   into a small fast student** (ViT-S/B or CoAtNet-0) trained on Ortho's predictions — the standard
   efficiency-track route.

**Combining our OWN models per label — measured 2026-09-24** (`fusion/perlabel_select.py`, c6_ortho +
ab_coatnet on the 681 scorable fold-0 studies; each choice made on a random half, scored on the other, 1,000
splits). Plain average **0.8268 < Ortho alone 0.8298**: averaging in a clearly weaker model costs. "Use each
model only where it is strongest" **0.8289, no better than Ortho alone** — twelve picks on ~340 studies are
noisy and it drops the averaging gain on close labels. "Verification" (Ortho, averaged only where the two
disagree) **0.8279, also below Ortho alone** — for AUROC there is no decision to verify, only another blend.
**The one rule that helped: take the winner where it is clearly ahead (|Δ| > 0.03 on the choosing half,
threshold fixed before the run), average where they are close: +0.0022 ± 0.0019 over Ortho alone, better in
88% of splits (~1.2σ) — suggestive, not established.** It ends up as Ortho alone on ACL, Lateral Meniscus,
Effusion, Synovitis and an average on MCL, the three OA labels, Contusion and Fracture. Pre-registered for
when a Stage A CoAtNet exists: same rule, same threshold, re-measured before it is used. It cannot be
applied between our model and the public stack: their members trained on these 691 studies.

**Open questions for the owner, asked 09-24, still unanswered:** rent cloud GPUs (~100 A100-h ≈ $150–250;
training anywhere is allowed, only inference must run on Kaggle)? What Mac (chip, memory) — could it take
evaluation or small runs? Open to a team merger before the **Oct 15** deadline?

**Local env:** `.venv` (Python 3.12) has kaggle, torch, timm, pydicom, opencv. `data/` is gitignored and
holds `train.csv`, the test CSVs, public label tables, `stage0/`, `stage_a/*/manifest.json` + `slots.csv`.

> ⛔ **Read [`PREMORTEM.md`](PREMORTEM.md) before Stage A.** Four defects in the plan are already
> known: Stage C is priced off a 6-tile benchmark and written for 54 windows (**5–10× over, and
> over the 12 h session cap**); Gate B's bar is built on `runs_port` 0.7323, which §3y states is
> **not a valid control**; the noise floor was never measured, so both paired gates may be
> unreadable at one fold; and no stage closes the teacher→truth gap, which caps the whole plan at
> ~0.8985. Stage 0 is unaffected — run it. Do not fund Stage A/B/C on the prices in `PLAN-C6.md`.

> ⛔ **D3 was amended on 2026-09-23** (`DECISIONS.md`). The NIfTI files are not what cost 0.085:
> the fork's own weights score the same on them as on their DICOMs (§3m). What differed is the
> input: our port trained on **one 3-slice window per series, at the centre**, while the fork trains
> on **12 slices across 20–80%**. ✅ **§2b measured it 09-23: +0.0441 ± 0.0058, 12/12 labels — the input
> is the main cause, about half the gap.** Our own training on public-style input reads 0.8037.

> ⛔ **The frontier's training recipe is public (read 2026-09-23) — `MODEL.md` §0.** A single
> CoAtNet on our own `slots64` geometry reads **LB 0.928**, above our whole blend. C6 copied its
> input and not its training. Read §0 before funding Stage B or C as written.

Last updated **2026-09-24** (handoff, Submission B, the public stack's runtime, two T4s); D3 amended and
§2b added **2026-09-23**; §0.3 re-pulled and §1.1 confirmed (auth works on `access_token`) **2026-09-23**. If today is much later than that, re-run §0.3 before trusting any
claim here about the outside world — five beliefs about it have already expired in this project,
one after 24 hours.

---

## 0. State of the world

### 0.1 What exists right now

| file | status | verified how |
|---|---|---|
| `PLAN-C6.md` | current plan, Stages 0–D | — |
| `notebooks/build_stage0_kernel.py` | ✅ **ran 09-23, all 4,407 studies — `DATA.md` §9** | smoke-tested against a synthetic fixture — at the real mount depth since 09-23 (the depth-1 fixture had hidden a `find_root` that Kaggle would fail) |
| `notebooks/stage0_fixture.py` | reproducible smoke test | r recovered to 0.01 on all 3 planes |
| `pipeline/slots64.py` | ✅ **Stage 0 kept all three assumptions (09-23, `DATA.md` §9)** | `python pipeline/slots64.py` self-test (now includes the 3-coronal case; `cor_2` bug fixed 09-23) |
| `fusion/perlabel_head.py` | ready, unused until Stage B | `python fusion/perlabel_head.py` self-test |
| `fusion/train_port.py` | has `OrthoNet`, never run | ✅ C-PRE resolved 09-23: its ImageNet constants are Ortho's own, see §4 |
| `notebooks/build_sampling_test_kernel.py` | **full run pushed 09-23 (T4) — §2b**; smoke passed all 3 checks | whole path smoke-tested off Kaggle on a synthetic DICOM fixture |
| `fusion/contract_audit.py` | fixed 09-23: `slices`/`band` read off the path `train_port` trains on | — |

### 0.2 What is NOT done

> ⛔ *Stale since 09-24 — §0.1 and §0.2 describe 09-23 morning. There is a C6 number (0.8298, step 5),
> `data/` exists locally again (CSVs, label tables, Stage 0/A manifests), and the handoff at the top is
> the current state.*

- **The first GPU runs were pushed 2026-09-23** (§2b sampling test; §2c backbone A/B smoke). No arm
  has been trained on a C6 cache. There is no C6 number yet.
- The local corpus was deleted 2026-09-07. **This repo is the only copy of the project.**
- `data/` does not exist. Anything reading `data/*.csv` will fail until a stage recreates it.

### 0.3 Facts with an expiry date

Re-measure before citing. Last pulled **2026-09-23**:

- Leaderboard: **4,244 teams**, top **0.958**, 10th (prize line) **0.955**, ours **0.914** @ rank
  2184. (09-07: 3,289 / 0.954 / 0.950 / rank 1446.)
- ⛔ **The public notebook tier is now 0.942–0.943** — read off the Kaggle pages, not the titles:
  `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` **0.943** (Apache 2.0; ~~6 min
  inference on 2×T4~~ **6 min was its commit run on 3 studies; the hidden run is "under 9 hrs",
  guarded at 8 h** — corrected 09-24 from its code and log; DINOv2 + two CoAtNet readers, training notebooks public as "RSNA Knee E11/E13
  train"), `romantamrazov/rsna-knee-dinosaur-v5` **0.943**, `andreluizpedroso/rsna-knee-tony-rank-ensemble`
  **0.942**. On 09-07 it was 0.917–0.922 and "leaders at 0.942+ publish nothing". **The free tier
  now sits +0.029 LB above our best submission.** Every member in §5.4 predates this.
- Public-source survey re-run 09-23 (`kaggle datasets list -s "rsna knee" --sort-by updated`).
  ✅ **Label tables benched 09-23** (`DATA.md` §11): `dreaddevelopment/rsna-knee-labels` (holds gold
  out, unscorable on it), `riadmohamed42/jev-knee-labels`, `hosseini7798/rsna-knee-label-extraction-jev`
  — none beats `steven_v2` unfitted. `tuyennguyenthicam/rsna-knee-expert-58` is **not a label table**:
  it holds `test.csv`, `test_series.csv` and test DICOMs. Not yet benched: weights
  `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` (1,895 downloads), `miguelbeta/rsna-knee-dinov2-weights`;
  caches `crtproduction/rsna-knee-cache-v3-*` (384×12, 320×9), `lvthanhxun/rsna-knee-v9-cache`.

- **Board intelligence, "best single model" discussion, read 2026-09-25 (self-reported public LB, single
  fold unless stated):** CoAtNet 2.5D at **224 px → 0.950** (36th: "my OOF predictions were already closer to
  the radiologists than the extracted labels... bigger models weren't helping... the labels were the
  bottleneck"); **Qwen 3.5 2B VLM**, LoRA on the language model + vision encoder fine-tuned, 384 px → **0.950** in
  ~3.5 h (54th); CoAtNet 288 full-train 0.938 → **0.949 from label work alone** — several weak-label sources
  (public, API, local) train teacher models, the teachers' predictions become pseudo-labels, a final model
  trains on those, "different teachers help different targets", API cost under $5 (71st); 7th place: 0.942 at
  288 px, "not sure resolution is such a big driver", OOF pseudo-labels correlate with LB; CoAtNet 384 0.937,
  "fusion is ineffective" (85th); small ResNet 224 → 0.936, +0.015 from combining labels (64th). **Read: the
  0.928 → 0.950 single-model gap is labels (teacher → student), not resolution or backbone size.** Our model
  reached label parity on gold-58 on 09-25 (`DATA.md` §10b), the stated precondition.

### 0.4 The plan as of 2026-09-23 — supersedes the stage order in `PLAN-C6.md`

What changed it: the frontier's training recipe is public and puts a **single** model at LB 0.928
(`MODEL.md` §0); C-PRE is resolved; Stage 0 kept `slots64`. So Stages B and C as written (a
DINOv2-small encoder, then a 50–160 GPU-h ViT-L ablation) are replaced by **the frontier recipe with
a backbone A/B** (§2c), which is cheaper and starts from a measured 0.928.

| step | what | GPU | status |
|---|---|--:|---|
| 1 | Stage 0 — measure the data | 0 | ✅ `DATA.md` §9 |
| 2 | §2b sampling test — why `train_port` lost | ~5 h | ✅ **input is the main cause: +0.0441 ± 0.0058, 12/12 up, ~52% of the gap** |
| 3 | §2c backbone A/B on the CC0 corpus, 9 epochs: `ortho` vs `coatnet` | ~15 h | ✅ **ortho wins: 0.8148 vs 0.8047, paired +0.0102 ± 0.0041, P 0.996** — the rule passes |
| 4 | Stage A — build the **64-slice, wide-span** cache from DICOMs (the 0.928 geometry) | 0 (CPU decode) | ✅ **built 09-23: 4,407 studies, 0 errors, 0 decode failures, slot fill = Stage 0 to 3 dp, GATE A passed** (`DATA.md` §10); attach `raahimnawaz/rsna-knee-stage-a-s{0..3}of4` |
| 5 | Train the A/B winner (**ortho**) on the Stage-A cache on **all** non-fold-0 studies | ~11 h | ✅ **0.8298, +0.0149 ± 0.0031 over ab_ortho, 11/12 up — Stage A pays** (`fusion/runs_c6_ortho/`). The all-studies retrain moves to Phase 2 |
| 6 | Inference kernel for our own model; blend it into the public 0.943 stack, rank-mean by family | ~1–2 h/sub | after 5 |
| — | `dinov3l` arm — explains *why* ortho won or lost | ~10 h | next quota week |
| — | Label tables (dreaddevelopment soft labels, JEV) — leak-check, bench | 0 (CPU) | ✅ **closed 09-23: none beats `steven_v2` without being fitted to gold; the per-label pick transfers at −0.0002 ± 0.0090** (`DATA.md` §11, `DECISIONS.md`). Keep `steven_v2` as targets |

**Constraints that shape the order:** 30 GPU-h/week; **at most 2 GPU batch sessions at once** (hit on
09-23); 12 h per session; deadline **2026-10-22**.

---

## 1. Prerequisites — probably no blocker at all

### 1.1 Kaggle credentials — probably already there

`~/.kaggle/` holds an `access_token` (38 bytes) and no `kaggle.json`. **That is the file the current
client wants.** This section used to say a `kaggle.json` was required, which was wrong. `kaggle`
2.2.x authenticates with `~/.kaggle/access_token` (or `$KAGGLE_API_TOKEN`) **first** and only falls
back to `kaggle.json` after that (`kagglesdk.get_access_token_from_env`, checked 2026-09-23 against
2.2.4). README and `requirements.txt` already said so.

Test it with §1.2's last line. Only if that fails (an expired token, most likely) make a new one:

```bash
# kaggle.com -> avatar -> Settings -> API -> "Generate New Token"; paste the token string into:
printf %s 'KGAT_...' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

### 1.2 A Python env with the `kaggle` package

The project venv was deleted with the corpus. Any 3.12 env works; Stage 0 needs `kaggle` only to
*push*, and nothing else locally.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -q kaggle
.venv/bin/python -m kaggle competitions list -s "rsna knee"   # proves auth works
```

### 1.3 What Stage 0 does NOT need

Stage 0 declares `dataset_sources: []` and `enable_internet: false` **deliberately** — it imports
nothing from `raahimnawaz/rsna-knee-code`, so the stale code Dataset is **not** a blocker for it.
That refresh becomes a blocker at Stage A, not before. See §5.1.

---

## 2. STAGE 0 — run this first `CPU only · no GPU quota · minutes`

### 2.1 Smoke it locally (optional, ~10 s, catches arithmetic errors not crashes)

```bash
.venv/bin/pip install -q pydicom pandas numpy
.venv/bin/python notebooks/stage0_fixture.py          # writes /tmp/s0fix/input
python3 - <<'EX' > /tmp/s0fix/work.py
import re, pathlib
print(re.search(r"WORK_SRC = r'''(.*?)'''", pathlib.Path("notebooks/build_stage0_kernel.py").read_text(), re.S).group(1))
EX
KAGGLE_INPUT=/tmp/s0fix/input KAGGLE_WORKING=/tmp/s0fix/out .venv/bin/python -u /tmp/s0fix/work.py
```

**Expected:** `r_median` exactly **10.00** Sagittal, **10.00** Coronal, **8.00** Axial;
`sample_n = 10` of 11 series (the 2-slice series is correctly excluded);
`pct_with_2plus_sagittal = 50.0`, `pct_with_2plus_coronal = 25.0`, both `pct_with_3plus_*` = `0.0`;
`root:` ends in `competitions/rsna-knee-abnormality-detection` and `images:` in `train_series`.
Anything else means the reader changed and the arithmetic is no longer trustworthy.

### 2.2 Smoke it on Kaggle — 20 studies, ALWAYS do this first

```bash
.venv/bin/python notebooks/build_stage0_kernel.py --out /tmp/s0 --limit 20
.venv/bin/python -m kaggle kernels push -p /tmp/s0
.venv/bin/python -m kaggle kernels status raahimnawaz/rsna-knee-c6-stage0
```

Every cache build in this project's history died hours in on something a 20-study run would have
caught. The smoke costs about a minute.

**Check in the log before going further:**
1. `root:` names the real competition mount, not a stray directory.
2. `train_series.csv (N, M) cols=[...]` — `Anatomical_Plane` and `Fluid_Sensitive` are present.
   If they are not, **stop**: §0.3's composition table cannot be built and no plane may be
   inferred by a rule here (`slot_cache.py:175`).
3. `n_errors` is 0 or near it.

### 2.3 Run it in full

```bash
.venv/bin/python notebooks/build_stage0_kernel.py --out /tmp/s0          # no --limit
.venv/bin/python -m kaggle kernels push -p /tmp/s0
# when Complete:
.venv/bin/python -m kaggle kernels output raahimnawaz/rsna-knee-c6-stage0 -p data/stage0/
```

Outputs: `stage0_series.csv` (one row per series), `stage0_spacing.csv` (the 400-series true-spacing
sample), `stage0_summary.json` (the aggregates).

### 2.4 What to do with the answer — decision rules, fixed in advance

**A · `r_median_overall` — the 3D question.** The threshold was set in `PLAN-C6.md` §2 *before*
this number existed. Apply it as written; do not renegotiate it against the number you got.

| result | consequence |
|---|---|
| **r ≥ 4** | isotropic 3D is **contraindicated**. Triad and MedicalNet-class encoders are excluded — they require isotropic 1 mm resampling, which would interpolate data never acquired. A 3D arm, if ever built, uses nnU-Net-style anisotropic kernels `(1,3,3)` with in-plane-only pooling. Deferred to Stage E. |
| **r < 4** | an isotropic 3D arm is admissible; §2's exclusion is void and Triad returns to the table **on anatomy grounds only** — it is still brain/breast/prostate with no MSK evaluation. |

Also compare `true_spacing_mm_median` against `tag_thickness_mm_median`. If they differ, there is
an inter-slice **gap**, and any downstream code using `SliceThickness` as the sampling interval is
wrong. Record the delta in `DATA.md`.

**B · `stack_depth_by_plane` — does `slots64.py` survive?**

| result | consequence |
|---|---|
| sagittal median **≥ 30** | 18 slices over 6–94% is a real sub-sample. Keep the spec. Its windows then span ~3+ source slices (2 × `slots64.source_step`), not neighbours (`MODEL.md` §3, deliberate). |
| sagittal median **20–30** | 18 slices is most of the stack; `pick_depths` repeat-padding becomes load-bearing. Keep, but check the p10. |
| sagittal median **< 20** | ⚠️ the spec asks for more slices than exist. **Reduce the slot counts** to the p10, or a large share of studies gets padded repeats that teach the encoder nothing. |

**C · `slots64_assumption` — the claim I made without evidence.**
`slots64.py` assumes two usable sagittals and two coronals per study.

| result | consequence |
|---|---|
| `pct_with_2plus_coronal` **≥ 60** | keep `cor_2`. |
| **< 60** | drop `cor_2` and redistribute its 8 slices, or most studies carry a permanently masked slot — 8 slices of zeros per study, paid at every forward pass. |

Same test for `sag_2` against `pct_with_2plus_sagittal`.

**D · Write it down.** Add the three tables to `DATA.md` under a new `## 7. C6 Stage 0` with the
date. **No AUC is produced or quoted at this stage** (§3b, `never-select-on-a-tiny-set`).

### 2.5 Failure modes seen before in this project

| symptom | cause | fix |
|---|---|---|
| kernel "Complete" but no output files | worker died; parent still exited 0 | check the log tail for `exited <rc>` |
| `no competition root under /kaggle/input` | competition not attached — **or** a root search that only looks one level down (the mount is `/kaggle/input/competitions/<slug>`; fixed in Stage 0 on 09-23) | the log lists what it saw: if `competitions` is there, it is the search; otherwise `competition_sources` in `kernel-metadata.json` |
| plausible table, wrong numbers | reader trusted filename order | the fixture catches this — filenames are shuffled UIDs |
| queue wait then instant import error | stale code Dataset | not applicable to Stage 0 (`dataset_sources: []`) |

---

## 2b. THE SAMPLING TEST — why our own models lose `one T4 kernel · ~4–6 GPU-h · smoke on CPU first`

**The question.** Everything trained here scores 0.69–0.77 and everything downloaded 0.85. D3 put
the last 0.0856 of that on the NIfTI files, and the record contradicts it (banner above). This run
changes **one variable**: `train_port.py`, fold 0, every setting as in `runs_port_repro`, but fed
the fork's own pixels (12 slices over 20–80%, built by the fork's own notebook cells from the
competition DICOMs). It is scored **paired** against `runs_port_repro` on the same 691 studies. The
builder's docstring has the full design and the known residue.

```bash
# 1. smoke -- CPU, no GPU quota, 64 + 16 studies, one epoch
.venv/bin/python notebooks/build_sampling_test_kernel.py --out /tmp/st --smoke
.venv/bin/python -m kaggle kernels push -p /tmp/st
.venv/bin/python -m kaggle kernels status raahimnawaz/rsna-knee-sampling-test
```

**Check in the smoke log before going further:**
1. `scratch candidate …: N GB free`. The full run's cache is **~29 GB** and it refuses below **~37 GB**
   free (a 10% margin plus 5 GB). If no candidate has that, the full run refuses at its first cell. Nothing in this repo has measured a
   Kaggle machine's disk, so this line is new information.
2. `partition check: 691/691 control studies are in the rebuilt fold 0`. The smoke runs the real
   check. Anything less means an upstream label table or header parquet changed since 09-06.
   **Stop there**: the paired read would be contaminated.
3. `pixel contract = the members' own: {… 'CACHE_SLICES': 12 …}` and `SMOKE: the whole path ran`.

```bash
# 2. the real run
.venv/bin/python notebooks/build_sampling_test_kernel.py --out /tmp/st
.venv/bin/python -m kaggle kernels push -p /tmp/st
# when Complete:
.venv/bin/python -m kaggle kernels output raahimnawaz/rsna-knee-sampling-test -p fusion/runs_forksample_kaggle/
```

**✅ RESULT, 2026-09-23** (`fusion/runs_forksample_kaggle/`): `runs_forksample` **0.8037** against
`runs_port_repro` **0.7596** on the same 681 scorable studies, paired **+0.0441 ± 0.0058** (7.6σ,
P 1.000), **12/12 labels up** — largest on MCL (+0.078), Baker's (+0.070), Fracture (+0.085).
The pre-registered call is **"THE INPUT IS THE MAIN CAUSE"** (Δ ≥ +0.043). Read it with its margin:
+0.0441 clears the line by 0.2σ, so the input explains **~52% of the 0.0856 gap** — the largest
single piece, not all of it. The remaining **~0.042** (0.8037 vs pilkwang 0.8453) is training and
evaluation: one seed vs four, last epoch vs best-on-holdout, site-grouped folds, label table.
Consequences as fixed below: C6's coverage direction is right (Stage A is built); **Gate B's control
is this run's 0.8037, not `runs_port`**.

**What to do with the answer. The rule was fixed in the builder's docstring before the run.** The
verdict is in `sampling_test_verdict.json`:

| paired Δ vs `runs_port_repro` | consequence |
|---|---|
| **≥ +0.043** (half the gap) | **The input is the main cause.** Our training reaches public strength on public-style input. C6's coverage direction is right for a reason D3 never gave. Fund Stage A, and build its Gate B control from this run rather than `runs_port` (PREMORTEM §3). |
| **≤ +0.015** | **The input is not the main cause.** What remains is training and evaluation: site-grouped folds, one seed vs four, last epoch vs best-on-holdout, the label table. **Do not fund Stage A on coverage grounds.** |
| between | Both contribute. Next, split the fork's reader from its sampling. |

**Also commit what the run exports under `provenance/`** (`inputs.json` and `folds_site.csv`) into the
repo's `provenance/`. That is the first hash record of the inputs, and the fold partition every later
gate is held to (AUDIT.md). The kernel refuses to run on inputs that differ from a committed record.

⚠️ **Do not compare the absolute number with pilkwang's 0.8453.** That figure is a 4-seed mean, and
per the fork's notebook its holdout is not site-grouped and its best epoch is picked on that same
holdout. The bar is our own run.

---

## 2c. THE BACKBONE A/B — OrthoFoundation's first fair test `one T4 kernel per arm · bench first`

**The question.** OrthoFoundation was only ever run frozen, at 160 mm, on the old 6-tile cache (C-5,
0.7710 — still the best encoder number trained here). C-PRE is resolved (§4), so nothing blocks a
fair test. This runs it **fully fine-tuned, in the frontier's own recipe (`MODEL.md` §0), on the
frontier's public CC0 corpus**, with the encoder as the only variable. Harness:
`fusion/train_windows.py`; kernel builder: `notebooks/build_backbone_ab_kernel.py` (the rule is in
its docstring, written before any arm had a number).

| arm | encoder | answers |
|---|---|---|
| `ortho` | OrthoFoundation-L | the arm |
| `dinov3l` | stock `vit_large_patch16_dinov3.lvd1689m` | `ortho − dinov3l` = what knee pretraining is worth |
| `coatnet` | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` | `ortho − coatnet` = whether Ortho is the encoder to use |

Same 2,871 training studies (seed 20260923), `steven_v2` targets, and the **same 691 fold-0 studies**
as `runs_port_repro` and the sampling test, scored by `score_oof.py`.

```bash
# 1. smoke -- CPU, 16 + 6 studies, one epoch (pushed 2026-09-23)
.venv/bin/python notebooks/build_backbone_ab_kernel.py --arch ortho --out /tmp/ab --smoke
.venv/bin/python -m kaggle kernels push -p /tmp/ab
# 2. bench -- T4, ONE full epoch, scored: prices the full run from a measurement (PREMORTEM §1)
.venv/bin/python notebooks/build_backbone_ab_kernel.py --arch ortho --out /tmp/ab --bench
# 3. full -- epochs chosen from bench.json so the run fits the 12 h cap with eval
.venv/bin/python notebooks/build_backbone_ab_kernel.py --arch ortho --out /tmp/ab --epochs N
.venv/bin/python -m kaggle kernels output raahimnawaz/rsna-knee-backbone-ab-ortho -p fusion/runs_ab_ortho/
```

**Check in the smoke log:** `ortho: strict load, sha256 ok, RoPE periods = theirs`; `partition
check: 691/691`; `corpus holds 4407 studies`; `val windows per study` median near 34; a finite loss.
**Read from the bench:** `bench.json` → `train_epoch_min_measured`, `peak_cuda_gb`,
`eval_sec_per_study`. Epochs for the full run = what fits in ~10 h with eval, same for every arm.

**Bench, measured 2026-09-23** (`fusion/runs_ab_ortho_bench/`, T4, 2,871 studies, k 12, bs 8 as 2×4):
**61.8 min per epoch** (358 steps, 2.56 s per 2-study step, 0.107 s per window), peak **6.9 GB**,
eval **0.87 s/study** (691 val ≈ 10 min; `avg3` another 10), 73 min wall in total. After ONE epoch
Ortho reads **0.7508** against `runs_port_repro`'s 30-epoch 0.7596, paired **−0.0088 ± 0.0071**
(1.2σ, P 0.10) — level after 1/30th of the training, but a single compressed schedule, **not the A/B**.
Per label it is up on MCL (+0.050), Effusion, Synovitis, Fracture and down on Baker's (−0.095), PF OA,
Medial Meniscus, ACL. Gold-58 0.762, reported only.
**Epochs for the full runs: 9, every arm** — 9 × 62 min + ~25 min eval + setup ≈ 10 h inside the 12 h
cap; 10 epochs leaves under an hour of margin. `dinov3l` has Ortho's architecture, so the same cost.

**✅ RESULT, 2026-09-24** (`fusion/runs_ab_ortho/`, `fusion/runs_ab_coatnet/`; 9 epochs each, identical
2,871 training and 691 validation studies, verified byte for byte):

| read (paired, 681 scorable of 691) | Δ | P(better) |
|---|--:|--:|
| **`ortho` − `coatnet`, last epoch — THE RULE** | **+0.0102 ± 0.0041** (2.5σ) | **0.996** |
| `ortho` − `coatnet`, `avg3` (secondary) | +0.0078 ± 0.0041 | 0.975 |
| `ortho` − sampling test (0.8037) | +0.0112 ± 0.0048 | 0.992 |
| `coatnet` − sampling test | +0.0010 ± 0.0046 | 0.565 |
| `ortho` − `runs_port_repro` (0.7596) | +0.0553 ± 0.0060 | 1.000 |

Macros: **ortho 0.8148**, coatnet 0.8047 (avg3 0.8150 / 0.8071). **Pre-registered call: Ortho is the
encoder to use.** Per label Ortho leads on Lateral Meniscus **+0.074**, ACL +0.044, Effusion +0.043,
Synovitis +0.036 and trails on Fracture −0.026, Lateral OA −0.024, PF OA −0.022 (7/12 up). Gold-58,
reported only: 0.857 vs 0.859. Cost: 55 vs 41 min/epoch, 8.6 vs 6.4 h. **Caveats, stated with the
result:** one seed per arm and the seed-to-seed floor is unmeasured (PREMORTEM §4) — the ±0.0041
is study resampling only; equal epochs, not equal compute (CoAtNet could have run ~12 in Ortho's
time). `ortho − dinov3l` (why it won: knee pretraining, or ViT-L) is still open.

**Rule (pre-registered):** knee pretraining pays if `ortho − dinov3l > 0` paired at P ≥ 0.95; Ortho
is the encoder to use if `ortho − coatnet > 0` paired at P ≥ 0.95; neither is a null, recorded as
one. Primary = last epoch; `avg3/` is secondary and not a second chance. One seed per arm.

**Step 5's rule, PRE-REGISTERED 2026-09-24 before its result** (`rsna-knee-c6-train-ortho`: our 64-slice
2–98% cache, 3,502 studies, 8 epochs): **Stage A pays if `c6_ortho − ab_ortho` > 0 paired on the same 691
studies, P ≥ 0.95**, primary = last epoch. It changes TWO things at once — geometry and +22% training
studies — so a pass says the package works, not which half did it. Either way, the next step is the
inference kernel: a pass ships the Stage A model; a null or a loss ships `ab_ortho` (the CC0 44-slice
geometry, whose test-time preprocessing the frontier's inference notebook already publishes).

⚠️ This corpus is the frontier's **starting** geometry (44 slices over 15–85%), not its 0.928 one
(64 over 2–98%). It is the same for every arm, which is what an encoder A/B needs; the absolute
number is not the frontier's.

---

## 3. Standing rules — violating any of these has already cost this project

1. **Never select on gold-58.** n=58, per-label SE 0.04–0.09. Selecting on it claims +0.0137 and
   delivers **−0.0034**. Use `score_oof.py` large-n (n≈681, site-grouped).
2. **The training loss is not evidence.** Workstream C's loss fell monotonically through 36.7% of
   its range and the arm lost by **4.8σ**.
3. **Pre-register every gate before the run.** §3t measured a null, the record wrote it down as a
   refutation, and it cost a genuinely diverse arm. §3y and §3z were pre-registered and survived.
4. **A null is a result.** Record it as a null, not as a refutation.
5. **Read a competitor's code, never its description.** Three claims about `pilkwang` came from its
   description; two were wrong.
6. **Never price a forward pass by parameter count — bench it.** Predicted 14×, measured 6.33×.
7. **All-member gold reads are inflated by +0.1474.** Only fold-resolved numbers compare.
8. **Two scales exist.** gold-47/58 ≈ 0.85–0.89 and LB ≈ 0.91–0.95, offset **+0.039**. Never
   compare across them. Price a future arm's LB gain at ~half its gold gain.
9. **Write geometry into the manifest.** §R happened because `box_mm` was recorded nowhere.
10. **Never benchmark a downloaded label table without the leak check.** `rayanbabur_soft` and
    `yunus_v5_merged` reproduce all 58 gold rows and score a fake **1.0000**.

---

## 4. ⛔ C-PRE — fix before Stage C, or Stage C answers the wrong question

> ✅ **RESOLVED 2026-09-23 — no fix needed.** OrthoFoundation's own pretraining code normalises with
> exactly these ImageNet constants (`ytrsk/OrthoFoundation` `datasets/augmentation.py:51`,
> `datasets/knee_multidata.py:45-47`, greyscale loaded as RGB). The check below was right to demand
> the source; the source agrees with the code. `PREMORTEM.md` §9.

`fusion/train_port.py:274-275` registers ImageNet constants inside `OrthoNet`:

```python
self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
```

and `fusion/orthofoundation_model.py` resolves **no** normalisation at all. Correct for DINOv2; an
**assumption** for OrthoFoundation-L, continued-pretrained on 1.25 M greyscale knee MRIs.

A competitor hit this exact bug swapping in RAD-DINO (`image_mean` 0.5307, `image_std` 0.2583,
native 518 px). Their arm lost from epoch 1, the gap widened every epoch, and **training loss was
lower** — fitting worse, not overfitting. It reads as "medical pretraining does not transfer."
<https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154>

**Required:** resolve `image_mean`/`image_std` from OrthoFoundation's own checkpoint or config. If
it ships none, measure them from the pretraining corpus or fall back to greyscale stats — and
**record which**, because a guessed constant is a confound either way. Assert the C0 control
resolves to byte-identical ImageNet constants so the comparison stays clean.

---

## 5. Stages A–D — the shape, and what each needs first

Full rationale in `PLAN-C6.md`. Gates are binding.

### 5.1 Stage A — the cache `8 kernels · ~9 GPU-h`

**Blocked on:** Stage 0's answer (slot spec may change), **and** refreshing
`raahimnawaz/rsna-knee-code` with `pipeline/slots64.py` + `fusion/perlabel_head.py`:

```bash
.venv/bin/python -m kaggle datasets version -p <code-dir> -m "C6: slots64 + perlabel_head"
```

Before writing a builder from scratch, **evaluate forking the published CPU pixel cache**
(`stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache`) — it persists decoded pixels as sharded
memory-mapped `.npy`. Check its crop and slice-position conventions against `slots64.py` first;
`check-whats-free-first` and §4b both say measure the free thing before building.

**GATE A:** manifest records `fov_mm 140`, `depth_range`, resolved spacing · slot fill ≥ Stage 0's
prediction per slot · render 20 studies at the 6% and 94% positions and confirm real anatomy, not
wrap or air. **No AUC.**

### 5.2 Stage B — do the data fixes pay? `2 kernels · ~14 GPU-h`

Encoder held at dinov2-small so cache and head are the only variables. **B0** = C6 cache + shared
pool; **B1** = C6 cache + per-label head.

**GATE B:** `B0 ≥ 0.7591` (the `runs_port` **0.7323 ± 0.0086** anchor + the crop's own **+0.0268**)
· `B1 − B0 > 0` paired · **Lateral Meniscus and MCL must both rise** vs 0.6422 / 0.6433.
**STOP if B0 < 0.7591** — re-open D2 before spending T4 hours on a 303 M encoder.

### 5.3 Stage C — does knee pretraining pay? `2 kernels · ~15 GPU-h`

**Do §4 first.** **C0** = stock `vit_large_patch16_dinov3`; **C1** = OrthoFoundation-L. Same
architecture, params, RoPE, head, cache, folds, schedule — `orthofoundation_model.py:1` maps Ortho
onto that exact timm model, so the only difference is the knee corpus. This is the ablation §3w-2
could not give, where domain and architecture moved together.

**GATE C:** `C1 − C0 > 0`, paired, fold 0, large-n. A null is a real result.

✅ **OrthoFoundation is now MIT-licensed** (a `LICENSE` file in `ytrsk/OrthoFoundation`, seen
2026-09-23). Previously: ⚠️ no LICENSE; the only grant was the authors' email of **2026-08-18**.
Preserve it — it is the entire claim. Competition §2.5.a carves out incompatibly-licensed
pretrained models, so this is not a shipping blocker.

### 5.4 Stage D — the blend `~6 GPU-h`

Where the model actually is: best single **0.8522**, 3-family blend **0.8932**. Equal-weight
rank-mean over families, **nothing fitted** — §2z measured refitting weights at **−0.0000**.
Members: C6's winner, `ft_b` (0.8522), public RadImageNet folds_v1 (0.8486), pilkwang (0.8516).
Screen on strength first; every arm ever rejected here was genuinely weak (0.6924–0.8025) and
every rejection was correct.

---

## 6. Closed — do not re-derive

- **Weighting label sources by accuracy** — reverse-log `−log(1−acc)` **−0.0005**, `acc^8`
  **−0.0005**, equal rank-mean **−0.0041**. Sharpening converges the blend *to* the best member.
  Cause: the sources are one read repeated (`steven_v4`↔`v2` **r=0.995**, LLM-to-LLM mean 0.819).
  **Re-opens only if a source appears at r < 0.7** — measure correlation before proposing it again.
- **Refitting blend weights** on site-grouped folds — −0.0000.
- **Post-hoc calibration / thresholds / priors** — AUROC is invariant to monotone transforms.
- **Crops as extra TTA windows** — dead on both instruments, pre-registered.
- **A C++ port for the efficiency track** — 93 s per 0.001 AUC; decode is already native.
- The full ledger is `DECISIONS.md`. **Consult it; do not extend it.**

---

## 7. The one-paragraph version

The labels are not the constraint — used as a predictor they read gold **0.8985** (≈ LB 0.938)
while the model reproduces its own training labels at **0.8486**. We are further from the labels we
have than those labels are from truth. The gap is the image path. Its largest measured piece is
**0.085** between our port and the fork on the same instrument, larger than every architecture
delta combined. **But D3's reading of it, "the NIfTI files", was amended on 09-23**: the fork's
weights score the same on those files, and what our port lacked was coverage, one centre window
per series. §2b measures whether that is the 0.085. Stage A is worth its price only if §2b says the
input is the cause, and Stage 0 still measures before anything is designed on it. Beating the
teacher afterwards is possible for one specific reason: **the model sees pixels the report never
described**, which is most of what synovitis is.
