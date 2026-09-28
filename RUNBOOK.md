# RUNBOOK — read this first, then `PLAN-C6.md`

**Purpose: any session can pick this up cold and know the next command to type.** `PLAN-C6.md`
says *why*; this says *what to run, what to expect, and what to do with the answer.*

## ⛔ GOAL CHANGED AGAIN 2026-09-27: THE GOAL IS 0.965 AUC

Set by the owner 09-27, superseding the 09-24 block below. **"Works" = a VALID improvement over the free public 0.943** — for
us two standard errors, by a rule written before the run, never by weights tuned on the public LB. **The public LB is ~30% of the
test (~400 studies; the final is the other ~70%)** — measured 09-27 by bootstrapping our own 691 OOFs at n = 400: one score
**±0.008**, two different models' difference ±0.005, a small blend change (a member at 1/3 into a fixed base) ±0.002 (at n = 925,
the private: ±0.005 / ±0.003 / ±0.001). So VALID = **≥ 0.947 for a blend change on the 0.943 stack** (+2 × 0.002) and **≥ 0.953
for a different standalone model** (+2 × 0.005). The top 25 (0.954–0.959) are one public-LB standard error apart; weights tuned on
the public 30% (the stack's 0.52 / 0.55 / 0.500001) may not hold on the 70%. Sub B-dread's pre-registered 0.944 stands as written,
but a read of 0.944–0.946 is NOT a valid improvement. The prize line and the efficiency track do NOT matter. **Only necessary work:**
every run must change a decision. Cut 09-27: the efficiency-track final and decode speed-ups, round-2 labels unless round 1 gains,
the normal-baseline, location / severity heads, Qwen, CoAtNet-from-CC0 (unless a later decision needs it), extra blend tests.
**Proposed (owner to confirm):** drop Friday's Stage B-`dread` run — it only separates input from labels, which changes no
decision (Stage B already paid +0.014) — and spend that slot on CNN teachers → the student ON Stage B directly. **✅ The owner, 09-27 17:05: "depending on how the CNN works we will do TS"** — CNN USABLE → swap Friday's task for CNN
teachers → Stage B student; CNN FAIL → keep Friday's Stage B-`dread` run, and the fastMRI+ teacher becomes the main teacher.

**⚖ HOLDING ITS WEIGHT — every part against the best AVAILABLE alternative (owner, 09-27: "use whatever's best available…
if any part of the pipeline isn't holding its weight, we get rid of it").** The base is the public 0.943 stack; a part of ours
stays only with evidence that it adds to the best available thing.
| part | ours | best available | evidence | verdict |
|---|---|---|---|---|
| system | — | **public 0.943 stack** (Apache 2.0) | the free floor | OUR BASE: every submission is judged against it (≥ 0.947 valid) |
| labels | `dread_soft` (public) | the same | +0.015 LB | keep; teacher → student on top |
| finished model | Ortho, 0.916 alone | **public CoAtNet, 0.928 alone** (CC0 weights) | at EQUAL setup Ortho > CoAtNet +0.010 (one run), gold tie; the 0.012 LB gap is setup (all ~4,350 studies, gold ×3, epochs on gold, SWA) | Ortho is NOT proven better than simply building on the public CoAtNet — **Sub B-dread (tonight) is its test in the stack**; fine-tuning the public CoAtNet on better labels is the untried best-available route — **its pipeline check PASSED 09-27 (0.9187 vs 0.9214)** |
| input | Stage B (ours) | the public 64-slice geometry | +0.014 LB | keep |
| training code | `train_windows` | the public training script | a port of it | keep (equivalent) |
| small CNN | ResNet34 | — | 0.808 vs 0.844 on the 691; as a TEACHER on gold +0.021–0.028 (Ortho +0.034) | cut as a main model 09-27 18:45; **back as the cheap teacher 20:15** (the 691 cannot judge teachers) |
| Qwen VLM, own labeler | — | — | failed their gates | cut (done) |
| outside data | fastMRI+ | the best image-graded data available | unproven | one test, then keep or cut |
| 691 hold-out | ours | — | costs ~0.010 on finals | train the finals on all data |

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

## ▶ NEXT SESSION STARTS HERE `handoff written 2026-09-25 ~02:30 PDT; updated 2026-09-26 12:55 PDT`

**▶ STATE AT 12:55 PDT 09-26 — three things in flight, each with its rule already written below:**
| what | where | when | read it with | its rule |
|---|---|---|---|---|
| **Sub A-c7** — Stage B alone on the LB (ref **56587668**) | ✅ **LB 0.915** (13:34) — Sub A 0.901, **+0.014**; 691 said +0.011, gold +0.012 | — | — | a same-labels model gain, confirmed on the board: the 691 ranks fixed-target arms correctly |
| **dread_soft retrain** — step 5 with only the labels changed | `rsna-knee-c6-train-ortho-dread` v2 (v1 stopped by provenance: the folds moved — fixed, `AUDIT.md`) | ~11 h after it starts | Qwen block below | LB ≥ 0.911 = labels are a lever |
| **VLM one-epoch learning check** | ⛔ **0.504 — FAIL, shelved** (15:20); constant predictions, a training defect (Qwen block) | — | — | ≥ 0.75 on the 691 → a full run next week |
Settled today: Qwen labels FAIL their gold gate (`steven_v2` stays); Stage B PASSES its 691 rule (+0.0112); the 09-26
audit (`AUDIT.md`) found no leakage and fixed three HIGHs in the Qwen path. dread_soft's LB result decides what the VLM's
full run trains on. To submit the dread retrain: `build_submit_kernel.py --ours-only --weights-kernel
raahimnawaz/rsna-knee-c6-train-ortho-dread --weights-sha <sha of its runs_ortho/last.pt> --tag dread` (download
`last.pt` for the sha, as done for c7). GPU: ~5 h left this week (estimate), resets Fri Oct 2 17:00 PDT.
**⏩ SPEED BENCH `pre-registered 2026-09-26 18:30 PDT, before the push`** (`notebooks/build_speed_bench_kernel.py`, `rsna-knee-speed-bench`, one T4, ~0.5 GPU-h; AUDIT 09-26 pass 4). The Stage B recipe on 300 studies, 40 identical micro-steps per variant: base (all 24 blocks checkpointed, eager) · ck12 · ck0 · comp (`torch.compile`) · comp12. **A variant is ADOPTED for later runs iff** its median s/micro-step is ≥ 10% below base, peak < 14 GB, its first-40 loss trace within 0.02 of base at every step, and it slows eval by less than it speeds training. Data loading is already ruled out (step 5: 1,751 × 2.685 s = 78.4 of 78.7 min). DDP on both T4s is adopted without a bench — measured on Stage B. New flags: `train_windows --ckpt-blocks N`, `--compile` (in place: `last.pt` keys unchanged, tested); `bench.json` now records data wait and the loss trace.
**⛔ SPEED BENCH READ 2026-09-26 20:58 — NOTHING ADOPTED, per the rule** (`fusion/runs_speed_bench/`). base 2.90 s/step, 6.05 GB ·
ck12 2.76 (−5%, 12.7 GB) no — under the 10% bar · ck0 **OOM** (14.5 of 14.6 GB) · comp 2.40 (**−17%**, loss Δ ≤ 0.0003) no — eval
2.53 → 5.86 s/study · comp12 2.17 (**−25%**, 12.4 GB, loss Δ ≤ 0.0005) no — eval +40%. Data wait 0.1–0.2% of a step (confirms
step 5's arithmetic). *Not a second chance, a caveat on the instrument:* eval was timed on 12 studies, so a one-time compile
(and a recompile per new window count) is spread over 12 — comp's +40 s total and comp12's +12 s (it likely found comp's kernels
in the on-disk cache) look like warm-up, not steady state. Unmeasured. If compile is revisited it needs its own pre-registered
test with eval run eager (only the training step compiled), which removes the question rather than re-measuring it.
**⏩ SPEED BENCH v2 `pre-registered 2026-09-27 11:20 PDT, before the push`** — the owner said yes 09-27 (4.5 GPU-h to Friday;
no cloud GPUs). Two changes, nothing else: (1) `--compile` now compiles the TRAINING step only — `predict()` runs under
`torch.compiler.set_stance("force_eager")`; (2) every variant runs as the real runs do, **DDP on both T4s** (compile has never
run under DDP here, and DDP adds ~2.5 GB). Variants: base · compT (compile, all 24 checkpointed) · compT12 (+ first 12 only;
may OOM under DDP — recorded, not a fault). **THE RULE, v1's unchanged:** a variant is ADOPTED for later runs iff its median
s/micro-step is ≥ 10% below THIS session's base, peak < 14 GB, first-40 loss within 0.02 of base at every step, and eval slows
by less than training speeds up. The fastest adopted variant is used by the Stage B + `dread_soft` run at Friday's reset.
Checked locally first (`vit_tiny`, 14-study synthetic Stage B corpus, 2 CPU processes, the kernel's own cell): all three rc 0,
DDP checksums identical across ranks with compile + checkpointing, eval flat (5.04 / 5.17 / 5.02 s/study); force_eager
unit-checked (four new eval shapes → zero dynamo frames; training still compiles). ~0.4 GPU-h.
**✅ SPEED BENCH v2 READ 2026-09-27 12:14 — `compT12` ADOPTED** (`fusion/runs_speed_bench_v2/`, from `7fa61c2`, T4×2 DDP, torch
2.10). base **3.465** s/step, 7.26 GB (Stage B's real run measured 3.52 — the bench is the real thing) · compT **2.797 (−19.3%)**,
7.02 GB, loss Δ ≤ 0.0007, eval +0.4% → passes · **compT12 2.475 (−28.6%)**, 13.27 GB (of 15.6 GB), loss Δ ≤ 0.0006, eval +0.9%
→ passes, fastest. No OOM, no traceback; DDP checksums identical across ranks in every variant. **Later runs use
`--compile --ckpt-blocks 12`.** What it buys: Stage B's 55 min/epoch → ~39; a 10.3 h session now fits the full schedule (`c7`
stopped at 86% of OneCycle) — so the next Stage B run differs from `c7` in that too, not only in its labels. Headroom is
~2.3 GB; if a long run ever OOMs, `--compile` alone (7.0 GB, −19%) is the fallback, already passed.

**⏩ STAGE B ON `dread_soft` `pre-registered 2026-09-27 16:40 PDT, before any push` — the owner said "go for it" 09-27.**
Kernel `rsna-knee-c7-train-ortho-dread`: `c7`'s recipe exactly (`PACKAGE`, Ortho, Stage B corpus, the same 3,502 / 691
studies, seed 42, cap 14 epochs fitted to a 10.3 h budget, T4×2 DDP) with TWO changes: `--targets labels_llm_soft.csv` (sha
`8cec477f`, attached from `dreaddevelopment/rsna-knee-labels`) and the adopted `SPEED` flags (`--compile --ckpt-blocks 12`) —
so it will also complete more of its schedule than `c7` did. Command: `build_backbone_ab_kernel.py --arch ortho --corpus
stage_b --n-train 0 --epochs 14 --targets data/public_llm_labels/dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv
--targets-from dreaddevelopment/rsna-knee-labels --tag dread` (`--smoke` first, on CPU: no GPU quota). **When:** the full run
needs ~11 GPU-h; the owner has ~4 until the quota resets **Fri 2026-10-02 17:00 PDT** — push then. **THE RULE, on the LB,
submitted ALONE through Sub A's path:** **PASS ≥ 0.926** (Stage A on `dread_soft` 0.916 + 0.010) → it becomes our main model and
the base for teacher → student; **null 0.906–0.926** → input and labels do not add, stay on Stage A for label work (cheaper);
**loss < 0.906**. Reported only: gold-58 paired vs `dread` and `c7`; the 691.
**✅ SMOKE PASSED 2026-09-27 17:05** (`rsna-knee-c7-train-ortho-dread` v1, CPU, from `e4306de`): targets `labels_llm_soft.csv` sha
`8cec477f`; 2 processes; checkpointing 12 of 24; encoder compiled; DDP checksums identical; epoch 0 loss 0.970; eval ran; done in
26.8 min; no traceback. **Friday: the same command without `--smoke` → v2 on T4×2.** (Since `e4306de`, `train_windows.py` changed
only in `5fb0423`: the `resnet34` arch and a `ckpt` guard that is True for Ortho — Ortho's path is unchanged.)
**Scheduled:** the desktop app's one-time task `rsna-push-stage-b-dread` fires **Fri 2026-10-02 17:20 PDT** (runs only while the
app is open; if closed, on next launch). It checks nothing is running, builds, verifies the flags, pushes once, and records here.
Do NOT "Run now" before the reset — it would start an ~11 GPU-h run on ~1 h of quota. (09-26's task sat 7 h on a permission
prompt — check on it Friday evening.)
**⛔ SUPERSEDED 2026-09-28 07:35 — the owner: "yes swap that for friday".** Sub B-dread (0.939) showed a report-label Ortho of this
class cannot lift the stack, so ~11 GPU-h on its twin changes no decision. Friday's slot goes to the students on teacher labels —
see **FRIDAY 10-02 PUSH LIST** below. This run is NOT pushed; the scheduled task now pushes only that list.

**♻️ WHAT WE TAKE FROM THE PUBLIC 0.943 STACK, AND THE TWO FINALS — owner Q&A 2026-09-28 07:45.** The stack is Apache 2.0: all of it
is ours to use (`REFERENCE.md` §3.2). It is a chain of rank blends whose vote rides on CoAtNets (~70%); its weights look tuned on the
public 30%; no member uses teacher-cleaned labels; and it already takes ~8–9 h of the **9 h** submission limit (T4). So:
1. **Its core model is our CoAtNet student's start** (maxspan-v5, 0.928 alone; check passed). A student is the SAME architecture, so it
   drops into the stack's CoAtNet slot at zero extra runtime — replace, don't add.
2. **Its other CoAtNets** (native384 v8 / dense v10, mattiaangeli's four) are further public starts for students if the first one gains.
3. **Its slow, small branch** (20 DINOv2 members ≈ 5% of the vote) is where runtime comes from for the Ortho student.
4. **Its code** (inference, slice-reversal TTA, rank-space joins) is reused as is — `build_submit_kernel.py` already wraps it.
**Two finals (rules 2.2.b), scored on the private 70%; Kaggle ranks a team by the better of the two.** So they should be strong and
DIFFERENT: (A) **the upgraded stack** — its structure, with our student(s) in its CoAtNet slot; (B) **our own blend** — equal-rank
families, nothing tuned on the public LB, the one that survives a shake-up best. Both chosen by pre-registered evidence, never by a
0.001 public difference.

**🗓 FRIDAY 10-02 PUSH LIST — ⛔ NOT READY** (the scheduled task `rsna-push-stage-b-dread`, Fri 17:20 PDT, pushes ONLY what this block
lists once it reads **✅ READY**, and nothing if it does not). To be written after today's CNN-teacher gold gate, each entry with its
exact build command, kernel id, flag checks and its pre-registered rule. Intended, in order (Kaggle runs two GPU sessions at once):
1. **CoAtNet student** — `--arch coatnet --init` the public 0.928 weights, student targets `data/student/targets_round1_cnn.csv`
   (`dread_soft` + CNN-teacher OOF; **gate PASSED 09-28, P 0.994**). ~6 GPU-h.
2. **Ortho student** — Stage B, the same student targets. ~10 GPU-h.
3. ~~**NYU teacher**~~ **CUT 09-28 16:25 (failed its gold gate on the Mac, P 0.21).** Was: Ortho on `raahimnawaz/rsna-knee-fastmri-plus-cache` alone (`--slots cor_1,cor_2 --allow-missing-labels`),
   pushed when a session frees. ~1–2 GPU-h.
If the CNN teachers FAIL their gate, the students train on `dread_soft` alone and this list says so.
**PROPOSED 2026-09-28 14:05 — before Friday, all free (owner to confirm):** (1) **the NYU teacher moves OFF Friday's GPU list onto the
Mac, tonight after fold 4** — a student should train on FINAL targets; if NYU passes after Friday, both students (~16 GPU-h) retrain.
ResNet34 recipe C (ImageNet, never saw competition labels — the design's rule) on the 556 NYU knees is ~15 min on the M5; gate as
designed (report + NYU beats report on its findings' macro, gold-58, P ≥ 0.95; mixed in only if the mix does not read below the CNN
alone), pre-registered before it runs. (2) **CPU smoke of both student kernels on Kaggle** (no quota; the 09-27 smoke took 27 min) with
the real targets file, so Friday cannot die on a flag. (3) **Pre-register the students' LB rules** before the push. (4) Keep the Mac
awake with the app open Friday 17:20 — the scheduled task needs it, and today an app self-update killed a run.
**Hardware (owner's what-if, 09-28): a Mac Studio (M5 Ultra).** This Mac is a base M5 (10 GPU cores, 16 GB): ResNet34 3.4 min/epoch vs
~2 on a T4. An Ultra (if like past Ultras, ~8× the GPU cores, 100s of GB memory) would run teachers, round 2 and likely the students
locally, 24/7, off the ~30 GPU-h/week quota — measure one epoch of each model first (MPS: no `torch.compile`, ViT-L untested). Finals
still infer on Kaggle's T4s (train locally, upload weights). Competition data must stay reachable only by team members (rules §7.b).
**Owner Qs 09-28 ~14:15.** (a) *Route the blend to the model best at each finding, or per study?* Per-finding weights need per-finding
scores we can trust, and we have none (gold-58 per-finding SE 0.03–0.09; the 691 rewards agreeing with reports): picks transfer at
−0.0002 ± 0.009 (gold) and 0.8289 vs 0.8298 (691). A per-study router has even less to learn from. Kept: the pre-registered
"winner only if clearly ahead (> 0.03), else average" rule (+0.0022 ± 0.0019, 09-24), re-measured before use on the students.
(b) *Strip Ortho?* Its inference is ~35 min (decode ~21 + GPU ~13 on two T4s); the runtime problem is the stack's 8–9 h — the room
comes from cutting its 20 DINOv2 members (~5% of the vote), not from trimming Ortho; distilling Ortho IS the student.
(c) *Studio (2025 = M4 Max or M3 Ultra):* code from GitHub; copy `data/stage_a` 30 GB + the NYU cache 3.7 GB + raptor 0.3 GB from this
Mac (external SSD / cable — not a download); Stage B from Kaggle only if students train there (8 × ~7.8 GB ≈ 62 GB, resumable
`curl`); the 168 GB of raw fastMRI archives stay behind. ~half a day incl. a one-epoch benchmark per model.
**Owner Qs 09-28 ~14:30.** (d) *A statistical second stage that reasons over the 12 predictions?* Measured 08-13 (IMPROVEMENTS §3r):
a fitted cross-finding stacker **+0.0033 on report labels, −0.0132 on gold** — it learns the REPORTS' co-occurrence; closed.
(e) *Test is graded on gold, so does extraction matter?* Yes: the model learns only what training labels teach — labels alone were
+0.015 LB. (f) *NYU as the instrument for per-finding weights?* No: image truth, but coronal-only, another protocol, 4–5 findings, one
reader — our competition models would be scored on inputs they never see at test. It is a teacher, not a ruler. (g) *MedGemma as a
TEACHER:* the runtime objection does not apply (teachers run offline); a 4-bit quantized 4B fits this Mac. Screen, if the owner
accepts the terms: zero-shot on the NYU 83-knee hold-out (no gold spent) — low odds (card: MRI 64.7% accuracy). (h) *Fewer DINOv2
members?* Yes — the branch is ~5% of the vote; cut it to make runtime room for the students, time the stack's cells first.
(i) *Gold ×3* = each gold knee repeated 3× per epoch; costs the gold instrument, so finals only. (j) *Drop Ortho for CoAtNet?* Friday
runs both at once (2 Kaggle sessions); to pre-register before the push: the Ortho student stays only if it lifts the stack +
CoAtNet-student blend by ≥ +0.004 public LB (2 × the blend-change SE).
**Owner Qs 09-28 ~14:40.** (k) *Are the models weighted for labels being only ~90% right?* Yes, in the way AUC can use: `dread_soft`
is soft (0 of 52,188 cells exactly 0/1; 14 levels, e.g. hedged reports ~0.3–0.4), the loss is BCE on those soft targets (+ pos_weight
for rare findings), and student targets turn report-vs-teacher conflicts into ~0.5, not a flip. A uniform "90%" correction cannot
move AUC (it changes no order); finding WHICH labels are wrong is the teacher's job. (l) *Qwen weights:* moved to the macOS Trash
(10.55 GB: Qwen3.5-9B-MLX-4bit + Qwen3.5-2B, `~/.Trash/qwen-weights-20260928-1413`), owner's request; `data/qwen_labels` kept;
`qwen_label.py` / `qwen_vlm.py` would re-download. (m) *MedGemma as the extractor:* no — the ~0.90 report ceiling; ~60% of reports
are not English (1,715 of 4,349 English by a crude test); 27B does not fit 16 GB even at 4-bit. (n) *Image truth NYU lacks (MCL, OA,
synovitis, fracture) elsewhere?* No allowed, usable source: OAI closed, SKM-TEA 155 knees on a research sequence (900 GB, Stanford
non-commercial terms), MRNet / KneeMRI are report labels, KneeCoT ruled out by the host. Those findings rely on the in-domain
teacher, round 2 and gold ×3 in the finals. The 168 GB fastMRI clinical DICOMs add KNEES (pseudo-labelled), not truth.
(o) *Is ~0.90 the realistic best for a report extractor?* For automated readers, yes: `steven_v2` 0.887 on gold; 13 tables + Qwen, none
better unfitted (`dread_soft` unscorable on gold, +0.015 on the LB). Perfect reading of the text: ~0.92–0.94 at most (hand read on 30
gold knees, DATA §1: +0.055 [+0.005, +0.111], one reader). Beyond that the truth is not in the text (DATA §12). The extractor is not
our ceiling: report + image teacher reads 0.915–0.922 on gold today.
(p) *Would a 0.92 reader help?* Modestly: +0.02 gold ≈ +0.01 LB single model at the ~½ transfer, less in the blend, and it overlaps
the teacher (both fix label errors). Route: Claude Opus graded read with the official cut-offs (`extractor/llm_extract.py`), iterate
on the 303 hand set (~$5–10), full corpus ~$60–150, rank-mean with `dread_soft`. Blocker: no cheap judge — `dread_soft` has no gold
rows, so only a GPU student on the LB could show it beats our labels; the same design on Qwen failed. Not before Friday; revisit only
if the teacher gate fails. (q) *Are our labels overfitted?* `dread_soft`: 0 gold rows (cannot be tuned on gold), proven by one
pre-registered LB read (+0.015 ≈ 3× a model-difference SE); its author may have tuned choices on the public LB — the private 70%
will show. `steven_v2`'s 0.887 may be optimistic (author-published; a gold-tuned prompt leaves no trace). Two public tables with
gold copied in were caught and excluded. The larger overfitting risk is OURS: every teacher arm and table was read on the same
58 knees (recipe C picked 1-of-3, ~+0.003–0.005 optimistic) — guarded by pre-registration and the LB as the judge.

**⏩ SMALL-CNN BASELINE `pre-registered 2026-09-27 16:45 PDT, before any push` — the owner said yes 09-27.** Question: does
the board's *"simple CNN at 224 px scores 0.94+"* (`REFERENCE.md` §3.1b) reproduce in OUR pipeline, at a fraction of Ortho's
cost — and so make teacher → student affordable (5 folds of teachers)? Arms (`build_backbone_ab_kernel.py --arch resnet34
--corpus stage_a --n-train 0 --epochs 8 --targets …labels_llm_soft.csv --targets-from dreaddevelopment/rsna-knee-labels`):
`resnet34.a1_in1k` at **224 px**, everything else `dread`'s recipe (Stage A corpus, `dread_soft`, the same 3,502 / 691
studies, seed 42, k 12, bs 8, head lr 1e-3, our attention head, 8 epochs fitted to a 1.0 h budget), in TWO arms that differ only
in backbone lr — **3e-5** (Ortho's) and **3e-4** (a usual CNN rate) — because Ortho's rate may starve a CNN. **Primary = the
better arm on the 691** (a 2-way pick: expect it to read ~+0.002 optimistic). **Compared paired on the 691 against `dread`**
(Ortho ViT-L, same labels and studies, 0.8440 — the 691 ranks fixed-target arms correctly, twice confirmed on the LB).
**THE RULE:** paired Δ vs `dread` ≥ **−0.010** → **USABLE**: it is the teacher for teacher → student; Δ > 0 at P ≥ 0.95 →
**BETTER**: also a main-model candidate (LB next); Δ < −0.010 → **FAIL** for THIS recipe (our attention head, 8 epochs,
intensity-only augmentation) — not a verdict on small CNNs; teachers stay Ortho. Reported: gold-58, min/epoch, GPU-h.
Checked locally (Stage A fixture, CPU): ImageNet weights load, checkpointing off, train → eval → OOF. ~2.6 GPU-h for both.
**⛔ READ 2026-09-27 17:10 — FAIL, for THIS recipe** (`fusion/runs_r34_lr3e5/`, `…_lr3e4/`; both 8/8 epochs, labels `8cec477f`, the
same 691). 691: **lr 3e-4 0.7915** (primary, the better arm), lr 3e-5 0.7349, `dread` (Ortho) 0.8440 → paired **−0.0525 ± 0.0051**,
0/12 labels up. Gold-58 (reported): 0.837 / 0.766 vs 0.912. **Cost: 1.9 min per epoch** (8 epochs ≈ 15 min; data wait 6%) —
~40× cheaper than Ortho. **Per the rule: teachers are NOT this CNN; the owner's branch → Friday's Stage B-`dread` stays scheduled
and fastMRI+ becomes the main teacher.** Diagnosis, not a second chance: the arm is UNDER-trained, not capped — 10× the lr bought
+0.057, and 8 epochs of 1.9 min is 15 minutes of training; the board's 0.94+ CNNs train longer and augment. Whether a proper CNN
recipe reaches Ortho is a NEW question (one ~1 GPU-h run), to be pre-registered only if the owner wants cheap teachers. **Where the CNN loses, per finding (lr 3e-4 vs Ortho, the 691):** Medial Meniscus **−0.115**, Lateral Meniscus **−0.110**, Contusion
−0.070, MCL −0.066, Fracture −0.052 … but Medial OA only −0.016, Synovitis −0.025, PF OA −0.025, Effusion −0.029. The gap is
largest on the SMALL structures (menisci, ligaments, bruises) and smallest on the large ones (joint-wide OA, fluid) — consistent
with resolution: at 224 px the 140 mm crop is 0.63 mm/px, against Ortho's 0.42 (336) and Stage B's 0.36 (384). A hypothesis to
check against the v2 arms, not a verdict.
**⏩ SMALL-CNN, PROPER RECIPE `pre-registered 2026-09-27 ~17:08 PDT, before any push` — the owner: "we should do proper testing
for the cnn".** The first read was under-trained (loss still falling at epoch 7, 0.824 → 0.702; 10× lr bought +0.057). New
question: does a CNN trained like the board's reach Ortho? Everything as the first read (resnet34 @ 224, Stage A, `dread_soft`,
the same 3,502 / 691, seed 42) except: **~22–24 epochs** (cap 24, fitted to a 0.75 h budget; ~2 min/epoch) and, in one arm, a new
training-only **per-window affine** (`--aug-affine 1`: rotate ±10°, zoom ±10%, shift ±5%, never a flip; unit-checked: eval
unchanged, `last.pt` keys unchanged). Arms: **A** lr 3e-4 · **B** lr 1e-3 · **C** lr 1e-3 + affine. **Primary = the best of the
three on the 691** (a 3-way pick, ~+0.003 optimistic), paired vs `dread` (Ortho, 0.8440). **THE RULE, unchanged:** ≥ −0.010 →
USABLE (the teacher: 5 folds ≈ 5 × 1 GPU-h); > 0 at P ≥ 0.95 → BETTER; below → CNNs are out of the teacher role for this
competition, fastMRI+ and Ortho carry it. ~2.7 GPU-h of the ~3 left this week. **Kaggle allows 2 GPU sessions at once** (C's push was refused: "Maximum batch GPU session count of 2 reached"); C is queued behind A / B by a local watcher. Pushed ~17:09 from `3333502`. **READ 18:00 (A, B; C pushed by hand 18:00 — the watcher died silently):** A lr 3e-4, 19 epochs **0.7929** (≈ the 8-epoch
0.7915: longer training alone bought nothing) · B lr 1e-3, 18 epochs **0.8076** (gold 0.892; Medial Meniscus 0.867 vs Ortho 0.908) —
paired vs Ortho −0.036: both below the −0.010 bar; C (lr 1e-3 + affine) decides. **Averaging B with Ortho (equal ranks, a
diagnostic): gold +0.0001, the 691 −0.0057 ± 0.0023 (2.5σ worse)** — a weaker member at equal weight dilutes the stronger one,
the Sub B lesson again. Rank correlation Ortho~CNN 0.78 (they do disagree; the CNN is just weaker). (Averaging 5 CNN folds would lift a TEST-time
ensemble by ~0.005–0.015, but a teacher labels each study with ONE fold model — the single model must be good.)
**🔎 WHY THE CNN LOSES — owner's question 2026-09-27 ~18:45 ("too small? wrong type? would the crop-and-classify CNNs fit?");
analysis of A / B only, C still running, no new run.** Per finding, B − Ortho on the 691 (`score_oof.py`): Lateral Meniscus
−0.081, MCL −0.074, Lateral OA −0.065, ACL −0.041, Medial Meniscus −0.041, PF OA −0.033, Baker's −0.027, Contusion −0.027, Medial
OA −0.017, Effusion −0.015, Synovitis −0.008, Fracture −0.008.
1. **Not "too small to learn".** B's last-epoch train loss **0.606 is below Ortho's 0.625** on the same labels and studies, yet it
   is 0.036 worse held out: it fits as well and generalises worse (schedules differ, so a rough sign). A bigger ImageNet CNN
   would fit the training labels better still, which is not what is missing. RSNA 2024 lumbar agrees (§4.5 of `REFERENCE.md`):
   ConvNeXt-small beat base beat large, and EfficientNet-V2-S is ResNet34's size (~21 M parameters).
2. **The two real differences from Ortho:** what it saw before (ImageNet photos vs OrthoFoundation's orthopaedic MRI, a 303 M
   DINOv3 ViT-L), and detail (224 px of a 140 mm field = 0.63 mm/px vs Ortho's 0.42). With the better lr the gap is no longer
   cleanly "small structures" (Fracture −0.008 but Lateral OA −0.065), so detail explains part of it, not all.
3. **Crops are the right shape of fix for the findings we lose** (menisci, MCL, ACL: small, at known places), with four costs:
   the host gives no locations (the lumbar winners had coordinates), so they come from anatomy — fixed mm crops (free, crude),
   TotalSegmentator MR (adds time to the ~9 h submission) or fastMRI+ boxes (coronal only); the stored Stage A corpus is 336 px,
   so a crop from it tops out at Ortho's 0.42 mm/px and finer needs a re-extract from DICOM; a multi-week build; and the
   §U→D2 warning stands — a measured case first.
4. **The cheap measured case, if wanted:** the CNN is ~40× cheaper than Ortho, so it is where to test whether zooming in helps
   before building anything for Ortho — B's recipe plus a zoomed centre crop as extra slots, ~0.7 GPU-h, judged on ACL + MCL +
   both menisci. NOT pre-registered; no quota until Friday's reset, and Friday is Stage B-`dread`. Priority stays: C's read →
   the fastMRI+ teacher tonight → Friday.
**⛔ C READ 2026-09-27 18:45 — the small CNN is OUT of the teacher role (the pre-registered rule).** C (lr 1e-3 + per-window affine),
17 epochs, **0.8057** (gold 0.890), paired vs B **−0.0019 ± 0.0048** (0.4σ: augmentation bought nothing). Primary = the best of the three
= **B 0.8076**, paired vs Ortho **−0.036** < −0.010 → FAIL. fastMRI+ and Ortho carry the teacher role; Friday's Stage B-`dread` stays.
One more clue for the diagnosis above: the affine raised C's final train loss (0.653 vs B's 0.606: less memorising) and held-out
did not move, so the ceiling is what the features can see, not over-fitting. Files: `fusion/runs_r34v2_C/`.
**A CNN-type model already nearly matches Ortho (owner's follow-up, 09-27 ~19:00: "can a more advanced setup make CNNs the path?").**
In §2c's equal-setup A/B the ImageNet CoAtNet (convolutional early stages, 384 px, ~74 M) read **0.8047 vs Ortho 0.8148 (−0.010)** and
beat it on MCL, Lateral OA, PF OA and Fracture; the public CoAtNet is **0.928 alone on the LB** (our Ortho 0.916), and it is already the
student's planned start. So "CNN vs ViT" is not the gap; ResNet34 at 224 px from ImageNet-1k is. What separates the two: 224 vs 384 px,
22 M vs ~74 M, ImageNet-1k vs 12k, attention in the late stages. The one medical-pretrained CNN ever run here (RadImageNet, 0.6924,
§3w-2) was the old pipeline — "our training was refuted, not the architecture". **Tried:** lr ×3, 8 → 19 epochs, affine augmentation.
**Never tried on a CNN:** more pixels, zoomed crops, medical pretraining, simple pooling instead of our attention head, more slices.
**⏩ CNN TEST — detail vs pretraining `pre-registered 2026-09-27 ~19:10 PDT, before any code or run` — the owner: "yes the zoomed cropped
test".** Why it changes a decision: in-domain k-fold teachers at Ortho's price are ~32 GPU-h (a week's quota); at a CNN's price they run on
the Mac for free. **Where:** the Mac (M5 GPU, fp16 autocast; benched 19:05 on a 96-window step: **3.9 min/epoch at 224, 9.2 at 336**,
~2× a Kaggle T4) — no quota. **Arms,** all B's recipe (Stage A, `dread_soft` sha `8cec477f`, the same 3,502 / 691, seed 42, k 12, bs 8,
head lr 1e-3, backbone lr 1e-3, **18 epochs fixed**, no affine, primary = last epoch):
- **B′ control** — ResNet34 at 224, whole image. Also the run-to-run floor: |B′ − B (0.8076, Kaggle)| > 0.010 → the Mac arms compare
  only with each other.
- **Z zoom** — B′ plus every window again as its centre 94 mm (the middle 67% of the 140 mm field) at 224 px = 0.42 mm/px, Ortho's
  density; both views in one attention bag. Before any run: 20 rendered studies show the joint line inside the box.
- **R resolution** — ResNet34 at 336, whole image: the same 0.42 mm/px as Z without cutting anything, at about Z's compute.
- **P pretraining** — RadImageNet ResNet-50 at 224 (1.35 M radiology images; input x/127.5 − 1; strict load).
**THE RULE:** paired on the 691 (`score_oof.py`). A lever **HELPS** iff arm − B′ > 0 at P ≥ 0.95; per finding and the small-structure
group (ACL, MCL, both menisci) reported. The CNN is **USABLE** again iff the best arm − Ortho `dread` (0.8440) ≥ **−0.010** paired (a
4-way pick, ~+0.004 optimistic) → the k-fold teachers run on the Mac. If two or more levers help and none reaches −0.010: **one**
combined arm, same rule, then stop. Z helps → zoom views become a student candidate (its own LB test). Nothing helps → CNNs and crops
are closed for this competition. **Needs:** the Stage A cache (~32 GB, our own `raahimnawaz/rsna-knee-stage-a-s{0..3}of4`), the
RadImageNet encoder (~90 MB, `mattiaangeli/radimagenet-resnet50-encoder`), `train_windows` on MPS + `--zoom` + a `radimagenet_r50`
arch. **Waiting on the owner's OK for the two downloads.** ~9 h of Mac time; fastMRI+ conversion goes first.
**🔍 OUTSIDE REVIEW OF THE PIPELINE — read against the code, 2026-09-27 ~19:30 (the owner pasted it, "your thoughts?").**
Adopted: (1) the NYU gate as a paired macro bootstrap, not per-finding deltas (item 3 of "HOW fastMRI+ FITS"); (2) a fastMRI+
hold-out as a teacher diagnostic, report-only. Already so: 336 px (Stage B 384 = 24 × 16); "quantile matching" IS
`build_student_targets.py` (each teacher's ranks mapped onto the base's own sorted values, then averaged). Rejected, with the
evidence: **the 691 as the judge of labels** — its reference is another LLM's read of the same reports; `DATA.md` §2 measured a
large-n substitute ranking label sets at Spearman 0.43 (p = 0.34) against gold, rewarding copies of the report — and gold-58 is not
as weak as the review's single-finding, unpaired SE (0.035–0.045): paired macro SE is **0.008** (measured tonight on
`build_student_targets --gold`: 0.8873 → 0.9216, +0.034 ± 0.008); **"the 691 is contaminated"** — students train only on the
3,502 (`train_windows` refuses train/val or train/gold overlap), the NYU teacher never trains on a competition knee, and its 691
predictions exist only for the finals on all data, where the 691 is spent by design; **interpolating slices** — invents data never
acquired (`PLAN-C6.md`), the real limit (channels 1.6–3.9 source slices apart) is recorded in `DATA.md` §10b and Stage B eases it;
**box-area severity + sigmoid for fastMRI+** — the student targets use only the teacher's ORDER, so its calibration never reaches
them, and area needs a joint segmentation and measures size, not presence; **zoom and RadImageNet in one arm** — it could not say
which helped; **gating on the joint-space findings only** — reported, but the gate stays on all findings; "RadImageNet cuts epochs
~40%" — no source. **Prompted by it, open:** fastMRI is NYU — if NYU also contributed competition knees, a teacher that saw a gold
or 691 knee would inflate its own gate. Check during conversion: scanner vendor / field strength / protocol against the
competition's site fingerprints, then thumbnails if any site matches.

**⏳ TEACHER → STUDENT — design 2026-09-27 (the owner: "we'll take the better labels teacher student approach"); its rule is
fixed once the small-CNN read picks the teacher.** Why now: gold-58 says the report label and an image model's OOF, averaged,
read **0.927** against 0.887 / 0.912 alone (`DATA.md` §12 item 2); the board's 0.950 single models credit this.
1. **Teachers:** 5 folds over the 3,502 training studies (the 691 and gold stay out), each predicting its held-out fifth →
   one OOF image prediction per training study. The small CNN if it is USABLE (~1 GPU-h a fold), else Ortho (~8 GPU-h a fold).
2. **Student targets:** per study and finding, the equal rank-mean of `dread_soft` and the teacher OOF, mapped back onto
   `dread_soft`'s own values (its graded levels and prevalence kept; only the ORDER moves — the same construction as the Qwen C2
   candidate). Nothing weighted, nothing fitted.
3. **Student:** `dread`'s model exactly (Ortho, Stage A, 8 epochs, the same 3,502), only `--targets` changed → the LB against
   0.916, so the difference is the targets. Judged on the LB (the 691 is another LLM's reading and would call a move toward the
   images a loss); gold-58 reported only. Round 2 (the student becomes the next teacher) only if round 1 passes.
Budget: CNN teachers ~6 GPU-h + a student ~11 → one week's quota after Friday's Stage B run.

**🎯 TARGET SYSTEM `2026-09-27, owner Q&A` — supersedes the roadmap just below (its efficiency final and prize-line framing are
cut).** Four layers, each piece kept only if its own gate passes:
1. **Data:** the competition's 4,349 training knees + reports; NYU fastMRI+ (1,172 image-read knees).
2. **Labels:** `dread_soft` (proven +0.015) · the NYU image teacher (gate: beats the report label on gold-58 for its findings) ·
   our k-fold teacher (only if the CNN passes) → **combined labels** (rank-mean, mapped onto `dread_soft`'s values).
3. **Models (students), one per family:** CoAtNet from the public 0.928 weights (`--init`; gate: its pipeline check) · Ortho on
   Stage B · CNN (only if it passes). Finals retrained on all data.
4. **Blend:** our families at equal ranks, nothing fitted → with the public 0.943 stack → **counts only at ≥ 0.947** (valid).
GPU: NYU teacher ~2 · CoAtNet student ~6 · Ortho student ~10 · finals on all data ~16 → ~34 h (+5 if CNN teachers) of ~90 left.
Honest landing ~0.95; 0.965 needs the teachers to correct the labels well beyond what reports carry.
**🏁 PATH TO #1 — the ledger, 2026-09-28 07:25 (live #1: 0.960, up from 0.959 yesterday).** Scoreboard truth: our best submission
is **0.939**, still under the free stack's 0.943. Measured levers, never yet combined in one model: **labels +0.015 LB** (Ortho 0.901 →
0.916), **Stage B input +0.014 LB** (0.901 → 0.915), **blending +0.015** (the stack over its best single, 0.928 → 0.943); on gold, not
yet on the LB: **report + image teacher +0.034** (Ortho) / +0.028 (CNN). The strongest start is the public CoAtNet, **0.928** alone,
checked in our pipeline. Stacked: CoAtNet student 0.928 + teacher labels + Stage-B-grade input + all-data finals → a ~0.95 single model;
two such students + the stack's other families → **0.96+, first-place territory**. The gains overlap, so each keeps its own gate. The
first LB proof point is the CoAtNet student (GPU reset Fri 10-02). What decides it: whether teacher labels move the LB the way
`dread_soft` did.

**🧭 WHAT'S MISSING — owner's question 09-27 ~20:00 ("if NYU helped, would our data be as good as it could be?"). No.**
The test's answer key is two MSK radiologists reading the IMAGES with the official cut-offs, borderline = negative (`REFERENCE.md` §2);
every training label we have is an LLM reading the REPORT. On gold-58 the training labels read 0.887 and our model 0.878 — level
(`DATA.md` §10b, "the model has reached its labels"): the label is the binding constraint on a single model. NYU, if it passes, lifts
~5 of 12 findings from another hospital's radiologists; MCL, the three OA findings and synovitis stay report-grade. Ranked:
1. **Image-read labels on OUR knees, with the official cut-offs** — the only lever aimed at the gap on all 12. (a) A radiologist on the
   team labels a few hundred training knees: allowed for training data, but they must JOIN the team before seeing an image, by **10-15**;
   ~5–10 min a knee → 300 knees ≈ 25–50 h. The public 0.928 counted the 58 gold knees ×3 in training — that is how much 58 were worth.
   (b) OAI MOAKS: image grades on thousands of knees for cartilage, effusion-synovitis, marrow lesions, menisci, cysts — our weakest
   findings; blocked on terms (the NDA / Stanford questions are Gmail DRAFTS since 09-26, unsent); a large download and weeks of work.
2. **What the model sees:** still ~half the slices, 76% of series shrunk, 15% of series unused; Stage B's partial fix was +0.014.
3. **More knees:** 3,200 → 4,349 was +0.010 in the public model's own ablation; finals on all data are planned.
4. **More strong, different models in the blend** (the CoAtNet + Ortho students).
2–4 are each ~+0.005–0.010 on a single model, less on the blend — the ~0.95 landing. Only 1 aims at 0.965; it cannot be sized without
trying. Owner decisions: a radiologist who could join by 10-15? send the OAI / Stanford terms drafts?
**⛔ 1(a) CLOSED 2026-09-28 13:50 — the owner: no radiologist.** Owner: "do we even need one with the NYU files?" — **NYU is not a
substitute for one; the in-domain image teacher is the closer one.** NYU = 556 knees (not 1,172), coronal only, a protocol no competition
study uses, labels on ACL-high-grade / Effusion / Contusion / Baker's (+ menisci), none on MCL, the 3 OA, synovitis, fracture; estimate
+0.003–0.010 single model. The k-fold image teacher on OUR knees reads report + teacher **+0.028–0.034 on gold, all 12 findings** — more
than NYU's ceiling on its five. So the image-read-labels lever is carried by: the image teacher (running) → NYU on its findings → round 2
(student as the next teacher) if round 1 gains. ~~OAI MOAKS~~ **closed 09-28** (the NDA reply: research-use only, share only with
DUC-listed researchers, silent on publishing weights — fails "equally accessible" and the winner's licence warranty; `REFERENCE.md`
§1.3). **Extra report fields (owner, 09-28: sex, left/right?) — no:** at test time there are no reports, and both already come from
the DICOM headers (`PatientSex` on 91%; side = tag on 50% + the IPP-x rule at 97.3%, already used by Stage B's canonical flip). Sex as
a model INPUT is the old F3 idea, never built; it has to be a training feature or a group prior (a monotone post-hoc shift cannot
move AUC), and the one group prior tried, F1's site prior, died. Report fields that could matter are severity and location — queued
(aux severity head; DATA §12 item 5's zoom first), behind teacher → student.
**Owner Q 09-28, "did we consider an already-trained LLM for extraction?"** Yes: 13 public LLM label tables benched on gold (none
beats `steven_v2` without touching gold; `dread_soft` won on the LB, +0.015), our own Qwen3.5-9B graded read (FAIL). A frontier pass
(`extractor/llm_extract.py`, Claude Opus, ~$60–150) was costed, never run — and stays unrun: every reader reads the same text (LLM↔LLM
r 0.82, best tables ~0.90 on gold), and the misses that matter are image truth absent from the report — the teacher's job, not a reader's.

**⏩ CNN TEACHERS — 4 folds on the Mac `pre-registered 2026-09-28 ~06:55 PDT, before any fold runs` (the owner, 09-27: CNN
teachers on the Mac).** Recipe C exactly: `resnet34.a1_in1k` at 224, Stage A, `dread_soft` (sha `8cec477f`), seed 42, k 12, bs 8, head
and backbone lr 1e-3, `--aug-affine 1`, **17 epochs fixed** (C's count), `--mps` (Apple GPU, bf16; the CUDA path unchanged; trial on 30
NYU knees: 0.15 s per 2-study step, 27× the CPU). Folds: `provenance/folds_site.csv` training folds 1–4 (821 / 918 / 924 / 839 — their
union is the 3,502 exactly, checked against `runs_dread_ortho/train_studies.csv`); fold K trains on the other three and predicts fold K
(its OOF), the 691 and gold-58. Teacher on gold = the mean of the four folds. **THE RULE** (gold-58, paired bootstrap, base `steven_v2`
— the only report table with gold rows): **PASS iff report + CNN teacher beats report alone at P ≥ 0.95.** PASS → student targets =
`dread_soft` + the CNN OOF (`build_student_targets.py`); the NYU teacher joins only if it passes its own gate AND the mix does not read
below the CNN alone (the weak-teacher lesson). FAIL → Ortho k-fold teachers (~25 GPU-h), decided then. Reported: each fold's 691 macro,
per finding, minutes per fold. Recipe C was the best teacher in last night's exploratory reads (a 3-way pick, ~+0.003–0.005
optimistic). ~1 h a fold on the Mac, ~4 h in all, no GPU quota.
**▶ RUNNING since 2026-09-28 08:53** (terminal tab "CNN teachers"). First start 07:45 was killed at the first step: macOS pickles the
memmapped corpus into each DataLoader worker (a full 32 GB copy each). Now `TW_WORKERS=0` (the default in `local_teacher_folds.py`) —
data loading only, the recipe is unchanged; 0.6 GB resident, **0.15 s per 2-study step = 3.4 min an epoch**, ~1 h a fold → all four
~13:00. Stage A shard 3 came in at 08:1x by a resumable `curl` on the output's direct link (the Kaggle CLI restarts from zero).
**Folds 1–2 DONE** (71.7 / 68.7 min; own gold-58 macro, report-only, one fold on 3/4 of the data: 0.874 / 0.872). **Fold 3 was
killed at 11:40:09, epoch 7 of 17:** the Claude desktop app relaunched itself for an update (`pmset` log: "Preparing to relaunch", then
ShipIt "Updating") and took the terminal tab with it — no error in `train.log`. **Restarted 13:36 as a detached process** (own session,
parent = launchd, under `caffeinate -i`; log `fusion/runs_teacher_cnn/runner.log`), so an app relaunch cannot kill it again; fold 3
starts over (the runner resumes per fold, not per epoch), same recipe, same 0.15 s/step. **Fold 3 ~14:47, fold 4 ~16:00 → the gold gate.**
**✅ GATE READ 2026-09-28 ~16:03 — CNN TEACHERS PASS.** All four folds done (71.7 / 68.7 / 69.2 / 70.8 min, 4.7 h on the Mac, 0 GPU-h).
Teacher on gold = the mean of the four folds' `oof_gold.csv` (last-epoch weights, the primary). `build_student_targets --gold`, base
`steven_v2`: **report 0.8873 → report + CNN teacher 0.9132, paired +0.0258 ± 0.0102, 95% CI [+0.0060, +0.0463], P 0.994 ≥ 0.95 → PASS.**
(09-27's exploratory single CNN C read +0.028 at P 0.995 — consistent.) Reported, not judged: own gold per fold 0.874 / 0.873 / 0.874 /
0.896; the 691 per fold 0.814 / 0.819 / 0.815 / 0.814, mean of four 0.832. Per finding on gold (report → both; indicative, 9–35
positives each): Effusion 0.853 → 0.930, Fracture 0.793 → 0.881, Contusion 0.831 → 0.891, Medial OA 0.931 → 0.965, Synovitis 0.790 →
0.825, Lateral Meniscus 0.862 → 0.873, PF OA 0.901 → 0.909, Baker's 0.946 → 0.958; flat or −0.01: ACL, MCL, Medial Meniscus, Lateral OA.
**Per the rule → student targets built:** `data/student/targets_round1_cnn.csv` (sha256 `f313476903287ab2…`) = `dread_soft` + the CNN
OOF (`fusion/runs_teacher_cnn/teacher_oof_mean.csv`: each of the 3,502 from its one unseen fold; the 691 the mean of all four). 4,183
of 4,349 studies have a teacher; the other 166 keep `dread_soft` exactly. The NYU teacher joins only via its own gate (PROPOSED: on the
Mac tonight, awaiting the owner). Next: the Friday push list → READY (targets to Kaggle as a private dataset, build + CPU smoke both
students, pre-register their LB rules).
**⏩ NYU TEACHER ON THE MAC `pre-registered 2026-09-28 16:08:00 PDT (commit 149cb0a; the run started 16:08:21), before it trains` (owner: "if it's less than an hour just do it
now").** ResNet34 recipe C exactly as the CNN teachers (`resnet34.a1_in1k`, 224, bb + head lr 1e-3, `--aug-affine 1`, 17 epochs, seed 42,
k 12, bs 8, `--mps`), ImageNet start — never saw a competition label. Train: the fastMRI+ cache (`data/external/fastmri/cache_stage_a`,
Stage A format), `--slots cor_1,cor_2 --allow-missing-labels`, labels `labels_fastmri_v1.csv` (ACL high-grade 61 · Effusion 100 ·
Baker's 111 · Contusion 97 of 556; the other 8 blank → no loss), its 473 training knees; val = the 83-knee patient hold-out
(report-only). Then eval only (`--init` its `last.pt`, `--epochs 0`, same slots) on the competition Stage A cache: the 3,502 + 691
and gold-58. **THE RULE** (gold-58, paired bootstrap 4,000 over studies, base `steven_v2`): (1) **PASS iff report + NYU beats report
alone on the macro over its 4 findings at P ≥ 0.95.** (2) It joins the student targets only if ALSO report + CNN + NYU does not read
below report + CNN on the 12-finding macro (point estimate) — the weak-teacher lesson. Either fails → the targets stay
`targets_round1_cnn.csv` and NYU is cut. Report-only: hold-out AUC per finding ("did not learn" vs "did not transfer"), gold per
finding, minutes.
**⛔ NYU READ 2026-09-28 16:25 — FAIL on both rules → NYU is CUT; the students train on `targets_round1_cnn.csv`.** Trained 16:08–16:19
(10.5 min on the Mac; the first launch died on a zsh word-split before any step; the first eval pass needed a `--train` list — 2
out-of-pool studies, never used at 0 epochs), competition eval 2.5 min. **(1)** report + NYU vs report on its 4-finding macro: **0.8914
vs 0.9057, −0.0143 ± 0.0176, CI [−0.048, +0.021], P 0.21 → FAIL.** **(2)** report + CNN + NYU vs report + CNN, 12 findings: **0.9079 vs
0.9131, −0.0053, P 0.08 → FAIL** (the weak-teacher lesson again). Report-only: NYU hold-out (83 knees) ACL 0.784 (6 pos) · Effusion
**0.962** · Baker's 0.718 · Contusion 0.765 — it LEARNED. On gold, NYU alone: ACL 0.776 · Effusion 0.835 · Baker's **0.525** · Contusion
0.804 — Baker's (fastMRI+'s "periarticular cyst", noisy by design) did not transfer, and ACL sits far below the report's 0.993. Effusion
alone looks useful (report + CNN 0.930 → + NYU 0.952), but that is a per-finding pick read off gold — the rule forbids it, and any test
of it on the same 58 knees is contaminated. Friday's push list drops item 3 (NYU teacher, ~1–2 GPU-h saved).
**Owner Q 09-28 ~16:20: will Lateral OA and synovitis always lag; crop to the finding and retry when AUC < 0.93?** They lag for
different reasons. Lateral OA is VISION, measured 09-25 (DATA §10b): its labels are clean (0.988) and not scarce (25.7%); the model
recovers the least of them — 2–3 mm cartilage is smaller than one ViT token — and the reports themselves are weakest there on gold
(0.830); the CNN teacher did not move it (0.830 → 0.825). Synovitis is LABELS: reports rarely address it (83.7% of cells), and the
teacher did help (0.790 → 0.825). AUC exists only over a set, never per knee, so "retry below 0.93" cannot run at test time; the working
form is an ALWAYS-ON zoomed second view (e.g. the lateral half of the canonical coronal at 2×) added BESIDE the full view — the one crop
design never run (crop-INSTEAD-of-context died twice, archive PLAN F2). It is DATA §12 item 5's zoom arm. Sized: +0.05 on two findings
= +0.008 macro; gold cannot judge one finding (11 Lateral OA positives, SE ~0.08), so only the LB can — after Friday, not before.
**Owner Qs 09-28 ~16:35.** (a) *Enough to move the needle?* Estimate +0.005–0.010 LB for a single student over `dread_soft` alone (the
gate's +0.026 is against the weaker `steven_v2`, and gold → LB has run at ~½). Real, not sufficient alone: stacked with the 0.928 start
and all-data finals it points at ~0.945–0.95 single, more in the stack. Still lacking, the student targets on gold: Lateral OA 0.825,
Synovitis 0.825, Lateral Meniscus 0.873, Fracture 0.881, Contusion 0.891; the model side on the 691: MCL 0.735, Lateral OA 0.796,
PF OA / Lateral Meniscus 0.812 — labels gaps (teacher, round 2) vs vision gaps (thin structures: zoom, resolution). (b) *Use the NYU
images anyway?* The fastMRI+ LABELS are done (their gate failed; an effusion-only pick is off gold). The one open use is the 10k
UNLABELLED clinical exams (168 GB, all planes) as pseudo-labelled extra knees (Noisy Student) — round 2, judged on the LB. (c) *Are we
leaving something out by testing NYU with a ResNet34, not Ortho?* Possibly a little: a bigger backbone lifts the numbers, but the
failure is transfer (Baker's 0.72 hold-out → 0.53 gold; ACL 0.78 vs the reports' 0.993), label definitions and one plane — not
capacity; and Friday's students train before an Ortho-NYU run could finish, so it changes no Friday decision. (d) *Switch to a better
model when a weak finding fires?* AUC has no firing moment — every knee gets a score; a conditional second model is a blend, measured
09-24: "verification" 0.8279 < Ortho alone 0.8298. The working version is ALWAYS averaging the two students and the pre-registered
clear-winner rule.

**🔬 TEACHERS ON GOLD — 09-27 ~20:15 (owner: "are our labels as good as they realistically could be?"; no radiologist; the OAI
email SENT by the owner). EXPLORATORY gold reads, not pre-registered — gold-58 is being spent; the student's LB is the real judge.**
`build_student_targets --gold`, base `steven_v2` (0.8873), one teacher's gold OOF each, paired bootstrap:
| teacher | its own gold | its 691 | report + teacher | paired Δ | P |
|---|--:|--:|--:|--:|--:|
| Ortho `dread` | 0.912 | 0.844 | 0.9216 | **+0.034 ± 0.008** | 1.00 |
| CNN C (lr 1e-3 + affine) | — | 0.806 | 0.9152 | **+0.028 ± 0.011** | 0.995 |
| CNN B (the rule's primary) | 0.892 | 0.808 | 0.9081 | +0.021 ± 0.012 | 0.96 |
| CNN A | — | 0.793 | 0.9050 | +0.018 ± 0.011 | 0.95 |
| pilkwang OOF (dinov2-small, free, all 4,407) | 0.840 | 0.843 | 0.8905 | +0.003 ± 0.009 | 0.64 |
| Ortho + pilkwang | | | 0.9086 | +0.021 | 0.99 |
| Ortho + CNN B | | | 0.9224 | +0.035 ± 0.012 | 1.00 |
**Two lessons.** (1) **The 691 cannot judge a teacher:** pilkwang ties Ortho there (0.843 vs 0.844) and is useless as a teacher on
gold (+0.003) — the 691's reference is a report read, so it rewards agreeing with reports; a teacher is worth what it knows BEYOND
them. So the small-CNN rule (judged on the 691) measured the wrong thing for the teacher role: **the CNN is cut as a MAIN model, not
as a teacher** — it carries 60–80% of Ortho's teacher value at ~1/40 of the cost. (2) **A weak teacher dilutes a strong one** (Ortho +
pilkwang +0.021 < Ortho +0.034): only teachers that pass gold on their own go into the mix. **So our labels are NOT as good as they
realistically could be:** today every model trains on report labels alone; report + an in-domain image teacher reads ~0.91–0.92 on gold
against 0.887, on all 12 findings — larger than NYU's realistic +0.005–0.019 on five. (`dread_soft` is a better base than `steven_v2`,
so the real gain is somewhat smaller.) **Proposed:** 4-fold CNN teachers (`--teacher-fold 1..4`, recipe C) — free on the Mac (needs the
Stage A cache) or ~3 GPU-h on Kaggle — → student targets → gold gate (pre-registered before the folds run) → students; 4-fold Ortho
teachers (~25 GPU-h) only if the CNN's targets fall short. The zoom / resolution / RadImageNet arms are judged on gold teacher value too.

**🧾 FREE-RESOURCES AUDIT — 09-27 ~20:30 (owner: "are we making the best use of free resources? innovate, don't invent").**
USING the best available: labels `dread_soft` (every other public table benched on gold 09-23, `DECISIONS.md`: none beats `steven_v2`;
two contain gold — leaks); the 0.943 stack as the base; the 0.928 CoAtNet as the student start (check passed); OrthoFoundation; the
public training script (ported); pilkwang's free OOF (tested tonight as a teacher: +0.003, useless); forum recipes; the Mac's GPU (now);
Kaggle CPU for checks. Our own builds with a public twin are kept on evidence (Stage B +0.014 LB over the public geometry).
**NOT using:** (1) **fastMRI knee DICOMs** — ~10,000 full clinical NYU knee exams, all planes, unlabelled; our access already covers
them (`knee_DICOMs_batch1` 137.0 GB + `batch2` 30.6 GB). The standard way to use them is Noisy Student / distillation: the best
available model (the public 0.928 CoAtNet, which never saw them) labels them, a student trains on ours + theirs. Our DICOM → Stage A
pipeline already exists (`build_slots64.py`). Evidence of size: more real-labelled knees (3,200 → 4,349) was +0.010 LB in the public
model's ablation; pseudo-labelled knees are worth less — unmeasured here. Costs: ~7 h download, disk, ~4 GPU-h to label, students 2–3×
longer; the agreement question (derived data on Kaggle) and the NYU-overlap check apply. Cheapest test first: the CNN on the Mac with
vs without them. **Download QUEUED 09-27 ~22:40 at the owner's word** (after the Stage A cache; resumable, to
`data/external/fastmri/dicom/`, git-ignored; ~7 h at ~7 MB/s → ~07:00). (2) **A team merger** (other teams' models + their 30 GPU-h/week each; deadline 10-15) — the owner's call.
(3) Kaggle TPU quota (20 h/week) and Colab's free T4 — skipped: a TPU port is invention, and Colab re-downloads 32 GB per session.

**🗺 (SUPERSEDED 09-27 by the target system above) ROADMAP IF TEACHER → STUDENT WORKS — proposed 2026-09-27, the owner to confirm; each step keeps its own rule.**
"Works" = student LB ≥ 0.926 (`dread` 0.916 + 0.010). Quota resets Fri 17:00 PDT (10-02, 10-09, 10-16); final 10-22 16:59 PDT;
merger / entry deadline **10-15**. Two finals, and they serve BOTH tracks.
| week | GPU (≈30 h) | what | judged on |
|---|---|---|---|
| 10-02 → 10-09 | Stage B `dread` 10.3 · CNN teachers 5 × ~1.2 · student ~5 (Stage A, T4×2 + SPEED) | Stage B read; round-1 labels; student #1 | LB ≥ 0.926 each |
| 10-09 → 10-16 | Ortho Stage B on student labels ~9.5 · round-2 CNN teachers ~6 (only if round 1 gained ≥ 0.010) · CNN 5-fold on student labels = an ensemble member with its own OOF | best single model; Ortho + CNN blend (equal families) | LB; merger decided by 10-15 |
| 10-16 → 10-22 | final models on ALL data (3,502 + 691): Ortho Stage B ~10 · CNN 5-fold ~6 · optional CoAtNet from the CC0 weights | the finals | recipes fixed earlier; the 691 is spent |
**Finals:** (1) the strongest blend, main track; (2) the fastest good model — CNN 5-fold, decode-optimised toward ~10 min — for
the efficiency track (at maxAUC ~0.96: 0.94 in 10 min scores ~0.062; 0.93 in 25 min ~0.111; lower wins). CPU work alongside:
`infer_windows` for `resnet34` (its `build_model` passes `img_size`, which a ResNet rejects), a multi-model submission, decode
speed-ups (`order_series` reads every header). **Honest landing:** single model ~0.93–0.94, blend ~0.94–0.95 — the 0.956 prize
line still wants outside data, image-read labels or a merger; the efficiency track is the realistic prize.

**Candidates from the owner's 09-27 questions — NOT pre-registered, ranked by evidence:**
- **Start from the 0.928 CoAtNet instead of ImageNet.** `dreaddevelopment/raptor-knee-maxspan` (CC0) is a CoAtNet already
  trained on THIS task (LB 0.928 alone); fine-tune it on teacher → student targets. ⚠️ It trained on all 4,349 studies with gold
  ×3, so for anything started from it **the 691 and gold-58 are no longer held out** — the LB is its only judge. Check first that
  its encoder keys load into timm `coatnet_rmlp_2_rw_384` (their head ≈ our `WindowMIL`).
- **Final models on ALL the data** (3,502 + 691, maybe gold ×3 as the frontier did): their ablation measured +0.010 LB from
  3,200 → 4,349 studies. Last, once recipes are fixed — it spends the 691 as a check.
- **A medical-pretrained CNN teacher:** RadImageNet ResNet-50 (1.35 M radiology images; CC-BY-NC-SA, carved out by §2.5.a,
  `DECISIONS.md`) instead of ImageNet, if the ResNet34 read makes CNNs worth it.
- **Knee pretraining's own worth is still unmeasured:** `ortho − dinov3l` (generic DINOv3-L) was never run (§2c).
- OAI-derived pretrained weights (KneeXNet-2.5D etc.): wait for the host — the 12th-place team has asked.

**🧰 PREPPED FOR IMPLEMENTATION `2026-09-27 ~18:15` (owner: "take the important findings and prep them") — built and tested
locally, nothing run on GPU:**
- `train_windows --slots cor_1,cor_2` (plane filter, train + eval) and `--allow-missing-labels` (masked loss; the default
  path verified byte-identical: same loss trace, identical OOF vs HEAD) — `bc56682`.
- `train_windows --init` (our `last.pt` or the public CC0 CoAtNet, strict) — `d9f473e`; launcher `--cpu`, `--extra-datasets`.
- launcher `--teacher-fold K` (K = 1–4: train on the other three training folds, label fold K; four runs = an OOF teacher
  label for all 3,502) — `bc56682`; `--predict-all` (an outside teacher, `--init` + `--epochs 0`, labels every training
  study and the 691) — `6c4bd3b`.
- `labeling/build_student_targets.py` — report label + any number of partial OOF teachers → student targets (each teacher
  mapped onto the base's values on its studies, then averaged; uncovered studies unchanged; audit list; `--gold` gate:
  with Ortho's OOF as the teacher, gold-58 0.8873 → 0.9216) — `e5143a6`.
- **Unblocked 18:40:** `h5py` 3.16.0 installed in `.venv` (owner OK) — the fastMRI+ converter can be written. Known from fastMRI+: v1
  labels = ACL (high-grade), Effusion, Contusion (fracture/contusion box), Baker's (periarticular cyst); meniscus and OA
  need the tear's SIDE from box position + knee side — v2.
**Order to run, each with its own rule when it is pushed:** NYU teacher (fastMRI+ corpus, `--slots cor_1,cor_2
--allow-missing-labels`) → `--predict-all` with `--init` on the competition knees → `build_student_targets.py --gold` (the
free gate) → student (`--init` public CoAtNet, `--targets` student table) → the stack at ≥ 0.947.

**🗺 HOW fastMRI+ FITS — design 2026-09-27 (download ETA ~21:00; not pre-registered).** Purpose: a TEACHER whose signal comes
from radiologists reading IMAGES and which has never seen a competition knee — the one thing report labels cannot give.
1. **Convert** (Mac, free): each of the 1,172 h5 volumes' ready image → our 3-slice windows; its boxes → per-knee labels
   (ACL high-grade 101 · meniscus tear 663 · effusion 142 · fracture / contusion 119 · subchondral edema 196 · cartilage
   full-thickness 122 · cysts 161). **Confirmed 19:10 on the two sample files:** coronal only, PD (`CORPD_FBK`) and PD fat-sat
   (`CORPDFS_FBK`) as SEPARATE files; ready image `reconstruction_esc` 320 × 320 over **140 × 140 mm = 0.4375 mm/px** (ours 0.4167 —
   a 5% resize), 3 mm slices, ~33 per file; a hashed `patient_id` per file (→ the hold-out goes by patient); Siemens Aera 1.5 T and
   Biograph mMR 3 T, `institutionName` NYU (the overlap check's fingerprint). Knee side: not in the header — still open.
   **Download STALLED** at 18:14 at 23.8 GB; **RESUMED 19:10** at the owner's word (links from the NYU email, kept in the session
   scratchpad, never the repo): the partial's last 64 KB matched the server byte for byte, so `curl -C -` continues it. Sizes:
   train **78,016,052,604** B, val **16,050,975,296** B; both in parallel, ~7 MB/s, **ETA ~22:00**; each restarts itself if it
   stalls (`--speed-limit`), under `caffeinate` so the Mac stays awake. A truncated .tar.xz still unpacks up to the
   cut, so the converter can be built on what is there.
2. **Teacher** (~1–2 GPU-h): a backbone that has NOT seen competition labels — Ortho or an ImageNet CoAtNet, NOT the public 0.928
   CoAtNet (it trained on our knees, so its predictions would echo their report labels) — fine-tuned on fastMRI+.
3. **Gate 1, free:** the teacher scores our gold-58 on the findings it covers — radiologists at another hospital, never seen.
   Useful only if it beats the report label there on those findings. **Tightened 09-27 (outside review):** PASS iff report +
   teacher beats report alone on the MACRO over the teacher's covered findings, **paired bootstrap P ≥ 0.95** — never per finding
   (measured: 6 findings paired SE **0.011**, 12 findings 0.008; one finding alone can sit at the ceiling, ACL 0.993). Beside it,
   report-only: the teacher's AUC on a **15% patient-level hold-out of fastMRI+** — if gold fails, it says "did not learn" (fix the
   training) from "did not transfer" (drop it).
4. **Teacher labels:** it predicts all 4,349 training knees' coronal series → combined with `dread_soft` → student targets.
5. **Student:** the best available start (public 0.928 CoAtNet via `--init`, if its pipeline check passes) trained on those
   targets → blended into the 0.943 stack → **valid only at ≥ 0.947**.
**✅ CONVERTED 2026-09-27 22:50 (Mac, 0.3 min) — `pipeline/build_fastmri_cache.py` + `labeling/fastmri_labels.py`, output
`data/external/fastmri/cache_stage_a/` (3.7 GB, git-ignored).** Both archives unpacked clean (1,172 h5 = 973 + 199, xz-verified).
**⚠ 1,172 FILES ARE 556 KNEES:** 513 patients have both PD and PD fat-sat (one exam, two files). Per knee: **ACL high-grade 61
(11%) · Effusion 100 · Contusion 97 · Baker's 111** (the "101 ACL" earlier counted files). Every estimate above that said 1,172
knees is on half the data it assumed. **⚠ ORIENTATION CORRECTED:** the stored images are UPSIDE DOWN relative to ours (kneecap
tissue centre on the most anterior coronal: competition 0.30 of height, NYU as stored 0.67, flipped 0.33; fastMRI+ boxes land on
anatomy only when flipped) → every slice is `np.flipud`-ed; slice order already matches (0 = anterior). The 19:10 "same way up"
read of two samples was wrong. Reused unchanged: `assign_series` (PD fat-sat → `cor_1`, 22 of 30 in the smoke), `pick_depths`
over 2–98%, the 2–98th percentile window, `crop_mm` at each file's own header mm/px. Edge tissue cor_1 **0.22 / 0.80 / 0.80**
(competition 0.22 / 0.81 / 0.75); cor_2 0.08 / 0.76 / 0.74 (0.19 / 0.81 / 0.73). Labels by the OFFICIAL criteria, written before
any teacher: ACL = high-grade only (low-grade 0), Effusion ungraded (over-calls small), Contusion = fastMRI+'s fracture/contusion
box (Fracture left blank), Baker's = periarticular cyst (noisy), the rest blank. Hold-out 83 knees by patient (seed 20260927;
ACL 6, Effusion 12, Contusion 14, Baker's 17 — report-only). Trial run: ResNet34, `--slots cor_1,cor_2 --allow-missing-labels`,
2 epochs on 30 knees, CPU — trains, evaluates, writes (the first "grad-norm nan" is the log's initial value, not a NaN).
**✅ NYU-OVERLAP CHECK 2026-09-28 07:10 — no sign that any fastMRI knee is a competition knee** (owner OK'd the scanner list:
`zhukovoleksiy/rsna-metadata-probe` `headers.parquet`, sha `bb2b67e2` = our Kaggle runs' record). NYU's 1,172 files: Siemens Aera 1.5 T /
Skyra / Biograph_mMR / Prisma_fit 3 T at NYU, HJD, TH; one protocol, coronal PD (TR ~2750–2800, TE 27–32) + PD fat-sat (TR ~2870–3000, TE
33), 15-channel coil, 320 matrix over 140 mm. The competition has those MODELS (Aera 522 studies, Skyra 184, Prisma 109, Biograph_mMR 2)
but not that PROTOCOL: its 2 Biograph_mMR studies run COR T2 FS (TR 3800 / TE 68) + COR T1 — no coronal PD; the only NYU-like TR / TE / 3 mm
coronals are 108 `MAGNETOM Prisma` studies (not Prisma_fit) at 384 × 384, 0.39 mm, software XA30 (2021+), paired with T1, not PD. Nothing
matches on model + protocol, so no thumbnail pass was needed. The converted set goes to Kaggle as a PRIVATE dataset (owner OK'd the
agreement question, 09-28). **Still open:** knee size vs the competition (needs the Stage A cache). ⚠ The NYU cache's manifest carries
the Stage A spec (`c6-64slice`, shard 0 of 1): never attach it beside the Stage A shards — `Corpus` would refuse the mixed set.

**What fastMRI+ could buy — an ESTIMATE, not a gate (owner's question 09-27 ~19:20; gold-58, report = `steven_v2`, the only report
table with gold rows).** Per finding, report → report + an image model (Ortho's in-domain OOF): ACL 0.993 → 0.994 (no room) · Effusion
0.853 → 0.933 · Fracture 0.793 → 0.868 · Contusion 0.831 → 0.887 · Baker's 0.946 → 0.962 · Lateral Meniscus 0.862 → 0.904. On the
12-finding macro: **v1's five findings +0.019** if the NYU teacher were as good on our knees as an in-domain image model (+0.024 with
the menisci, v2); a PERFECT teacher on them tops out at +0.049 (+0.064). Priced at ~half on the LB (`DECISIONS.md` 4c-4) and
discounted for NYU's handicaps (another hospital, coronal only, 1,172 knees, ACL high-grade 101): **a student single model ~+0.003 to
+0.010 LB, ~+0.005 central; less inside the 0.943 stack** (tomorrow's Sub B-dread read gives the member-at-1/3 exchange rate).
`dread_soft` is a better base than `steven_v2`, so the real room is somewhat smaller. Worth its ~2 GPU-h as the only image-read
truth we have; not a route to 0.965 on its own.
*Owner, 19:55: "could it be 2.5%+?"* — **+0.025 is the CEILING, not a forecast:** it is what a PERFECT teacher on v1's five findings
would buy a single model (+0.049 gold × ~½); v2's ceiling ~+0.032. On the final blend, 0.943 + 0.025 = 0.968 would beat today's first
place (0.959) by 0.009. Realistic: single model +0.003 to +0.010, blend less (exchange rate from Sub B-dread tomorrow).
Limits: one plane (ACL and effusion are read best sagitally), another hospital's scanners, one radiologist, side of a meniscus
tear from box position only. Upload a derived cache to Kaggle only after the owner reads the agreement's redistribution terms.

**⏩ THE PUBLIC 0.928 CoAtNet IN OUR PIPELINE — a CHECK, not an experiment (`2026-09-27 17:40 PDT`; owner: "sure download").**
`dreaddevelopment/raptor-knee-maxspan` (CC0-1.0; `raptor_ft_coatnet_v5_full_swa.pt`, sha256 `20d45e07`, 293 MB, local at
`data/external/raptor/`) is OUR layout: 486 tensors, every shape equal, three names differ (`backbone.`→`encoder.`,
`clsW`→`w`, `clsb`→`b`); `train_windows --init` now loads it strictly (tested locally, eval-only with `--epochs 0`). Its
checkpoint records **gold-58 0.9214** (SWA over epochs 14/12/15; gold was in its training at ×3, so this is NOT a held-out read).
The check: that model through OUR Stage A cache and OUR eval, on Kaggle CPU (no GPU quota), must reproduce **0.9214 ± 0.010** on
gold-58 → our pipeline reads images as theirs does, and fine-tuning it on teacher → student labels is a drop-in. Outside ±0.010 →
find the input difference before building on it.
**✅ READ 2026-09-27 19:15 — PASS** (`rsna-knee-c6-train-coatnet-raptor-check`, Kaggle CPU, 115 min; init sha `20d45e07`, 486
tensors strict, 72.9 M encoder, `--epochs 0`): gold-58 **0.9187** (recomputed locally from `oof_gold.csv`: 0.9187), **−0.0027** from the
recorded 0.9214 — inside ±0.010. Our Stage A cache + our eval read images as its own pipeline did: the public CoAtNet is a drop-in
start for the CoAtNet student (step 4). Per finding: ACL 0.982 · MCL 0.982 · Med Men 0.965 · Lat Men 0.901 · Med OA 0.971 · Lat OA
0.836 · PF OA 0.830 · Effusion 0.976 · Synovitis 0.804 · Baker's 0.944 · Contusion 0.914 · Fracture 0.921 (NOT held out: gold was in
its training ×3). Files: `fusion/runs_coatnet_raptor_check/`. The student started from it is judged on the LB only.

**⏩ SUB B-dread — our new-labels model in the public 0.943 stack `pre-registered 2026-09-27 16:40 PDT, before the build`.**
The owner asked 09-27. The stack was re-pulled and is UNCHANGED since Sub B (23/23 code cells verbatim), so this is a clean A/B
against Sub B's **0.936**: same stack, same rule (ours one of three equal families, 1/3, nothing fitted), only our member moves
from 0.901 alone to **0.916** alone. `build_submit_kernel.py --public-dir <pull> --weights-kernel
raahimnawaz/rsna-knee-c6-train-ortho-dread --weights-sha 8b4cdbd8… --tag dread --alone-lb 0.916`. **THE RULE:** PASS (a blend
with our model beats the stack alone) iff LB **≥ 0.944**; else it is not a final candidate. Either way the Δ vs Sub B is the
read: what +0.015 of member strength buys at 1/3. **No re-weighting on the public LB afterwards** (`DECISIONS.md` 2z).
Written before the result: I expect it BELOW 0.943 — the stack's core (`REFERENCE.md` §3.2) is a 0.928 CoAtNet at ~42% of
the vote plus 7+ more members; ours at 0.916 is new diversity (an orthopaedic-MRI ViT-L) but weaker than that core.
**✅ SUBMITTED 2026-09-27 16:29 PDT** (ref **56621730**, `rsna-knee-sub-b-ortho-blend-dread` v1, from `e4306de`). Commit run: weights
sha ok; gate max |Δp| **0.00079**; `ARMS: public stack (2 families, as published) + ortho_stage_a = 3 families; w_ours 0.3333`;
the public publish cell COMPLETE, 3 rows; no traceback. The hidden rerun is the stack's (~8–9 h) — read it tomorrow.
**⛔ READ 2026-09-28 06:45 — LB 0.939: FAIL by the rule (≥ 0.944), as predicted.** +0.003 over Sub B (0.936) for +0.015 of member
strength alone (0.901 → 0.916) at 1/3: **the exchange rate is ~0.2**, and the blend is still 0.004 BELOW the stack alone (0.943). A family at
1/3 of the tuned stack pays only if it is about as strong as the stack's own core (the 0.928 CoAtNet) — so the route is STRONGER students,
above all the CoAtNet student (it starts at 0.928), and then our own blend, not a 0.916 Ortho added at 1/3. No re-weighting on the LB.

**✅ 2026-09-27 09:48 PDT — dread RETRAIN SUBMITTED** (ref **56614330**, `rsna-knee-sub-a-ortho-dread-nall` v1, built from
`cffd681`; the owner said "just submit it when it's done"). Run: 8/8 epochs (time budget hit at epoch 7 iter 1567, as step 5),
the SAME 3,502 train / 691 val studies as step 5 (lists identical), labels sha `8cec477f` as pre-registered; `last.pt` sha256
`8b4cdbd8c784f131`. Commit-run gate: 8 fold-0 studies rebuilt from DICOMs, max |Δp| **0.00079**; `last_meta.json` read (Stage A,
spec c6, RoPE = trained); ARMS `ortho_stage_a only`; 3 rows. **Reported only, NOT the verdict** (`fusion/runs_dread_ortho/`):
gold-58 **0.9121** vs `c6_ortho` 0.8777, paired **+0.034 [+0.015, +0.055]**, 11/12 up (Fracture 0.778 → 0.906, MCL 0.887 →
0.959; Contusion the one down); vs Stage B (`c7`) +0.022 [−0.000, +0.047]. The 691: 0.8440 vs 0.8298, +0.0142 (4.0σ).
**The verdict is the LB against the rule above: ≥ 0.911 PASS · 0.891–0.911 null · < 0.891 loss.**
**✅ READ 2026-09-27 10:14 PDT — LB 0.916: PASS. The labels are a lever for our model.** Same model, same studies, only the
targets changed: 0.901 → **0.916 (+0.015)**, above Stage B's 0.915 on the older, cheaper input. Gold-58 said +0.034; the board
gave 44% of it (`DECISIONS.md`'s "price an arm's LB gain at ~half its gold gain" holds). **Per the rule written before the run:
`dread_soft` (`labels_llm_soft.csv`, sha `8cec477f`) is the training target for every later run, and teacher → student moves up.**
Next, NOT yet pre-registered: Stage B's recipe on `dread_soft` (the two gains, +0.014 input and +0.015 labels, were each
measured alone; whether they add is the question that run answers).
**⏩ OUR-TWO BLEND `pre-registered 2026-09-27 10:40 PDT, before any blend number was computed`.** The owner asked whether a
blend with the new labels was tried (no) and has ~4.5 GPU-h until Fri — Stage B on `dread_soft` (~11 h) cannot run this week.
Candidate, ONE, nothing fitted: the equal per-label rank-mean of `dread` (Stage A, `dread_soft`, LB 0.916) and `c7` (Stage B,
`steven_v2`, LB 0.915). **Offline screen** (OOFs in hand, 0 GPU): build the submission only if on gold-58 the blend's paired
point estimate vs `dread` alone is ≥ 0 (gold cannot resolve a small gain; this only refuses a blend that reads worse) — the
691 is reported beside it, not judged (it is another LLM's reading and trained-label-biased). **LB rule:** PASS (the blend
becomes the main-track candidate) iff LB ≥ **0.921** (best single 0.916 + 0.005); below that, keep `dread` alone.
Not a candidate: `dread` into the public 0.943 stack — ours is still 0.027 below it, and at 0.901 the same move cost 0.007.
**⛔ SCREEN READ 10:45 — FAILS, not built.** gold-58: blend **0.9109** vs `dread` 0.9121, paired **−0.0012 [−0.012, +0.010]**,
P 0.42 (vs `c7` +0.021). The 691 (reported, not judged): blend 0.8523 vs `dread` 0.8440 vs `c7` 0.8410 — the 691 rewards
adding a `steven_v2` model by construction, which is why it was not the judge. Rank ρ `dread`~`c7` 0.83 on the 691. The two
split by finding on gold (`c7` Contusion 0.941 vs 0.888; `dread` Fracture 0.906 vs 0.782), but per-label picking on gold-58 is
CLOSED (`DECISIONS.md`: −0.0002 ± 0.009 out of sample). Diagnosis, not a second chance: on image truth, an old-label model
adds nothing to a new-label one; blend only models trained on `dread_soft`.

**Candidates queued behind tonight's reads** (`DATA.md` §12, from the owner's 09-26 questions; none pre-registered yet): ① an auxiliary SEVERITY head at fixed `steven_v2` targets — the 691 can judge it, one run; ② teacher → student if dread says labels are the lever; ③ MedGemma — owner must accept its HF terms first; ④ MRNet / OAI MOAKS / fastMRI+ — blocked on the host ruling the owner has not yet asked for (`REFERENCE.md` §1.3).

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
- **⛔ READ 2026-09-26 10:16 PDT — BOTH FAIL; `steven_v2` STAYS.** Labeler finished 10:15 (4,407 reports, letter mass
  median 0.9995, min 0.907; manifest: MLX weights `938d8919`, mlx-lm 0.31.3). `bench_qwen_labels.py --write`, once:
  **C1** Qwen alone 0.8765, **−0.0108** [−0.030, +0.009], P 0.14, frozen-six −0.039 (breached) → FAIL.
  **C2** rank-mean 0.8928, **+0.0054** [−0.006, +0.018], P 0.82, frozen-six −0.008 (ok) → FAIL (bar +0.010, CI > 0).
  Diagnostic only, not a second chance: the two Qwen reads, ours and the public Qwen3-8B, both lift **Fracture
  (+0.108 / +0.091)** and Contusion — two readers, but ONE set of 58 studies (18 fracture positives), and per-label
  picking is closed (`DECISIONS.md`). If fracture is ever revisited, it needs an instrument other than gold-58.
  C2's +0.005 is inside the noise this gate was built to refuse; it is not "almost a pass". Budget: 2 of 3 gold reads
  on Qwen-family candidates are spent. **Next per the FAIL branch: `dread_soft`** — covers all 3,502 training studies,
  holds out every gold study (so gold-58 stays clean for the report-only read), judged on the LB against Sub A's 0.901:
  `build_backbone_ab_kernel.py --arch ortho --corpus stage_a --n-train 0 --epochs 8 --targets <dread_soft csv> --targets-from dreaddevelopment/rsna-knee-labels --tag dread` (attached, hash-checked: the embedded 2.1 MB notebook was refused with a 400).
  Not pushed: it is ~11 GPU-h and not a Qwen run, so it waits for the owner's word.
- **⏩ `dread_soft` RETRAIN — the owner said go 2026-09-26 10:20; rule `pre-registered 10:25 PDT, before the push`.**
  Kernel `rsna-knee-c6-train-ortho-dread`: step 5 exactly (Ortho, Stage A, the same 3,502 studies, seed 42, 8 epochs,
  10.5 h budget, one T4) with `--targets labels_llm_soft.csv` (sha256 `8cec477f`, held to `provenance/inputs.json`).
  **THE RULE — judged on the public LB, submitted ALONE through Sub A's path at every window:**
  **PASS (the labels are a lever for OUR model) if LB ≥ 0.911** (Sub A 0.901 + 0.010); **null** within ±0.010;
  **loss** below 0.891. PASS → `dread_soft` becomes the target for every later run (the VLM's full run included) and
  teacher → student moves up; null → table labels are not our lever, go to teacher → student or the model side.
  Reported only, never a second chance: gold-58 paired against `c6_ortho`'s `oof_gold.csv` (honest here — neither
  model saw gold) and the 691 (NOT evidence: it is scored against another LLM reading of the reports).
  ⚠️ To submit it, `build_submit_kernel.py` needs the weights kernel and sha as flags (today they are constants for
  `c6-train-ortho`), and its gate must check against THIS run's `oof_all.csv`. Build that while it trains.
- **VLM bench's job** (`--arch qwen35vl --bench`, after Stage B frees the second GPU session, ≤ 1 GPU-h): (a) 40 fp16
  steps on a T4 with a finite loss throughout; (b) s per micro-step → epochs that fit a 10.3 h session; (c) eval s/study
  at 12 windows; (d) peak memory. It prices the full run; it is not a number. **The CPU smoke passed 10:17** (v4, from
  `3b49bcd`: pinned transformers 5.17.0 over the image's 5.0.0, pinned weights, 348 M of 2,230 M trainable, slot headers
  from the manifest, 2 steps, eval, finite OOF) after three fixes it forced: an in-process reload, two fp32 host
  copies, and no checkpointing on CPU.
- **✅ VLM BENCH READ 2026-09-26 12:40** (`rsna-knee-c6-train-qwen35vl` v5, from `d6c598a`, T4×2, 16 min): **fp16 holds**
  — 40 steps, finite loss throughout, the GradScaler settled from 65,536 to 2,048 (a few overflow steps skipped, by
  design); **peak 13.2 GB** of 15; DDP checksums identical. **Cost: 9.75 s per 2-study micro-step → ~2.4 h per epoch**
  on two T4s (3,502 studies), eval 1.15 s/study at 12 windows. So a 10.3 h session holds **~3 epochs** (step 5 had 8).
  The 40-step read — 0.466 on 60 val studies, gold 0.508 — is an untrained model and says nothing either way.
  **What the bench does NOT show: that this arm learns here.** Proposed next, for the owner: a ONE-epoch learning
  check (~2.8 GPU-h), rule written now: **continue to a full run only if the 691 macro after one epoch ≥ 0.75**
  (Ortho's one-epoch bench on the older corpus read 0.7508); below it, the arm does not learn fast enough for T4
  budgets and is shelved, not tuned on the 691.
- **⛔ VLM LEARNING CHECK FAILS — read 2026-09-26 15:20** (`rsna-knee-c6-train-qwen35vl-e1`, `fusion/runs_qwen35vl_e1/`):
  one full epoch (437 steps, 131 min on T4×2, DDP checksums identical) reads **0.5041** on the 691 (bar 0.75), gold
  0.495. **Shelved, per the rule.** It is not a slow learner, it is a DEFECT: its predictions are constant across studies
  (per-label std ~0.0003 in probability, against Ortho's ~0.26; Spearman with Ortho −0.14..+0.14) — it learned each
  finding's prevalence and nothing from the pixels. Diagnosed locally on real Qwen3.5-2B weights (fixture inputs): the
  PRETRAINED model does pass images to the answer positions — very different inputs move the label-token states 7–19%
  and the logits up to 0.26 — identically in fp32 on CPU and fp16 on MPS, so the fp16 forward is not it. The training
  log is where it goes wrong: the GradScaler fell 65,536 → 256 in the first 150 steps (grad-norm spikes to 15) and stayed
  there. **Checked against the trained `last.pt` (5.2 GB, 15:56):** the frozen LM is untouched (2.9e-8, as it should
  be); **the 24 vision blocks did not move** (median relative change 4.4e-4 — fp16 save rounding) while the vision
  MERGER moved 2.9% and the LoRA factors grew (B from 0 to norm ~0.18); the head sits at its init. **The fp16-underflow
  explanation first written here is REFUTED by measurement:** on the real 2B model in fp32 the vision-block gradients
  per weight are median 1.6e-4–5.7e-4 (merger 2.4e-4, LoRA B 4.8e-4) — ~1,000× above where fp16 at scale 256 loses
  precision (2.4e-7) — and gradient checkpointing passes them (24/24 blocks, locally, on or off). Yet on Kaggle the
  blocks never updated, and the logged total grad-norm (~2) is smaller than the vision blocks alone should give (~10).
  So on the GPU path (CUDA + autocast + DDP + checkpointing) the vision blocks most likely received NO gradient — cause
  **not identified**; nothing on CPU reproduces it. The cheap next diagnostic is a ~15-min GPU run printing per-group
  gradient norms each step. Not scheduled: the arm is shelved by its rule, and this week's GPU is committed to dread.

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

**✅ STAGE B PASSES ITS RULE — read 2026-09-26 12:20 PDT** (`fusion/runs_c7_ortho/`, in-kernel and again locally):
`c7_ortho` **0.8410** vs `c6_ortho` 0.8298 on the same 681 scorable of the 691, paired **+0.0112 ± 0.0029 (3.8σ, P 1.000)**,
8/12 up (ACL +0.029, Baker's +0.034, Lateral OA +0.029, Lateral Meniscus +0.022; Fracture −0.020, Medial Meniscus −0.003).
`avg3` +0.0107 ± 0.0030 (secondary, agrees). Gold-58, reported only: **0.8897 vs 0.8777, +0.012 ± 0.009, P 0.92** — same
labels, so it is a model gain, and gold agrees in direction. DDP checksums identical on both ranks. Cost: 3.52 s per
2-study micro-step per T4, 55 min/epoch on two T4s, eval 2.9 s/study, peak 8.5 GB. ⚠️ **The schedule did not finish:**
fit-budget planned 13 epochs from a 2.79 s probe; the run measured 3.52 s (+26%) and the time budget stopped it at epoch
11 iter 185 (~86% of OneCycle). Fixed for later runs: the probe's margin is ×1.35, not +10% (`train_windows.py`).
**What it buys, per the 09-26 correction above:** a better model on the SAME labels. Before building on it, check the LB:
that needs `infer_windows.py` to learn Stage B (spec c7 + `--canon` in `build_study`, 384 px, the slot/depth embedding and
`pos` from `last_meta.json`) and its `--check-against` gate against `runs_c7_ortho/oof_all.csv`.
**✅ BUILT 2026-09-26 12:20.** `infer_windows.py` now reads the checkpoint's own `last_meta.json` (spec, canon, res,
slots, slot/depth embedding, RoPE periods); without one it is exactly Stage A as submitted. Proven off Kaggle by
`notebooks/infer_parity_fixture.py` on `canon_fixture.py`'s four knees: every window, slot and depth from the DICOM
path identical to the `--canon` cache (max |Δpixel| 0), and the no-canon control differs on every study (139–187).
⚠️ A random-weight model's prediction check passed EVEN the no-canon control (a random ViT is blind to pixels), so
off Kaggle the proof is the tensor check; on Kaggle the trained model's `--check-against` is the gate.
`last.pt` sha256 `406eedaa520ce2cc` (downloaded 12:16, 607 MB). Build and push, when a GPU session is free:
`build_submit_kernel.py --ours-only --weights-kernel raahimnawaz/rsna-knee-c7-train-ortho --weights-sha 406eedaa520ce2ccdaf3eb681ed70a9724912e7e37f9543ae43965225dd00733 --tag c7 --desc "..."`
→ `rsna-knee-sub-a-ortho-c7-nall`. Submit only on GATE PASSED and `ARMS: ortho_stage_a only` (the label is historical).
**Its LB is the read that matters:** A (step 5) scored 0.901; Stage B alone vs that is a same-labels model gain.

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

- **Re-pulled 2026-09-27 10:00 PDT:** top **0.959** (two teams), 10th **0.956**, 50th 0.952, 100th 0.947, 200th 0.944.
  Nobody is at 0.965 (the owner asked 09-27 for a plan to it — none of the levers below is measured to reach it).
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
