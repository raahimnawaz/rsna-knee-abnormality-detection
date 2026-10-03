# How to win from here: brainstorm and plan (2026-10-03)

## 0. The short version

- **Where we stand.** Our public score is 0.939, about 1,940th. Two numbers suggest our true level is about 0.938–0.940. The three public 0.961s carry roughly +0.003 to +0.005 of public-LB selection, so the private winner probably lands near 0.957–0.960 on the public scale. The board is still rising: it moved +0.016 in the 25 days to 09-06, so gold (top ~20 of 4,999) will need about 0.957–0.960 public by 10-22.
- **Where the current trajectory lands.** Kaggle-only, solo, one big run a week, members slotted into someone else's 8–9 h stack: about 0.943–0.946. That is the silver range, not a win.
- **The pending (A) read, due about 16:30, will probably be 0.942–0.943.** To lift the stack +0.004, a replacement core has to score about 0.947 on its own. Our student is 0.931. Treat the core-swap route as closed once (A) reads below 0.947, and stop spending on it.
- **Three owner decisions now matter more than any modelling choice.**
  1. Rent training GPUs, capped at $300.
  2. Pay for a frontier-LLM criteria-graded label read, capped at $150.
  3. Decide on a merger by 10-12. The deadline is 10-15.
- **One free change matters more than any single model:** use 4–5 submissions a day as the image-truth instrument. Only the ~6-minute commit run counts against quota (RUNBOOK.md:244).
- **The honest odds.**

  | Path | Public score | P(gold) | P(win) |
  |---|---|---|---|
  | Solo, with money and the full program | 0.948–0.953 | ~5–12% | ~1% |
  | The same plus a merger with an independent 0.950–0.955 team | 0.955–0.959 | ~30–45% | ~3–6% |

  A 0.95-class new member (a fixed small CNN, the VLM, or a label breakthrough) is the only solo route to the gold zone.

---

## 1. What went wrong

The biggest losses were calendar time and compute, not modelling choices.

**1. Compute was limited by our own choices, and even the free compute was half-used.** Cost: ~0.010–0.020 LB never built (estimate).
- About 90 of the ~270 free T4-hours available since early August were used.
- Kaggle bills per session, not per GPU, yet the Stage A, CoAtNet and CNN runs used one T4 of two. The CoAtNet student, dread, c6 and every ResNet run show `world 1` in summary.json. Source: notebooks/build_backbone_ab_kernel.py:326-327. That is ~48 single-GPU session-hours and ~14–20 quota-hours wasted. AUDIT.md:255 flagged it on 09-26, and the CoAtNet student still ran on one GPU a week later.
- The dread and c6 Ortho runs were cut to 8 epochs by the 12-h session limit; the public core trained 16.
- The Mac sat idle 09-29 → 10-02 (RUNBOOK.md:46).
- Cloud GPUs were priced on 09-24 at $150–250 for ~100 A100-h and declined on 09-27 (archive RUNBOOK-2026-09-28:1218, :132).
- As a result each GPU-week buys one big experiment. That is why the round-1 teacher had to be the weakest model in the project.

**2. Dormancy, and a 3–6 week lag between diagnosis and action.** Cost: every lever now in the score arrived 3–6 weeks late. When active, the project went 0.914 → 0.939 in about 11 days.
- 32 of 58 days have no recorded work.
- No LB submission was made between 08-18 (0.914) and 09-26: 38 days.
- Lags from diagnosis to action:

  | Lever | Diagnosed → acted on | Lag |
  |---|---|---|
  | Dense slice coverage | 08-18 §3z → 09-24 | 37 days |
  | Labels as the lever | 08-13 D4 → 09-27 | 45 days |
  | Copying the frontier's training script | by 09-06 → 09-23 | 17+ days |
  | All-data training | 09-23 → 10-03 | 10 days |

**3. The label lever was dismissed, and its cheapest form was never run.** Cost: the largest measured lever (+0.015) arrived about 6 weeks late, and the API read is still +0.003–0.010 left unclaimed.
- D4 ("labels are not the binding constraint") was restored on 09-06 (DATA.md:39-60).
- Stage L, a frontier-LLM read with the official criteria, priced at $25–150 (DATA.md:257-275), was scheduled for 09-07..09-15 and never ran.
- Only a 4-bit 9B Qwen was tried. It lost overall but read Fracture +0.108 better (DATA.md:530-535).
- dread_soft was benched as "not scorable" on 09-23. The LB reversed D4 on 09-27 (0.901 → 0.916).

**4. The leaderboard, the only large image-graded instrument, went largely unused.** Cost: dozens of image-truth reads forgone during the weeks the label and input levers were found.
- About 13 submissions out of ~290 allowed.
- The public LB holds ~400 radiologist-read studies, about 7× gold-58.
- The belief that "a submission costs ~2 h" still stands in DECISIONS.md:87, though RUNBOOK.md:244 retracted it on 10-02.
- DATA.md:682 says "no instrument can confirm a one-finding fix". That is wrong for a single-column swap: with 11 columns identical, the macro change is ΔAUC_k/12, and only column k contributes noise.

**5. Gates were aimed at questions they cannot answer, and the same gate is in today's plan.** Cost: ~17 GPU-h (CoAtNet student 7.4, Ortho student ~10) and the 10-02 week, with no LB proof.
- Round 1's gate scored the targets (report + teacher) on gold-58: 0.887 → 0.913, P 0.994 (archive :542).
- That is a report-plus-image ensemble, and test studies have no report. A gate of that shape almost cannot fail.
- Students trained on those targets read flat:
  - CoAtNet student 0.931 vs the core's 0.933.
  - Ortho student ~0.927, against ~0.930 if the input and label levers simply added (RUNBOOK.md:63).
  - On gold the Ortho student read 0.911 vs dread's 0.912, so the teacher was about −0.01 on gold.
- The ratio was roughly 5×: gold said +0.026, the LB said ~0 to +0.005.
- Plan item (1) from this morning gates round 2 on the same artifact ("report + Ortho OOF" already reads 0.927).

**6. We modified things before copying them exactly, and acted on attributions nobody had checked.** Cost: about 3 weeks in August and 16 days without local pixels. An unexplained gap of up to ~0.012 sits inside every Ortho number.
- D3: 0.085 was blamed on the NIfTI files and the 458 GB corpus was deleted. The real cause was input sampling (DECISIONS.md:17, :124).
- train_windows ports the frontier's script with "deliberate" differences (train_windows.py:25-38):
  - a confidence weight `0.25+0.75|2y−1|` applied to every target type (:755-760);
  - no gold epoch pick;
  - k=12 random windows.
- The confidence weight was built for steven_v2's "not said" = 0.5. Carried over silently, it:
  - down-weights dread_soft's 0.45 level to 0.33;
  - down-weights teacher-report conflicts to 0.25, against 0.925 for agreeing cells. The student therefore learns least from exactly the cells the teacher corrected.
- Stage A dread Ortho reads 0.916 vs the frontier core's 0.928. The architectures differ, so not all of that gap is setup, but none of it has ever been broken down.

**7. The endgame was built inside someone else's runtime, and that runtime was never measured.** Cost: about 5 days of the endgame critical path, plus 7.3 GPU-h and 3 submissions on a route that by construction could not move the stack.
- The stack takes 8–9 h of 9. Its per-cell cost has never been timed (RUNBOOK.md:242-247), so (C) has been blocked since 09-28.
- The "0.2× exchange rate" rests on two LB points that are each ±0.002–0.005.
- A one-parameter correlation fit (ρ ≈ 0.89) gives the real requirements:
  - adding a family at 1/3 needs about 0.943 alone for +0.004;
  - replacing the 29% core needs about 0.947.
- The CoAtNet student was a 7-epoch, lr 3e-5 restart from a converged SWA checkpoint, with 2.4% relative weight change. It was always going to land near the core.

**8. Closed solo resources, plus heavy process overhead.** Cost: about 1 day a week of critical-path wall clock, and much of the owner's attention.
- No merger, no radiologist, no host question in 7 weeks, no API spend, no TPU or Colab.
- 13,390 lines of markdown. 39 of the 50 commits since 09-28 are docs-only.
- Ops failures: a permission-prompt stall (7 h, 09-25), the Mac asleep (~12 h, 10-02/03), an app update killing a fold, a GPU-only log line killing a 2.5-h CoAtNet run.

**The recurring pattern, still active today.** The process guards hard against false positives, which are cheap at our rank. It lets in false negatives and scarcity, which are what cost us.

| Pattern | Instances |
|---|---|
| A careful measurement aimed at a question it cannot answer | The round-2 gold gate; single LB reads on effects forecast at 0 to +0.005; a 224-px ResNet (~0.88 LB-equivalent) screening levers meant for 0.93 models |
| An underpowered or buggy null closed by a rule | VLM (frozen vision tower, never diagnosed); small CNN (forum recipe never run); NYU-N (weak teacher); zoom (P 0.943, one seed); augmentation (ResNet only) |
| Scarcity treated as a law | One big run a week; ~1 submission a day; the 8–9 h stack treated as fixed |

---

## 2. Where we fall short of a winner

The rows overlap, so do not add them up. The total gap from 0.939 to ~0.957 is about 0.018.

| Dimension | Winner | Us | Est. LB cost |
|---|---|---|---|
| Training compute | A100/H100; hundreds of GPU-h in the endgame (lumbar 2nd: A100, ~24 h; mammo 1st: ~8 days for 4 folds) | 30 T4-h a week, half the GPUs idle in many runs, ~90 of ~270 free T4-h used | Enabler for 0.010–0.020 |
| Iteration rate | 20–60 trained models a week on a cheap proxy, port only the winners | ~3 big single-seed runs a week; the main recipe never tuned | 0.005–0.015 |
| Leaderboard use | 50–150 pre-registered reads (median 117 in one comparison set) | ~13 | 0.005–0.010, through better decisions |
| Ensembling | Folds × seeds × 2–3 families, timed for the 9 h | One seed per model; our members slotted into a public stack at 1/3; our own 4-fold ResNet mean (+0.016 on the 691) never served | 0.005–0.010 |
| Labels | Several weak sources, including an API LLM, strong OOF teachers, judged on images (71st: +0.011; 64th: +0.015 from combining) | One public table (dread_soft) plus a weak-teacher round judged by an artifact | 0.005–0.012 |
| Training recipe | Training bags contain the target region; augmentation; EMA; seeds | k=12 of 54–82 windows (35–71% of positive bags miss a 2–4-window lesion at Stage B); intensity augmentation only; confidence weight; no EMA | 0.003–0.010, mostly the weak focal findings |
| Model diversity | CNN + ViT + another family | Ortho ViT-L plus the public CoAtNets; CNN cut on the wrong recipe; VLM shelved for a bug | 0.003–0.008 |
| Inference engineering | Measured runtime, shared decode, room for members | Stack untimed; each branch decodes again | 0.002–0.005 (room) |
| Team | Most top RSNA finishes are teams of 2–5 | Solo by choice | 0.005–0.012 |
| Operations | Always-on queue | Mac sleep, permission prompts, a page-long RUNBOOK entry per result | ~0.3 iteration over 19 days |

---

## 3. What could work

Expected gains are analyst estimates unless marked as measured. The one-line verdict on each group is in bold.

### 3.1 Labels. Spend dollars before GPU-hours, and judge labels by a student's score on images, never by scoring the targets themselves.

| # | Idea | Mechanism; evidence for / against | Expected LB | Cost | First signal | Fastest decisive test |
|---|---|---|---|---|---|---|
| L1 | **Plain BCE for soft and teacher targets** (drop the confidence weight) | The weight mutes conflict cells 3.7× and graded dread levels about 3×. The public core does not use it. For: train_windows.py:755-760, build_student_targets.py:14-16. Against: down-weighting uncertain cells is a legitimate variant; unmeasured. | +0.002–0.008 | A flag | Tonight (Mac) | B′ ± weight, 2 seeds, paired on the 691. The targets are fixed, so the 691 is a valid judge. |
| L2 | **A student-scored label lab** (Mac ResNet34 students submitted alone to the LB, paired) | Only the LB can see label effects. Calibrate first: steven_v2 vs dread_soft students must show ≥ +0.008. For: gold-scored students tracked the LB in all 3 known cases. Against: a weak student may react to labels differently from ViT-L or CoAtNet; the resnet serve path is unverified. | 0 directly; it makes every other label decision possible | ~2.5 Mac-h, 2 submissions | Sun | Calibration pair. The instrument is valid only if Δ ≥ +0.008. |
| L3 | **Frontier criteria-graded read** of the 4,407 reports (Opus 5.5 Batch; Sonnet costs about half) | Official cut-points verbatim, borderline = negative, graded fields plus location, post-op status, mention flag and a quote, read in the native language. For: dread_soft (a better grading of the same text) gave +0.015; Qwen read Fracture +0.108; the host permits commercial LLMs. Against: text ceiling ~0.90; LLM readers correlate; OA severity is often missing from the text. | +0.003–0.010 single-model | $80–140 (pilot ~$5) | Gold reader read in ~24 h | Freeze the prompt on the 86 hand-labelled + ~100 non-gold reports. Score on gold-58 against steven_v2 (valid, since it is one text source): ≥ +0.010, CI > 0, no finding down more than 0.02. |
| L4 | **targets_v2 as one bundle** | (a) Rank-mean of readers mapped onto dread_soft's levels. (b) Monotone levels P(grader+ \| grade) fitted on gold-58. (c) Mention-masked imputation for Synovitis (87.6% unmentioned), Fracture and Contusion, using a teacher trained with those cells masked, so it is not circular. (d) An LLM re-reads the top 5% of image-vs-report disagreements. Each piece is below resolution; the sum may not be. | +0.003–0.012 single-model | ~1 day plus (c) ~5 Mac-h | ~10-07 | One student vs the dread_soft student, paired on the LB. Adopt only if ≥ +0.008. |
| L5 | **Equal-strength self-distillation test**, before any teacher spend | Round 1 met none of noisy student's preconditions: teacher ≥ student, a noised student, enough steps. This test is ResNet34 trained on targets from ResNet34 folds vs on dread_soft, plus a noised arm. | Decides ~20 GPU-h | ~2.5 Mac-h, 1–2 submissions | Sun–Mon | The teacher route continues only if LB Δ ≥ +0.008. |
| L6 | **Gold-58 in the finals' training** (weight 3), decided now by prior | The only grader-drawn labels; they teach the borderline convention. Cannot be measured, but costs nothing. | +0.000–0.004 | 0.5 day (flag, relax the train∩gold refusal for finals) | Never measurable | Sanity check: the 691 must not drop on the Mac. |
| L7 | Training-only auxiliary heads: location, horn, post-op, PCL, LCL, osteophytes (from L3's fields) | Separates constructs that the 12 labels mix. The 691 can judge it because the main targets stay fixed. Against: aux gains are usually small. | +0.000–0.006 | 1 day + 1 Mac run | ~10-07 | Paired on the 691, P ≥ 0.95. |
| L8 | Report-bias term per language or site, used in training only and dropped at test | Absorbs over-calling (Turkish and Croatian ~21–22%). Against: confounded with real case mix (ComBat hurt). | +0.000–0.005 | 0.5 day | 2 days | Low priority; run only if a slot is spare. |
| L9 | Reconstruct dread_soft's gold rows (if its pipeline is public) | Lets L3 be gated against the incumbent, not against steven_v2. | 0 directly | ~1 h | Same day | The owner checks dreaddevelopment's notebook. |

### 3.2 Model. Our models are short of signal per bag and short of compute, not short of capacity.

| # | Idea | Mechanism; evidence for / against | Expected LB | Cost | First signal | Fastest decisive test |
|---|---|---|---|---|---|---|
| M1 | **Fix the training bag**: k=36, or slot-stratified k=12, or a full-bag head | Eval on 12 of 54 windows loses −0.048 against all 54, and −0.0017 at 36 (archive :1122-1128). At Stage B, 35–71% of positive bags miss a 2–4-window lesion. Against: the 0.928 core trains at k=12; k=36 costs ~3× per epoch. | +0.003–0.010, focal findings | Mac ~4 h | Overnight | B′ vs k=36 (6 epochs) vs stratified, paired on the 691 after the seed floor is known. |
| M2 | **The forum's small-CNN recipe done properly** | Fix BatchNorm (micro ≥ 8 or frozen BN); antialiased resize; mean/max or GeM pooling instead of attention MIL; 288–384 px; in21k EfficientNetV2-S or ConvNeXt-S; EMA. Two top-100 self-reports at 0.94+. Ours sits at ~0.88 LB-equivalent, and resolution alone gave R +0.009. It is 40× cheaper per run than Ortho. | Third family +0.003–0.008; at ≥0.93 it becomes the fast instrument | Mac sweep (free), then ~9 session-h or ~$5 for 5 seeds at 384 | Tonight / 2–3 days | One fold at the forum recipe: ≥ 0.835 on the 691 (~0.907 LB by the Ortho mapping) earns an LB read. |
| M3 | **Stage B CoAtNet**, all data, DDP, a real learning rate, 8 epochs, from the public init | Gives CoAtNet the input that gave Ortho +0.014 on the LB. CoAtNet beats Ortho on Lateral OA (+0.024), PF OA (+0.022), MCL and Fracture. Per-finding Spearman between the two families is 0.80. Against: fine-tuning the core moved nothing at lr 3e-5; the init saw the 691, so only the LB can judge it. | Alone ~0.935–0.945; +0.005–0.010 over the better of it and Ortho | ~4.5 session-h or ~$3 | 1 day | Read alone via --ours-only, paired against 0.931. |
| M4 | **Ortho 5-fold, used three ways** | r1 is already fold 0. Each fold produces OOF for a teacher, leak-free embeddings, and a fold member. Fold and seed averaging: ResNet 4-fold gave +0.016 on the 691. | Ensemble +0.003–0.007; labels +0–0.008 only if L5 passes | ~20 session-h or ~$10–20 | 1 day on cloud | Read the free 2-model family now: Ortho-all + r1 vs Ortho-all alone. |
| M5 | Full-bag or sequence head (GRU or 2-layer transformer) on frozen per-window embeddings | Trains the aggregator without subsampling; cross-slice context (the meniscus "≥2 images" criterion). Every recent RSNA winner has a sequence stage. Against: embeddings of all-data models are in-sample. | +0.002–0.008 | ~1.7 session-h to extract; Mac minutes | ~1 day | r1 embeddings, head arms paired on the 691 against r1's 0.8534. |
| M6 | EMA of weights everywhere | Cheap variance reduction on noisy labels. | +0.001–0.004 | ~30 lines | Overnight | B′ ± EMA, 2 seeds. |
| M7 | **Unshelve the VLM** (Qwen3.5-2B): a 15-minute per-group gradient-norm diagnostic | The vision tower never received gradient, a recorded defect (archive :930-946), not a result. 54th place reports 0.950 single with this model. A distinct family; ~11 min of inference. | +0.003–0.008 if it reaches ~0.94; P(works by 10-15) ~25% | 0.25 GPU-h, then ~10 T4-h or ~$4 | 1 day | 40 steps logging vision vs merger vs LoRA gradient norms, with and without checkpointing and DDP. |
| M8 | Augmentation arm on the big models (affine); mirror-swap with Medial↔Lateral label swap | Ortho and CoAtNet have only ever had ±10% intensity. The mirror-swap is speculative: the knee is not mirror-symmetric. | 0–0.005; mirror −0.002 to +0.004 | 1 run each | 2 days | Run only on cloud, after M1/M2. |
| M9 | A 448–512 px CNN family (Stage C cache built on Kaggle CPU, which is quota-free) | Resolution has been the most reliable input lever. Against: returns diminish; inference ×2. | +0.002–0.006 | CPU free + ~$10 | 3 days | Lower priority. |
| M10 | Dense-slice TTA for Stage B | The window curve still rises from 36 to 54 windows; the core gains +0.005 from denser test sampling. | +0.001–0.004 | ~20 lines + 0.7 session-h | 1 day | r1 with spec c7 vs c7-dense on the 691. |
| M11 | Per-finding spatial attention or a zoom view for cartilage | A 2–3 mm defect is one of ~144 cells in an averaged map. Zoom gave MCL +0.052 and Lateral OA +0.042 on the CNN. | +0.002–0.008 | ~40 lines, 1 Mac arm | 1 day | Only if M2 succeeds. |

### 3.3 Data

| # | Idea | Evidence | Expected LB | Verdict |
|---|---|---|---|---|
| D1 | NYU 7,140 knees pseudo-labelled by a strong teacher that never saw them (the public core, run on the Mac) | N with a weak teacher was +0.0035, P 0.75; +1,149 real knees gave +0.010. Against: fastMRI's non-commercial terms vs a CC BY-NC winner obligation (REFERENCE §1.3 unread). | +0.000–0.006 | Low priority. Only after the owner clears the licence, and never on a cloud VM without reading the agreement. |
| D2 | Harvest the other public 0.942–0.943 notebooks (romantamrazov dinosaur-v5, tony-rank-ensemble); re-pull the public list daily | The free tier went 0.899 → 0.943 in 6 weeks; a new 0.946+ public notebook would reset the base. | +0.001–0.004 or a base reset | Yes, 1–2 h each. |
| D3 | Transductive self-training on the hidden test inside the 9 h | Adapts to the test's scanners. Against: no runtime room; risky. | 0–0.006 | Park. |

### 3.4 Compute

| # | Idea | Evidence | Expected LB | Cost / test |
|---|---|---|---|---|
| C1 | **Rent datacenter H100/A100 for training; Kaggle only for inference and commit runs** | No rule restricts where training happens (COMPETITION_RULES.txt:110 only requires documenting the environment). The host's ruling allows external compute (REFERENCE.md:24-47). Ortho Stage B costs 0.105 s/window on a T4 at ~10 TFLOPS achieved; an H100 in bf16 should be ≥6–12×. | Enabler for +0.010–0.020 | Cap $300; expect $150–250 for ~60–80 H100-h. Test: 1 H100-hour, pull one Stage B shard, 40-step bench, compare s/window with 0.105. |
| C2 | Both T4s on every Kaggle run (DDP, or 2 arms per session) plus checkpoint/resume across 12-h sessions | Kaggle bills the session: the 10-03 quota arithmetic shows 17.2 h charged for one 1-GPU run and one 2-GPU run. | 0 directly; ~2× throughput | 0.5–1 day |
| C3 | Mac on a 24/7 queue (on power, caffeinate, launchd); orchestration on an always-on VM | ~13 h of critical-path delay in 10 days came from prompts, sleep and crashes. | ~0.5–1 day a week recovered | 2–4 h |
| C4 | Train inside the quota-free scoring rerun (LB readout only) | Up to 9 h on 2×T4 per slot at zero quota. Against: invisible outputs; a timeout burns the slot; check the forum and host rulings first. | Indirect | **Only if C1 is refused.** |
| C5 | Colab Pro+ (~$50 for ~40 A100-h), then a Kaggle TPU port | Fallback if a new vendor is unacceptable. | Like C1, smaller | The TPU port is 2–3 days of risk; last resort. |

### 3.5 Ensembling and endgame

| # | Idea | Mechanism / evidence | Expected LB | Cost | Test |
|---|---|---|---|---|---|
| E1 | **Decompose the stack on the LB**: chain alone, CoAt branch alone, minus calibrator, minus DINOv2 | The `_pipeline_stage.csv` seam exists (build_submit_kernel.py:702, 740). This prices every endgame option. | 0 directly; decides +0.001–0.005 of options | Chain-alone 1–2 h of build; CoAt-branch-alone 0.5–1 day; 4 slots | Chain-alone is submittable today. |
| E2 | **Profile the stack per cell, then add a bit-exact I/O read-ahead warmer and shared decode** | Parsing is ~3.1 ms/slice, but Kaggle decode measured 2.6–5.5 s/study/worker, so ~85% is file-open latency, and 7–10 branches each re-read ~210k files. | Frees 1–3 h → room for 2–4 members | ~1–2 quota-h + 1–2 days | 100 train studies behind a fake test root; per-phase timestamps. |
| E3 | **Final 1 = stack with the Ortho family in the chain's 30% slot** (or added at 1/3 if the chain is strong) | Chain ~0.925 → swap gives ~0.946; chain ~0.915 → ~0.948. Also removes the slowest branches. | +0.001–0.005 | 0.5 day | Chain-alone read first: if ≤ 0.928, swap; otherwise add at 1/3. |
| E4 | **Final 2 = the full public CoAt branch + the Ortho family** (+ M3 or M2 if ≥ 0.930 alone), equal rank, nothing fitted, shared decode | Projected 0.945–0.947, vs 0.939–0.941 for this morning's "Ortho + student". Its public score is unbiased, while the stack carries ~10 LB-chosen knobs worth ~+0.002–0.003 of selection. | Public −0.001 to +0.004 vs 0.943; private +0.002–0.006 vs the stack | ~1 day | Read v0 on Tue 10-06. |
| E5 | Fail-loud experiment builds plus a hidden-runtime probe | build_submit_kernel.py:525-531 neutral-fills failed studies silently in the hidden run, so a degraded read looks like a weak model. | Protects every read and both finals | 1–2 h | Next build. |
| E6 | **Per-finding probes / single-column swaps** | Other 11 columns at 0.5 → that finding's public AUC = 12×LB − 5.5, ±0.006 at 3 decimals. A one-column swap resolves ±0.012–0.015 on that finding. Use to direct effort, never to pick per finding. | +0.002–0.005 via the weak four | ~12 slots | Synovitis-only probe of Ortho r1. |
| E7 | **Pre-registered final-selection rule** (written by 10-15) | ≤4 candidates per final; ignore public gaps < 0.004 between structurally different candidates; discount stack-shaped candidates by ~0.002–0.003; require measured runtime margin ≥45 min. | ~+0.001 expected private | 1 h | — |
| E8 | **Merger with an independent 0.950–0.955 team** | Our ~0.946 + their ~0.953 at ρ 0.85 gives ~0.956; two 0.953 pipelines give ~0.959. Adds 30 quota-h a week. What we bring: a non-CoAtNet ViT-L Stage B family (Spearman ~0.80 with CoAtNet) and the label work. | +0.005–0.012 | Social; we have ~13 subs, well within the cap | Owner decision by 10-12. |
| E9 | Efficiency track | Score = (max−AUC)/(max−0.5) + runtime/32400. A 0.93 model in 15 min (~0.095) beats a 0.945 in 1 h (~0.146). It must be one of the two finals. | Prize money, not rank | — | Not unless it costs nothing. It conflicts with making Final 2 strong. Check the host's daily efficiency board once. |

### 3.6 Process

| # | Change | Why |
|---|---|---|
| P1 | **4–5 LB reads a day from a ledger, each with its rule written first.** Model vs model only if \|Δ\| ≥ 0.010; blend changes only if ≥ 0.004; never pick among more than 3–4 near-equal variants. | ~95 slots remain. Hidden reruns are free. Best-of-10 at SE 0.005 inflates ~+0.008. |
| P2 | **The 691 as a calibrated proxy for the Ortho family**: LB ≈ 1.091 × 691 − 0.004 (4 points, r 0.996, residuals ≤ 0.0014; checked by hand). 0.95 needs ~0.874 on the 691. | Same-hour decisions. Pre-register a check: the next held-out Ortho run must land within ±0.004. CoAtNets from the public init are excluded (that init saw the 691). |
| P3 | **Measure the seed floor once**: B′ seed 2 tonight. Never close a lever whose effect is smaller than seed noise. | ResNet recipes one flag apart rank-correlate at 0.81. |
| P4 | **Pre-registration and 2-SE rules for the finals only.** Development takes anything that holds across 2 seeds or 5-fold OOF and is confirmed in the stacked sum on the LB. | The current rules filter out the +0.002–0.005 gains the endgame is made of. |
| P5 | A 10-minute end-of-day log instead of RUNBOOK essays. Fix DECISIONS.md:87 ("~2 h per submission") so a fresh session does not ration submissions again. | ~1 day a week of attention. |

### 3.7 Closed correctly, and closed too early

| Closed correctly (keep closed) | Closed too early (reopen) |
|---|---|
| Fitted stackers, per-finding picking, calibration fitted on gold | Small CNN as a family: the forum recipe was never run (M2) |
| ComBat / site harmonisation (−0.013 to −0.032: real case mix) | VLM: shelved for a bug, not a result (M7) |
| fastMRI+ teacher (P 0.21) | Frontier-LLM label read: closed by analogy to a 9B 4-bit model (L3) |
| Qwen labels as a *replacement* table | Combining label sources: closed on the targets-as-predictor instrument (L4) |
| Zoom as a recipe *lever* (P 0.943) | Zoom as an ensemble *member* (low priority) |
| The core-swap route (after (A) reads) | Synovitis decoupling: the test used a circular teacher (L4c) |
| NIfTI-as-cause (D3 amended) | "No instrument for a one-finding fix" (E6) |
| Site prior | "Blend of our own two models": decided with a weak ResNet; the real pair is untested (E4) |
| | Augmentation on the big models; seed and fold ensembling ("~+0.001") |
| | NYU with a strong teacher (low priority; licence-gated) |
| | Cloud GPUs and the merger: owner closures, now reconsidered with prices and medal rules |
| | Gold-58 in training or epoch-picking for the finals (L6) |

---

## 4. How a winner would play the next 19 days

### 4.1 Decisions only you can make (answer today if possible)

| # | Question | My recommendation |
|---|---|---|
| Q1 | **Rent training GPUs, capped at $300?** Datacenter or "secure cloud" H100/A100, private encrypted disk, data deleted at the end. | **Yes.** This decides whether we get one more label/teacher iteration (Kaggle-only) or four or five. The 4-fold Ortho teacher is ~$10–20 on cloud versus 2/3 of a quota week. |
| Q2 | **API label read, capped at $150?** Opus 5.5 Batch; a ~$2 gold-58 pilot gates the full run. | **Yes.** It is the only label lever that costs dollars rather than GPU. |
| Q3 | **Merger by 10-12** (Kaggle deadline 10-15)? Please verify on Kaggle's progression page, but as far as all six analysts know, team medals count toward Competitions Master (1 gold + 2 silver); only Grandmaster needs a solo gold. What medals do you hold? | **Open conversations this weekend with 2–3 teams at 0.950–0.955 that describe non-stack pipelines. Merge if by 10-12 our best public score is < 0.950.** No solo path we can see reaches gold with better than ~10–15% probability. |
| Q4 | Use 4–5 submissions a day under the ledger rules (P1)? | **Yes.** |
| Q5 | Gold-58 in the finals' training (weight 3), giving up gold as a sanity read for those runs? | **Yes**, decided now by prior. |
| Q6 | Keep the Mac on power, lid open, with permission prompts pre-approved through 10-22? Move orchestration to an always-on VM? | **Yes to both.** |
| Q7 | Final 2 = full CoAt branch + Ortho family, untuned, even if it reads 0.002–0.003 below Final 1 on public? | **Yes.** It is the private-LB hedge. |
| Q8 | What happened 08-20..09-05 and 09-08..09-22? If more dead days are possible, front-load the finals' training. | Plan assumes you are available about 2 h a day minimum. Labels freeze by 10-09 regardless. |

### 4.2 Parallel tracks and dated sequence

**Sat 10-03 (today)**
- **By 12:00.** Answer Q1–Q3. Send merger feelers.
- **Today's 2 expiring slots (reset at 17:00 PDT).**
  - Slot 1: **chain-alone** (E1a), if its build passes the seam checks by ~15:30.
  - Slot 2: **the existing 4-fold ResNet34 alone** (diagnostic for M2), if the `--ours-only` resnet path loads.
  - Fallback for either slot: a Synovitis-only probe of Ortho r1 (E6).
- **~16:30, (A) reads.** Pre-registered: ≥ 0.947 keeps the core-swap route. Otherwise it is closed for good, which is expected at ~0.942–0.943. Do not re-read it.
- **Cloud.** VM up; pull one Stage B shard; 40-step bench. Run a tiny bf16 → fp16 inference parity check through `infer_windows --check-against`.
- **~18:00.** Download the Ortho-all weights.
- **Kaggle (~4 h of quota left until 10-09 17:00).** Push the stack per-cell timing run on 100 train studies (~1.5 h) tonight. Keep ~1 h for parity checks of cloud weights.
- **Mac overnight queue, in priority order:**
  1. B′ seed 2 (seed floor).
  2. B′ with plain BCE (L1).
  3. Forum-recipe CNN: mean/max pooling, micro 8, antialias, 288 (M2).
  4. k=36 for 6 epochs (M1).
  5. Slot-stratified k=12 (M1).
  6. Recipe C on steven_v2 (L2 calibration).

**Sun 10-04**
- **LB (5 slots).**
  - Ortho-all alone (P2 check: 691 not available for an all-data run, so this is the base read).
  - Ortho-all + r1 family.
  - Label-lab calibration pair: steven vs dread ResNet students (2 slots).
  - One probe.
- **Cloud.** Start Ortho folds 1–4. Each saves OOF, per-window embeddings, last.pt and avg3. Start the Stage B CoAtNet all-data run (M3, lr not restarted at 3e-5 from SWA). Use DDP everywhere.
- **API.** Freeze the prompt. Run gold-58 (~$2). Gate per L3.
- **Mac.** Read last night's arms, then the L5 equal-strength pair.

**Mon 10-05 – Wed 10-07**
- **Cloud.** Folds finish → OOF + embeddings. Ortho-all seed 2. If M2 passes on the Mac, a 5-fold CNN at Stage B 384. VLM 15-minute diagnostic on Mon (M7).
- **API.** Full batch on Mon if the gate passed. Build targets_v2 Tue–Wed (L4).
- **Engineering.**
  - Multi-weight, shared-decode `infer_windows` (Final 2 skeleton) on Mon–Tue.
  - CoAt-branch-alone build (E1b) on Mon.
  - Fail-loud builds (E5).
  - I/O warmer, if the timing shows decode dominating (E2).
- **LB, ~4–5 a day.**
  - CoAt-branch-alone (Mon).
  - Stage B CoAtNet alone (Tue).
  - **Final 2 v0** (Tue).
  - **Final 1 v0**, chain-slot swap or add at 1/3 depending on the chain read (Wed; 8–9 h read).
  - Label lab: round-1 targets vs dread student, and plain BCE vs weighted.
  - Probes on the weak four.
- **Mac.** Head arms on r1 embeddings (M5), EMA (M6), aux heads (L7) if L3's fields have landed.

**Thu 10-08 – Fri 10-09**
- **Thu night: LABEL FREEZE.**
  - `targets_final` = dread_soft or targets_v2 (the LB decides).
  - Loss: plain or weighted.
  - Gold weight 3.
  - Round-2 teacher targets only if L5 read ≥ +0.008.
- **Fri: RECIPE FREEZE** (k, EMA, augmentation, CNN family yes/no, VLM yes/no).
- **Fri 17:00 quota reset.** Kaggle goes to commit runs, CPU cache builds and fallback.

**Sat 10-10 – Wed 10-14: final training (cloud)**
- Ortho all-data × 3 seeds on the final targets + gold.
- Stage B CoAtNet × 2–3 seeds.
- CNN family if it passed.
- The fold models remain members.
- Re-fit a stack-core-slot CoAtNet on new labels **only** if the label lab showed ≥ +0.008. Otherwise skip it; no third core attempt.
- Assemble Final 1 v1 and Final 2 v1 and read each once.
- **Merger decision Mon 10-12** (complete by 10-14).

**Thu 10-15.** All recipes frozen. Merger deadline. The selection rule (E7) is written and committed.

**10-16 – 10-20.** No new ideas. Runtime hardening (≥45 min measured margin on at least one final). One confirming read per candidate, ≤4 candidates per final. Buffer for failures: stack reads take 8–9 h, so at most ~2 a day.

**10-21.** Last candidate submitted by 08:00 PDT.

**10-22.** Select both finals by the rule before 12:00 PDT. Nothing after that.

### 4.3 Budget

| Resource | Allocation |
|---|---|
| Cloud GPU | Cap $300. Plan ~60–80 H100-h, ~$150–250: folds ~5 h, Ortho seeds ~6–9 h, CoAtNet ~5 h, CNN family ~5 h, VLM ~3 h, round-2/v2 students ~5 h, bag/aug variants ~10 h, 2× slack. |
| API | Cap $150 (gold pilot ~$2; full batch $80–140; adjudication $10–25). |
| Kaggle quota | ~4 h to 10-09 (timing + parity). 30 h for the 10-09 week (commit runs, Stage C CPU builds which are free, fallback training). 30 h for the 10-16 week (commit runs + reserve). |
| Mac | 24/7 queue: proxy recipes, label lab, heads. |
| Submissions | ~95: ~12 probes, ~25 decomposition/family reads, ~25 label lab, ~15 final candidates, ~15 slack. |

**If Q1 is "no" (Kaggle-only fallback, stated honestly).** One label iteration in total:
- 10-09 17:00: folds 1+2 in 2 sessions (both GPUs).
- ~22:00: folds 3+4.
- 10-10: a round-2 Ortho student (8.6 h).
- The 10-16 week: one DDP Stage B CoAtNet (~4.5 h) + one Ortho seed (8.6 h) + commit runs.
- Use C4 (training inside the rerun) for LB screens, after checking the rules.

Expect the top of the "solo, no money" row in §5.

### 4.4 Kill criteria

| Bet | Kill if... | By |
|---|---|---|
| Cloud | Bench < 4× a T4 per dollar, or fp16 parity on Kaggle fails | 10-05 |
| Core swap (A) | Reads < 0.947 | Today |
| L2 label lab | Calibration pair Δ < +0.008 (the instrument cannot see labels) | 10-05 |
| L3 frontier read | Gold gate < +0.010 or any finding down > 0.02. Stop at $2. | 10-05 |
| L4 targets_v2 | Paired LB < +0.008 → freeze dread_soft | 10-08 |
| L5 / round-2 teacher | LB Δ < +0.008 → no teacher targets; folds serve as members only | 10-06 |
| L1 plain BCE | 691 paired Δ ≤ seed floor → keep the current loss | 10-04 |
| M1 bag size | Not > 0 at P ≥ 0.95 and above the seed floor → keep k=12 | 10-05 |
| M2 small CNN | < 0.835 on the 691 after 2 arms × 2 seeds → instrument only, no family | 10-07 |
| M3 Stage B CoAtNet | Alone < 0.933 → not in Final 2 | 10-07 |
| M7 VLM | Cause not found by 10-06, or alone < 0.930 by 10-12 | 10-06 / 10-12 |
| Any third family | Alone < 0.930 → stays out of the finals | 10-12 |
| Solo path | Best public < 0.950 → merge if a partner exists | 10-12 |

### 4.5 What must be true at each checkpoint

- **By Fri 10-09.**
  - Cloud-trained weights served on Kaggle with sha and shape parity, or the no-cloud plan is committed.
  - Labels frozen, with an LB go/no-go.
  - Stack per-cell timings known.
  - Chain-alone and CoAt-branch-alone read.
  - Ortho family ≥ 3 members (all-data + r1 + folds).
  - Stage B CoAtNet read.
  - Small-CNN and VLM verdicts.
  - Final 1 v0 and Final 2 v0 each read once.
  - Seed floor known.
  - ≥ 20 ledgered reads.
- **By Thu 10-15.**
  - Merger decided (and done, if yes).
  - Recipes frozen.
  - Final-training seeds done or running.
  - Both finals' v1 read.
  - Measured runtime margin ≥ 45 min in at least one final.
  - Selection rule written.
- **By Thu 10-22.** Both finals chosen by the rule from ≤ 4 candidates each, all of which scored by the night of 10-21.

---

## 5. The honest odds

Board drift of about +0.003–0.005 by 10-22 is assumed. All probabilities are judgement calls with wide error bars.

| Path | Plausible public | Private rank band | P(gold, top ~20) | P(top 10) | P(win) |
|---|---|---|---|---|---|
| A. Status quo: Kaggle-only, solo, this morning's plan | 0.943–0.946 | ~200–600 | ~1–2% | <1% | <0.5% |
| B. Solo, no money, all the free fixes (LB 4–5/day, both T4s, Mac 24/7, decomposition, chain slot, CoAt-branch Final 2) | 0.945–0.949 | ~100–300 | ~3–5% | ~1% | <0.5% |
| C. Solo + $300 GPU + $150 API, full program | 0.948–0.953 | ~40–200 | ~5–12% | ~2–4% | ~1% |
| D. C, plus a 0.95-class member lands (P ≈ 20%) | 0.953–0.957 | ~10–60 | ~25–40% (conditional) | ~10–20% | ~2–4% |
| E. C + merger with an independent ~0.952 team | 0.955–0.959 | ~5–40 | ~30–45% | ~12–25% | ~3–6% |

**Reading this.** An outright win is a long shot on every path. The top 10 are within ~0.003 of each other and will reshuffle privately; that is the only reason it is not zero. Gold is the realistic "winning" target. It needs money plus either a merger or a breakthrough member. The 0.965 goal is above the projected private winner (~0.957–0.960 public-scale) and should be retired as a planning number.

---

## 6. This morning's plan, kept and corrected

| Earlier item | Verdict | What changes, and why |
|---|---|---|
| (1) 4 more Ortho folds (~20 GPU-h) → round-2 targets, gated on gold-58 vs 0.913 | **Changed** | The gate scores report + teacher, an ensemble that cannot exist at test time. It cannot fail (report + Ortho OOF already reads 0.927), and round 1 showed it overstates the effect ~5×. Keep the folds, run on cloud for ~$10–20, as members, OOF and embeddings. Use their labels only if the Mac equal-strength test (L5) or the LB label lab passes ≥ +0.008. Without cloud they lose priority to the Stage B CoAtNet and an Ortho seed. |
| (2) Auxiliary "where" outputs on the Mac | **Kept, demoted** | It needs location fields that only the frontier read (L3) provides cleanly. Run after 10-06. The 691 can judge it because the main targets are fixed. |
| (3a) All-data Ortho tonight (~0.937) | **Kept** | Read alone on Sunday, plus the free Ortho-all + r1 family read. |
| (3b) Own blend = Ortho-all + CoAtNet student | **Changed** | The student roughly equals the core (0.931 vs 0.933). Use the **full public CoAt branch** (raptor ×4 + mattiaangeli ×4, ~70% of the stack's vote) + an Ortho family at equal rank, with nothing fitted. Projected 0.945–0.947 vs 0.939–0.941. This becomes Final 2. |
| (3c) Round-2 student in the 10-16 week | **Changed** | Too late for a label decision. Labels freeze Thu 10-08; the 10-10..10-14 window trains finals only. |
| (4) One Mac run of the forum small-CNN recipe | **Promoted** | Tonight's top Mac job, together with the BatchNorm and antialias fixes, the bag-size test, plain BCE and the seed floor. If it reaches ≥ 0.835 on the 691, it becomes the fast instrument and a third family. |
| Estimate: own blend ~0.935–0.945; lifting the stack to 0.947 needs a ~0.95 core | **Corrected** | A core replacement needs ~0.947 alone. A better seam is the chain's 30% slot (+0.001–0.005, decided by the chain-alone read). With the full CoAt branch, Final 2 is ~0.945–0.947. |
| Not in the plan | **Added** | Cloud and API decisions; the merger decision by 10-12; stack decomposition and timing; plain BCE; bag-size fix; Stage B CoAtNet; VLM diagnostic; fail-loud builds; per-finding probes; 4–5 reads a day; gold in the finals; a written selection rule; label freeze 10-08; recipe freeze 10-09; everything frozen 10-15. |