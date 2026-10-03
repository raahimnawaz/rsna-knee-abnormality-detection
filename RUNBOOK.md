# RUNBOOK — the current state, on one page

**Any session starts here.** This page holds only what is true now and what runs next. It was rewritten 2026-09-28
~20:45 PDT. Everything before that (every pre-registration, read and owner Q&A) is in
[`docs/archive/RUNBOOK-2026-09-28.md`](docs/archive/RUNBOOK-2026-09-28.md). An older reference to a RUNBOOK § or
line points there.

**Keep it one page.** When a pending read lands, give it one line under *Settled* and move its full block to
[`docs/LOG.md`](docs/LOG.md) (append-only, newest last). Owner Q&A goes to the doc that owns the topic
(`DATA.md` / `MODEL.md` / `REFERENCE.md`) or to `docs/LOG.md`, not here.

## Goal — 0.965 AUC (owner, 2026-09-27)

A **valid** improvement over the free public 0.943 stack: pre-registered, at least 2 SE, never tuned on the public LB.
The public LB is ~30% of the test (~400 studies). There, one score is ±0.008, a model-vs-model difference ±0.005 and a
blend change ±0.002. So valid = **≥ 0.947 for a blend change on the stack** and **≥ 0.953 for a different standalone
model**. The prize line and the efficiency track do not matter. Only necessary work: every run must change a
decision. **Entry and team merger 10-15; final 10-22 16:59 PDT.** Two finals; Kaggle ranks a team by the better one.

## Scoreboard

| | LB | note |
|---|--:|---|
| #1 on the board | **0.961** | 10-02 10:45 (10th 0.958; was 0.960 / 0.956 on 09-28) |
| free public stack (`mattiaangeli` bend-the-knee, Apache 2.0) | 0.943 | our base; ~8–9 h of the 9 h limit |
| its core, public CoAtNet maxspan-v5 alone (CC0) | 0.928 | our CoAtNet student's start (the author's notebook) |
| the same weights through the STACK's core code (`--core-alone`, ref `56779801`) | **0.933** | (B)'s paired baseline, 10-02 |
| **ours, best** — Sub B-dread (Ortho `dread` at 1/3 in the stack) | **0.939** | still below the free stack |
| ours alone — `dread` (Ortho, Stage A, `dread_soft`) / `c7` (Ortho, Stage B, `steven_v2`) | 0.916 / 0.915 | Sub A (Ortho, Stage A, `steven_v2`) 0.901 |

**Measured levers, never yet combined in one model:**
- labels `dread_soft`: **+0.015 LB**;
- Stage B input: **+0.014 LB**;
- CNN image teacher → student targets: **+0.026 on gold-58**, P 0.994 (LB unmeasured);
- blending: the stack is +0.015 over its best single model;
- all-data finals: +0.010 LB in the public model's own ablation.

**Exchange rate:** a member at 1/3 in the stack moves it ~0.2× its own gain (Sub B-dread). So strong students must
**replace** the stack's core, not be added to it.

**Path to #1.** The CoAtNet student starts at 0.928 and adds teacher labels, Stage-B-grade input and all-data finals
→ a ~0.95 single model. Two such students plus the stack's other families → 0.96+. The gains overlap, so each keeps
its own gate. The first LB proof is the CoAtNet student, after Friday.

## Running now — nothing (the Mac is idle since R finished at 11:39 on 09-29)

## Pending reads — each rule was written before its result

1. **The students on the LB, Sat–Sun after Friday** (owner confirmed 09-28 ~18:40):
   - **(A)** upgraded stack (the CoAtNet student in the maxspan-v5 slot): PASS iff **≥ 0.947** → final candidate A.
   - **(B)** CoAtNet student alone, reported (no gate). **Read (owner 10-02: "sure"): student-alone − public-alone,
     the same `--core-alone` notebook, only the weight file differs**, not vs 0.928 (the stack's core code samples up
     to 96 slices and adds a reversed pass, so 0.928 is another notebook's number). |Δ| < 0.010 is noise (2 × the
     ±0.005 model-vs-model SE). **The public-alone half read 0.933** (ref `56779801`, from `3f15198`): the stack's
     core code is worth +0.005 over the author's notebook, on the same weights.
   - **(C)** Ortho student: joins (A) as a family only if it lifts (A) by **≥ +0.004**. Otherwise Ortho is cut from
     the finals.
   - If (A) fails, the teacher route has not shown on the LB. Next are round 2, more knees and zoom, each screened
     first.

## 🗓 FRIDAY 10-02 PUSH LIST — ✅ READY

The scheduled task `rsna-push-stage-b-dread` (Fri 17:20 PDT) pushes ONLY what this block lists. Kaggle runs two GPU
sessions at once; both entries go together.

**READY, re-read 2026-09-29 ~13:45.** Entry 1 is now the ALL-DATA CoAtNet student (owner 09-29: "yes train on
them"). Its CPU smoke from `b90ab08` read clean at 13:30:
- targets sha `75757be8`; partition check 691/691;
- **train 4,349** (every non-gold knee), val = the **58 gold** knees only (train ∩ gold 0);
- init strict, 486 tensors, sha `20d45e07`;
- epoch 0 ran; no traceback.

Entry 2 (Ortho) is unchanged: its v2 smoke from `b5331ff` read clean 09-28 ~20:50. Each entry is built from a
detached worktree pinned to the commit its smoke ran, so later commits cannot change what runs. Both commands below
were test-built at ~13:40 09-29 (not pushed) and pass every flag check.

1. **CoAtNet student, ALL DATA** — kernel `raahimnawaz/rsna-knee-c6-train-coatnet-student-r1-all`. Build (one line):
   `cd /Users/raahimnawaz/rsna-knee-abnormality-detection && rm -rf /tmp/rsna-friday && git worktree prune && mkdir -p /tmp/rsna-friday && git worktree add --detach
   /tmp/rsna-friday/src-coat b90ab08 && cd /tmp/rsna-friday/src-coat && /Users/raahimnawaz/rsna-knee-abnormality-detection/.venv/bin/python notebooks/build_backbone_ab_kernel.py
   --arch coatnet --corpus stage_a --n-train 0 --epochs 7 --targets /Users/raahimnawaz/rsna-knee-abnormality-detection/data/student/targets_round1_cnn.csv --targets-from
   raahimnawaz/rsna-knee-student-targets --tag student-r1 --extra-datasets dreaddevelopment/raptor-knee-maxspan
   --train-args '--init /kaggle/input/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt' --all-data --out /tmp/rsna-friday/coat`.
   **Flag checks:** prints `sha256 75757be8c9b5c8d1`, `n_train=ALL (4,349, nothing held out)`, `mode=full epochs=7`,
   `from b90ab08`, no "uncommitted"; `kernel-metadata.json`: id as above, `"enable_gpu": true`,
   `dreaddevelopment/raptor-knee-maxspan` and `raahimnawaz/rsna-knee-student-targets` in `dataset_sources`; the
   `.ipynb` under 1,000,000 bytes (test build: 692,175). Push dir `/tmp/rsna-friday/coat`. ~7.3 GPU-h (one T4,
   ~62 min/epoch on 4,349).
2. **Ortho student** — kernel `raahimnawaz/rsna-knee-c7-train-ortho-student-r1`. Build, after entry 1's (its own
   worktree): `cd /Users/raahimnawaz/rsna-knee-abnormality-detection && git worktree add --detach /tmp/rsna-friday/src b5331ff && cd /tmp/rsna-friday/src &&
   test "$(git rev-parse --short HEAD)" = b5331ff && /Users/raahimnawaz/rsna-knee-abnormality-detection/.venv/bin/python
   notebooks/build_backbone_ab_kernel.py --arch ortho --corpus stage_b --n-train 0 --epochs 14 --targets
   /Users/raahimnawaz/rsna-knee-abnormality-detection/data/student/targets_round1_cnn.csv --targets-from raahimnawaz/rsna-knee-student-targets --tag student-r1 --out
   /tmp/rsna-friday/ortho`. **Flag checks:** `sha256 75757be8c9b5c8d1`, `mode=full epochs=14`, `from b5331ff`, no
   "uncommitted"; id as above, `"enable_gpu": true`, the eight `raahimnawaz/rsna-knee-stage-b-s{0..7}of8` in
   `kernel_sources`, `raahimnawaz/rsna-knee-student-targets` in `dataset_sources`; `.ipynb` under 1,000,000 bytes
   (test build: 689,467). Push dir `/tmp/rsna-friday/ortho`. ~10 GPU-h (T4×2, fitted to 10.3 h). It keeps the 691
   holdout: for Ortho the 691 is a clean check.
3. ~~**NYU teacher**~~ **CUT 09-28 16:25** (failed its gold gate on the Mac, P 0.21) — not pushed.

**Rules:** see *Pending reads* item 1.

**What happened 10-02** (the Mac slept on battery with the lid shut, so everything ran late):
- The 17:20 scheduled task started 17:26, read the block, then sat on a permission prompt again and built nothing.
  Stopped at 18:05; pushed by hand. Every flag check passed, sizes 692,175 / 689,467.
- **⏳ PUSHED 10-02 18:07 PDT** raahimnawaz/rsna-knee-c7-train-ortho-student-r1 v3 (built from b5331ff). RUNNING.
- **❌ PUSHED 10-02 18:07 PDT** raahimnawaz/rsna-knee-c6-train-coatnet-student-r1-all v2 (built from b90ab08): **ERROR**
  8 min after its start. `train_windows.py:702` logged `len(model.encoder.blocks)`, and a CoAtNet (timm MaxxVit) has
  `.stages`, no `.blocks`. GPU only: the CPU smoke never enters that branch (no amp). Everything before it was right
  (targets `75757be8`, train 4,349, val = 58 gold, init strict 486 tensors).
- **Fix `dc70f5c`:** the log line only; nothing that trains changed. Run locally past it and to the end (`--mps`,
  `TW_FORCE_CKPT=1`, real Stage A, the public init, 2 epochs, gold, `--save-weights`, avg3): exit 0. Its `last.pt` +
  `last_meta.json` meet the core cell's checks (meta equal, 486 tensors, key set and shapes = the public core's).
- **⏳ PUSHED 10-02 20:36 PDT** raahimnawaz/rsna-knee-c6-train-coatnet-student-r1-all v3 (built from dc70f5c: entry 1's
  command with `dc70f5c` for `b90ab08`; 692,987 bytes; the only code difference is that fix). Due ~Sat 04:15.

## 🗓 WEEKEND RUN SHEET (written 10-02 ~11:30; each step submits only if its check line is in the log)

All builds run from `main` HEAD with no "uncommitted" warning. Check the public stack is unchanged first: `kaggle
kernels pull mattiaangeli/bend-the-knee-to-speedy-raptors-the-original -m`, notebook sha256 starts `2fc5a420`. At most 2
GPU sessions run at once, and the Ortho student holds one until ~Sat 05:00, so push the commit runs one at a time.
**The steps fire from the open desktop session: the Mac must be on power with the lid open.**

1. ~~Fri 17:12: the Friday push list~~ **done late, see the push list:** Ortho 18:07, CoAtNet 20:36 after a fix.
2. **Sat ~04:15, CoAtNet student COMPLETE →**
   1. `kaggle kernels output raahimnawaz/rsna-knee-c6-train-coatnet-student-r1-all -p fusion/runs_coatnet_student_r1_all
      --file-pattern 'runs_coatnet/(last\.pt|last_meta\.json|summary\.json|oof_gold\.csv)$'`, then `shasum -a 256` its `last.pt`.
   2. **(A):** Open item 2's build → push → submit only if the log has `[core] ... relative change` and `[core] ARMS:`.
   3. **(B):** `build_submit_kernel.py --public-dir data/public_notebooks/bend-the-knee-2fc5a420 --core-alone --core-kernel
      raahimnawaz/rsna-knee-c6-train-coatnet-student-r1-all --core-sha <sha> --tag student-r1 --out /tmp/alone` → push →
      submit only if the log has `[core]`, `[alone] ARMS: ... student` and `0 failed`.
3. **Sat ~05:20, Ortho student COMPLETE →** download `runs_ortho/(last.pt|last_meta.json|oof_all.csv|summary.json)` the
   same way, and read its 691 score (`score_runs_ortho.txt`). **Ortho alone:** `build_submit_kernel.py --ours-only
   --weights-kernel raahimnawaz/rsna-knee-c7-train-ortho-student-r1 --weights-sha <sha> --tag student-r1 --out /tmp/suba-os`
   → push → submit only if the log has `GATE PASSED` and `[sub A] ARMS:`. Its LB is (C)'s `--alone-lb`.
4. **Sat ~13:30, the reads** (`kaggle competitions submissions`): (A) vs 0.947; (B) student − public; Ortho alone. One line
   each under *Settled*. **(C) waits for the owner:** Stage B Ortho inside the 8–9 h stack has never been timed (Open
   item 4), and a run over 9 h fails.

## Open — build or decide before it is needed

1. **Friday 17:20:** the list is READY; be at the Mac with the app open (the task runs only then), or push by hand.
   **Preflight 10-02 ~10:45: green.** Both kernel ids COMPLETE (nothing running); `rsna-knee-student-targets` holds
   `targets_round1_cnn.csv`; `raptor_ft_coatnet_v5_full_swa.pt` present; `stage-b-s0of8` COMPLETE; the public stack is
   unchanged upstream (notebook sha `2fc5a420`, same as `data/public_notebooks/`). The task stalled on its first
   permission prompt on 09-25, so someone must approve its prompts at 17:20.
2. **Replace-the-core submit mode: BUILT and tested locally 09-28** (`build_submit_kernel.py --core-*`; test
   `notebooks/core_swap_fixture.py`; detail in `docs/LOG.md`). Not yet run on Kaggle. **(A) on Saturday:**
   1. When the CoAtNet student finishes (~Sat 00:40), download its `runs_coatnet/last.pt` and take
      `shasum -a 256`.
   2. Build: `.venv/bin/python notebooks/build_submit_kernel.py --public-dir data/public_notebooks/bend-the-knee-2fc5a420
      --core-kernel raahimnawaz/rsna-knee-c6-train-coatnet-student-r1-all --core-sha <that sha> --tag student-r1 --out
      /tmp/subcore`. Check first that the stack is unchanged upstream.
   3. Push the commit run on the freed session.
   4. Submit only if the log shows both `[core]` lines (the relative change vs the public weights, then `[core] ARMS:`)
      and `COMPLETE`.

   For **(C)**, add `--add-family --weights-kernel <Ortho student> --weights-sha <sha> --alone-lb <its LB>`.
3. **(B) BUILT 10-02: `build_submit_kernel.py --core-alone`** (owner: "yes start b"). It runs only the stack's core,
   maxspan-v5 + its reversed pass at 6:1, on both T4s, with the stack's own code: public cells 1–3 and cell −2's Raptor
   code, verbatim. With `--core-kernel/--core-sha/--tag` it loads our student; without, the public weights (the paired
   baseline, which needs no student and can run any time). A last cell refuses the run unless the load log shows that
   one file, once per GPU, at its sha256; the commit run also refuses any failed study. Test:
   `notebooks/core_alone_fixture.py` PASSED (both builds run as built on CPU over synthetic DICOMs; study-by-study
   parity exact; the control differs; four refusals). Why (B) is worth it: the fail-branch diagnostic (`docs/LOG.md`
   09-28), and rule 4 needs the student's own score before our own blend (final B). Submit next to (A).
   Student build: `.venv/bin/python notebooks/build_submit_kernel.py --public-dir data/public_notebooks/bend-the-knee-2fc5a420
   --core-alone --core-kernel raahimnawaz/rsna-knee-c6-train-coatnet-student-r1-all --core-sha <sha> --tag student-r1 --out
   /tmp/alone`. Baseline: the same without `--core-*`/`--tag`, `--out /tmp/alone-pub`.
   **(C)** needs the Ortho student's own LB first (`--ours-only --weights-kernel/--weights-sha --tag`; the build
   refuses `--add-family` without `--alone-lb`). It adds a Stage B Ortho (96 slices, 384 px) to the stack. Sub
   B-dread fitted a Stage A Ortho (~35 min); Stage B's cost inside the stack has never been timed. That is item 4.
4. **Runtime room for students in the stack:** it already takes ~8–9 h of 9. Cut its 20 DINOv2 members (~5% of the
   vote), after timing its cells. **Owner Q 10-02: "drop DINOv2 while testing to save GPU?" No.** The hidden-test
   rerun does not count against the weekly quota (Kaggle: it runs on a private server; only the version save counts,
   i.e. the stack's ~6 min commit run on 3 studies). So the cut saves no GPU-h, only turnaround and 9 h room. It
   would also move the base under the ≥ 0.947 gate: "~5% of the vote" is a weight estimate, not an LB delta, and a
   blend change reads ±0.002. Cut only when (C) needs the room, and then test and ship the same cut stack.
5. **Save the `avg3` weights** of every run a final might use (never worse on the 691: +0.0000 to +0.0024).
6. **No team merger** (owner 09-28: "I want that Kaggle master"). Solo, on our own 30 GPU-h/week.
7. **No "safe" final** (owner 09-28). Both finals aim to win. **(A)** is the upgraded stack: its structure, with our
   students in its CoAtNet slot. **(B)** is our own blend: equal-rank families, nothing fitted on the public LB, the
   one most robust to the shake-up. Both are retrained on all data (3,502 + 691) with gold ×3, recipes fixed before,
   and chosen by pre-registered evidence, never by a 0.001 public difference.

**GPU:** 30 h/week, reset Fri 17:00 PDT on **10-02, 10-09 and 10-16**. At most 2 sessions at once, 12 h each, 9 h per
submission run. Friday spends ~16. Small CNNs run free on the Mac (M5, `--mps`, ResNet34 ~3.4 min/epoch alone).

## Settled — one line each (detail in the archive, `DECISIONS.md`, `DATA.md`)

- **Labels:** `dread_soft` (sha `8cec477f`) → LB 0.916 vs 0.901, PASS 09-27. It is the base target of every run.
- **Input:** Stage B → 691 +0.0112 (P 1.000); LB 0.915 vs 0.901. Speed: `--compile --ckpt-blocks 12` (−28.6% per step).
- **CNN teachers** (4 folds, ResNet34, Mac): gold 0.8873 → 0.9132, +0.026, P 0.994 → PASS. Student targets
  `data/student/targets_round1_cnn.csv`, sha `75757be8`, per-file rank mapping (AUDIT 09-28).
- **The public CoAtNet in our pipeline:** gold 0.9187 vs its recorded 0.9214 → a drop-in start (gold was in its
  training, so the LB is its only judge).
- **Z (zoom), 09-29 → HELPS** (pre-registered): 691 0.8129 → **0.8288**, paired **+0.0159 ± 0.0041** (3.8σ), 10/12
  up. Small structures: ACL +0.019, MCL +0.046, Med Men +0.011, Lat Men +0.015; also Lateral OA +0.042 and Fracture
  +0.026. Gold-58, reported only, points the other way: −0.017 ± 0.011. Cost ~2× compute. Zoom is a student
  candidate but not servable yet (`infer_windows`, and it cannot sit in the public core slot).
- **R (ResNet34 at 336, whole image), 09-29 → zoom STOPS** (pre-registered): Z − R **+0.0070 ± 0.0045, P 0.943** is
  under 0.95. Most of Z's gain over B′ was resolution: R − B′ +0.0089, P 0.979. Our 336/384 px students already have
  that. The one crop-specific gain is MCL (+0.052 over R), and per-finding picking is closed. A near-miss, not a
  second chance.
- **N (7,140 NYU pseudo-knees), 09-29 → null:** +0.0035 ± 0.0051, P 0.75. NYU image work is parked until after
  Friday's LB read.
- **Cut:**
  - ResNet34 as a main model: −0.036 vs Ortho on the 691; it stays as the cheap teacher.
  - NYU fastMRI+ teacher (P 0.21).
  - Qwen labels and the Qwen VLM.
  - The blend of our own two models.
  - Sub B-dread as a final (0.939 < 0.944).
  - A radiologist on the team (owner 09-28).
  - OAI MOAKS (its terms).
- **Closed, do not re-derive** (`DECISIONS.md`): weighting or per-finding picking of label sources, per-finding or
  per-study routing, a fitted cross-finding stacker (−0.013 on gold), calibration / thresholds (AUROC is
  rank-invariant), re-fitting blend weights, ComBat.

## Standing rules — each has already cost this project

1. **Never select on gold-58.** It claims +0.0137 and delivers −0.0034. It is spent on teacher gates; the LB judges
   labels.
2. **Pre-register every gate before the run; never renegotiate it after.** A null is a null, not a refutation.
3. **The 691 judges fixed-target comparisons only** (architecture, input). It cannot judge labels or teachers: its
   reference is another LLM's read of the reports.
4. **A weaker member at equal weight costs** (Sub B −0.007; the CNN −0.006 on the 691). Blend only strong members.
5. **Price an LB gain at ~½ its gold gain.** Gold and LB are different scales; never compare across them.
6. **Commit before building a kernel.** Before submitting a blend, check its `ARMS:` line names every family.
7. **Leak-check any downloaded label table.** Four public tables have gold baked in and score a fake 1.0000.
8. **A leaderboard number older than a few days is not evidence.** Re-pull the board.
9. **Run long local jobs detached under `caffeinate -i`.** An app update killed a run on 09-28.
10. **Commit and push after each substantive step.** Sessions restart; unpushed work is invisible to the next one.
11. **A CPU smoke skips every GPU-only branch** (anything behind `amp` or CUDA). Before an architecture's first GPU run
    on new trainer code, force those branches locally (`TW_FORCE_CKPT=1 --mps`, a few studies, to the end). 10-02: 2.5 h.
12. **An unattended step needs the Mac on power with the lid open, and no permission prompt.** The scheduled task has
    stalled on a prompt twice (09-25, 10-02); `caffeinate` does not survive a shut lid.
