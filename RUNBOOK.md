# RUNBOOK — read this first

The current state of this competition entry, on one page. Working branch **`labels-small-backbone`** (pushed;
remote `raahimnawaz/rsna-knee-abnormality-detection`, private, the only copy). `main` = `showcase` is the portfolio
tree of 10-03 (the ViT-L pipeline): no work goes there.

| if you need | read |
|---|---|
| what was run, when, and the rule written before each result | [`docs/LOG.md`](docs/LOG.md): dated blocks, newest at the end |
| why the weak findings are weak | [`docs/REVIEW-2026-10-12.md`](docs/REVIEW-2026-10-12.md) |
| this page as it was on 10-13 (that day's per-read rules, the arms v3 replaced, the audits' proposals) | `git show c5ab90c:RUNBOOK.md` |
| anything before 10-11 | `git show 9c23daf:RUNBOOK.md` (also `:DATA.md`, `:REFERENCE.md`) |

Last updated 2026-10-16 14:40 PDT.

## 0. Right now (10-16 14:40 PDT)

- **Board (10-16 10:00, 6,098 teams):** #1 0.968, gold ~0.963 (rank ~22). **0.958 is 433 teams (ranks 123–555), so
  silver (~304) and bronze (~609) both sit inside 0.958.** Ours: **0.957** (`57241351` and B), rank 825. A new free
  base is public: `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` (10-15 17:03), read 10-16: the 0.9576
  notebook with per-finding Raptor weights changed (Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35), nothing new trained.
  Page with the pipeline drawn: "Knee Blend Review", https://claude.ai/artifact/TiphXndYsMErAy497AYjnG.
- **Read 10-16 by their rules (LOG 10-16):** `57241351` 0.957 → nothing claimed · **B 0.957 → R6 + R7 in one decode is
  the ConvNeXt half from now on** · **D 0.954 → D is the OAI-free final** (was 0.945).
- **The cartilage-measurement test: CLOSED (read 10-16 14:25, LOG).** On 800 held-out training knees the measurement
  model reads Lateral OA 0.654 and PF OA 0.619 (rule ≥ 0.85); independent of ours (ρ ≈ 0.2) but nearly no signal:
  +0.00008 on the macro. Nothing is left that targets the weak four this week.
- **The week is locked (owner 10-16): no new arms.**
  1. **Fri 10-16 17:00 PDT, Kaggle GPU reset (30 h, the last):** v3 fold 0 (standing go, §4), then folds 1–4 two at
     a time, on `targets_teacher_v3`.
  2. **The main final's base is the free 0.958** (read, LOG 10-16). Whether B's R6 + R7 replaces its R6 + flip under
     its weights is one read: **one slot, owner's go**.
  3. **Before the finals:** test the mount of our copy of heliosli's code dataset (free CPU kernel, as
     `rsna-knee-pintest-r6v1`), then build both finals on our private copies (`--replace-dataset`) and **without**
     `--read`, with the vendored-MONAI tree check (review finding 3, `cartilage_measure.VENDOR_TREE_SHA`) added to
     the R6 / R7 worker's path.
  4. **Finals by 10-22 16:59 PDT:** the main final = the best free base with R6 + R7 as its ConvNeXt half + our member
     (v3 if it passes its fold-0 check, else the five teacher folds); the OAI-free final = **D**. No new arms.
- **Kaggle GPU before Fri 17:00:** little or none. Submissions: 5 a day, reset 17:00 PDT.
- **Owner's calls:** no rented GPU; A, B, D approved, C not (10-15). 10-15 ~15:15: "train our own models when
  necessary"; the weak-four plan of that afternoon was cut to this one test (LOG). Still theirs: S6fall into D;
  deleting 4 stale remote branches.

## 1. Goal and the owner's directions

- **0.97 macro AUROC.** Do not reframe it. Two finals, 10-22 16:59 PDT. Solo: no merger.
- **Use whatever is available** (public weights, models, data; owner 10-12 21:17).
- **Ask before a batch of submissions and before spending GPU hours.** A go covers what it named and nothing later.
  **Standing go (10-13 12:37):** v3 fold 0, then folds 1–4, then a second label round from v3's own held-out
  predictions; and two submissions, v3 five folds alone and v3 inside the public blend.
- **Keep one final OAI-free** (thread 748276 challenges OAI's eligibility; no host reply).
- This page stays a current-state page; dated detail goes to the LOG. No Claude attribution in this repo's commits.
- **OAI data never enter this repo, GitHub or Kaggle** (DUC §3). They live in `~/oai_data/` (mode 700) only. Any public write-up needs the OAI acknowledgement and an NDA Study (DUC §11–12; LOG 10-14).

## 2. Where we are

| | public LB | |
|---|--:|---|
| #1 / gold line / silver line | 0.967 / 0.963 / 0.957 | 10-15 10:04 |
| free public base, newest (heliosli, 10-15 17:03, not yet read) | 0.958? | 433 teams at 0.958 (10-16) |
| **free public recipe** (0.9576 notebook, §0) | **0.957** | 246 teams (10-15) |
| **ours: the 0.957 recipe + our five teacher folds** | **0.957** | `57241351`; B (R6 + R7 in one decode) the same, `57244451`. Our best; OAI-derived |
| ours: the old free recipe + our five teacher folds | 0.956 | `57160505` |
| **D: R6 + R7 + our five consensus folds, no Raptor** | **0.954** | `57244659`. **The OAI-free final** |
| the public R7 x4 alone (`heliosli`) | 0.949 | `57160504`; OAI-free. Its mirror pass: 0.949 (`57160736`), stays off |
| public single model (`nartaa` CoAtNet, 2,399 OAI knees in training) | 0.949 | |
| ours: five folds, teacher labels | 0.949 | `57161107`; holds the CoAtNet's opinions, so not OAI-free |
| ours: one model, pseudo-label round (`targets_teacher_v2`) | 0.946 | `57162950` |
| ours: five folds, consensus labels | 0.945 | `57112913`. OAI-free; D's member |
| ours: one model, consensus labels | 0.938 | `57098827`, the paired baseline of recipe changes |

**What a leaderboard number is worth** (resampling our held-out predictions at the public part's ~390 studies):
one score ± 0.008 (1 SD); two of our models that differ in labels only ± 0.003, so ~+0.006 is the smallest
readable gain; two unrelated models ± 0.005; one finding's AUROC ± 0.02–0.03.

**Where the error is** (single-finding probes, five consensus folds): Synovitis 0.884, Lateral OA 0.896, PF OA 0.896,
Lateral Meniscus 0.932 — together ~60% of what is left; the other eight ~0.966. OAI's MOAKS reads grade exactly
those four. 0.97 needs them near 0.95.

**The model we train** (`fusion/train_windows.py`): ConvNeXt-small (in12k → in1k, 384 px) on the Stage B cache (7
series slots, 96 slices, 140 mm field, left knees mirrored to right); windows of 3 slices → per-finding attention →
12 logits; soft report targets, plain BCE; five folds held out by site; ~1.9 GPU-h a fold on two T4s.

## 3. The plan for the last week (Kaggle only)

**What a submission is now** (`notebooks/build_public_blend_kernel.py`, `notebooks/build_r6r7_kernel.py`): the public
0.9576 notebook (`data/public_notebooks/heliosli-0-9576-16a9d953/`, sha256 `16a9d953…`; build from this saved copy,
heliosli may change theirs) → Raptor 384 + 224 (nartaa, OAI-trained) | the ConvNeXt half (R6 + flip, or R6 + R7 in one
decode from `raahimnawaz/rsna-knee-r6r7-dual`) → their 50 / 50 rank blend → our five folds appended by per-finding
rank (0.25 on eight findings, 0.10 on Lateral Meniscus, Lateral OA, PF OA, Synovitis). D drops the Raptor cells and
mounts no Raptor weights.

**The expert panel's plan (10-15; five independent reviews, LOG) and where each item stands:**

| | item | status |
|---|---|---|
| A | fail loudly: `--read` for reads; private copies of every outside dataset | built; the code dataset copy's mount untested |
| B | R6 + R7 in one decode (heliosli's two-group worker path; two worker lines patched so R6's Baker's stacker runs) | **0.957, same tick as `57241351`: adopted** |
| C | v3 on `targets_consensus_v3` (OAI-free labels) | **not approved**: v3 stays on `teacher_v3` |
| D | the OAI-free final as a blend: R6 + R7 + our consensus folds, no Raptor | **0.954: the OAI-free final** |
| E | our weight from v3's standalone score, fixed before reading it: ≥ 0.953 alone → 0.33 / 0.15, else 0.25 / 0.10 | a rule, for v3's read |
| next | S6fall (0.950, OAI-free by its author) into D | proposed, ask |

Every reviewer's ceiling for the week: **+0.001 to +0.003 true on the main final (~0.958–0.960)**. Gold needs several
independent members of 0.955 or better; 0.97 needs the weak four near 0.95 (today 0.88–0.93), i.e. image-graded labels
for exactly those findings plus compartment-level models: not reachable with what exists this week.

**Tested and cut (10-15; detail in the LOG):** building a small or crop CNN (owner) and the public small CNN
(`goodpjw2008`'s reader: its author measured it adds nothing to this base); fixed half-crops of our cache (not
knee-centred); fine-tuning with R7's trainer; distilling public members into ours; R7's mirror pass (0.949 = 0.949);
blend-weight tuning on the public board (zero slots); single-finding probes for weights; consistency rules between
findings (flat or negative on gold and the 4,349); fixing knee-side errors (3.1% disagree with the report, but the
model reads anatomy: swapping is worse); "tricompartmental" label fixes (no excess noise); the second label round and
the separate OAI-free v3 twin; run-to-run noise (no GPU for it); OAI images and a rented GPU (owner).

**Finals:** the best blend by the reads above, and one OAI-free (today: D if ≥ 0.950, else R7 alone 0.949 or the
consensus folds 0.945). Both built on our private dataset copies, without `--read`.

## 4. v3, the Kaggle recipe (owner 10-13 12:37; everything fixed before any run)

| part | what |
|---|---|
| labels | `targets_teacher_v3`: `targets_teacher_v1` + the five teacher folds' held-out predictions at 0.75 of every cell (1.5× v2's dose) |
| pooling | `--spatial-attn` |
| mirror | `--mirror 0.5` (medial ↔ lateral; Meniscus and OA targets swapped, MCL masked) and `--mirror-tta` at inference |
| windows | `--k 24 --micro 1` |
| schedule | 16 epochs; submit `avg3/last.pt`, gated against `avg3/oof_all.csv` |

- One fold per kernel, ~4.9 h on two T4s (estimate). Everything is built and CPU-tested (`mirror_fixture.py`, 66
  checks; Kaggle CPU smoke `rsna-knee-c7-train-convnext-small-v3s-cv` clean 10-13). Nothing has trained.
- **Fold 0 is a not-broken check. Stop and tell the owner if**, against `runs_cnxs_teach1_f0` (847 macro 0.8804) on
  the same studies, v3's `avg3/oof_all.csv` is more than **0.008** below on the macro, or more than **0.03** below on
  MCL / either Meniscus / Medial or Lateral OA, or it ran fewer than 16 epochs. Otherwise folds 1–4 go.
- A second label round (with hours to spare): `build_consensus_targets.py … --own-oof <v3's five cv_avg3> --w-own
  0.75` on `targets_teacher_v1`, then five more folds. The OAI-free twin: the same recipe on `targets_consensus_v3`.

## 5. Decided and closed (each by a rule written before its result; detail in the LOG)

| question | result |
|---|---|
| ConvNeXt-small against ViT-L | small wins (LB 0.938 against 0.927, 5× cheaper): **the workhorse** |
| five folds against one | +0.007: **ensemble** |
| teacher labels against consensus | +0.006, kept: **finals' labels** |
| denser slices (Stage C) | fail: **Stage B stays** |
| R7's mirror pass at inference | 0.949 = 0.949: **off** |
| training on the 58 gold studies | ~+0.0005: **not done; gold stays held out** |
| cartilage / meniscus measurements (SKM-TEA segmenter) as a member | 0.654 / 0.619 alone on Lateral / PF OA, ρ 0.2: +0.00008 macro. **Closed** (10-16) |
| OAI images, a compartment CNN, traumatic-cartilage relabels, Synovitis from Effusion | repeats or measured dead (LOG 10-15 / 10-16) |
| closed, do not re-propose | a bigger backbone; a stronger LLM re-reading the reports; centre crops; an nnU-Net crop; Qwen report labels; a per-site prior |

## 6. Budget

- **Kaggle GPU:** a small remainder until Fri 10-16 17:00 PDT (the owner's quota page is the truth), then 30 h for
  the last week. Two GPU sessions at once. All of it is this competition's (casmi26 dropped 10-13).
- **Submissions:** 5 a day, reset 17:00 PDT (00:00 UTC). A commit run is ~10–15 GPU-min; the scored rerun is free.
- **Off Kaggle: none** (owner 10-15: no rented GPU). This M5 laptop is ~5× slower than Kaggle (measured): not a
  training option.

## 7. How to run things

`K=.venv/bin/kaggle`, `PY=.venv/bin/python`, `C=rsna-knee-abnormality-detection` (system Python has neither).
Kernels are generated from committed code: **commit first**, build into a scratch folder, push, never edit a notebook
on Kaggle. The images are only on Kaggle; `data/` here holds tables (git-ignored).

```bash
# one fold of v3 (folds 1..4: --folds k --tag v3-f<k>; slug < 50 chars). CPU smoke first: --smoke --tag v3s
$PY notebooks/build_backbone_ab_kernel.py --arch convnext_small --corpus stage_b --folds 0 --n-train 0 --recipe v3 \
    --time-budget-h 10 --targets data/student/targets_teacher_v3.csv \
    --targets-from raahimnawaz/rsna-knee-student-targets --tag v3 --out <dir>
$K kernels push -p <dir>          # then read `fit-budget: … -> 16 epoch(s)` in its first minutes

# results: small files only (the weights break the download); paired scoring on the same held-out studies
$K kernels output raahimnawaz/<kernel> -p fusion/runs_<name> --file-pattern '.*\.(json|csv|txt|log)$'
$PY fusion/score_oof.py fusion/runs_cnxs_teach1_f0/runs_convnext_small/fold0 fusion/runs_<name>/runs_convnext_small/fold0/avg3

# weights sha256, read where the file is
$PY notebooks/build_weights_sha_kernel.py --kernels raahimnawaz/<training kernel> --out <dir> && $K kernels push -p <dir>

# a submission kernel: ours alone (an ensemble is comma lists), or inside a public blend
$PY notebooks/build_submit_kernel.py --ours-only --weights-kernel raahimnawaz/<k1>[,<k2>] \
    --weights-sha <64 hex>[,<64 hex>] --tag <short> --desc "<what this is>" --out <dir>
$PY notebooks/build_public_blend_kernel.py --help       # the public notebook as stage 1 (exact --cell-patch lines only), ours as stage 2
#   --read: a read (our stage skipped / late / failed -> the submission errors); finals leave it off
#   --replace-dataset [public/slug=]ours/slug: mount our private copy (heliosli overwrote R6 on 10-14; kernels cannot pin a version)
#   --drop-cells / --final-from / --drop-dataset / --w-ours STRONG,WEAK: e.g. D (no Raptor, the R-family's output as the base)

# on R6 + R7 in one decode (bundle raahimnawaz/rsna-knee-r6r7-dual, 0aa1bca3; rebuilt by build_r6r7_bundle.py):
NB=data/public_notebooks/heliosli-0-9576-16a9d953        # the saved 0.9576 notebook (sha256 16a9d953), not a fresh pull
$PY -I notebooks/build_r6r7_kernel.py $PWD $NB <out> --read --tag r6r7-teach1cv5 \
    --desc "our five ConvNeXt-small folds on targets_teacher_v1"                     # B (57244451): teacher folds, 0.25 / 0.10
$PY -I notebooks/build_r6r7_kernel.py $PWD $NB <out> --weights-kernel <cons2 kernels> --weights-sha <their 5 shas> \
    --drop-cells 2,3,4,5 --final-from ours_sub.csv --drop-dataset nartaa/rsna-knee-publication-swa-weights-20261007 \
    --w-ours 0.35,0.20 --read --tag oaifree-r6r7-cons5 --desc "…"                    # D (57244659): OAI-free
# a final: the same lines without --read, plus --replace-dataset for each outside source (our copies, LOG 10-15 "A")
# read in its log before submitting: `sha256 ok` · `GATE PASSED (n/n)` · 0 tracebacks
$K competitions submit $C -k raahimnawaz/<kernel> -v <n> -f submission.csv -m "<what, against what, and the rule>"

# off Kaggle (a rented GPU): folds file, one fold's run, the cache pull, packing weights as a private dataset
$PY notebooks/offkaggle.py pull --dest <cache>          # our Stage B cache, ~62 GB, untested at that size
$PY notebooks/offkaggle.py train --fold 0 --corpus <cache> --labels data/student/targets_teacher_v3.csv \
    --folds data/student/folds_site_stage_b.csv --out <runs>          # --dry prints the command only
$PY notebooks/offkaggle.py pack --runs <runs> --slug <slug> --out <dir>   # prints the datasets-create + submit lines

# OAI labels (the output stays in ~/oai_data; the script refuses a path git could push)
$PY labeling/oai_moaks_targets.py --moaks ~/oai_data/mri_assessments/"Semi-Quant Scoring_ASCII"/kMRI_SQ_MOAKS_BICL00.txt \
    --out ~/oai_data/derived/oai_moaks_targets_v1.csv

# a pseudo-label table: the teacher table + fold models' held-out predictions; then a new dataset version
$PY labeling/build_consensus_targets.py --labels-dir data/public_llm_labels --gold data/train.csv \
    --teacher fusion/runs_coatnet_oai_teacher_pa/runs_coatnet/oof_all.csv --w-teacher 0.5 \
    --own-oof <fold oof_all.csv files> --w-own <w> --out data/student/<table>.csv
$K datasets version -p <folder with EVERY table already there + the new one + dataset-metadata.json> -m "<what>"

# a submission at a set hour, surviving the session (not a reboot); plan format: notebooks/submit_plan_2026-10-13.json
$PY notebooks/submit_at_reset.py --plan <plan.json> --not-before "YYYY-MM-DD 17:02:00" --detach; tail data/submitter.log

# offline tests (no GPU)
$PY notebooks/mirror_fixture.py --out <dir>             # the mirror, 66 checks, ~1 min
$PY notebooks/canon_fixture.py --out <dir> && $PY notebooks/infer_parity_fixture.py --fixture <dir>
$PY notebooks/public_blend_fixture.py                   # the blend kernel's cells
```

**Things that have already cost a run or an hour.**
- A kernel slug over 50 characters is refused with a bare 400. Keep `--tag` short.
- Push a kernel that consumes another kernel's output only after the source reads COMPLETE. A third GPU kernel cannot
  start while two run.
- `--fit-budget` cuts epochs on a slow first epoch: v3 gets `--time-budget-h 10` so it keeps 16 (the modeling review,
  10-15: at k 24 the cold-cache probe projects ~8.4 h for 16 epochs, a 6% margin under 9.0; 10 still fits the 12 h
  session). Read `fit-budget: … -> 16 epoch(s)` and fold 0's real epoch time: over ~19 min, five folds do not fit 30 h.
- `--train-args=--flag` needs the `=` when the value is one token starting with `--`.
- A CPU smoke does not reach GPU-only branches, GPU memory, fp16 or the epoch time.
- `--k 24` with two studies a forward is over a T4's memory: `--micro 1`.
- zsh does not split `$var` in a `for` loop; macOS has no `timeout`.
- The Mac sleeps with the lid shut and background watchers stop; a session's background task dies with the session
  (use `submit_at_reset.py --detach`). `kaggle kernels output` can hang after a sleep: retry.
- CoAtNet on Stage B needs its own argument package (`COATNET_PACKAGE`).

## 8. Labels and weights

Tables are in `data/student/` (git-ignored) and in the private Kaggle dataset `raahimnawaz/rsna-knee-student-targets`.

| table | sha256 | what it is |
|---|---|---|
| `targets_teacher_v3.csv` | `6f786dda…` | teacher_v1 + the five teacher folds' held-out predictions at 0.75. **v3's labels** |
| `targets_consensus_v3.csv` | `26401559…` | consensus_v2 + the five consensus folds' held-out predictions at 0.75: **the OAI-free twin** |
| `targets_teacher_v2.csv` | `fa98f943…` | the same at 0.5 (one-fold read 0.946) |
| `targets_teacher_v1.csv` | `2980e8d6…` | consensus_v2 + the public OAI-trained CoAtNet at half of every cell. **Best read so far** |
| `targets_consensus_v2.csv` | `a1ed4034…` | seven public readers of the reports on `dread_soft`'s levels; Synovitis masked where unread |
| `~/oai_data/derived/oai_moaks_targets_v1.csv` | local only | 2,400 OAI baseline knees, MOAKS → our findings (constants in the script's docstring). Share ≥ 0.5: Medial Men 41%, Lateral Men 21%, Medial OA 21%, Lateral OA 19%, PF OA 43%, Effusion 20%, Synovitis 14%, ACL 3%. Two-read agreement 0.89–0.94 |

`folds_site_stage_b.csv` (`b8d82239…`, same dataset): the fold of each of the 4,349 non-gold studies. Never train on
a table that contains the gold rows (`tasmeemreza`, `yehezkielhaganta`, `dhiyaneshbabu`, `lvthanhxun` fused,
`gperdrizet`), and never on `lixin73`'s: `score_oof.py` scores against it.

Weights stay on Kaggle as kernel outputs (`runs_convnext_small/fold<k>/last.pt`); full sha256s in the LOG.

```
teacher labels, five    raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-cv      fold 0      1c58b2bc…
                        raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-f12-cv  folds 1, 2  683857fa…, 479cf64b…
                        raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-f34-cv  folds 3, 4  4921d096…, c910af26…
consensus, five         raahimnawaz/rsna-knee-c7-train-convnext-small-cons2-cv       fold 0      8fc45272…
                        raahimnawaz/rsna-knee-c7-train-convnext-small-cons2-f12-cv   folds 1, 2  a3113be6…, d147aa3c…
                        raahimnawaz/rsna-knee-c7-train-convnext-small-cons2-f34-cv   folds 3, 4  eb3bb162…, 8b0768b0…
public CoAtNet re-saved raahimnawaz/rsna-knee-c7-train-coatnet-oai-teacher-pa        (all)       f900846d…
```

Caches: Stage B = `raahimnawaz/rsna-knee-stage-b-s{0..7}of8` (in use, ~62 GB). Stage C = `…-stage-c-s{0..7}of8`
(not adopted).

## 9. Standing rules

1. Never select on gold-58. Report it only.
2. Write the rule before the run; a null is a null.
3. The held-out fold screens recipe changes at fixed labels; the leaderboard judges labels.
4. A weaker member at equal weight costs a blend.
5. Commit before building a kernel; push after each step.
6. Leak-check any downloaded label table against gold before it trains anything.
7. A leaderboard number older than a few days is not evidence: re-pull before quoting.
8. Before proposing an experiment, search the history for it (`git log --all --grep`, the LOG).
9. Record a result where the text that proposed it still stands, not only in the LOG.

## 10. What is in the repo

| path | what |
|---|---|
| `fusion/train_windows.py`, `infer_windows.py`, `folds.py`, `score_oof.py` | trainer, inference, site folds, paired held-out scoring |
| `fusion/error_budget.py`, `score_gold.py` | the review's tables; gold-58 scoring (report only) |
| `fusion/orthofoundation_model.py`, `qwen_vlm.py`, `window_curve.py` | the 10-03 ViT-L / VLM work. Not on the current path, but the kernel builders embed them: **do not delete** |
| `fusion/runs_cnxs_*`, `runs_coatnet_oai_teacher_pa` | this branch's run records: config, held-out and gold predictions, scores (no weights) |
| `fusion/runs_c6_ortho`, `runs_port_repro`, `runs_c7_ortho` | 10-03 records still read: the training kernel embeds the first two as its paired control; `score_oof.py` / `score_gold.py` use the third in their examples |
| `pipeline/` | DICOM → cache: `build_slots64.py` (the builder), slot layouts `slots96.py` (Stage B, in use), `slots118.py` (Stage C), `slots64.py` (Stage A); provenance, site fingerprints |
| `labeling/build_consensus_targets.py` | every label table of §8, including the pseudo-label step |
| `labeling/oai_moaks_targets.py` | OAI MOAKS → our findings; its data never in a repo |
| `extractor/metrics.py` | the competition metric |
| `provenance/` | the site folds and input hashes the training kernels check |
| `notebooks/build_*_kernel.py` | generators: training (`backbone_ab`), submission (`submit`, `public_blend`, `r6r7`, `r7_read`), weights sha, the Stage A cache |
| `notebooks/build_r6r7_bundle.py` | heliosli's R6 + R7 as one two-group bundle (each checked against its manifest) |
| `notebooks/*_fixture.py` | offline tests on synthetic DICOMs |
| `notebooks/offkaggle.py` | training on another GPU and the way back for its weights. Never run on a GPU |
| `notebooks/submit_at_reset.py` | the detached submitter; `submit_plan_2026-10-13.json` is its format |
| `README.md` | the portfolio description as of 10-03, with a status note |

Removed 10-15 (in history at `c5ab90c` and on `showcase`): the same-report CNN teachers
(`local_teacher_folds.py`, `build_student_targets.py`), the public-stack core fixtures and their run records, the
ViT-L student run records. Local-only legacy run folders (weights and logs, ~1.5 GB) and unused local tables (Qwen
labels, NYU pseudo-labels, round-1 CNN targets, the synbase control, an old CoAtNet v5 checkpoint; ~295 MB) moved to
`~/rsna-knee-legacy-runs/`. `data/` now holds only what live code reads.

**Outside the repo, on this Mac:** `~/oai_data/` (OAI tables, mode 700, never in a repo); `data/public_notebooks/`
(git-ignored: the saved public notebooks the builders take as `--public-dir`, and S6fall's solo notebook);
`~/rsna-knee-work/2026-10-15/` (the panel brief, the page source, the one-off audit scripts; its README lists them);
`~/rsna-knee-legacy-runs/` (moved-out legacy runs and tables). Private Kaggle datasets: `raahimnawaz/rsna-knee-r6r7-dual`,
`…/rsna-knee-r6-frozen-fe055406` (R6 v1), `…/rsna-knee-publication-swa-weights-20261007`, `…/rsna-knee-s6fall-weights`,
`…/rsna-orthofoundation-fold0-epoch7-20261004` (copies under the same slugs), `…/rsna-knee-student-targets`.

**Left on purpose until after 10-22** (tech-debt pass, 10-15): the ViT-L / Qwen / DINOv3 branches and their modules
inside the live trainer (~450 lines, imported only when those archs are chosen, embedded by the builders): removing
them changes the code v3 and every submission kernel run, days before the deadline. The README's 10-03 body, to be
rewritten with the public release (OAI acknowledgement included). Four stale remote branches (`claude/*`,
`readme-handoff-rewrite`): deleting them is the owner's call.
