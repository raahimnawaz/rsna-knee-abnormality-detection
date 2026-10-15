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

Last updated 2026-10-15 10:30 PDT.

## 0. Right now

- **Our best is 0.956, rank 408 of 6,037** (`57160505`: the free 0.956 recipe + our five teacher folds). It was rank
  198 on 10-13; the board moved under it. Pulled 10-15 10:04: #1 0.967, gold to ~rank 22 (0.963), **silver to ~301
  (0.957)**, bronze to ~603 (0.956).
- **A free 0.957 has been public since 10-14** (246 teams at 0.957, ranks 136–381; 927 at 0.956). Two notebooks,
  read and not run by us: `heliosli/this-is-new-version-0-9576-in-one-click` = `nartaa`'s Raptor CoAtNet 384 px
  + its 224 px checkpoint (0.35 of that half) and heliosli's **R6** bundle (`heliosli/rsna-knee-r6-frozen-fe055406`,
  four folds) **with flip TTA on**, per-finding ranks 50/50 (the title claims 0.9576); and
  `bliverrigpdoge/r7-raptor-knee-centred-convnext` = the old 0.956 at 0.8 + knee-centred crop models (private
  weights) at 0.2. **Any blend of ours now starts from the 0.9576 notebook**, not the 0.956 one (`6406e923`).
- **OAI: access granted 10-15** (DAR 26933, the owner's, to 2027-10-15). On disk: OAI's MRI assessment tables only
  (27 MB, no images) in `~/oai_data/`. Baseline MOAKS reads → our findings by `labeling/oai_moaks_targets.py`:
  **2,400 knees** → `~/oai_data/derived/oai_moaks_targets_v1.csv` (§7). **No OAI images yet:** the baseline image
  package is ~983 GB zipped / ~2 TB unzipped (read on NDA 10-15, not re-checked) and goes to a rented GPU only.
- **Nothing has trained since 10-13, and no GPU is rented.** Kaggle GPU: ~0.7 h until its last reset, **Fri 10-16
  17:00 PDT**, then 30 h. No submission since 10-13 evening; 5 a day, reset 17:00 PDT. **Finals 10-22 16:59 PDT.**
- **Waiting on the owner:** (1) which GPU, rented by them; (2) the OAI image download (their NDA login, on that
  machine); (3) a go for each submission in §3 that has no standing go.

## 1. Goal and the owner's directions

- **0.97 macro AUROC.** Do not reframe it. Two finals, 10-22 16:59 PDT. Solo: no merger.
- **Use whatever is available** (public weights, models, data; owner 10-12 21:17).
- **Ask before a batch of submissions and before spending GPU hours.** A go covers what it named and nothing later.
  **Standing go (10-13 12:37):** v3 fold 0, then folds 1–4, then a second label round from v3's own held-out
  predictions; and two submissions, v3 five folds alone and v3 inside the public blend.
- **Keep one final OAI-free** (thread 748276 challenges OAI's eligibility; no host reply).
- This page stays a current-state page; dated detail goes to the LOG. No Claude attribution in this repo's commits.
- **OAI data never enter this repo, GitHub or Kaggle** (DUC §3). They live in `~/oai_data/` (mode 700) and on the
  rented machine. Any public write-up needs the OAI acknowledgement and an NDA Study (DUC §11–12; LOG 10-14).

## 2. Where we are

| | public LB | |
|---|--:|---|
| #1 / gold line / silver line | 0.967 / 0.963 / 0.957 | 10-15 10:04 |
| **free public recipe** (0.9576 notebook, §0) | **0.957** | 246 teams |
| **ours: the old free recipe + our five teacher folds** | **0.956** | `57160505`, rank 408. Our best; OAI-derived |
| the public R7 x4 alone (`heliosli`) | 0.949 | `57160504`; OAI-free. Its mirror pass: 0.949 (`57160736`), stays off |
| public single model (`nartaa` CoAtNet, 2,399 OAI knees in training) | 0.949 | |
| ours: five folds, teacher labels | 0.949 | `57161107`; holds the CoAtNet's opinions, so not OAI-free |
| ours: one model, pseudo-label round (`targets_teacher_v2`) | 0.946 | `57162950` |
| **ours: five folds, consensus labels** | **0.945** | `57112913`. **Our only OAI-free model: today's OAI-free final** |
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

## 3. The plan for the last week (proposed 10-15; the owner decides the GPU and the submissions)

**Track A — a rented GPU, the main bet: a public R-family model retrained on better labels, with OAI's knees.**

| step | state |
|---|---|
| Rent a 32 GB-class card (R7 trains 176 windows a study). Four cards run the four folds at once: the same GPU-hours, a quarter of the wall clock. DUC §8: encrypted disk, no public ports, wiped at the end | **not rented** |
| Pick the model: R7's trainer is public (`jessky/rsna-knee-local-cv-cnx50`, LOG 10-13 14:15); R6 (the new recipe's member) is not yet read. Read R6's bundle first: no GPU needed | open |
| Competition DICOMs onto the box (`kaggle` CLI), the model's own pixel cache from them (~3–6 h, estimate) | not started |
| OAI baseline images onto the box (the owner's NDA login). First test there: download speed, and whether only the series of the 2,400 MOAKS knees can be pulled. The MOAKS knees through the same cache builder (OAI has sagittal + coronal, no 2-D axial) | not started; **the largest unknown** |
| Labels: reports + the OAI-trained CoAtNet + the model's own held-out predictions + ours (`build_consensus_targets.py`); OAI knees from `oai_moaks_targets_v1.csv`, unscored cells masked | MOAKS done; the table not built |
| Train four folds × 16 epochs (~55–65 GPU-h with OAI, estimate; ~$30–65) → `offkaggle.py pack` → a private Kaggle dataset → a submission kernel | not started |

Rule, written before any run: R7 retrained must read **≥ +0.006 over R7 alone (0.949)** to be shown; in the blend,
it replaces the public member it was retrained from and is read against that blend. The one measured OAI gain is
+0.003 to +0.005 on a single model (`nartaa`).

**Which findings OAI supervises (open):** proposed: the three OA, both menisci, Effusion, Synovitis and ACL;
masked: MCL, Fracture (not graded), Baker's (presence only, one project) and Contusion (in an OA cohort a marrow
lesion is mostly degenerative). `nartaa` used PF OA, Lateral OA and Synovitis only.

**Track B — Kaggle, the fallback (standing go): v3 five folds** from Fri 10-16 17:00, two at a time (§4). ~24.5 h
of the last 30 h.

**Submissions:**
1. *Asked, not approved:* the 0.9576 notebook unchanged as stage 1 + our five teacher folds at `57160505`'s weights
   (0.25 on ACL, MCL, Medial Meniscus, Medial OA, Effusion, Baker's, Contusion, Fracture; 0.10 on Lateral Meniscus,
   Lateral OA, PF OA, Synovitis), built with `notebooks/build_public_blend_kernel.py`. **Needs a small change
   first:** the builder pins the old notebook's slug (`PUBLIC`) and its intro text describes the old recipe. **And a
   time check:** the new notebook doubles R6's GPU time with its flip pass; if our stage runs out of the 9 h, the
   scored rerun keeps the public file and the slot reads a 0.957 that says nothing. ~10–15 Kaggle GPU-min for the
   commit run. Read against 0.957: ≥ 0.958 ours stay in; 0.957 read the rank inside the block; ≤ 0.956 ours come out.
2. *Standing go:* v3 five folds alone; v3 inside the blend in place of the teacher folds.
3. *To be asked:* the retrained model alone, and in the blend.

**Finals:** the best blend, and one OAI-free (today the five consensus folds 0.945; v3's OAI-free twin if trained).

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
| closed, do not re-propose | a bigger backbone; a stronger LLM re-reading the reports; centre crops; an nnU-Net crop; Qwen report labels; a per-site prior |

## 6. Budget

- **Kaggle GPU:** ~0.7 h until Fri 10-16 17:00 PDT (the owner's quota page is the truth), then 30 h for the last
  week. Two GPU sessions at once. All of it is this competition's (casmi26 dropped 10-13).
- **Submissions:** 5 a day, reset 17:00 PDT (00:00 UTC). A commit run is ~10–15 GPU-min; the scored rerun is free.
- **Off Kaggle:** a rented 5090 ~$0.20–1.34/h (Vast.ai / RunPod snapshots, 07–08 2026). This M5 laptop is ~5×
  slower than Kaggle (measured): not a training option.

## 7. How to run things

`K=.venv/bin/kaggle`, `PY=.venv/bin/python`, `C=rsna-knee-abnormality-detection` (system Python has neither).
Kernels are generated from committed code: **commit first**, build into a scratch folder, push, never edit a notebook
on Kaggle. The images are only on Kaggle; `data/` here holds tables (git-ignored).

```bash
# one fold of v3 (folds 1..4: --folds k --tag v3-f<k>; slug < 50 chars). CPU smoke first: --smoke --tag v3s
$PY notebooks/build_backbone_ab_kernel.py --arch convnext_small --corpus stage_b --folds 0 --n-train 0 --recipe v3 \
    --time-budget-h 9.0 --targets data/student/targets_teacher_v3.csv \
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
$PY notebooks/build_public_blend_kernel.py --help       # the public notebook unchanged as stage 1, ours as stage 2
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
- `--fit-budget` cuts epochs on a slow first epoch: v3 gets `--time-budget-h 9.0` so it keeps 16.
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
| `notebooks/build_*_kernel.py` | generators: training (`backbone_ab`), submission (`submit`, `public_blend`, `r7_read`), weights sha, the Stage A cache |
| `notebooks/*_fixture.py` | offline tests on synthetic DICOMs |
| `notebooks/offkaggle.py` | training on another GPU and the way back for its weights. Never run on a GPU |
| `notebooks/submit_at_reset.py` | the detached submitter; `submit_plan_2026-10-13.json` is its format |
| `README.md` | the portfolio description as of 10-03, with a status note |

Removed 10-15 (in history at `c5ab90c` and on `showcase`): the same-report CNN teachers
(`local_teacher_folds.py`, `build_student_targets.py`), the public-stack core fixtures and their run records, the
ViT-L student run records. Local-only legacy run folders (weights and logs, ~1.5 GB) moved to
`~/rsna-knee-legacy-runs/`.
