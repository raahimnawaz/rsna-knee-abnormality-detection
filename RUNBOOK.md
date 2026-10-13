# RUNBOOK — the current state, on one page

**Any session starts here, and for most work this page is enough.** Working branch: `labels-small-backbone` (pushed).
`main` = `showcase` is the portfolio tree of 10-03: no work goes there.

| if you need | read |
|---|---|
| a kernel name, a full sha256, a gate line, the exact wording of a rule | [`docs/LOG.md`](docs/LOG.md): every run and read since 10-11, dated |
| why the weak findings are weak | [`docs/REVIEW-2026-10-12.md`](docs/REVIEW-2026-10-12.md): the error budget |
| anything before 10-11 | `git show 9c23daf:RUNBOOK.md` (also `:DATA.md`, `:REFERENCE.md`, `:docs/REVIEW-2026-10-04.md`) |

Last updated 2026-10-13 15:28 PDT.

## 0. Right now

- **Owner 10-13 ~12:37: "Change of approach for the last 9 days: stop testing one change at a time."** One combined
  recipe, **v3**, on the ConvNeXt-small trainer: stronger pseudo-labels, spatial attention, mirror augmentation with
  a mirrored pass at inference, 24 windows, 16 epochs, the averaged weights (§4 has the recipe, the dose and the
  rule; LOG, last block, has the owner's message in full). It replaces the one-arm-at-a-time queue.
- **Built and tested on CPU, 10-13 12:40–13:10:** the mirror code (`notebooks/mirror_fixture.py`, 66 checks, the
  geometry against the real DICOM path), `targets_teacher_v3` and its OAI-free twin (§7; in the private Kaggle
  dataset), the recipe under two-process DDP on this laptop, the off-Kaggle runner and the way back for its weights
  (`notebooks/offkaggle.py`, `--weights-dataset`). **The Kaggle CPU smoke of the full recipe on the real cache is
  clean** (`rsna-knee-c7-train-convnext-small-v3s-cv`, COMPLETE 13:09: the v3 table found by its sha, fold 0's
  3,502 / 847, two processes, mirror + spatial attention + two-pass predictions, `avg3`, 0 tracebacks). A smoke does
  not reach GPU memory, fp16 or the epoch time: read `fit-budget: … -> 16 epoch(s)` in fold 0's first minutes.
- **Nothing has trained. The GPU is the open question:** the owner's message left it blank ("I am getting [FILL
  IN]"). Kaggle has ~0.7 h until Friday 17:00's reset, and five v3 folds are ~24.5 h of that last 30 h week (§5).
  **Ask the owner which GPU, and where the 62 GB Stage B cache should be pulled to** (it has not been pulled).
- **Standing go (owner 10-13 12:37; no need to ask per run):** fold 0 of v3 as a not-broken check → the other four
  → a second label round from v3's own held-out predictions if hours remain. Stop and tell the owner only if fold 0
  is clearly worse than the current model (§4 says what that means). **Standing go for two submissions:** v3 five
  folds alone, and v3 inside the public blend in place of the current folds. Ask before any other.
- **Tonight's three submissions (§4) are armed again.** The 12:26 submitter was not running at 12:37 (it had gone
  with its session). `notebooks/submit_at_reset.py` was armed at 12:42 as its own process (pid 96147): it sends the
  blend, the five teacher folds and the pseudo-label student at **17:02 PDT**, or on wake if the Mac sleeps through
  it, and never an item twice. **After 17:02: `tail data/submitter.log` and `$K competitions submissions $C`;** if
  the Mac was rebooted, re-arm with the command in §6. Then read the three scores by their rules into §2–§4.
- **The direction since ~15:10 (owner; LOG, last three blocks): build on the public R7 ("robot B") if it is the
  best single family, feed it better labels, and add OAI's knees to it.** Its trainer is public; its run was 176
  windows a study, 16 epochs, 4 folds on a 32 GB-class card: **not a Kaggle job** (~100 h on two T4s and over a T4's
  memory, by estimate). On a rented 5090: ~35–45 h of machine, ~$20–45, for its recipe on a pseudo-label table
  (reports + the OAI-trained CoAtNet + R7's own held-out predictions + ours); with OAI's ~2,400 knees ~55–65 h,
  ~$30–65. All estimates. **v3 is now the fallback, for the case of Kaggle's T4s only.**
- **Two reads approved 15:10, both kernels COMPLETE and gated 15:25** (`notebooks/build_r7_read_kernel.py`; rules
  in §4, gate lines in the LOG): `rsna-knee-sub-r7-alone` (R7 as published; its score alone exists nowhere) and
  `rsna-knee-sub-r7-alone-flip` (its mirror pass on, one line changed; `FLIP_TTA_READY` in both workers).
  **A second submitter (pid 99722) is armed for 17:06** with `notebooks/submit_plan_2026-10-13-r7.json`: the two R7
  reads, then tonight's three if the 17:02 submitter (pid 96147) has not sent them. **After 17:10: `tail
  data/submitter.log` and `$K competitions submissions $C`; five should carry the new UTC day.** The flip read can
  fail by running late (two passes of four folds): then it raises and shows no score. `jessky`'s public
  pseudo-labelled folds were checked and are NOT a read worth a slot (the small recipe, no public inference
  settings, nothing to gate against).
- **OAI's images need the owner first:** the NDA / OAI account and its data-use terms are theirs to sign; then the
  download (~200–400 GB, unverified), a MOAKS → findings mapping we would write, and R7's cache builder on OAI's
  series. The forum's challenge to OAI's eligibility has no host reply. Measured by its one public user: +0.003 to
  +0.005 on one model.
- **Still open: which GPU.** Nothing trains until it exists.
- **The golds stay held out** (the owner asked again 12:37; LOG): 58 studies are worth ~+0.0005 as training rows
  and are the one check that is not biased against pseudo-labels. Not changed without the owner's word.
- **OAI-free line for a final:** the five consensus folds (0.945, ref `57112913`) are it today. Its v3 version is
  ready to train (`targets_consensus_v3`, §7) if hours allow: the same recipe, ~24.5 h more.

## 1. Goal and the owner's directions

- **0.97 macro AUROC** (10-11). Do not reframe it. Final submission **10-22 16:59 PDT**, two finals.
- **Use whatever is available to us** (owner 10-12 21:17; an earlier "our own models only" note is withdrawn). Public
  weights, models and data may be used and built on. Not asked, so not assumed: submitting the public blend as it
  stands (the owner once called that "useless").
- Solo: no merger.
- **Ask before a batch of submissions and before spending GPU hours.** A go covers what it named and nothing later.
  **In force since 10-13 12:37, a standing go that names:** v3 fold 0, then folds 1–4, then a second label round
  from v3's own held-out predictions; and two submissions, v3 five folds alone and v3 inside the public blend.
  Everything else is still asked first.
- Keep this page a current-state page; dated detail goes to the LOG. No Claude attribution in this repo's commits.

## 2. Where we are

| | LB | |
|---|--:|---|
| #1 | 0.966 | two teams; three at 0.965, three at 0.964, 0.963 at rank 9–19 (pulled 10-13 11:42; re-pull before quoting) |
| medal lines, by Kaggle's rule on 5,938 teams | | gold to rank ~21 (0.962), silver to ~297, bronze to ~594. **Both the silver and the bronze line are inside the 0.956 block** |
| best single models reported on the forum | 0.953–0.960 | five-fold ConvNeXt-T 0.959 (rank 5–6, "only the competition data"), a single-fold 0.956, a "basically single" 0.960. Those that say how: pseudo-labels, no outside data (0.953), labels alone (0.938 → 0.949) |
| **the free public recipe** (`nartaa` 384 px + `heliosli` R7, 50/50 rank blend at full precision) | **0.956** | **583 teams, rank 152–734**; 14 public notebooks, none higher. 0.957 is rank 90–151 |
| older free blends (S6 at 30%, or written at six decimals) | 0.954 | 447 teams, rank 799–1,245 |
| public single model (`nartaa` CoAtNet, trained with 2,399 OAI knees) | 0.949 | its 224 px sibling in the same dataset: 0.945 |
| **ours: five folds, consensus labels** | **0.945** | ref `57112913`, **rank 1,792 of 5,938**. Our best submission, and our only OAI-free one |
| **ours: one model, teacher labels** | **0.944** | ref `57122696`. Our best single model |
| ours: one model, consensus labels | 0.938 | ref `57098827`. **The baseline every recipe change is paired against** |
| ours: one model, `dread_soft` labels | 0.933 | ref `57117178` |
| ours before this branch (Ortho ViT-L, all data) | 0.929 | ref `56811647` |

**How much a leaderboard number is worth (measured 10-13 by resampling our own held-out predictions at the public
part's size, ~390 studies; the private part is ~910):**

| what is read | noise, 1 SD | so |
|---|--:|---|
| one macro score | ~0.008 (0.005 on the private part) | our "0.945" is 0.945 ± 0.008 as an estimate of the model; the top of the board is inside one SD of itself |
| two of OUR models that differ in labels only | ~0.003 | the LB can show a label gain of ~0.006 or more; anything smaller is not readable |
| two unrelated models | ~0.005 (09-27) | a difference under ~0.010 says little |
| one finding's AUROC, as a level | 0.02–0.03 | the probe table below is the public part's truth, not the test's |
| one finding, two of our models (a probe pair) | ~0.010 | "two ticks decide" (0.024) is ~2.4 SD: a sound rule |

**The model.** ConvNeXt-small (in12k → in1k, 384 px) on the Stage B cache: 7 series slots, 96 slices, 140 mm field
at 384 px, left knees mirrored to right. Windows of 3 consecutive sampled slices → global average pool → per-finding
attention over the windows → 12 logits. 12 epochs, 12 random windows a study per step, plain BCE on soft report
targets, no geometric augmentation. One fold = ~1.9 GPU-h on both T4s (7.8 min/epoch); five folds, held out by site.

**Where the error is: the public part of the test, by single-finding probes on the five-fold consensus model.** The
readout rounds to ±0.006; the sampling noise of each level is 0.02–0.03 (above). The three weak findings are ~0.07
under the rest, so the grouping holds; their order does not.

| finding | AUROC | share of what is left | what limits it (review §2.2) |
|---|--:|--:|---|
| Synovitis (measured on fold 0) | 0.884 | 18% | the labels: reports rarely address it |
| Lateral OA | 0.896 | 16% | the model's vision: the report reads gold better than the image model |
| PF OA | 0.896 | 16% | the model's vision |
| Lateral Meniscus | 0.932 | 10% | both |
| MCL | 0.968 | 5% | — |
| the other seven, by subtraction | mean 0.966 | 36% | — |

0.97 needs those first three near 0.95 with everything else near 0.975. They are also the three findings the public
0.949 model's OAI knees supervised (its author's write-up). Gold-58 gives the same order but reads every weak finding
0.03–0.07 lower: do not size anything from gold.

**Where we stand against the outside (audit 10-13 11:55).**
- **OAI is not what separates the top from us.** Rank 5 and rank 6 both state competition data only (thread 748295),
  and the forum's best singles read 0.953–0.960; the ones that say how name pseudo-labels and label work, with no
  outside data. The review's "0.97 needs image-graded supervision" (§0.6, §6) is corrected there.
- **Our models are ~0.015 behind those pipelines on every instrument we share with them.** A rank-29 team reports
  five-fold OOF 0.937 against its LLM labels and gold-58 0.938 for LB 0.961 (thread 748011). Ours: five consensus
  folds OOF 0.900 against `dread_soft` and 0.905 against their own targets, gold-58 0.931, LB 0.945; five teacher
  folds 0.907 / 0.915 / 0.931, LB read tonight. Other tables than theirs, so indicative only.
- **What a member like ours adds to the free 0.956 recipe is ~0, measured by others:** a 0.929 reader at full weight
  cost its own author ~0.002 on that base (he now runs it at a tenth); `nartaa`'s 224 px checkpoint (0.945 alone)
  moved a rank inside the block and not the displayed score; ours at 1/3 on the older stack was 0.941 against 0.943
  (10-04). Nothing we own is shown to beat or to lift what is free.
- **Rules risk on everything OAI-derived** (the free recipe, and our `targets_teacher_v1/v2` and teacher folds):
  thread 748276 (10-11, 15 votes, no host reply) argues that NDA has barred China and five other countries since
  04-2025, that rule 2.6.b names geo-restrictions, and that KneeCoT was ruled out for uneven access. A participant's
  argument, not a ruling. Keep one final OAI-free.

## 3. Decided (each by a rule written before its result; detail in the LOG)

| question | result | outcome |
|---|---|---|
| ConvNeXt-small against ViT-L | 691 held-out 0.8654 against 0.8534; LB 0.938 against 0.927; 5× cheaper | **the workhorse** |
| five folds against one | LB 0.945 against 0.938 | **ensemble: +0.007** |
| consensus labels against `dread_soft` | LB +0.005 (held-out +0.0049) | kept, not shown (bar +0.010) |
| teacher labels against consensus | LB +0.006 (held-out had said +0.0138) | **kept**, not shown. Use them for finals |
| Synovitis: mask unread cells or fill from Effusion | probes 0.884 against 0.872: one tick | a tie; the mask stays |
| denser coronal and axial slices (Stage C cache) | held-out macro +0.0030 ± 0.0023; Lateral OA / PF OA / MCL −0.0002 ± 0.0049 | **FAIL: Stage B stays.** It ran 11 epochs of 12 |
| the public OAI-trained model run on our cache | gold 0.920 (ours 0.928) | usable as a teacher; its LB in our input is read 3 of §4 |
| affine augmentation | −0.0019 ± 0.0048 on ResNet34, 09-27 (`75ab54c`) | not run on this recipe; held |
| a bigger backbone | ViT-L lost by 0.011; the forum's best singles are ConvNeXt-Tiny, ResNet50, EfficientNetV2-S at 224 px | not proposed |
| a stronger LLM re-reading the reports | every reader reads the same text; best tables ~0.90 on gold, our image model 0.93 | not run (~$5–20; the judge is the cost) |

**OAI images themselves: not pursued, the owner's call.** The rules side looks open (host: allowed if reachable
without institutional sign-off; one participant reports individual approval at once, unverified by us; the host has
not said yes or no). The images cannot go on Kaggle, so training needs another GPU: this M5 laptop is ~5× slower than
Kaggle (17.7 img/s against ~90, measured), a rented 5090 is ~$0.65/h (whether OAI's terms allow a rented machine is
not answered). The 0.949 author gained +0.003–0.005 from 2,399 baseline knees. Sizes and costs: LOG, 10-12 evening.

**Closed before this branch (`9c23daf`), with what has changed since:**
- *Teacher rounds from the same reports:* the round that read ~0 on the LB used four ResNet34 teachers. Top teams
  report their best singles from pseudo-labels of strong models, so a round from our own five folds is **re-opened
  and running** (§4, read 4).
- *The fastMRI+ teacher:* what failed was a ResNet34 on four findings judged on gold-58, not fastMRI+ as data. Its
  ceiling for us is low (coronal only, no synovitis label); the 556-knee cache is still on Kaggle in Stage A format.
- *Our model as the public stack's core:* lost with our 0.93 models. With the direction changed and ours at 0.945,
  adding the public checkpoint to OUR ensemble is open (§4).
- Still closed: centre crops and zoom; an nnU-Net crop; Qwen report labels; a per-site prior.
- **Training on the 58 gold studies, repeated or not: not run, and not proposed** (owner's question 10-13 ~12:30; LOG).
  They are 696 cells against 49,784, the fold → all-data step (+847 studies) was worth ~+0.002, and repeating a
  study adds no information. Free only as rows in a final all-data fit, at the cost of that model's gold read.

## 4. Open work

**Today's reads, 17:00 PDT reset (three go, by the owner's word of ~12:00; submit only after the gate lines read clean, §6).**

| # | submission | kernel (`raahimnawaz/…`, v1) | gate | against | rule |
|---|---|---|---|---|---|
| 1 | the five teacher folds, one ensemble | `rsna-knee-sub-a-ortho-teach1-cv5-nall` | passed (5/5) | 0.945 (`57112913`) | kept unless ≥ 0.005 below; shown at ≥ +0.010; at ≥ 0.945 the finals' candidate |
| ~~2~~ | ~~Synovitis probe, teacher-label fold 0~~ **NOT submitted (10-13 12:00):** no decision hangs on it; the teacher labels are kept either way | `rsna-knee-sub-a-ortho-teach1-f0-nall-p-synovitis` | passed | 0.532 (`57117247`) | two ticks decide whether the teacher moved Synovitis |
| ~~3~~ | ~~the public checkpoint alone, in our input~~ **NOT submitted (10-13 12:00):** the blend runs it in its own notebook and the fine-tune it would have sized is dropped | `rsna-knee-sub-a-ortho-oai-pa-nall` | passed | its own 0.949 | no pass / fail: near 0.949 → use it as it is and skip the fine-tune; ~0.940 or lower → our input costs it and the fine-tune has room |
| 4 | the pseudo-label student alone. Kept in: the one LB point on the lever proposed for Friday's hours, and only ≥ 0.950 or ≤ 0.940 would move that plan | `rsna-knee-sub-a-ortho-teach2-f0-nall` | passed | 0.944 (`57122696`) | kept unless ≥ 0.005 below; shown at ≥ +0.010 |
| **5** | **the free 0.956 recipe + our five teacher folds** (`notebooks/build_public_blend_kernel.py`, from `6bc69f9`) | `rsna-knee-sub-pub956-teach1cv5` | passed (5/5), 12:23 | 0.956, the recipe as published (rank 152–734) | below |

**Read 5, written 10-13 12:15 before its commit run finished.** The public notebook
(`varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956`, sha256 `6406e923…`: `nartaa` 384 px with the
224 px checkpoint at 35% of that slot, R7's four folds, per-finding ranks 50/50) runs unchanged; our five folds run
after it and join by per-finding percentile rank at **0.25 on ACL, MCL, Medial Meniscus, Medial OA, Effusion,
Baker's, Contusion and Fracture** (our probes read 0.966–0.968 there) **and 0.10 on Lateral Meniscus, Lateral OA,
PF OA and Synovitis** (0.88–0.93). Fixed before any run, fitted to nothing. What they rest on: within-class rank
correlation of our folds with the public CoAtNet 0.85–0.88 (in our input, so an upper bound), under which a member
at par adds ~+0.002 a finding at 0.25 and one 0.03 behind adds ~0 at 0.10. **Expected: +0.001 macro, between −0.001
and +0.002.** In the scored rerun any fault, or under an hour left of the nine, keeps the public submission.
**Rule:** the paired noise of such a pair is ~0.0007 at 390 studies (resampled on a stand-in pair), one displayed
tick is ~1.4 of that. **≥ 0.957 → our folds stay in and the per-finding weights are the next thing to test;
0.956 → read our rank inside the block against the recipe's own (its author's team was ~207th on 10-13), nothing
claimed; ≤ 0.955 → our folds cost it and come out.** It cannot say which finding moved.

```bash
C=rsna-knee-abnormality-detection   # each: $K competitions submit $C -k raahimnawaz/<kernel> -v 1 -f submission.csv -m "<message>"
# 1  "ConvNeXt-small x5 ALONE on targets_teacher_v1: read vs the five consensus folds 0.945 (57112913); kept unless >= 0.005 below, shown at >= +0.010"
# 2  "PROBE Synovitis only: the teacher-label fold-0 model (0.944) against the consensus model's probe 0.532 (57117247); two ticks decide"
# 3  "The public OAI-trained CoAtNet (nartaa) AS IT IS in OUR input (Stage B), alone: what our input costs it against its own 0.949"
# 4  "ConvNeXt-small ALONE, one model, targets_teacher_v2 (pseudo-label round): read vs the 0.944 model (57122696); kept unless >= 0.005 below, shown at >= +0.010"
# 5  "The free 0.956 recipe as published (nartaa 384 + 224 px at 35% of that slot, heliosli R7 x4, ranks 50/50; notebook 6406e923) + OUR five ConvNeXt-small folds on targets_teacher_v1 as one more stage: per-finding ranks, ours 0.25 on ACL MCL MedMen MedOA Effusion Bakers Contusion Fracture and 0.10 on LatMen LatOA PFOA Synovitis, fixed before any run; >= 0.957 ours stay in, 0.956 read the rank in the block, <= 0.955 ours come out (6bc69f9)"
```

The fifth slot is kept for an ensemble with the public checkpoint. **Not approved: the owner decides how to ensemble
after these scores.** What the forum shows works: every public blend runs `nartaa`'s notebook unchanged as stage 1,
in its own preprocessing, and blends per-finding percentile ranks. For us that is the Submission B path of
`build_submit_kernel.py` (last used 10-03, needs pointing at `nartaa/rsna-knee-0949-anatomical-mirror`), with
weights fixed beforehand and theirs heavier on PF OA, Lateral OA and Synovitis.

**Added by the audit of 10-13 11:55 (proposed, NOT approved; no training GPU):** the base to build on is no longer
the 0.949 notebook but the free 0.956 recipe (`heliosli/0-957-latest-dataset-score-in-one-click`: `nartaa` 384 px +
R7's four folds, per-finding ranks 50/50, **scores written at full precision**: six decimals tie R7's small
probabilities and cost 0.002, which is the whole 0.954 → 0.956 step), with `nartaa`'s 224 px checkpoint at 35% of
the CoAtNet slot (`varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956`). Our five folds would join at
a small weight fixed beforehand (≤ 20%; rule 4). **The read:** a submission that becomes our best shows its rank
inside the 0.956 block, which follows the unrounded score (the 224 px checkpoint moved 233 → 156 there), so the
recipe with and without ours can be told apart below 0.001. That needs the unchanged recipe once as the paired
baseline, which the owner has called useless as a submission: ask. Our own CSV is already written at full precision
and our outputs sit in 0.06–0.99 (checked: ≤ 4 ties of 847 at six decimals).

**After the scores (written 10-13 10:40, before any of them).** Read 1 at ≥ 0.945 → the five teacher folds are the
finals' candidate until something beats them; with read 6a's +0.006 it is the second read of the same label change.
Read 4: the student's held-out gain is +0.0021, under one SD of what the LB can read, so **"kept" by its rule buys
nothing: five folds on `targets_teacher_v2` (~9.5 GPU-h) are trained only if read 4 is ≥ +0.006 over 0.944**; a
smaller or negative read leaves `targets_teacher_v1` as the finals' labels. A stronger dose (the OOF at full weight,
or other teachers per finding, as the forum describes) is a different arm and would need its own go.

**Two reads of the public R7, approved 10-13 15:10 (rules written 15:17, before either kernel was built).**

| read | kernel | against | rule |
|---|---|---|---|
| A: R7's four folds alone, as published | `rsna-knee-sub-r7-alone` | the CoAtNet's 0.949, our 0.945 | a level, ± 0.008. **≥ 0.950 → R7 alone is the OAI-free final candidate** in place of the five consensus folds; it is also the baseline for R7's recipe on a pseudo-label table (shown at ≥ +0.006 over it) |
| B: R7 alone + its mirror pass | `rsna-knee-sub-r7-alone-flip` | read A | **≥ +0.002 → the pass goes on wherever R7 is used; −0.001 to +0.001 → not shown, stays off (it doubles R7's GPU time); ≤ −0.002 → off** |

**Tonight's three, reviewed again 15:15 at the owner's word ("make sure those 3 are actually worth it"):** read 5
is the one that matters (the only one that can become our best score and a final); read 1 is worth it moderately
(the evidence for the CoAtNet's opinions in a label table, readable at ≥ +0.006); read 4 is weak (it changes
nothing in about five outcomes of six) and stays only because its slot would otherwise be empty and it costs no GPU.

**v3, the combined recipe (owner 10-13 12:37; everything here was fixed before any run). Since 15:10 the fallback
for a Kaggle-only week: R7's own recipe on a 5090 comes first (§0).**

| part | what | state |
|---|---|---|
| labels | `targets_teacher_v3`: `targets_teacher_v1` with the five teacher folds' held-out predictions at **0.75** of every cell (v2 was 0.5: reports 12.5%, public CoAtNet 12.5%, our held-out models 75%). Cells move 0.050 on average (v2 0.033); 4.9% cross 0.5 (v2 3.4%): **1.5× v2's dose, not a different regime** | built, in the Kaggle dataset |
| pooling | `--spatial-attn` | built 10-12, in the smoke |
| mirror | `--mirror 0.5`: half the training draws are the knee mirrored medial ↔ lateral (a sagittal slot read backwards at depth 1 − pos; coronal and axial slices flipped), Medial ↔ Lateral Meniscus and OA targets swapped, MCL masked. `--mirror-tta`: every prediction is the mean of the plain and the mirrored pass, MCL from the plain pass | built 10-13, 66 offline checks |
| windows | `--k 24 --micro 1` (24 windows a forward per GPU, as today; the batch stays 8 studies) | flag existed |
| schedule | 16 epochs; submit `avg3/last.pt` (mean of the last three epochs' weights), gated against `avg3/oof_all.csv` | existed |

- **One fold per kernel, ~4.9 h on two T4s (estimate: twice the measured k 12 epoch, × 16, + four evaluation
  passes); five folds ~24.5 h; a second label round the same again.** §5.
- **Fold 0 is a not-broken check. "Clearly worse" = stop and tell the owner:** against `runs_cnxs_teach1_f0` (847
  macro 0.8804, the 0.944 model) on the same studies, v3's `avg3/oof_all.csv` more than **0.008** below on the macro,
  or more than **0.03** below on any of MCL, Medial / Lateral Meniscus, Medial / Lateral OA (a mirror or swap fault),
  or fewer than 16 epochs. Otherwise folds 1–4 go. It is not a test for a gain: the 847 under-credit pseudo-labels.
- **The two submissions with a standing go:** v3 five folds alone (against the five teacher folds, tonight's read 1;
  a label-and-recipe pair, readable from ~+0.006); v3 in the public blend in place of the teacher folds, same
  weights as read 5 (`build_public_blend_kernel.py`), against read 5.
- **A second label round** = `build_consensus_targets.py … --own-oof <v3's five cv_avg3 files> --w-own 0.75` on
  `targets_teacher_v1` again, then five more folds. Only with hours to spare.
- **The OAI-free twin:** the same recipe on `targets_consensus_v3` (`targets_consensus_v2` + the five CONSENSUS
  folds' held-out predictions at 0.75; 2,404 unread Synovitis cells are filled by those models alone). Not started.
- **What v3 cannot tell us:** which of its five parts moved the score. The owner's choice.

**Superseded by v3 on 10-13 12:37 (kept for the record: each arm's own rule and the audit's view of it):**

| arm | cost | state | rule |
|---|--:|---|---|
| spatial attention (`--spatial-attn`), for Lateral OA and PF OA | ~1.9 h | built, CPU smoke clean; the owner moved it to Friday. **Recommended to hold (audit 11:55):** the public R7 already ships this on our backbone, with its trainer; and the bar is 2 SE of study sampling alone, run-to-run noise never having been measured (§6) | paired on the 847 against the 0.938 model: macro ≥ +0.004, or the mean of Lateral OA and PF OA ≥ +0.015 → into the finals |
| fine-tune of the public CoAtNet on our input and `targets_teacher_v1`, all data, 6 epochs | ~3.5 h (estimate) | CPU smoke clean; one push away. **Recommended to drop (10-13 audit; the owner asked for it, so the owner confirms):** its best case is the checkpoint's own 0.949 + ≤ 0.005, which the LB cannot read, and running the public notebook unchanged as stage 1 gives the known 0.949 for no GPU hours. **And it is already public:** `syouyatobita/rsna-knee-g231-nf1-weights` (10-11) is this checkpoint fine-tuned 6 epochs on all data; no public notebook uses it and it carries no score | LB only (its init saw every study): ≥ +0.005 over read 3, or it did not pay |
| `--k 24` | ~3.3 h | not built. **Recommended to drop (audit 11:55):** its expected gain is under what the 847 can resolve, and the public 0.949 model also trains at 12 windows a study | macro ≥ +0.004 → finals; below +0.002 closes bag size |
| finals: five new folds | ~9.5 h | **only if an arm above passes** (or read 4 ≥ +0.006); otherwise the five teacher folds already trained are the finals | alone on the LB |

**What the audit of 11:55 recommends the hours go to instead (proposed, NOT approved).** Every recipe arm above is
worth +0.004 at best on a 0.945 ensemble that sits 0.011 under the free recipe; as a ≤ 20% member of a blend that is
under 0.001. The one lever with outside evidence of +0.010 or more is the labels: `Raymond Yuen` 0.938 → 0.949 on
labels alone, the 0.951–0.953 singles that say how are pseudo-labelled, rank 5 and 6 use competition data only (their
method is not public). Our one round
(read 4) was a half dose on one fold, judged on the 847 against a report reader, which cannot credit a model for
out-ranking the report (§6). So: after tonight's reads, **one full pseudo-label round as five folds (~9.5 h)**, read
five folds against five folds on the LB (readable from +0.006). The dose and the teachers need writing down first.

## 5. Budget

- **GPU:** 30 h/week on Kaggle, reset Fri 17:00 PDT; **10-16 is the last reset**. By our count ~0.7 h is left until
  then once the student finishes (the owner's quota page read 10.2 h on 10-12 20:46; that page is the truth).
  **casmi26 shares the quota.** Two GPU sessions at once, five CPU sessions.
- **Submissions:** 5 a day, reset 17:00 PDT (00:00 UTC). 25 used before today's four.
- **v3 against that:** ~4.9 h a fold on Kaggle (estimate), ~24.5 h for five, of the last 30 h week, which casmi26
  shares. A second label round or the OAI-free twin (~24.5 h each) does not fit on Kaggle beside it.
- **A GPU off Kaggle** (the owner's 12:37 message says one is coming, not which): `notebooks/offkaggle.py` (§6).
  It needs this repo, `~/.kaggle` credentials, `data/train.csv`, the private targets dataset (tables and
  `folds_site_stage_b.csv`) and the Stage B cache, **~62 GB** as eight kernel outputs (4,407 studies × 96 × 384²
  bytes). `offkaggle.py pull` has not been run at that size, and this page's own note says a 99 MB kernel output
  once broke the download: try one shard first. A rented 5090 was estimated at ~$0.20–0.99/h and several times
  two T4s; this M5 laptop is ~5× slower than Kaggle (measured 10-12), ~24 h a fold: not an option for five.

## 6. How to run things

`K=.venv/bin/kaggle` and `PY=.venv/bin/python` (system Python has neither). Kernels are generated from committed
code: **commit first**, build into a scratch folder, push, and never edit a notebook on Kaggle. The images are only
on Kaggle; `data/` here holds tables.

```bash
# one fold of the baseline recipe (fold 0 of five, site-held-out). Change --targets / --tag for another label table.
$PY notebooks/build_backbone_ab_kernel.py --arch convnext_small --corpus stage_b --folds 0 --n-train 0 --epochs 12 \
    --time-budget-h 3.0 --targets data/student/targets_consensus_v2.csv \
    --targets-from raahimnawaz/rsna-knee-student-targets --tag <tag> --train-args=--allow-missing-labels --out <dir>
$K kernels push -p <dir>                     # build with --smoke (and another --tag) first for a CPU wiring check

# results: small files only (the 99 MB weights break the download)
$K kernels output raahimnawaz/<kernel> -p fusion/runs_<name> --file-pattern '.*\.(json|csv|txt|log)$'
$PY fusion/score_oof.py fusion/runs_cnxs_cons2_f0/runs_convnext_small/fold0 fusion/runs_<name>/runs_convnext_small/fold0

# weights sha256, read where the file is
$PY notebooks/build_weights_sha_kernel.py --kernels raahimnawaz/<training kernel> --out <dir> && $K kernels push -p <dir>
$K kernels output raahimnawaz/rsna-knee-weights-sha -p <dir>          # weights_sha.json

# a submission kernel: one model, an ensemble (comma lists), or one finding (--probe)
$PY notebooks/build_submit_kernel.py --ours-only --weights-kernel raahimnawaz/<k1>[,<k2>] \
    --weights-sha <64 hex>[,<64 hex>] --tag <short> [--probe "Lateral OA"] --desc "<what this is>" --out <dir>
$K kernels push -p <dir>
# read in its log before submitting: `sha256 ok` · `GATE PASSED (n/n)` · `[sub A] ARMS:` · 0 tracebacks
$K competitions submit rsna-knee-abnormality-detection -k raahimnawaz/<kernel> -v <n> -f submission.csv \
    -m "<what, against what, and the rule>"

# ---- v3 (§4). One fold per kernel; --recipe v3 = the flags and 16 epochs, defined once in the builder.
$PY notebooks/build_backbone_ab_kernel.py --arch convnext_small --corpus stage_b --folds 0 --n-train 0 --recipe v3 \
    --time-budget-h 9.0 --targets data/student/targets_teacher_v3.csv \
    --targets-from raahimnawaz/rsna-knee-student-targets --tag v3 --out <dir>      # folds 1..4: --tag v3-f1 … (slug < 50)
#   9.0, not the ~5 h it should take: --fit-budget cuts epochs if 16 x (measured epoch x 1.35) does not fit, and the
#   epoch time at k 24 with spatial attention is an estimate. 9 h keeps 16 epochs up to 25 min an epoch, with 3 h
#   of the 12 h session left for the four evaluation passes. Check `fit-budget: … -> 16 epoch(s)` in the first minutes.
#   CPU smoke first: add --smoke and --tag v3s. Not-broken check after fold 0 (§4's numbers):
$PY fusion/score_oof.py fusion/runs_cnxs_teach1_f0/runs_convnext_small/fold0 <v3 fold0>/avg3
# submission: the AVERAGED weights. --weights-sha = sha256 of each fold's avg3/last.pt (build_weights_sha_kernel.py)

# ---- v3 off Kaggle (the same lists, package and flags; fold 0's lists are identical to the Kaggle run's, in order)
$PY notebooks/offkaggle.py pull --dest <cache>                                   # ~62 GB, untested at that size
$PY notebooks/offkaggle.py train --fold 0 --corpus <cache> --labels data/student/targets_teacher_v3.csv \
    --folds data/student/folds_site_stage_b.csv --out <runs>                     # --dry prints the command only
$PY notebooks/offkaggle.py pack --runs <runs> --slug rsna-knee-v3 --out <dir>    # one flat private dataset per fold;
#   it prints the `kaggle datasets create` lines and the build_submit_kernel line (--weights-dataset a,b,… --weights-sha …).
#   build_public_blend_kernel.py takes --weights-dataset too. The kernel's commit run gates them as it does a kernel's.

# ---- the 17:00 submitter (survives the session; not a reboot). The plan file lists kernel, version, message, key.
$PY notebooks/submit_at_reset.py --plan notebooks/submit_plan_2026-10-13.json --dry-run     # what it would send now
$PY notebooks/submit_at_reset.py --plan notebooks/submit_plan_2026-10-13.json --not-before "2026-10-13 17:02:00" --detach
tail data/submitter.log

# the arms that waited for GPU before v3 (§4): the baseline line above with
#   spatial attention:  --tag cons2-sa "--train-args=--allow-missing-labels --spatial-attn"
#   the fine-tune:      --arch coatnet --all-data (no --folds) --epochs 6 --time-budget-h 5.0 --tag oai-ft \
#                       --targets data/student/targets_teacher_v1.csv \
#                       --extra-datasets nartaa/rsna-knee-publication-swa-weights-20261007 \
#                       "--train-args=--init raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt --allow-missing-labels --eval-bs 1 --enc-chunk 16"

# a pseudo-label table: the teacher table again, plus our fold models' held-out predictions (one file per fold kernel)
$PY labeling/build_consensus_targets.py --labels-dir data/public_llm_labels --gold data/train.csv \
    --teacher fusion/runs_coatnet_oai_teacher_pa/runs_coatnet/oof_all.csv --w-teacher 0.5 \
    --own-oof fusion/runs_cnxs_teach1_{f0,f12,f34}/runs_convnext_small/cv_last/oof_all.csv --w-own 0.5 \
    --out data/student/targets_teacher_v2.csv
# a new table goes into the Kaggle dataset as a new version: a folder with EVERY table already there, the new one
# and a dataset-metadata.json ({"id": "raahimnawaz/rsna-knee-student-targets", "title": …, "licenses": [{"name": "other"}]})
$K datasets version -p <dir> -m "<what was added>"

$PY fusion/error_budget.py                                          # the review's tables
$PY notebooks/canon_fixture.py --out <dir> && $PY notebooks/infer_parity_fixture.py --fixture <dir>   # offline tests
$PY notebooks/mirror_fixture.py --out <dir>                         # the mirror: 66 checks, ~1 min, no GPU
```

**Judging a change.**
- *Recipe or input, labels fixed:* fold 0, paired with `score_oof.py` against the 0.938 model on the same 847
  held-out studies. Bar so far: macro ≥ +0.004 (paired SE ≈ 0.002). The LB then confirms only the final ensemble.
  **⚠️ That SE is study sampling only (audit 10-13 11:55): no recipe has ever been run twice, so run-to-run noise is
  unmeasured.** One hint that it is not small: the 0.944 model and the pseudo-label student share seed, split and
  recipe, their targets differ by 0.02–0.05 a cell, and single findings moved −0.009 to +0.011 between them. Until a
  repeat of the baseline exists (~1.9 h), read +0.004 as "not shown" in both directions: the dense arm's +0.0030
  "fail" and a +0.004 "pass" are the same reading.
- *Labels:* the held-out number overstates (it is scored against another report reader). The LB decides, one model
  against one model: kept unless ≥ 0.005 below, shown at ≥ +0.010. The LB's noise for such a pair is ~0.003 (§2), so
  "kept" alone is not evidence of a gain and must not buy GPU hours; ~+0.006 is the smallest readable gain.
  **For pseudo-labels the 847 errs the other way:** the reference is a report reader, so a student that out-ranks
  the report from the image is scored as wrong there. The pseudo-label student's +0.0021 is not a size for that
  lever, and nothing we have can see a label gain under +0.006.
- *The repo's numbers reproduce* (same audit): the four paired reads of §3 to the fourth decimal, the noise table of
  §2 (0.0075 one score, 0.0030 a label-only pair at 390 studies), gold in no OOF file (0 of 58).
- *One finding:* `--probe`; score = (5.5 + AUROC) / 12, one tick = 0.012 on that finding, two ticks decide.

**Things that have already cost a run or an hour.**
- A kernel slug over 50 characters is refused with a bare 400. Keep `--tag` short.
- Push a kernel that consumes another kernel's output only after the source reads COMPLETE.
- On a newly built cache the first epoch is slow and `--fit-budget` then cuts epochs: pass `--time-budget-h 4`.
- `--train-args=--flag` needs the `=` when the value is a single token starting with `--`.
- A CPU smoke does not reach GPU-only branches; force those locally before an architecture's first GPU run.
- zsh does not split `$var` in a `for` loop; write the list out.
- The Mac sleeps with the lid shut and background watchers stop with it; Kaggle runs carry on. Anything that must
  happen at a set hour needs the lid open. Take times for the LOG from `date`, not from memory.
- `kaggle kernels output` can hang for hours after a sleep: run it with a timeout and retry.
- CoAtNet on Stage B needs its own argument package (`COATNET_PACKAGE`); the ViT one fails on `--ckpt-blocks`.
- A third GPU kernel cannot start while two are running: build it, push it when one finishes.
- A submitter armed as a session's background task stops with that session (twice on 10-13). Use
  `submit_at_reset.py --detach`, and check `data/submitter.log` and the account after the hour.
- `--k 24` with two studies a forward is 48 windows: over a T4's memory (24 windows peak at 8.1 GB). `--micro 1`.
- A CPU smoke of one epoch never reaches the averaged weights; the small-backbone smoke now runs two.

## 7. Labels and weights

Targets are in `data/student/` (git-ignored) and in the private Kaggle dataset `raahimnawaz/rsna-knee-student-targets`.

| table | sha256 | what it is |
|---|---|---|
| `targets_teacher_v3.csv` | `6f786dda…` | teacher_v1 with the same held-out predictions at **0.75** of every cell. **v3's labels** (§4) |
| `targets_consensus_v3.csv` | `26401559…` | consensus_v2 with the five CONSENSUS folds' held-out predictions at 0.75, no outside teacher: **the OAI-free twin** |
| `targets_teacher_v2.csv` | `fa98f943…` | teacher_v1 with our five teacher-label folds' held-out predictions at half of every cell (a pseudo-label round). **Under test** (§4, read 4) |
| `targets_teacher_v1.csv` | `2980e8d6…` | consensus_v2 with the public OAI-trained CoAtNet's predictions at half of every supervised cell, and alone on unread Synovitis. **Best so far** |
| `targets_consensus_v2.csv` | `a1ed4034…` | seven public readers of the same reports on `dread_soft`'s levels; Synovitis masked where unread. **The baseline's** |
| `dread_soft` | `8cec477f…` | the public graded table both are built on (`dreaddevelopment/rsna-knee-labels`) |

`folds_site_stage_b.csv` (`b8d82239…`, same dataset): the fold of each of the 4,349 non-gold studies, for training
off Kaggle. Builder: `labeling/build_consensus_targets.py [--teacher <oof_all.csv>] [--own-oof <files>] [--w-own]`
(§6); with `--teacher` alone it rebuilds `targets_teacher_v1` byte for byte, and with `--w-own 0.5` v2 (checked
10-13). Never train on a table that
contains the gold rows (`tasmeemreza`, `yehezkielhaganta`, `dhiyaneshbabu`, `lvthanhxun` fused, `gperdrizet`), and
never on `lixin73`'s: `score_oof.py` scores against it.

Weights stay on Kaggle as kernel outputs (`runs_convnext_small/fold<k>/last.pt`):

```
consensus, five folds   raahimnawaz/rsna-knee-c7-train-convnext-small-cons2-cv       fold 0  8fc45272b82839f4439f4f7a069d2b5754fc9191063a0cc2a3eae63a9c4e6e58
                        raahimnawaz/rsna-knee-c7-train-convnext-small-cons2-f12-cv   fold 1  a3113be6535d7d6e8b6c441c9000ae66d62ea2cf86408eeece58575a4205e00d
                                                                                     fold 2  d147aa3c816ad1ad2b23c8202b1f534112514c667660d4cf84b0e163c7ed08b3
                        raahimnawaz/rsna-knee-c7-train-convnext-small-cons2-f34-cv   fold 3  eb3bb1627d24a5cfa7118ee6721b6b154e970431223f5b9820dd5deca6c7af7d
                                                                                     fold 4  8b0768b0e78e80ecaa6cf8826102d44f8c9f79bc487863047da703aa364eecc8
pseudo-labels, fold 0   raahimnawaz/rsna-knee-c7-train-convnext-small-teach2-cv      fold 0  4cc8c3d45263bd705b86ff0c8b98d391935bdd553a0076268cd5f2dabf41ae02
teacher labels, five    raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-cv      fold 0  1c58b2bcfe5adc0168fffba48f38735e5888ce86a3291db3ab6cb96155d93b20
                        raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-f12-cv  fold 1  683857fad528ce5cc3518bb8e5be3314bfcfc89ff9d276a587c6c3a038aa5de6
                                                                                     fold 2  479cf64b1b5123f4684183e4f048be8b09fba8c34e4b03223aec3e3f97c88d75
                        raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-f34-cv  fold 3  4921d0968d3616a90e3ec22948f40d60529f9f1ab3f4ed458b510da2a862269b
                                                                                     fold 4  c910af266a41af08042d48da83f2db6e7a43deeb7b6365531c5cab30b0fd4685
public CoAtNet, re-saved raahimnawaz/rsna-knee-c7-train-coatnet-oai-teacher-pa       (all)   f900846d3e821721cf571a9f0a477021bc0c2e0e71ddb5565dc5687752fbd0c1
dread_soft, fold 0      raahimnawaz/rsna-knee-c7-train-convnext-small-dread-cv       fold 0  cdedbae5509faa6093f3f97d83366f8d08037e0221308140b2ebe86edcb7f3f1
```

Caches: Stage B = `raahimnawaz/rsna-knee-stage-b-s{0..7}of8` (in use). Stage C = `...-stage-c-s{0..7}of8` (built
10-12, tested, not adopted).

## 8. Standing rules

1. Never select on gold-58. Report it only.
2. Write the rule before the run; a null is a null.
3. The held-out fold screens recipe changes at fixed labels; the leaderboard judges labels.
4. A weaker member at equal weight costs a blend.
5. Commit before building a kernel; push after each step.
6. Leak-check any downloaded label table against gold before it trains anything.
7. A leaderboard number older than a few days is not evidence.
8. Before proposing an experiment, search the history for it (`git log --all --grep`, `9c23daf`'s docs).
9. Record a result where the text that proposed it still stands, not only in the LOG.

## 9. What is in the repo

| path | what | status |
|---|---|---|
| `pipeline/` | slot layouts (`slots96.py` = Stage B, `slots118.py` = Stage C, `slots64.py` = Stage A), DICOM → cache builder, provenance | live |
| `fusion/train_windows.py`, `infer_windows.py`, `folds.py`, `score_oof.py` | trainer, inference, site folds, paired scoring | live |
| `fusion/error_budget.py` | the review's tables | live |
| `labeling/build_consensus_targets.py` | the label tables of §7, including the pseudo-label step | live |
| `notebooks/build_*_kernel.py` | generators for the cache, training, weights-sha and submission kernels | live |
| `notebooks/*_fixture.py` | offline tests on synthetic DICOMs; `mirror_fixture.py` is the mirror's | live |
| `notebooks/offkaggle.py` | v3 on another GPU: the folds file, one fold's run, the cache pull, packing weights as datasets | live, never run on a GPU |
| `notebooks/submit_at_reset.py`, `submit_plan_*.json` | the detached submitter and the list it sends | live |
| `fusion/runs_cnxs_*`, `runs_coatnet_oai_teacher_pa` | this branch's run records: config, held-out and gold predictions, scores (no weights, no logs) | live |
| `fusion/orthofoundation_model.py`, `qwen_vlm.py`, `local_teacher_folds.py`, `window_curve.py`, `score_gold.py`, `labeling/build_student_targets.py`, the `core_*` fixtures, the other `fusion/runs_*` | the ViT-L, VLM, same-report-teacher and public-stack work up to 10-03 | legacy: kept for the showcase, not on the current path. Two files are still read: the training kernel embeds `runs_c6_ortho/oof_all.csv` and `runs_port_repro/oof_all.csv` as its paired control |
| `README.md` | the portfolio description as of 10-03, with a status note for this branch | |
