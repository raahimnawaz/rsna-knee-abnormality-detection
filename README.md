# RSNA Knee Abnormality Detection

Twelve-label knee-MRI classification, macro-AUROC. Final submission **2026-10-22**.

---

## START HERE — state as of 2026-08-12 (end of day)

> # WE ARE AT 0.899, AND THE FREE PUBLIC CEILING IS NOW REACHED
>
> **Banked: submission `55465252` = 0.899** (was 0.891). Top **0.946**, 10th/prize **0.935**.
> **Read `PLAN.md` §9a for the F-series — that is the current plan.** The E-series is closed.
>
> **AMENDED 2026-08-12 late, `IMPROVEMENTS.md` §3g — F2 got much cheaper.** §9b claimed the fork's
> members verify a fingerprint on their *pixel contract*, so no crop could reuse them. Reading
> `fingerprint()` instead of our own summary of it: the input is a **seeded synthetic bag of random
> bytes**, and the docstring says *"it cannot check that the pixels reaching it are the right
> pixels."* `img_size` trips it; **`CROP_MM` and `SLICE_BAND` do not.** So crops can go in as extra
> TTA windows on the frozen members — **no training run** — pooled per target exactly as §3d does.
> That is the same move that paid +0.008. The weights are CC0, 1.54 GB, and the A/B is local and
> paired. **Run each member on its own held-out fold only**, or memorised studies will hide the
> effect.
>
> ### The four things a fresh session most needs to know
>
> 1. **The port is CLOSED as an ensemble member (§2y).** Paired against the fork's own OOF:
>    **0.7323 vs 0.8434, −0.111 at 15.4σ, 0/12 labels**, and **no rank-blend weight helps at any
>    value.** Not a duplicate-member problem either — rank correlation 0.639. It had diversity and
>    is simply not good enough. **Give it no more compute as a member.**
> 2. **The fork ships its OOF** — `pilkwang/rsna-knee-weights::oof.npz`, 368 KB, all 4,407 studies,
>    `gold_mask` = 58. Imported to `fusion/runs_pilkwang`. **The fork is a local arm now; stop
>    comparing against a web-page number.** This is what made §2y, §3b and §3f possible in an hour.
> 3. **Never select on gold-58 (§3b).** A weight chosen on 58 studies claims +0.0137 and delivers
>    **−0.0034**, negative in **92%** of draws; bagging does not fix it. Reliable selection starts
>    around **n ≈ 400**. Gold-58 can *evaluate* a fixed decision, never *choose* one.
> 4. **A submission costs ~2 h, not 74 s (§3e).** The 74 s is the members' forward passes; a real
>    run also decodes the whole hidden test from DICOM. ~15 runs/week against a 30 h quota.
>    **Batch post-hoc experiments; never spend a run on one +0.002.**
>
> **The live board, 2026-08-12 (§2x):**
>
> | | LB | |
> |---|--:|---|
> | our own pipeline, best estimate | ~0.76 | never submitted; closed **as an ensemble member** (§2y), not as a pipeline — it is still the only local arm that can fine-tune |
> | the fork, submitted 2026-08-09 | 0.891 | rank 400 / 1,276 — 73 teams tied |
> | **OURS NOW — pilkwang + per-target TTA pooling** | **0.899** | `55465252`, §3d/§3e |
> | **10th place — the last prize** | **0.935** | only 10 teams clear it; was 0.934 that morning |
> | top | 0.946 | |
>
> **All four numbers moved against us since the 08-10 read** (§2x): top 0.942→0.946, 10th
> 0.926→**0.934**, field 908→**1,276**, and our banked 0.891 decayed **230 → 400** for doing
> nothing. **A banked score is not a banked rank.**
>
> **0.891 is not a floor we built — it is a commodity 73 teams share, and it is not even the free
> one.** The whole public field is compressed into **0.891–0.900**; 326 teams sit at or above
> 0.899. So the real bar is not our 0.043 to 10th, it is the **~0.035 from the free public plateau
> to the prize** — and that is what the severity-labels and anatomical-crops bets have to clear.
> They are not garnish on an ensemble; they are the whole game. Measure them on **0.899**.
>
> Phase 0 spent its entire budget rebuilding, in our own code, an architecture that is free to
> download — reaching 0.7229, with one fold worth +0.0171. **Rule 6 of this README already said
> "the fork is the base, not a reference." It was written and not followed.**
>
> **Every number from here is a delta on top of 0.891.** The next moves are: verify the live
> leaderboard (the figures below are stale), **rank-mean `pilkwang` + `prvsiyan`** — two published
> notebooks, two lineages, a Kaggle run rather than a training job — and submit. The port does not
> get more compute until it can show that **adding** it to that ensemble helps; at ~0.76 it would
> hurt. Full reasoning and the live kernel survey in `IMPROVEMENTS.md` **§2w**.
>
> ### ✅ E1 RAN 2026-08-12 17:22 UTC — and found four errors, all pessimistic `§2x`
>
> It cost minutes and moved the prize cutoff +0.008, the field +40%, and our rank −170. The
> corrected figures are in the table above and in `REFERENCE.md` §3.1; the analysis is §2x.
>
> **The rule stands for next time: a leaderboard number older than a few days is not evidence.
> Re-run E1 before any submission decision.** This is the third belief about the outside world
> this project has had expire in six days.
>
> The measurement apparatus below is not wasted — site-grouped folds, the report-OOF instrument,
> K16, the guards, resume. **It is how you judge an ensemble honestly. It was never going to BE
> the ensemble.**

> ### CLOSED ROUTES — do not re-derive these `index, 2026-08-12`
>
> Each was measured, not argued. The section named is where the evidence lives.
>
> | route | verdict | where |
> |---|---|---|
> | our port as an ensemble member | **no weight helps**, −0.111 at 15.4σ | §2y |
> | K16 from DICOM header rules | 56.9–60.8% vs ~50% chance; resolved by *measurement* instead | §2n, §2m |
> | our rule extractor as a label source | 0/12 labels, negative in a rank-mean | §2f |
> | per-label fusion of public label readers | near-duplicates (|r| 0.87–0.95); loses to best single | §2i |
> | rank-mean `pilkwang` + `prvsiyan` (old E2) | prvsiyan already *contains* pilkwang | §3c |
> | re-fitting blend weights on site-grouped folds | **−0.0000** | §2z |
> | **harmonising scanner/site away (ComBat-style)** | **−0.013 to −0.032** — case mix is real | §3f |
> | selecting anything on gold-58 | claims +0.0137, delivers −0.0034 | §3b |
> | post-hoc calibration / thresholds / priors | AUC is invariant to per-label monotone transforms | §3a |
> | forking prvsiyan for its 0.906 | needs a **private** dataset; degrades to 0.899 | §3c |
> | CoPAS, MRI foundation models, Gold Loss Correction | surveyed and rejected, with reasons | §3a |
> | a C++ port for the efficiency track | 93 s per 0.001 AUC; decode is already native | `PLAN.md` §6.1 |

**Baseline: macro 0.7229 ± 0.0048**, site-grouped report-OOF over 2,612 studies. That is the
honest number; every figure this project produced before 2026-08-10 was inflated by ~0.024 of
site leakage (§2j) and measured on a 37-study instrument that could not resolve 0.04 (§2g).
Leaderboard top is **0.946** (live, §2x); the best *downloadable* public kernel is **0.899**.

> **LB figures on this page are the live 08-12 read (§2x); anything else in the repo dated 08-10
> is stale.** Two known-wrong attributions, corrected: `0.899 let me cook` is `aadigupta7686`, not
> `prvsiyan`, and **no public `Yash Bishnoi` / B3 kernel exists** — searches for `yash`, `b3` and
> `efficientnet` return nothing, so that 0.903 is a *writeup*, and reproducing it is a training job
> rather than a download. The best kernel you can actually download is `aadigupta7686` at 0.899.

> ### The port exists, it trains, and fine-tuning is worth **+0.0171 ± 0.0088 (1.9σ)** `2026-08-11`
>
> Step 4 is built. `data/tiles336` (7.31 GB of uint8 pixels) + `fusion/train_port.py`
> (dinov2-small@336, `UNFREEZE_LAST=6`). Fold 0 trained end to end on site-grouped folds, in our
> own code. **Paired against a freshly re-run frozen-cache control on the 493 studies they
> share: 0.7052 → 0.7223** (§2q). The control reproduces §2j's gold-37 **0.7465** exactly, which
> is what makes it a control.
>
> **Read that as "the mechanism §2e diagnosed is real", not as a score.** One fold, 1.9σ, and the
> per-label sign test is 6/12. Its value is that the pipeline can now test two things the frozen
> cache structurally could not — the anatomical crops (it could not fine-tune) and any
> label-quality change (it could not resolve one).
>
> **The gap is not seven increments.** We sit near LB 0.76 by the §5 conversion; an unmodified
> fork is **0.891 — and that is a 20-member rank-mean of published weights, not a single model**
> (§2e). Reaching ~0.89 is a *download*, and everything between here and there is public and
> chunky: the remaining 18% of the corpus (data ≈ 2× resolution, §2d), ensembling,
> multi-resolution, TTA, gold-in-training. **The 0.04 between the best visible public 0.903 and
> the 0.942 top is what is unexplained by anything public**, because every public team trains on
> the same handful of report-derived label tables and shares their ceiling. Two candidates aim at
> it: the **severity-thresholded label read** (host-confirmed, a few dollars of API, `REFERENCE.md`
> §2.1) and the **anatomical crops** (built, unbuilt tiles, aimed at the mm-scale labels).
>
> **Step 5 is NOT a reproduction gate and cannot be one — `IMPROVEMENTS.md` §2s, 2026-08-11.**
> `pilkwang` publishes no local number at all (`REFERENCE.md` 3.1 records `—`); its 0.891 is a
> leaderboard score from `infer_from_package()`, 20 rank-meaned published weights, no training.
> **There is nothing to land near.** The fold-0 run started 2026-08-11 is the **label-swap arm**
> and is read against a *predicted delta*: it should land **0 to 0.021 below `runs_port`**, that
> being the gold-58 label-quality gap (`steven_v2` 0.8873 vs `pilkwang_v2` 0.866). Read it
> asymmetrically — scoring through `lixin` handicaps the gate arm (0.947 vs 0.866 correlation),
> so a negative delta is ambiguous and only a positive one is clean.
>
> **Do not spend 18 h on folds 1–4 to answer the reproduction question. That run cannot answer
> it.** The target lives on the leaderboard, so step 6 answers step 5 and not the reverse.
>
> ### And the first submission goes NOW, beside step 5 `§2t-1, 2026-08-11`
>
> Not for measurement — for **risk retirement**. `kaggle_03_submit.py` has never executed against
> a real test DICOM: the 9 h cap, no-internet, weights-as-a-Dataset, degenerate series, a study
> missing a plane. Schedule risk, not score risk, and none of it is discoverable locally. It also
> replaces a CV↔LB conversion currently interpolated from **two foreign anchors** while carrying
> the claim that our gap is 0.19 rather than 0.22 — and it yields the single-model LB point that
> tests the boring hypothesis (§2t-5).
>
> **The compute budget, which this plan never had (§2t-3): 72 days, ~40 h/week local.**
> An item that does not name its cost in folds is not a plan item.
>
> **REVISED 2026-08-12 (§2v): a fold is ~1.6 h, not 3.6.** The 3.6 h figure was taken on a laptop
> that was asleep for ~2 of those hours. Held awake with `caffeinate -i`, the same three epochs run
> 10.2 / 9.5 / 9.5 min against 20.3 / 16.4 / 9.9 — identical losses, half the wall clock. So the
> budget is **~40 five-fold experiments, not ~20**. The constraint was never as tight as the plan
> believed, and the reason we thought otherwise is that nobody checked whether the machine was
> awake while it was being measured.

> ### Code review of the step-4 body — 15 findings; the A-cluster is FIXED `2026-08-11`
>
> §2r. Nothing here touches fold 0 or the paired +0.0171: those ran on protocol tiles at depth
> 0.5, the one configuration the medial/lateral findings cannot reach. Three of them (§2r-A1, A3,
> A4) had to close before `--slots anatomical` ran: the promise that a sagittal slab is never
> built without its K16 direction bit was made in three docstrings and in this README, and was
> implemented in none of them.
>
> **ALL THREE ARE FIXED, same day — §2r "As fixed".** One refusal became **four guards**: the bit
> must exist; `SAGITTAL_LR=1` is required at build time; a new cache is checked against every
> manifest in its directory *before the first NIfTI read*; and the K16 refusal is now **per
> series**, so a series with no bit gets `False` in the mask instead of a coin flip. Verified with
> coverage forced to half — `sag_med`/`sag_lat` fell to **13/30 while `sag_pf` stayed at 100%**
> from the same volume, i.e. **34 tiles that would have been coin flips**. Guard 3 caught the live
> hazard in test (`sagittal_lr_slice_flip: True vs False`).
> **The hazard is still live in one sense:** `data/tiles336` remains `SAGITTAL_LR=0`. The guards
> refuse a bad build; they do not rebuild anything. The ~21 min protocol rebuild is still owed.
>
> Two more from that review, **both now fixed too**: `train_port.py`'s `NameError` on the summary
> write when the neutral reference is missing (§2r-B2 — it fired after ~13 h of training, and
> recorded `n_oof` wrong on every run besides), and `score_oof.py`'s paired σ (§2r-B6). On B6 the
> **damage was smaller than reported** — measured over four `PYTHONHASHSEED`s before the fix, the
> delta was **identical at +0.0171 every time** (it is order-invariant by construction) and only
> the SD moved, ±0.0086–0.0091, i.e. 1.9–2.0σ. `score()` was never affected at all, so
> **0.7229 ± 0.0048 was always reproducible**. Post-fix all seeds give `+0.0171 ±0.0088 → 1.9σ`,
> which is what §2q published. See §2t-6.

### The plan, rewritten 2026-08-12 (§2w)

**Phase 0 is closed.** It built a measurement apparatus and a local trainer, both of which we
keep, and a score 0.13 below one we already had. The ordering below replaces it.

| # | step | cost | state |
|---|---|---|---|
| **E1** | **Verify the live leaderboard and both lineages' real scores** | minutes | **NOT RUN — DO THIS FIRST.** Every LB figure in this repo was read **08-10**. The 08-12 survey verified only which *kernels exist*, not what they score (§2w) |
| **E2** | **Rank-mean `pilkwang` + `prvsiyan`** — two published notebooks, two lineages | 1 Kaggle run, no training | the cheapest +0.01–0.02 on the board |
| **E3** | **Submit it** | 1 of 5 daily | also retires the never-executed inference path (§2t-1) and yields the single-model LB point that tests the boring hypothesis (§2t-5) |
| **E4** | **Make the port earn a slot** — does *adding* it to the ensemble help? | 1 scoring run | at ~0.76 it would hurt. **No more compute until it clears this bar** |
| **E5** | Differentiators as deltas *on the ensemble* — severity labels, anatomical crops | ~1.6 h/fold | blocked: the label bet still has no valid instrument (§2s) |
| **E6** | **Efficiency track, co-primary** | TBD | $18,000 over three places, thinner field; `ryanholbrook/…-efficiency-lb` makes it measurable |

**Kept from Phase 0, and still true:** labels are `steven_v2` (§2i-c); the instrument is 6.7×
tighter than gold-37 *at fixed targets* (§2g); folds are site-grouped and our leakage is +0.024
(§2j); the port trains and fine-tuning is worth +0.0171 ± 0.0088 on one fold (§2q); K16 is
resolved by measurement (§2n); the four medial/lateral guards and per-epoch resume exist
(§2r, §2u).

**The label-swap arm** (the step formerly called the reproduction gate) reached epoch 3 of 10
under `caffeinate` and is checkpointed at `fusion/runs_gate/fold0_last.pt`. It is **paused, not
abandoned** — resume costs ~70 min and it is the prerequisite for E4. But E1–E3 come first,
because they decide whether the port is worth finishing at all.

**Step 4 as built.** Five files, all committed and pushed:

| | |
|---|---|
| `pipeline/slot_cache.py` | the uint8 pixel cache. **Built: `data/tiles336`, 7.31 GB, 17,403 tiles over 3,599 studies, 21.3 min.** Slot fill matches FINDINGS §3.2 exactly — axial non-FS at 19.7% against a documented 19.4%. Six anatomical slots defined and **not yet built** |
| `fusion/train_port.py` | dinov2-small@336, `UNFREEZE_LAST=6`, `LR_BACKBONE=8e-6`, `LR_HEAD=1e-3`, a `SlotHead` reconstruction at `SLOT_PRIOR_STRENGTH=0.55`, site-grouped folds. 10.9M of 22.0M params trainable. **Fold 0 took 3.6 h, not the budgeted 2.6 — see §2p, the machine has 17.2 GB and thrashes** |
| **`fusion/score_oof.py`** | **the single scoring definition. If a number is going next to 0.7229 it comes from here.** Pairs two arms on shared studies and bootstraps the delta paired, not unpaired (§2o, §2q) |
| `notebooks/kaggle_01e_direction_measure.py` | K16, **resolved**: 8,048 sagittal series, 50.4% reversed, cross-validated 21/21 against an independent instrument |
| `pipeline/resolve_slice_direction.py` | the K16 gate. Refused the header-rule route at 56.9/60.8/56.9% (§2n); `--measured` is the route that worked |

**Read §2l before touching the crop geometry.** The in-plane axes are canonical per plane
(nearest signed LPS axis unanimous **132/132**, median obliquity 2.4–8.2°), so the geometry
kernel's "374 distinct IOP rows" is float obliquity, not mixed conventions. That, plus
`canonicalise` mirroring onto `'R'`, is what licenses detector-free boxes: **increasing column
index is medial** on axial/coronal, **anterior is low row index** on axial and **low column
index** on sagittal. Confirmed visually on built tiles, not just derived.

**K16 is RESOLVED — but not the way the plan said, and that closes a route it budgeted 3.7 h for
(§2n).** The header-rule route *failed*: three candidate sort keys over all 24,371 series scored
against the 51 the thumbnails settle give InstanceNumber **56.9%**, filename **60.8%**,
SliceLocation **56.9%**, versus ~50% chance. Not a sampling limit — `inst` and `loc` are already
at |rho| = 1.000, so a full header pass exports identical signs. **Do not re-attempt it.**

What worked was measuring the bit instead of inferring it: `kaggle_01e_direction_measure.py` →
`resolve_slice_direction.py --measured` → **`data/slice_direction_resolved.csv`, 8,048 sagittal
series, 50.4% reversed**, cross-validated **21/21** against the 01c thumbnails — a genuinely
independent instrument, and the same 100% bar the header rules failed.

> **HAZARD before building the anatomical slabs.** They need `SAGITTAL_LR=1`; `tiles_protocol`
> was built with `SAGITTAL_LR=0`. Without the flip, "slice 25% is medial" is exactly inverted for
> the 43% of studies that are left knees. `slot_cache.assert_caches_compatible()` **now does
> raise** — **FIXED 2026-08-11 (§2r "As fixed"), after a day in which it was defined and never
> called from anywhere.** `build()` calls it against every manifest already in the output
> directory *before the first NIfTI read*, and it caught this exact hazard in test:
> `sagittal_lr_slice_flip: True vs False`. A second guard refuses any `needs_direction` build
> with `SAGITTAL_LR` off, which the compat check structurally cannot see. **Rebuild protocol
> under the same flags first (~21 min)
> and check the two manifests by hand** until §2r-A3 is closed. Fold 0's result is unaffected:
> protocol tiles sit at depth 0.5, where a reversal maps the middle slice to itself.

**The anatomical slots are designed in and ready to build** — medial, lateral, patellofemoral,
intercondylar notch (`REFERENCE.md` §4.3–4.4). Six of them, as an 84 mm box at the same 336 px,
which is **0.25 mm/px against 0.48** — twice the effective resolution over the compartment where
the failing labels live, for identical compute. That is §2d's diagnosis answered directly.

**One recorded divergence from the fork, for whoever runs the gate.** Its slots are
`SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1` — it separates
fat-suppression from fluid-sensitivity. **That split is not reconstructable from the competition
metadata**: `Fluid_Sensitive` and `Fat_Suppression` are byte-identical over all 24,371 series, so
those columns yield exactly 3 planes × 2 and nothing finer. If the gate misses, recovering the
finer split from `SeriesDescription` / `EchoTime` — already held for every series in
`data/external/dicom_headers_zhukovoleksiy.parquet` — is the first thing to try.

**Before running anything on a fresh clone:** `data/` is gitignored — see "Regenerating the
derived data" under Setup. ~2 minutes, no GPU.

**Two open questions worth one forum post each:** whether MRNet is admissible (asked twice, host
answered only the LLM half — `REFERENCE.md` §1.3), and how many studies are bilateral (§2k).

---

## Read these in order

| doc | what it holds |
|---|---|
| **`PLAN.md`** | strategy, architecture, timeline, efficiency-track maths |
| **`FINDINGS.md`** | measured facts about the data — label coverage, languages, series structure |
| **`IMPROVEMENTS.md`** | **running friction log.** Open decisions, ranked weaknesses, resolved-bug provenance. Read before touching the extractor. |
| **`REFERENCE.md`** | **external ground truth — host rulings, the official label criteria, forum facts checked against ours, literature.** New 2026-08-10; four retracted beliefs are why it exists |
| `COMPETITION_RULES.txt` | the competition rules, verbatim |
| `labeling/README.md` | hand-labelling workflow |

## The five facts that shape everything

1. **58 of 4,407 training studies carry labels (1.3%).** The rest must be derived from the
   free-text radiology reports.
2. **Reports span 9 languages, 61% non-English** (English, Spanish, Turkish, Croatian, Greek,
   German, Bulgarian, Dutch, French). English-only clinical NLP is useless here.
3. **Gold labels are not a function of the report text** — they look like independent expert
   image reads. Careful full-text reading agrees with gold on 84.7% of labels, but ~2/3 of
   that gap is one-directional threshold error and is recoverable. See `IMPROVEMENTS.md` §2b.
4. **The leaderboard hit 0.932 within 48 hours of opening, on forks of public DINOv2
   notebooks** — and weak labels for all 12 findings are public too. So 0.9 is table stakes,
   the backbone cannot differentiate us, and "the extractor is the solution" is a claim to be
   earned rather than assumed. Measured in `PLAN.md` §7.1. Live 2026-08-12 (§2x) the top is
   **0.946**, and an unmodified public fork scores **0.891 at rank 400/1,276** — so the entire
   spread from mid-table to first is about **0.05 AUC**, and "0.9 is table stakes" is now
   literal: 326 teams are at or above 0.899. The claim in the last sentence of point 4 was tested on 2026-08-10 and
   **failed**: see `IMPROVEMENTS.md` §2f.
5. **Local scores and leaderboard scores are not the same scale, and the conversion is
   known.** Two public anchors give it: a baseline notebook reports OOF 0.632 → LB 0.664, and
   the highest visible solution reports OOF 0.8544 / cross-fitted gold-58 0.8568 → LB 0.903.
   Interpolating, **our 0.719 gold-37 is worth roughly LB 0.75.** So the gap to the 0.942 top
   is about **0.19**, not the 0.22 the raw numbers suggest — real, large, and not a measurement
   artefact. Equally: that solution's report-holdout OOF and its gold-58 agree to **0.002**,
   which is the proof that a *local* instrument can work here (see "Where this goes next" §0).

## Does our extractor actually beat the free one?

> ## **NO. RETRACTED 2026-08-10 — the moat is inverted.**
>
> Measured against what is *currently* free rather than against week-one's free, on the same 58
> gold studies, reproducible with `python extractor/bench_public_labels.py --download`:
>
> | label source | macro AUROC | SE |
> |---|---:|---:|
> | `stevenleehans/llm_labels_v4_blend` | **0.893** | 0.015 |
> | `stevenleehans/llm_labels_full` | 0.878 | 0.016 |
> | `pilkwang/report_labels_v2` | 0.866 | 0.016 |
> | `lixin73/labels_llm_gpt56sol` | 0.835 | 0.018 |
> | `pilkwang/report_labels_v1` | 0.813 | 0.019 |
> | **ours (rules)** | **0.777** | 0.021 |
>
> **We are last of six, by +0.116 — about 4.5× the combined SE, and the only extractor number
> in this project that clears its own noise floor.** Per label: **0/12**. And ours is not
> additive — rank-mean of the top two public readers is 0.890, and adding ours takes it to
> 0.887. Full result and the post-mortem in `IMPROVEMENTS.md` §2f.
>
> The comparison below is not wrong, it is *stale*: `nekkon`'s CSV is a binary rule set from
> week one, and §5 flagged the comparison as answering a question the field had moved past on
> 2026-08-09. It stayed in this position, phrased as a settled decision gate, for a day longer
> than the evidence supported. **The lesson is the one this file keeps re-learning: a
> comparison is only as current as its baseline, and "we measured that" has an expiry date.**

The original week-2 decision gate (`PLAN.md` §7), reproducible with
`python extractor/compare_methods.py`:

| label source | vs gold (n=58) | vs hand labels (n=83) |
|---|---|---|
| **rules (ours)** | **0.777 AUC / 0.749 bal-acc** | **0.864 / 0.862** |
| public (`nekkon`) | 0.672 / 0.672 | 0.757 / 0.757 |

Both metrics are reported because the public set is binary: with a single operating point its
AUC collapses to balanced accuracy, so an AUC-only table would flatter our graded targets for
free. The gap survives both metrics and both references.

**The two references are not independent, and that had to be checked.** The hand labels are
read from the *reports* — same modality the extractor works from, same person, and the
labelling UI highlights terms from the same `glossary.json` that drives the rules. So the
right question is whether switching reference gold→hand preferentially flatters *us*:

| source | vs gold | vs hand | lift |
|---|---:|---:|---:|
| rules | 0.775 | 0.864 | **+0.089** |
| public | 0.672 | 0.757 | **+0.085** |

Near-identical. Both sources are report-derived and both gain the same amount from a
report-derived reference, so the gap is not an artefact of shared vocabulary. *(Measured on
the pre-§2.2 extractor, which is the version the concern was raised against.)*

The table also re-prioritises `IMPROVEMENTS.md` by measurement. Hand-labelled reading scores
**0.838** on the gold studies it covers, so the headroom is ~0.06 — and it is concentrated:

- **Medial OA 0.720 → 0.957 by hand.** Compartment attribution (§2.2) was worth more than
  every other extractor fix combined; the first pass of it is now done and took Medial OA
  bal-acc 0.720 → 0.787.
- **Synovitis 0.621 rules / 0.639 public / 0.652 by hand.** Careful human reading barely beats
  a regex, so this is the report-only ceiling, not a bug — §2.1 is overweighted. Let the vision
  model learn it off the Effusion correlation instead.

## Layout

```
data/                  competition CSVs + derived label tables      (gitignored)
extractor/             report -> 12-label extraction, method A and method B
  rule_extractor.py      the rule extractor (clause-scoped, 9 languages). stdlib only
  run_extract.py         run over corpus + evaluate on gold. NB: works at import time
  llm_extract.py         method B via the Claude Batch API. --dry-run needs no key
  compare_methods.py     every label source scored head-to-head. the ORIGINAL decision gate,
                         stale: its baseline is nekkon's week-one binary CSV
  bench_public_labels.py the same gate re-pointed at the four public LLM readers. THIS is the
                         live comparison, and we lose it 0/12. IMPROVEMENTS.md 2f
  metrics.py             auc / bal_acc, shared. separate module so importing it is safe
  diagnose.py            bootstrap CIs + targeted failure diagnostics
  verify_claim.py        stress-tests the "labels aren't report-derived" claim
  calibrate_states.py    fits the soft-target ladder to P(gold=1|state). IMPROVEMENTS.md 1.3a.
                         The fit LOST (0.743 -> 0.699); kept as a measurement, not a setting
  to_fork_table.py       our labels -> the 25-column table pilkwang's notebook reads.
                         The __conf column is NEW and a guess -- see its docstring
labeling/              hand-labelling workflow
  glossary.json          multilingual terms for 12 findings + negation/uncertainty/severity
  compartment_patch.py   adds _side_* / _compartment_struct. IMPROVEMENTS.md 2.2
  build_labeler.py       generates the offline labelling UI
  model_labels.csv       labels produced so far (86/303, all 30 blind gold done)
  item_id_map.csv        item_id -> StudyInstanceUID. the durable fingerprint, no report text
  rekey_labels.py        binds labels to StudyInstanceUID; --check catches sample drift
  eval_model_labels.py   vs gold and vs rule extractor
  error_direction.py     splits disagreement into calibration vs true divergence
pipeline/              shared by BOTH machines -- the one definition of preprocessing
  preprocess.py          DICOM *and NIfTI* -> model input. Fingerprinted; see "Preprocessing parity"
  build_cache_local.py   the FROZEN-EMBEDDING cache, M5 from NIfTI. 1,062 studies/h @224.
                         Provenance only -- a cached embedding cannot fine-tune (IMPROVEMENTS 2e)
  bench_port.py          times the training port for real: 28.5 img/s, 2.6 h/fold. IMPROVEMENTS 2h
  bench_cache_build.py   times the 336 slot-pixel cache: ~16 min for the corpus. IMPROVEMENTS 2h
  validate_nifti.py      5 checks that the NIfTI repackaging matches the DICOMs. All pass
fusion/                the differentiator (PLAN.md 3.3). Trains on the M5, MPS
  model.py               slice transformer -> attention pool -> series attention -> 12 logits
  dataset.py             cached features -> padded batches. `python fusion/dataset.py` self-tests
  folds.py               5-fold, grouped by patient-proxy. NB: there is no patient column
  train.py               training loop + pooled-OOF gold eval. --synthetic needs no cache.
                         Also writes oof_all.csv -- every study, which is the real instrument
  instrument_test.py     proves oof_all is 6.7x tighter than gold-37, and finds the one thing
                         it CANNOT arbitrate: label sources. IMPROVEMENTS.md 2g
notebooks/             Kaggle-side only; the 570 GB of pixels never come local
  kaggle_01_dicom_audit.py    did (0020,0060) Laterality survive? + decode cost per syntax
  kaggle_01b_patients_laterality.py  header-only pass over all 4,407: PatientID + laterality
  kaggle_01c_series_geometry.py  per-series geometry + thumbnails so the NIfTI conversion can
                              be validated locally. CPU-only, no GPU lottery, ~5 min
  kaggle_02_dinov2_cache.py   frozen DINOv2 -> ~2.4 GB of per-slice features, shardable.
                              `--self-test` runs the whole scheduler locally, no DICOMs needed
  kaggle_03_submit.py         test DICOM -> submission.csv. No internet; weights from a Dataset
eda_01_labels.py       label coverage, prevalence, co-occurrence
eda_02_langs.py        language identification + series structure
eda_03_langid.py       lingua-based language detection (supersedes the heuristic)
eda_04_metadata_baseline.py   series metadata alone scores 0.471 on gold. "do not retest" was
                       written from n=58 and is retracted -- a public probe gets 0.5954 from the
                       same four columns over 4,407 (IMPROVEMENTS 2i-b). Still not a shortcut
```

## Setup

```bash
python -m pip install -r requirements.txt
# auth: new-style KGAT_ token at ~/.kaggle/access_token (legacy kaggle.json no longer works)

# all four CSVs are needed -- eda_01 reads test.csv/test_series.csv too
for f in train.csv train_series.csv test.csv test_series.csv; do
  python -m kaggle competitions download -c rsna-knee-abnormality-detection -f "$f" -p data
done

python eda_01_labels.py            # writes data/lang_guess.csv   <- eda_03 needs this
python eda_03_langid.py            # writes data/lang_detected.csv
python extractor/run_extract.py    # writes data/pseudo_labels.csv  (extractor RETIRED as a
                                   # target source -- IMPROVEMENTS 2f. Still needed for
                                   # extract_states.csv, which 2k flags as unexploited)
```

### Regenerating the derived data `data/` is gitignored, so this is the only record

Competition data is licensed and does not belong in git, which means **none of the tables the
current plan depends on survive a fresh clone.** Rebuild them in this order — total ~2 minutes,
no GPU:

```bash
# 1. targets: the public LLM labels, and the benchmark that chose them (IMPROVEMENTS 2f, 2i-c)
python extractor/bench_public_labels.py --download     # -> data/public_llm_labels/
python -c "
import pandas as pd
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
   'Effusion','Synovitis',\"Baker's\",'Contusion','Fracture']
src='data/public_llm_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv'
tr=pd.read_csv('data/train.csv')
pd.read_csv(src).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID') \
  .reindex(tr.StudyInstanceUID)[L].reset_index().to_csv('data/targets.csv', index=False)
"

# 2. scanner fingerprints -- NO Kaggle run needed, the probe author published the headers
mkdir -p data/external
kaggle kernels output zhukovoleksiy/rsna-metadata-probe -p data/external
mv data/external/headers.parquet data/external/dicom_headers_zhukovoleksiy.parquet
python pipeline/site_fingerprint.py    # expect 265 fingerprints, top 20 = 45.5%

# 3. both fold sets. Build BOTH: the gap between them is the site-leakage number (2j)
python fusion/folds.py --out data/folds.csv
python fusion/folds.py --group-by site --out data/folds_site.csv
```

`pipeline/site_fingerprint.py` prints a warning if the fingerprint count leaves 200–340. **Take
it seriously** — §2j records a version of that file which silently put 24% of the corpus in one
group while reporting a healthy count.

**Run `eda_01` before `eda_03`.** `eda_03_langid.py` reads `data/lang_guess.csv` to diff the
lingua result against the old heuristic, and `eda_01_labels.py` is the only thing that writes
it. Skipping straight to `eda_03` fails with `FileNotFoundError`.

**On a fresh clone, re-key the hand labels before trusting any evaluation** —
`labeling/model_labels.csv` joins on `item_id`, which is assigned *positionally* after a
shuffle (`sample_for_labeling.py:70-71`) and is not bound to `StudyInstanceUID`. If `lingua`
resolves to a different version, the language strata shift, the shuffle lands differently, and
those 86 labels silently describe the wrong studies. Hence the exact pin in `requirements.txt`.

```bash
python labeling/sample_for_labeling.py     # regenerates labeling_sample.csv
python labeling/rekey_labels.py --check    # fails loudly if the mapping drifted
```

Whole text pipeline runs in **6 s** on CPU (measured on an M5, 2026-08-07).

Then compare every label source, which needs no GPU either:

```bash
python extractor/compare_methods.py --fetch   # pulls the public weak labels into data/
python extractor/compare_methods.py           # the table above
```

## Kaggle-side work

**Read the competition's code, never its description.** Every finding on 2026-08-10 came from
pulling the notebook itself; three claims in "what is wrong" §6 had been written from the public
description and two of them were wrong — including "it ensembles two backbones", and the belief
that 0.891 was a *training* score when `find_weights()` makes it inference from 20 published
members in 74 s.

```bash
python -m kaggle kernels pull raahimnawaz/rsna-knee-lb-baseline-fork -p /tmp/fork -m
```

Re-pull it rather than trusting any summary in these documents, including this one.


The images are 570 GB and stay on Kaggle. The scripts run there in this order — the audits come
first, because if laterality were unrecoverable then four of the twelve labels would be
unreliable and that changes what the feature cache is worth building against. **Both audits are
done** (`FINDINGS.md` §6); the cache is not.

```
notebooks/kaggle_01_dicom_audit.py           -> /kaggle/working/dicom_audit.json        [done]
notebooks/kaggle_01b_patients_laterality.py  -> /kaggle/working/laterality_check.json   [done]
notebooks/kaggle_02_dinov2_cache.py          -> /kaggle/working/features  (publish as a Dataset)
```

**A bad Kaggle GPU draw is the single biggest way to lose a session here.** Kaggle assigns a
Tesla P100 on roughly four of five draws and `accelerator` in `kernel-metadata.json` does not
reliably override it; its PyTorch dropped Pascal, so a P100 cannot run this at all — while
`torch.cuda.is_available()` returns True on one. `pick_device()` checks compute capability
against the installed wheel's arch list in the first seconds of `main()` and refuses, so a bad
draw costs seconds and re-running is a viable strategy rather than a way to burn the weekly
quota. Run `python pipeline/preprocess.py` for the fingerprint; the guard prints the GPU it drew.

**Watch the PROBE line.** `kaggle_02` prints measured throughput at 25 / 100 / 400 series with
the implied hours for the shard, and warns when the rate is near single-worker. Three earlier
attempts burned 21 h, then two ~1 h sessions, then 9 h before anyone could tell the pool had
stopped parallelising. If that warning fires, kill the session.

The cache is the reason local work is **no longer text-only**. Frozen DINOv2 embeddings are
~2.4 GB for the whole corpus, so the §3.3 fusion head — slice transformer, attention pool,
series-type embedding, series attention — trains on a laptop in minutes per experiment. The
backbone is the same one everyone else has; the fusion layer, the labels, and external data are
where the edge has to come from.

`kaggle_02` shards via `SHARD` / `N_SHARDS` and skips finished studies on restart, because
decode is the bottleneck and this will not complete in one session.

## Training (the M5 is enough)

Two machines, one pipeline. Pixels never leave Kaggle; the laptop only ever sees vectors.

```
Kaggle   kaggle_01 audit  ->  kaggle_02 cache  ->  publish Dataset   [~2.4 GB]
M5       download  ->  fusion/folds.py  ->  fusion/train.py          [MPS, minutes/experiment]
Kaggle   upload fold*.pt  ->  kaggle_03_submit.py  ->  submission.csv
```

```bash
python fusion/dataset.py                              # self-test: shapes, masks, degenerate studies
python notebooks/kaggle_02_dinov2_cache.py --self-test # scheduler, resume, spawn pool. No DICOMs
python fusion/train.py --synthetic                    # whole loop on random features. No cache
python fusion/folds.py                                # writes data/folds.csv
python fusion/train.py --features data/features
python fusion/train.py --features data/features --limit 500   # fast iteration; keeps all gold
```

### Building the cache locally (the route that actually worked)

```bash
# 1. pixels: one NIfTI per series, ~178 GB zipped / ~386 GB extracted. Delete each zip as you go
python -m kaggle datasets download -d davidadekanmi/rsna-knee-nifti-part1 -p data/nifti --unzip

# 2. prove the repackaging matches the DICOMs BEFORE building anything (5 checks)
python pipeline/validate_nifti.py --geometry data/series_geometry.csv --thumbs data/series_thumbs.npz

# 3. build. IMG_SIZE feeds the fingerprint, so 224 and 518 caches cannot be confused
IMG_SIZE=224 caffeinate -dimsu python pipeline/build_cache_local.py --validated --out data/features_224
caffeinate -dimsu python pipeline/build_cache_local.py --validated --out data/features
```

Measured on a 16 GB M5: **1,062 studies/h at 224** (~4.2 h for the corpus), ~26 h at 518. The
`--validated` flag is a deliberate speed bump — without it the builder warns on every run that
checks 4/4b have not been re-earned for the current corpus revision.

**The NIfTI mirror is incomplete.** Parts 1–12 and 16–18 exist; 13–15 return 403. That is
**81.7% of studies** and **47 of 58 gold**. Re-check periodically — parts have appeared twice.

`kaggle_02 --self-test` is the one that guards a Kaggle session rather than a laptop run. It
drives the whole scheduling loop on synthetic series — prefetch window, per-study completion
accounting, the resume path, the throughput probe — and it constructs the **real spawn pool** and
asserts it fans out across workers, pins each to one thread, and does not re-search the mount on
re-import. Every one of those has cost a session at least once.

**`--synthetic` is not a toy.** It emits the exact shapes, dtypes, masks and edge cases the real
cache produces — single-series studies, unknown series types, minimum slice counts — so every
consumer is testable before a DICOM has been decoded. On random features it scores macro **0.518**
on the 58 gold studies. That is the point: chance is the correct answer, and anything meaningfully
above it would mean a leak.

**The M5 laptop is sufficient — no Studio, no 980 Ti.** The fusion head is 3.7M parameters over
cached vectors, so the binding constraint is RAM for the cache, not the accelerator. Measured on a
16 GB M5: **2.18 GB peak** with the full cache, the model and optimizer steps all live. The 980 Ti's
6 GB would fit too, but Maxwell has no tensor cores and PyTorch has deprecated sm_52, and it cannot
help with the one genuinely heavy step — the frozen backbone — because that runs on Kaggle either
way. Everything here is fp32 on purpose: 3.7M parameters gain nothing from mixed precision.

**Do not containerise the training.** Docker Desktop on macOS has no Metal passthrough — containers
run in a Linux VM and there is no Apple-silicon `--gpus`. A dockerised run trains on CPU. torch MPS
needs native macOS. Reproducibility comes from the pins in `requirements.txt` plus the preprocessing
fingerprint below.

### Preprocessing parity — the failure with no symptom

`pipeline/preprocess.py` is the single definition of DICOM → model input, imported by the cache
builder *and* the submission notebook. If those two ever disagree, the model is fed a distribution
it never trained on and simply scores badly — no traceback, no warning, and it would surface only
as an unexplained CV/LB gap. So every constant that changes a feature value is hashed into
`PREPROCESS_VERSION`, written into the cache manifest, and asserted by `kaggle_03` before it reads
a single study.

The manifest has to *travel with the weights* for that assert to mean anything: `kaggle_02` writes
`_shard*.json` beside the feature cache, `fusion/train.py` copies it to `manifest.json` beside
`fold*.pt`, and `kaggle_03` refuses to run at all if it cannot find one. A submission that cannot
prove which preprocessing built its heads is the exact failure this section is about, so it exits
rather than warning.

```bash
python pipeline/preprocess.py    # prints the manifest, including the fingerprint
```

Change `IMG_SIZE`, `TARGET_MM`, `FOV_MM`, `SLICES_PER_SERIES`, the model, or the canonical side, and
the fingerprint changes and the existing cache is invalid. That is the intended behaviour.

### What the folds can and cannot do

`fusion/folds.py` groups on a hash of the report text, because **`train.csv` has no patient
column** — the only same-patient signal available is that 150 studies share a report with another
(bilateral knees or follow-ups). It is a floor, not a solution: two studies on one patient with
genuinely different reports still leak and we cannot detect it.

Gold is scored **pooled out-of-fold, never per fold**. 58 gold studies over 5 folds is 8–16 each,
and MCL lands at zero positives in two of them.

## Where this stands, and what is wrong — 2026-08-09

The pipeline works end to end and produces **macro AUC 0.719** (37 gold studies, images only,
224px, 60% of the corpus, no tuning). Defects are listed **by cost**, and every claim here is
measured.

> The headline was `0.743` until 2026-08-09. That run's artefacts were destroyed by K17 and the
> `_nifti_axes` fix then deleted 43 mis-reformatted cache entries; `0.719` is the same
> configuration on the corrected cache. Two identical re-runs agree to three decimals on every
> label, so **training is deterministic** and A/B differences here are real. See
> `IMPROVEMENTS.md` §1.3a.

### 1. Resolution — the 224 cache throws away half the detail, and it is the biggest lever

`normalise_and_resample` resamples to `TARGET_MM` = 0.35 mm/px and centre-fits to
`round(FOV_MM/TARGET_MM)` = **457 px**, the full 160 mm knee. Then `imagenet_normalise`
interpolates that straight down to `IMG_SIZE`:

```
224   0.714 mm/px over the 160 mm FOV    261 tokens/slice   <-- the 0.35 resample is discarded
518   0.309 mm/px  ~= as designed      1,374 tokens/slice
```

**The labels this costs are exactly the ones failing.** Against the extractor's own per-label
AUC as a control, vision *beats* text on six labels (Medial OA +0.198, Effusion +0.120) and
collapses on six (Lateral Meniscus −0.332, Fracture −0.274, ACL −0.206). Spearman between the
two columns is **−0.17 (p=0.60)** — label quality does not predict vision performance, which
rules out the pseudo-labels as the ceiling. What the split tracks is the physical size of the
finding: centimetre-scale fluid and bone morphology succeed, millimetre-scale tear and fracture
lines fail. At 0.71 mm/px those lines are about one pixel. Full table and method in
`IMPROVEMENTS.md` §2d.

This outranks the two ordering defects below: they target the four medial/lateral labels,
resolution targets six including the two at chance. Since §2 and §3 force a rebuild anyway,
**rebuild at 518 rather than 224** and both land in one job.

### 2. Slice direction is mixed — this blocks a valid submission

`validate_nifti.py` check 4b, stratified across all six series types (n=51):

```
forward  66.7%      Axial     12/12 forward
reversed 33.3%      Coronal   14/18 forward
                    Sagittal   8/21 forward   <-- majority REVERSED
```

The NIfTI affine carries no direction cosines, so nothing in the file distinguishes the two, and
`load_series_nifti` does not flip. `load_series` — which `kaggle_03` uses at test time — always
sorts ascending by `ImagePositionPatient` projection. So roughly a third of cached series run
opposite to the test path, in the axis medial/lateral discrimination depends on, and
`PREPROCESS_VERSION` cannot see it because the conversion happens upstream of the fingerprint.

Not predictable from plane: a plane rule scores ~72%. It must be resolved per series.

**This depresses 0.719 rather than inflating it.** The first verdict said "100% forward" because
every thumbnail in that sample was `Axial_0` — see §8.

### 3. Sagittal handedness was never corrected — K18

`canonicalise()` mirrors left knees onto the right, but only **in-plane**, and only for axial and
coronal. For sagittal, medial/lateral is the **slice axis**, and `vol[:, :, ::-1]` is the only
reversal in the pipeline — nothing ever touched axis 0. The docstring's argument that a sagittal
flip "would mirror the knee front-to-back for no gain" is true of the in-plane axis and never
considered the other one.

`spatial_order` sorts ascending along the slice normal; for sagittal that normal is the patient's
left-right axis, on which medial is +x for a right knee and −x for a left one. So one sort gives
lateral→medial for one knee and medial→lateral for the other, and `FusionHead.slice_pos` is a
**learned per-index** embedding — slice 5 meant lateral in one study and medial in the next.

Exposure: sagittal is the largest plane at **9,864 of 24,371 series (40.5%)**, and **1,894 of
4,407 studies (43.0%)** are left knees. Four of twelve labels are medial/lateral pairs.

Code is written and **gated off** behind `SAGITTAL_LR_SLICE_FLIP` — one switch for both readers,
because a per-call decision would canonicalise the test set and not the training cache. It is an
XOR against §2's direction bit, so the two corrections compose; it cannot be turned on until that
bit exists. Off is a byte-level no-op and the fingerprint is unchanged.

### 4. The leaderboard anchor exists — and it is not ours

| | score | note |
|---|---:|---|
| LB top | ~~0.940~~ **0.946** | ~~908~~ **1,276** teams (live 08-12, §2x) |
| ours, submitted 2026-08-09 | **0.891** | unmodified fork of `pilkwang/rsna-knee-baseline-v1`; rank ~~230/908~~ **400/1,276** |
| our own pipeline | — | never submitted |

So Phase 1's first half is done and the reference implementation reproduces. **The whole spread
from rank 400 to rank 1 is ~0.055 AUC**, which sets the scale for everything below — and note
that the spread is that tight *because* the public field is stacked in a 0.891–0.900 band (§2x).

What is still unmeasured is the mapping from *our* CV to the LB. `0.719` is pooled-OOF on 37
**enriched** gold studies — §1.1 of `FINDINGS.md` records the 58 as "clearly curated to cover
every finding", positive rates 16–60% — while the public LB is ~390 natural-prevalence studies.
Ours is the harder instrument: on an enriched set the negatives for ACL are frequently knees
positive for meniscus or OA, so the model must separate pathology from pathology rather than from
healthy. AUC is insensitive to the prevalence *ratio* (`PLAN.md` §8) but not to how hard the
negatives are. **0.719 and 0.891 are not on the same scale.**

In its current state — 224 px, 60% of the corpus, a third of series back-to-front — our own
pipeline would very likely score *below* the 0.891 already banked. Submitting it is therefore not
a score move. It is still worth one run, because `kaggle_03_submit.py` has **never executed
against a real test DICOM** and every untested path in this repo has held defects. Treat the
first own-pipeline submission as a dry run of the inference path that returns the CV↔LB mapping
as a side effect.

### 5. The moat comparison is stale — `RESOLVED 2026-08-10, against us`

> Re-pointed and measured: **ours 0.777, best public 0.893, 0/12 labels won, negative
> contribution to a rank-mean.** `IMPROVEMENTS.md` §2f. The section below is what was known
> before that measurement; the "re-point it" instruction has now been carried out and the
> answer was that the extractor is not the moat and never becomes one.

`0.777 vs 0.672` was measured against `nekkon`'s published label CSV. The canonical public
notebook today (`pilkwang/rsna-knee-baseline-v1`, 251 votes) ships its own extractor: clause
scoped, multilingual, emitting `(score, confidence)` pairs, negation scoped by punctuation,
laterality by tag with a geometry fallback, slices sorted by IPP, `Anatomical_Plane` used, flip
augmentation refused for the same medial/lateral reason as ours — and it independently found the
Greek **MICRO SIGN U+00B5** issue that `FINDINGS.md` §2.2 treats as a distinguishing discovery.

The §7.2 A/B therefore answers a question the field has moved past. Re-point it at that
extractor — which we have now forked and run: **0.891**, see §4.

### 6. The architecture cannot fine-tune — this is the gap

Read from `pilkwang`'s notebook itself rather than its description:

```python
UNFREEZE_LAST = 6      # trainable transformer blocks, from the output end
LR_BACKBONE   = 8e-6   # the encoder is adapted, not retrained
```

**It fine-tunes the last six encoder blocks. Caching frozen embeddings makes that impossible for
us** — at any resolution, under any head, with any labels. That is why resolution bought +0.013
and why everything lands near 0.70 whatever changes downstream. Full analysis in
`IMPROVEMENTS.md` §2e.

Two claims this section previously made were wrong, from reading the description instead
of the code: it uses **one** backbone (not DINOv2 + EfficientNet) and it is DINOv2 **small** — not the base
checkpoint we run. The 224/336 claim was right and my denial of it was wrong: `RUNS` carries
both, and `CACHE_IMG` is the cache size rather than the only run. **And as attached it does not
train at all** — `find_weights()` short-circuits `main()` into inference from 20 published
members in 74 s, so 0.891 is a published-weights score. It also already ships per-diagnosis slot
attention, confidence-weighted targets and rank-mean TTA — three things listed as our
differentiators.

**The compute is small and the advantage is ours.** `N_SLOT = 6`, `GROUP = 3`: a study is six
encoder inputs, not the ~155 slices we embed. On the M5 that is ~12 min/epoch and ~2 h for a full
`EPOCHS = 10` run, against Kaggle's 8 h budget, 30 h weekly quota and a GPU lottery that refuses
four draws in five. **A 10–20x iteration advantage on the architecture that works** — that, not
518, is the asymmetry worth having. Its pixel cache is ~9 GB against 458 GB of NIfTI.

### 7. The corpus — 81.7% available, 60% actually cached

The NIfTI mirror has parts 1–12 and 16–18; **13–15 return 403**. That is 3,599/4,407 studies and
47/58 gold. Re-check periodically — parts appeared twice on 2026-08-09.

Separately from availability, the built cache currently holds **2,649 of 4,407 studies (60%)**,
which is why the pooled OOF covers 37 of the 58 gold rather than 47. Finishing the build against
what is already downloaded is free data before any new part appears.

### 8. Defects found by review, now closed

`kaggle_03` still carried a fourth hand-written copy of the embedding loop — on the *test* side,
under a docstring in `preprocess.embed` asserting that copies had been consolidated so train and
test could not drift. Migrated, and verified bit-identical. And `IMPROVEMENTS.md` §1.3 stated the
shrink picked `m = 20` when the cross-fitted folds actually run 20/50/20/50/50.

**K17 — a `--synthetic` smoke run overwrote the real result directory, and the guard passed it.**
`fusion/train.py --out` defaulted to `fusion/runs` regardless of `--synthetic`, so the smoke
command in its own docstring destroyed the 0.743 checkpoints, OOF and summary; the directory is
gitignored, so nothing survived. Worse, synthetic mode wrote **no** manifest, leaving the genuine
one in place to vouch for random-tensor weights — `assert_matches()` reads only
`preprocess_version` and would have passed a submission of pure noise. Now: `--out` resolves to
`fusion/runs_synthetic` under `--synthetic`, synthetic runs write a self-marking manifest, and
`assert_matches()` refuses it before the version check.

### 9. The failure mode this project keeps repeating

> **A second species, added 2026-08-10 — and it is the more expensive one.** K13–K18 are claims
> about the *data* that were never measured. §9.1, §2e and §2f are claims about the *world
> outside this repo* — what is possible, what is required, what is already free — that were
> never re-measured after the first time. "Local work is text-only" (retired 2026-08-09), "the
> extractor caps the vision model" (retired 2026-08-10), "the moat is real" (retired
> 2026-08-10). Each was true or plausible when written, load-bearing for weeks, and cheap to
> check. **The failure is not believing them; it is that a claim about a moving field was
> recorded once and then treated as a constant.** Rule 5 below is the fix, and rule 4 of "Where
> this goes next" is the operational form of it.

K13, K14, K15, K16, K17, K18 and the defects found on 2026-08-09 are one species: **a claim about the
data written as reasoning and never measured**, guarded by a self-test that shares the same
assumption. The clearest case is a docstring that justified picking the slice axis by voxel
spacing as "~10x separation, unambiguous"; measured across all 19,859 series the minimum ratio is
**1.005** and 84 series are under 1.5x, which silently reformatted 27 axial acquisitions into
sagittal ones.

Four rules follow, and they are cheap:

1. No claim about the data in a docstring without the measurement that produced it.
2. Every self-test must build the **real** backbone at least once. K14 and the `no_grad`
   regression both hid behind an injected `embed_fn`.
3. Any validation that samples must print its **coverage per stratum**. The Axial-only verdict
   passed three documents unchallenged because nothing printed the breakdown.
4. When a correction is axis-dependent, enumerate **every** axis before concluding. K18's
   docstring reasoned correctly about the in-plane axis and never mentioned the slice axis, so
   half the fix read as the whole of it for three months.
5. **A claim about the outside world carries an expiry date.** Any statement of the form "the
   public X is worse than ours" / "X is not possible here" / "X requires Y" gets re-measured
   before it is allowed to justify a week of work. Cost so far: four Kaggle sessions (§9.1) and
   five days of extractor work (§2f).
6. **A cost estimate is not a feasibility check.** `PLAN.md` §9 Phase 0 step 2 priced the K16
   export at two levels of detail — "2–3 header reads per series (~50k opens, ~20 min)" with
   "~700k opens, 3.7 h" as the fallback — and even flagged that the per-open latency underneath
   both was itself unmeasured. It never asked the prior question: *is the bit in the headers at
   all?* It is not (§2n). Both numbers were correct and both routes were dead. The check that
   settles it — join three candidate rules against the 51 series whose answer is already known —
   is a few seconds of pandas and existed the whole time. **When a plan states an estimate to
   two significant figures, that is the moment to ask what it is assuming, because precision
   reads as diligence and hides the assumption underneath it.**

> **A third species, added 2026-08-10 — the artifact that was already paid for.** §2m: the
> stratified thumbnails that correct K16's first verdict were generated by *our own* kernel on
> 08-09 at 22:39, and only the CSV from that run was ever downloaded. `validate_nifti` spent a
> day reporting the pre-fix "100% forward, 1/6 types" answer the docs already recorded as false,
> with no symptom other than a stale caveat nobody re-read. **When a fix is applied to a Kaggle
> script, the download is part of the fix** — a re-run whose output is never pulled leaves the
> local instrument answering the old question.

## Where this goes next

> **REWRITTEN 2026-08-10 (second time that day), after `extractor/bench_public_labels.py`
> returned 0.777 vs 0.893.** The previous version of this section is preserved below under
> "Superseded plans". It was built on one diagnosis — *the architecture cannot fine-tune* —
> which is correct but was treated as the only one. §2f showed a second constraint of similar
> size that had been sitting in plain sight since 2026-08-07, and showed why the test that was
> supposed to detect it could not.
>
> This section has now been reordered four times in three days. That churn is real and §0 is
> still the diagnosis of it, but the fix has changed: the previous version's answer was to stop
> measuring locally and let the leaderboard decide. That was wrong too, and expensively so —
> see §0.
>
> **The churn is over, and the ledger below is why.** Four things were measured on 2026-08-10
> and three of them cancelled work rather than adding any. Every row now carries the
> measurement that put it there, so a future reorder has to beat evidence rather than
> reinterpret it.

### The ledger — everything, with the evidence that settled it

**Works — keep, do not re-litigate.**

| | evidence |
|---|---|
| Report-derived OOF as the instrument | ±0.0046 over 2,612 vs gold-37's ±0.031 — **6.7× tighter** (§2g) |
| Site-grouped folds | our own leakage measured at **+0.024**, ~5σ; baseline is now **0.7229** (§2j) |
| Public LLM labels as targets | `steven_v2` **0.887** vs ours 0.777 on gold-58, 4.5σ (§2f, §2i-c) |
| The training port is affordable | **28.5 img/s**, 2.6 h/fold, cache builds in **~16 min** (§2h) |
| NIfTI conversion | 5 checks against the DICOMs, all pass (`pipeline/validate_nifti.py`) |
| Laterality | tag on 50%, geometry fallback agrees **97.7%** at `x < −62` (`FINDINGS.md` §6.2) |
| Fold assignment ungrouped | measured call; no patient linkage exists (4,407 IDs / 4,407 studies) |
| Four of twelve labels | Baker's **0.919**, Medial OA **0.913**, Effusion **0.863**, Lateral OA 0.824 |

**Dead — closed, with the number that closed it. Do not reopen without new evidence.**

| | why |
|---|---|
| Rule extractor as a label source | **0/12** labels, and *subtracts* from a rank-mean (§2f) |
| Frozen-embedding architecture | cannot fine-tune at any resolution, under any head (§2e) |
| The 518 rebuild (~22 h) | **+0.013**, inside the CI (§2d) |
| Soft-target calibration ladder | fitted against gold and **lost**, 0.743 → 0.699 (§1.3a) |
| Series-metadata shortcut | 0.5954 grouped-honest over 4,407 — real but far under the frontier (§2i-b) |
| Per-label reader fusion | readers are near-duplicates, |r| 0.87–0.95 (§2g) — killed before built |
| Report-hash fold grouping | the leak cannot occur for an image-only model (`fusion/folds.py`) |
| "The leaderboard is the instrument" | unaffordable: 5/day against ~8 h runs and a 30 h quota |
| Gold-37 as arbiter | ±0.031 cannot resolve 0.04; superseded, kept as the *check* on the instrument |
| Further extractor refinement | §2.2 was worth **+0.002** on gold |
| **K16 by header sort-key rule** | InstanceNumber **56.9%**, filename **60.8%**, SliceLocation **56.9%** against ~50% chance (§2n). The 3.7 h full-header fallback exports *identical* signs — \|rho\| is already 1.000. Measure the bit instead |

**Promising — ranked by (measured payoff) / (measured cost).**

| | expected | cost | basis |
|---|---|---|---|
| 1. **The training port** | the primary constraint | 16 min + 2.6 h/run | §2e diagnosis, §2h cost |
| 2. **The three labels at chance** | **~0.08 macro** | per-label specialists | Fracture 0.494, Lat Meniscus 0.526, Contusion 0.603 |
| 3. **External data (MRNet)** | supervises exactly those three | licence + a fine-tune | 1,370 exams, image-read; MRNet reaches ACL 0.965 |
| 4. **Finish the corpus** | **+0.024** per 1.6× data | download time | 1,000 → 2,649 measured |
| 5. **Gold-58 into training** | 58 image-read labels currently discarded | free | the fork does it; we hold all 58 out |
| 6. **Our head vs `SlotHead`** | unknown — largest unmeasured claim | one instrument run | `PLAN.md` §7.1, never once compared |
| 7. **Rank-mean over seeds** | mechanical, ~+0.01 | linear in runs | every public solution does it |
| **NEW: severity-thresholded label read** | unknown, plausibly large | a few $ of API calls | **HOST-CONFIRMED 2026-08-10**: "the image-based labels uses multiple readers with stricter image-based thresholds". An independent audit puts report-reading at FP 25 vs FN 17 — over-calling, the predicted direction. `REFERENCE.md` §1.4, §2.1 |
| **NEW: anatomical crops as slots** | targets exactly the failing labels | cheap — volumes are already in mm space | fingerprint biometrics and the RSNA-2024 lumbar winner independently answer our failure mode the same way: localize, then embed crops (`REFERENCE.md` §4.3–4.4) |
| **NEW: text as auxiliary supervision** | unknown | one head | `extract_states.csv` has held structured attributes since week one and **nothing has ever consumed it**; the public tables ship 12 numbers and nothing else (§2k) |

**Unknown → now measured, and it binds.**

- **External MRI datasets (MRNet, OAI, fastMRI+, SKM-TEA) — asked twice on the forum, NOT
  answered by the host.** Free but click-through. §2.6.a/§2.5.a argue yes; nobody official has
  said so. It gates a Phase 2 lever, so **ask directly** — one forum post, costs nothing.
  `REFERENCE.md` §1.3.
- **Bilateral studies exist** — the host confirms both knees are occasionally scanned under one
  `StudyInstanceUID`, with labels for one knee. Nothing here detects that, and `canonicalise()`
  would mirror both to the same handedness. Needs a count before it is worth fixing (§2k).
- ~~Site leakage is worth 0.053~~ **MEASURED for us: +0.024** (§2j). Original note:
  **the public probe's 0.053** (`zhukovoleksiy/rsna-metadata-probe`, §2i-a): DICOM headers
  with no pixels score 0.6516 under random folds and 0.5981 grouped on a scanner fingerprint.
  **`data/folds.csv` is ungrouped, so our §2g instrument carries an unknown share of this.**
  Fixed-target A/Bs mostly survive (both arms inflate together); the **reproduction gate does
  not**, because it compares our number against someone else's. Site-grouped folds are now a
  Phase 0 prerequisite. The fingerprint recipe is public: `Manufacturer |
  ManufacturerModelName | SoftwareVersions | ImagingFrequency | ReceiveCoilName`, 265 values.

### What the rules actually say `READ 2026-08-10`

Read in full, not summarised from the competition page. Four things settle open questions:

| § | text | consequence |
|---|---|---|
| ~~**2.4.b**~~ | ~~"not to transmit... the Competition Data to any party not participating"~~ | ~~Sending report text to a hosted LLM API is out.~~ **WRONG — RETRACTED 2026-08-10, same day, on a host statement.** The Host has ruled explicitly that hosted LLMs are **permitted** and that sending report text to an external API for label extraction "will not, by itself, be considered prohibited PRIVATE SHARING". It is governed by §2.6.b (accessible to all, minimal cost), not §2.4.b, and PRIVATE SHARING means sharing with *other participants or teams*. Full quote in `REFERENCE.md` §1.1. **The error is the same class as reading a competitor's notebook from its description (§2e): a rule read without its official interpretation.** |
| **3.6.b** | "You are permitted to publicly share Competition Code... deemed to have licensed the shared code under an OSI-approved license" | **Using the public LLM label tables is fine.** They are public Kaggle datasets, equally accessible to every participant. Whether their *producer* complied with 2.4.b is not our exposure, and nothing prohibits consuming publicly shared derived work. The Phase 2 risk note is **closed**. |
| **2.6.a / 2.5.a** | external data must be "publicly available and equally accessible to all Participants... at no cost"; and "input data or pretrained models with an incompatible license... you do not need to grant an open source license for that data" | **MRNet is very likely admissible.** Free on request, so accessible at no cost, and 2.5.a carves out its research-only licence explicitly. The winner licence is **CC-BY-NC 4.0** — non-commercial, unusually compatible. **No forum-disclosure requirement appears anywhere in the rules.** |
| **2.2.a / 3.4.b** | 5 submissions/day, 2 final; no hand labelling "of the validation dataset or test data records" | The submission budget that killed "the leaderboard is the instrument" is confirmed. 3.4.b constrains *test* labelling only — `labeling/` works on train reports and is unaffected. |

And the target is worth stating plainly: **prizes run to 10th place** ($5,000), plus **$18,000
across three efficiency prizes** — `PLAN.md` §6 is a live second route, not a footnote. Tenth on
the main board is **0.934** (live 08-12, §2x — it was 0.926 two days earlier, so treat it as a
lower bound on what a prize costs, not a target).

### 0. There is no working instrument, and the leaderboard is not the fix

Every macro this project has produced, across every experiment:

```
0.695   0.699   0.708   0.719   0.744        range 0.049
```

against a macro CI of **±0.038**. Almost every result is statistically indistinguishable from
almost every other. `PLAN.md` §7.2 diagnosed this for the extractor on 2026-08-07 and called it
"out of instrument". The fix chosen then was a vision model — scored on the **same 37 gold
studies**, so it inherited the blindness rather than escaping it.

**The previous plan's answer was "the leaderboard is the instrument". Retract that.** It is a
working instrument and an unaffordable one: five submissions a day, ~8 h per training run, a
30 h weekly quota, and a GPU lottery that refuses four draws in five. A plan that can only
learn one bit per submission cannot run 20 experiments, and 20 experiments is what the gap
needs. Worse, it moved the project *away* from local measurement at exactly the moment local
measurement was about to find §2f — which cost nothing but a download and was invisible to the
leaderboard entirely.

**The real fix is that the 37-study instrument is the wrong local instrument, not that local
instruments do not work.** The highest visible public solution validates on a held-out fifth of
the corpus against *report-derived* targets and reports **OOF 0.8544** against **cross-fitted
gold-58 0.8568** — two disjoint checks agreeing to 0.002. That is a local instrument with
~880 studies and 150–350 positives per label instead of 37 and 5–19, roughly **5× less noise**,
and it is free to run.

Three things have to be right for it to work, and all three are cheap:

1. ~~**Group the split by a hash of the report text.**~~ **CORRECTED 2026-08-10 before it was
   built — `fusion/folds.py` already tested this and rejected it, and its reasoning beats the
   public notebook's.** The mechanism the 0.899 notebook names — "scores the model on a target
   whose source it trained on" — **cannot happen for an image-only model**: the report is the
   target's source and never an input, so two studies sharing a target vector leak nothing
   through the text. `folds.py` also measured the cost: forcing the 49 template groups whole
   damaged fold balance to 664–1,077 studies per fold, to prevent a leak that does not exist.
   183 of 4,407 studies share a report; the largest group is 37 studies on one Turkish
   boilerplate normal.

   **The concern that does survive is a different one, and report-hash is only a proxy for it.**
   37 studies carrying one radiologist's boilerplate are plausibly one site and one scanner, so
   a model can learn *site appearance → that site's label distribution* and carry it to holdout
   members of the same group. That is **site** leakage, it would exist between same-site studies
   whether or not they share report text, and it is the thing to measure. There is no site
   column: `study_meta.csv` carries only UID, PatientID (unique per study — no patient linkage
   exists in this dataset), laterality and geometry. `Manufacturer` / `InstitutionName` /
   `StationName` would have to come from a header pass — which is nearly free if it rides along
   with the `kaggle_01c` slice-direction extension already required by step 3.

   So this becomes a measurement, not a mandate: report holdout OOF with and without template
   groups held whole, and adopt grouping only if the gap is real.
2. **Validate the proxy against gold, then stop looking at gold.** Report-derived OOF is the
   instrument; cross-fitted gold-58 is the check that the instrument is pointed at the right
   thing. If they diverge, the proxy is wrong. If they agree, use the proxy — it is the one
   with the error bars.
3. **Put the 58 gold studies into training at elevated weight.** They are the only labels in
   this project read from *images* rather than from text. We currently hold all 58 out to
   protect a 37-study evaluation that cannot resolve 0.04 — paying our scarcest supervision to
   buy a number we then cannot read.

### The rules (these bind the steps below)

1. **A local report-holdout OOF is the instrument.** Gold-58 is the check on the instrument,
   cross-fitted, never the arbiter. The leaderboard is the final audit — a few times, not
   every experiment.
2. **One change per measurement**, whether the measurement is local or a submission.
3. **Nothing below the instrument's resolution** — but note the instrument is now ~5× finer, so
   this forbids much less than it used to.
4. **Check what is free before building it.** §2f cost five days by not asking. Before any new
   component: search Kaggle Datasets, pull the top notebooks' code, and measure against them.
5. **Read the code, not the description.** Three claims about `pilkwang` came from its
   description and two were wrong; one of them cost a Kaggle run.
6. **The fork is the base, not a reference.** It scores 0.891 and its inference path
   demonstrably works. `kaggle_03_submit.py` has never executed against a real test DICOM.

### Two standing decisions `SET 2026-08-10`

These bind how every step below is built, not just which steps run.

**Reproduce the fork, then diverge — but in our own code.** The port's job is to hit a
reproduction gate against a known-good configuration, because a baseline that reproduces is the
only thing that makes a later A/B trustworthy. That is a reason to copy the fork's
*configuration*, not its file. So `pipeline/` and `fusion/` grow modules that reproduce
`pilkwang`'s slot scheme, `UNFREEZE_LAST=6` and its optimiser settings, written as ours, with
the reproduction gate as the test. If a component here cannot be explained from first
principles it does not ship, no matter what it scores.

**Rank-weighted, both objectives.** Chase the leaderboard, but nothing enters the repo that
could not be defended line by line. Practical consequences: forks are measurement baselines and
never the submitted artifact; public label tables *are* shipped, because using better data is
not the same as using someone else's model, and §2f is the honest record of why; ensembling is
allowed and must be reproducible from our own checkpoints.

### Phase 0 — supervision, then instrument, then port (this week)

Ordered so that each step is validated by the one before it. Nothing here needs a Kaggle GPU.

1. **Swap the labels. (~15 min.)** `extractor/bench_public_labels.py --download` already pulls
   them. Ship **`steven_v2` as `data/targets.csv`**, replacing `pseudo_labels.csv` everywhere,
   and stop there. ~~`steven_v4`~~ — **corrected 2026-08-10 (§2i-c):** v2's derivation is
   published and our measurement reproduces it exactly (0.8873, Synovitis 0.678→0.790, and it
   differs from v1 in that one column and no other). `v4_blend` measures 0.8927 with no
   published derivation, and **+0.0054 is below what 58 studies can resolve**. Take the one
   whose provenance can be defended. Rules cleared — see "What the rules actually say". ~~Then fuse per target, weighted by each reader's measured per-label
   accuracy, the way the 0.903 system does.~~ **Cut 2026-08-10 before building it:** that
   technique pays when readers are independent, and §2g measures these at mean |r| **0.87–0.95**
   — `steven_v4` predicts `lixin` at AUC **0.9998**. There is no diversity to fuse, which is
   why §2f's five-reader rank-mean (0.885) *loses* to `steven_v4` alone (0.893).
2. **Build the instrument. `DONE 2026-08-10 — it works.`** `fusion/train.py` now writes
   `oof_all.csv`; `fusion/instrument_test.py` scores it. Measured on the cache that already
   exists: **gold-37 ±0.031 vs report-OOF ±0.0046 over 2,612 studies, 6.7× tighter**, with the
   two landing 0.026 apart. §2g. Two corrections came out of building it, both of which
   cancelled planned work rather than adding any:
   - **Report-hash grouping is not required** — see the corrected item 1 above.
   - **The instrument is valid at fixed targets only.** It cannot arbitrate label *sources*,
     because the reference is itself a label source. Gold-58 keeps that job; §2f has already
     settled it. Everything the port needs is a fixed-target comparison, so this costs nothing.
2b. **Site-grouped folds. `DONE 2026-08-10 — our leakage is +0.024.`** `pipeline/
   site_fingerprint.py` + `fusion/folds.py --group-by site`. No Kaggle notebook was needed:
   the probe author published `headers.parquet`, 24,371 series × 43 fields, so
   `kaggle kernels output` replaced the whole header pass — **rule 4 paying for itself a third
   time.** Same targets, same cache, folds the only difference: **0.7468 ungrouped → 0.7229
   site-grouped, a gap of +0.0239 at ~5σ.** That is larger than the resolution effect this
   project dismissed as unmeasurable and the same size as the entire label swap. §2j.
   **The honest baseline is now 0.7229 ± 0.0048.**

3. **Time it before building it. `DONE 2026-08-10 — gate passed at 1.29×.`**
   `pipeline/bench_port.py` + `pipeline/bench_cache_build.py`. §2e's cost model was inferred and
   is now run: **28.5 img/s training** → 15.4 min/epoch, **2.6 h per fold-run**, 12.9 h for
   5 folds × 10 epochs. And the cache build is **~16 minutes** for the whole corpus, not hours —
   it stores pixels rather than embeddings, so its cost is a NIfTI read, not an encoder pass.
   §2h. The entry price for the architecture that *can* fine-tune turns out to be a coffee
   break; the architecture that cannot cost 21 h, 9 h and four Kaggle sessions.
4. **Port the training. (~2 h/run)** The frozen-cache architecture cannot fine-tune (§2e), so
   it cannot be fixed downstream. Build the ~9 GB pixel cache at 336 (26,442 slot images) from
   the NIfTI already on disk and train `UNFREEZE_LAST=6` locally. K16's slice-direction bit is
   on the critical path here — NIfTI carries no `ImagePositionPatient`, so extend `kaggle_01c`
   over all 24,371 series first (CPU-only, ~20 min); K18 handedness rides along.
5. ~~**Reproduce the fork's own configuration before changing one line of it.** If a local run
   with *its* labels does not land near its published score, the port is wrong and every
   comparison after it is noise. This is the gate; do not pass it by reasoning.~~
   **REWRITTEN 2026-08-11 (§2s).** "Its published score" does not exist as a local number —
   `REFERENCE.md` 3.1 records `pilkwang`'s local column as `—`, and 0.891 is a 20-member
   rank-mean of published weights on the leaderboard. Converting cannot rescue it: the two
   `(local, LB)` anchors give a band 0.02 wide before the ensemble gain is counted.
   **What this step is now:** train the port on the fork's labels and read the delta against
   `runs_port` — predicted **0 to −0.021** from the gold-58 label gap. A *plausibility band*,
   not a gate. **And before it launches: state the scoring reference and show it is neutral to
   both arms.** Four measurements have been lost to skipping that (§2d, §2i, §2o, §2s).
6. **Submit once**, as a dry run of the inference path, and record the CV↔LB mapping for *our*
   pipeline rather than the interpolated estimate in "The four facts" §5.

### Phase 1 — the labels that are at chance

Each label is 1/12 of the score, so a label left at chance forfeits `(M − 0.5)/12` ≈ **0.029**
no matter how good the other eleven are. Ours, on gold-37: **Fracture 0.494, Lateral Meniscus
0.526, Contusion 0.603**. Those three alone are worth ~0.08 of macro — more than the entire
remaining gap to the public frontier.

The per-label pattern is diagnostic, and it matches the physics. Against MRNet (Stanford,
n=1,370, image-read labels): our ACL **0.702** vs their 0.965; our meniscus **0.634 / 0.526**
vs their 0.847. Meanwhile our Effusion 0.863, Baker's 0.919 and Medial OA 0.913 are already
reasonable. **Gross-appearance findings work; findings that need fine local texture at a
specific anatomic site are at chance.** That is the signature of a frozen natural-image ViT
plus attention pooling over ~155 slices diluting the two or three that carry the finding — so
Phase 0 step 3 is the first-order fix, and these are what to check it against.

Then, in order: per-label specialists for the three at chance (the 0.899 notebook adds a
Synovitis specialist worth +0.084 on that label alone); rank-mean across seeds, then
resolutions, then backbones; our fusion head against `SlotHead` — **the largest unmeasured
claim in the project**, called "likely ahead of the public forks" in `PLAN.md` §7.1 and never
once compared.

### Phase 2 — the levers that measured largest

1. **External data.** MRNet (1,370 exams, image-read ACL / meniscus / abnormality) supervises
   exactly the three labels we are worst at, with expert reads rather than report text. OAI
   covers the OA three. **Gated on a rule, not on effort — and the rule has not been read.**
   The Kaggle rules page is JS-rendered and needs a logged-in browser; what is confirmed from
   elsewhere is only that this is a code competition, **≤9 h runtime, internet off, entry
   deadline 2026-10-15**, final 2026-10-22. Read the rules page before any MRNet work.

> **RISK, OPEN — read the rules before Phase 0 step 1 ships, not after.** `IMPROVEMENTS.md`
> §1.1 flagged in week one that Competition Rule 4.b (Data Security) may forbid sending report
> text to a hosted LLM API. That concern was about *us* running one. It now applies to the
> public label tables we are adopting: several were plausibly produced that way, and adopting
> their output is not obviously the same act as producing it. One signal that the question is
> live rather than paranoid — the 0.903 system's author went to the trouble of serving
> **Qwen3.6-35B locally**, at temperature zero, rather than calling an API.
>
> This does not change the measurement in §2f, which stands. It changes whether the best label
> source is *usable*, and the fallback if it is not is the local-LLM route §1.1 originally
> scoped — now much cheaper than in week one, because §2h shows the M5 handles work of this
> size and the reports are 4,407 short documents. **Do not build on the swap until this is
> read.**
2. **Data.** 1,000 → 2,649 studies was **+0.024**. The corpus is at 60% of 4,407. Finish it.
3. **Gold at weight 3.0** — folded into Phase 0 step 2, listed here because it is a lever, not
   only a validation fix.

### What the public frontier actually does, for reference

Read from the code (`extractor/bench_public_labels.py` pulls the labels; the notebooks come
from `kaggle kernels pull`). Two lineages, and the CNN one is currently ahead:

| | `pilkwang` / 0.899 line | Yash B3 line (0.903, highest visible) |
|---|---|---|
| backbone | DINOv2-**small** @336 | **EfficientNet-B3, single-channel, ImageNet** |
| trainable | `UNFREEZE_LAST=6`, `LR_BACKBONE=8e-6`, `LR_HEAD=1e-3` | full |
| study input | 6 slot images (`N_SLOT=6`, `GROUP=3`) | 3 fluid-sensitive, plane-diverse series |
| slices | 3/slot | 12/series train, **32 infer** |
| pooling | `SlotHead`, per-target priors 0.55 | **max over slices, then mean over series logits** |
| targets | conf-weighted `W = 0.25 + 0.75·conf` | per-target fusion of two readers by measured accuracy |
| aug | multi-window TTA | rotation, gamma, scale; **no horizontal flip** |
| ensemble | rank-mean, 20 members | mean of 5 fold sigmoids |
| cost | 10 epochs | 12.4 h, 5 folds |

Three of those are worth stating as flat corrections to our design. **A study is six encoder
inputs, not ~155** — the frontier does 20× less encoder work and gets more out of it, because
it adapts the weights instead of pooling frozen slices. **Max-pooling beats attention pooling**
for this task in their own ablation, for the reason above: the finding lives on two slices.
**No horizontal flip** — it breaks medial/lateral, which is 6 of our 12 labels.

### Explicitly not doing

- **Any further extractor work.** §2f: 0/12 labels, negative in fusion. `rule_extractor.py`,
  `glossary.json`, `calibrate_states.py` and the hand-labelling UI stay for provenance and for
  the disagreement-detector hypothesis in §1.1, and that is all.
- **The full 518 rebuild** (~22 h). Measured **+0.013**, inside the CI.
- **The frozen-feature architecture** as the route. Kept for provenance and for the head
  comparison in Phase 1.
- **Leaderboard-driven iteration.** Rule 1.

### Superseded plans (provenance — the reordering itself is the evidence)

| dated | operative claim | why it was replaced |
|---|---|---|
| 2026-08-08 | extractor is the critical path; cache needs Kaggle | §9.1: the cache builds locally on the M5 |
| 2026-08-09 | resolution is the ceiling | tested: **+0.013**, inside the CI (§2d) |
| 2026-08-10 (am) | trainability is the gap; the leaderboard is the instrument | true but incomplete, and the instrument claim was unaffordable (§2f, §0) |

## Status — 2026-08-10

- [x] Data logistics, language ID, series structure
- [x] Rule extractor v1 — macro AUC **0.777** on gold, 95% CI [0.74, 0.82]
- [x] Compartment attribution (`IMPROVEMENTS.md` §2.2) — ~1,000 studies off the flat 0.45.
      Invisible in gold macro (±0.038 CI); justified on corpus evidence, per §0
- [x] Hand-labelling UI + all 30 blind gold studies labelled
- [x] ~~**Our labels beat the public weak labels** on gold and on hand labels — the moat is
      real~~ **RETRACTED 2026-08-10.** True against `nekkon`'s week-one binary CSV, false
      against the four LLM readers published since: **ours 0.777, best public 0.893, 0/12
      labels won, negative contribution to a rank-mean.** `IMPROVEMENTS.md` §2f.
      `extractor/bench_public_labels.py` is the live gate; `compare_methods.py` is the stale one
- [x] **The public LLM labels are measured and adopted** — the extractor track is closed as a
      source of training targets, and §1.1 ("where does the LLM extractor run?") closes with it:
      it does not have to run at all
- [x] Series-metadata shortcut tested and rejected (0.471, below chance)
- [x] **Soft-target ladder fitted against gold — and the fit LOST** (`IMPROVEMENTS.md` §1.3a).
      `absent` is 52% of the target matrix and sits at 0.08 against a measured 0.167; correcting
      that scored **0.743 → 0.699** (both arms pre-K17, on the pre-deletion cache whose default
      baseline is now 0.719; the sign and mechanism are unaffected — see `IMPROVEMENTS.md`
      §1.3a). How far each label's `absent` was raised predicts how much
      AUC it lost (corr **−0.776**), because the `absent` bucket is heterogeneous and re-targeting
      it to its mean teaches the mean instead of the discrimination. Better calibration, worse
      ranking. Kept as a measurement, not a setting
- [x] **Training code complete and tested end-to-end on synthetic features** — fusion head,
      dataset, folds, training loop, submission notebook. Everything that does not need pixels
      is done; training is blocked only on the cache
- [x] **Laterality answered** (`kaggle_01` + `kaggle_01b`, `FINDINGS.md` §6.2) — the tag survives
      on 50% of the corpus, and the geometry fallback agrees 97.7% at the `x < −62` boundary, not
      the obvious `x < 0` (89.3%). Tag first, geometry second, source recorded per series
- [x] **Cache-build harness hardened** (`IMPROVEMENTS.md` §6) — the GPU guard, the spawn pool and
      the preprocessing-parity assert all had faults that cost or would have cost a session.
      Fixed and self-tested locally; **not yet run against real DICOMs**
- [x] **The cache is built — locally, from NIfTI, not on Kaggle** (`PLAN.md` §9.1 + its
      correction). Five Kaggle attempts died on the GPU lottery, the 9 h cap and ~19 ms/open on a
      network mount; none are properties of the data. `pipeline/build_cache_local.py` builds it on
      the M5 at a measured **1,062 studies/h** at 224. The conversion was validated against the
      DICOMs first (`pipeline/validate_nifti.py`, 5 checks). In-plane layout is `as-is` at
      **median r = 1.0000**, best for 98% of series across all six types — identical pixels, so
      the repackaging is faithful. **Slice direction is not**: see "What is wrong" below
- [x] **First vision-model result: macro AUC 0.719** on 37 gold studies, images only, 224px, on
      60% of the corpus with default hyperparameters. A floor, not a ceiling — 518px, the rest of
      the corpus and any tuning at all are still unspent. Synovitis scores **0.685** despite being
      the extractor's *worst* label (0.607), which is the first evidence for §2.1's option (b)
- [x] **Training is deterministic** — two identical re-runs give 0.719 and 0.719, agreeing to
      three decimals on every label. Every A/B in these documents measures a real difference with
      no run-to-run floor beneath it; the ±0.038 CI remains the *sampling* limit on n=37
- [x] **The pseudo-labels are not the ceiling — measured** (`IMPROVEMENTS.md` §2d). Vision beats
      the extractor on six of twelve labels and Spearman between the two per-label columns is
      **−0.17 (p=0.60)**. The bottleneck is the image path: the 224 cache resolves **0.71 mm/px**
- [x] **Resolution tested and it is small** (2026-08-10). 1,000-study 518 cache, 3.88 h, trained
      against the same 1,000 studies at 224: **+0.013 macro**, inside the ±0.038 CI. Right where
      predicted — Lateral Meniscus +0.067, Fracture +0.051 — and too small to plan around. The
      same run measured 1,000 → 2,649 studies at **+0.024**, so **data is worth ~2× resolution**
- [x] **The real diagnosis: no working instrument.** All four macros this project has produced —
      0.695, 0.699, 0.708, 0.719 — span 0.024 against a ±0.038 CI, so none of its conclusions are
      supported by the measurement that produced them. The leaderboard is the instrument and has
      been used once. See "Where this goes next"
- [x] **First LB submission — 0.891**, an unmodified fork of `pilkwang/rsna-knee-baseline-v1`,
      2026-08-09. Rank **230/908**; LB top is **0.940**. *(Live 08-12: same score, rank
      **400/1,276**, top **0.946** — §2x.)* Phase 1's first half is done and the
      reference implementation reproduces. Our *own* pipeline has still never been submitted, and
      in its current state would likely score below this — see "What is wrong" §4
- [ ] ~~**DINOv2 feature cache built and published (`kaggle_02`)**~~ — superseded by the local
      build above. Kaggle-side remains the path for the TEST set, which `kaggle_03` always did.
      The five failed attempts are tabulated in `PLAN.md` §9
- [ ] Hand-labelling — 86/303 done; the remaining 217 are the only validation set we will have
- [ ] **First own-pipeline submission** — a dry run of `kaggle_03`, which has never executed
      against a real test DICOM. Needs two Kaggle Datasets that do not exist yet
      (`rsna-knee-fusion` with `fold*.pt` + `manifest.json`, and `dinov2-weights` — there is no
      internet in the kernel). `rsna-knee-code` exists and needs updating to HEAD
- [ ] **The §7.2 A/B** — but against the *current* public extractor, not `nekkon`'s CSV. See
      "What is wrong" §5
- [ ] LLM extractor (method B) — host undecided (`IMPROVEMENTS.md` §1.1); ~$44 batched on the
      API, or ~30–90 min a pass on a 5090
- [ ] External data — MRNet, OAI, fastMRI+ cover ~6 of 12 labels with real expert image reads
