# Log — append-only, newest last

The detail behind [`RUNBOOK.md`](../RUNBOOK.md): a pre-registration once it has been read, the numbers, the
diagnosis, and owner Q&A that belongs to no other doc. Date and time every entry (PDT). Never edit an old entry;
correct it with a new one that names the old.

Everything before 2026-09-28 ~20:45 PDT is in [`archive/RUNBOOK-2026-09-28.md`](archive/RUNBOOK-2026-09-28.md).

---

**2026-09-28 ~20:45 — the RUNBOOK was cut to one page** (owner: "do 1"). Why: it had grown to 169 KB (1,729 lines).
Its state, pre-registrations and Q&A were interleaved, and parts contradicted the present. Its closing summary still
said "the labels are not the constraint", which D4 reversed on 09-27. The README still gave the 09-24 state. A cold
session reading top to bottom could act on a stale sentence. The full text is frozen in the archive, unchanged apart
from a header. Also decided: **no "safe" final.** The owner declined locking the public stack as a fallback ("2 is for
cowards"), so both finals aim to win (RUNBOOK, *Open* item 6).

**2026-09-28 ~20:50 — the Friday list is READY: v2 smokes read, build pinned to `b5331ff`.** Both v2 smokes (Kaggle CPU,
from `b5331ff`, pushed ~19:04) read at 20:40. Ortho (`rsna-knee-c7-train-ortho-student-r1` v2): targets
`targets_round1_cnn.csv` sha `75757be8c9b5c8d1`, not `steven_v2`; partition check 691/691 in fold 0; train 3,502 | val
691; strict load, RoPE = theirs; epoch 0 loss **0.9177** (v1: 0.9177); DDP weight checksums identical per rank; 18.8
min; no traceback. CoAtNet (`rsna-knee-c6-train-coatnet-student-r1` v2): the same targets sha; init by its full path
`/kaggle/input/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt` (sha `20d45e07`), 486 tensors strict, with no
6-minute `rglob` (15.0 min vs v1's 22.0); epoch 0 loss **0.6574** (v1: 0.6574); no traceback. Identical epoch-0 losses
across v1 and v2 show the off-by-default `--zoom` left training untouched. **Why pinned:** `train_windows.py` changed
again after `b5331ff` (`--extra-corpus`, `be32fdc`), and the Friday task builds from the working tree. So its build
commands create a detached worktree at `b5331ff` and build from there. Every embedded file is tracked, including the
two `oof_all.csv` controls. Test builds at 20:40, not pushed: both print `from b5331ff`, `sha256 75757be8c9b5c8d1`,
`mode=full` with 7 / 14 epochs, `enable_gpu` true, the right datasets and shards, 0.69 MB each.

**2026-09-28 ~20:45 — owner Q: "is the overall direction going well?"** The direction is right; the scoreboard does not
show it yet. Right: the bottleneck is found and confirmed on the LB (labels +0.015, only the targets changed); the model
has reached its labels on gold (0.878 vs 0.887); the teacher passed (+0.026 gold, P 0.994, free on the Mac); the base is
now the best available (the public 0.928 CoAtNet, checked in our pipeline); Friday is READY. Not yet: 0.939 < the free
0.943, and the combined path has no LB read. The first real proof is Sub (A), the upgraded stack, which needs ≥ 0.947
(~Sun 10-04). Three GPU resets remain, so each week is one iteration and a lost Friday costs a third of what is left.
Recommended (not decided): (1) build the critical path first. That means the replace-the-core submit mode and timing
the stack's cells, before any side screen. (2) Write both branches of (A)'s read before Saturday, so week 2's GPU is
committed the hour it lands. If (A) passes, students go into the stack's other CoAtNet slots, plus round 2. If it fails,
diagnose from the student-alone read. (3) A team merger by 10-15 is the one lever that adds GPU quota.

**2026-09-28 ~21:40 — the replace-the-core submit mode is BUILT (owner: "do that").**
`build_submit_kernel.py --public-dir <pub> --core-kernel K --core-sha S --tag T` builds the upgraded stack. Our
CoAtNet student's `last.pt` becomes the public core: `maxspan-v5` (w 0.60) and `maxspan-v5-reverse` (0.10), which
share one file. Everything else runs as published.

What it adds to the notebook:
- a CPU cell after public cell 1. It finds our `last.pt` by sha256 and checks `last_meta.json` against the slot
  (CoAtNet, Stage A 2–98%, 336 px, no slot embedding, no zoom). It renames the keys back to the public names and
  requires the key set to equal the public file's exactly, casting each tensor to its dtype. It writes the result
  under a NEW name and adds its folder to `ASSET_ROOTS`.
- the two ARMS entries' `file`, swapped, with seam checks that refuse a moved upstream.
- a cell before the publish cell. It reads the stack's own `raptor_checkpoint` load log and stops the run, so no
  submission.csv exists, unless the core loaded our file at its sha256 and never the public v5.

**Bug caught by the test, before Kaggle:** public cell 1 wraps its asset lookup in a ONCE-ONLY catalogue
(`_speed_asset_catalogue`) and a memo. A folder added to `ASSET_ROOTS` after the first lookup is invisible, and our
own lookup of the public file makes that first lookup. Built naively, the swap would have died on Kaggle at
`expected one pinned asset`, after the queue wait. The core cell now drops the catalogue after adding our folder.
The builder refuses to build if that cache changes upstream.

`notebooks/core_swap_fixture.py` (~7 s on the M5, all passed):
- a student `last.pt` saved as `train_windows` saves one, strict into WindowMIL, then perturbed;
- the build;
- the core cell as built, which refuses a Stage B checkpoint;
- the public stack's own `load_model` / `find_weight_file` on the swapped ARMS: our file, strict, shared by both arms.
  Without our folder, the lookup FAILS rather than falling back to the public weights;
- logits: the public code on the swapped core vs our WindowMIL on `last.pt`, |Δ| **0.00e+00**. Control: vs the
  public weights, 1.24;
- the check cell passes on the real load log and refuses a public-v5 load, a missing load and a wrong sha256;
- the builder refuses the misuses.

Old modes are unchanged: B, A and B-dread rebuilt from HEAD and from this tree are identical apart from the commit
stamp. **Not yet run on Kaggle.** The first commit run is (A) itself, on the CoAtNet student's freed session
(~Fri 23:30).

**2026-09-28 21:35 — B′ read (the Mac control for Z and N).** `fusion/runs_cnntest_Bp`: ResNet34 at 224, Stage A,
`dread_soft`, 18 epochs on the M5 (MPS). It scores **0.8129** on the 691 against Kaggle's run of the same recipe (B,
`runs_r34v2_B`) at 0.8076, paired +0.0053 ± 0.0046 (1.1σ, 7/12 up). That is inside the pre-registered 0.010, so
Z and N are judged against B′ as planned, and Mac and Kaggle arms can be compared. The NYU pseudo-labels are written
(21:16). Z is at ~9 min/epoch → ~00:15; N follows, → ~02:00.

**2026-09-28 ~21:45 — owner Q: "is B′ being what it is a good result?"** Yes, for its job. B′ is a control, not a
candidate. It matched Kaggle's run of the same recipe (+0.0053, noise), so the Mac bench is trustworthy. As a model,
0.8129 is well below Ortho `dread`'s 0.8440 on the same 691, which is why ResNet34 is cut as a main model and kept
only as a teacher. The 691 scale is not the LB scale: its reference is another LLM's read of the reports, so Ortho's
0.844 there was 0.916 on the LB. **What it sets for tomorrow:** the paired SE against B′ is ~0.0045, so Z or N must
beat it by roughly **+0.008** to reach P ≥ 0.95. A smaller real gain reads as a null.

**2026-09-29 08:40 — Z and N read against B′ (rules pre-registered 09-28 18:55 and ~20:00).** All three are ResNet34 at
224, Stage A, `dread_soft`, the same 3,502 / 691, seed 42, on the M5.
- **Z (`--zoom 0.6714`: every window again as its centre 94 mm, one attention bag), 179 min: HELPS.** 691 **0.8288** vs
  B′ 0.8129, paired **+0.0159 ± 0.0041**, 3.8σ, P 1.000, 10/12 up.
  - Small structures: ACL 0.817 → 0.836, MCL 0.686 → 0.732, Medial Meniscus 0.889 → 0.900, Lateral Meniscus 0.811
    → 0.826.
  - Also Lateral OA 0.754 → 0.796, Fracture 0.861 → 0.887, Baker's +0.023, Contusion +0.014.
  - Down: PF OA −0.009, Synovitis −0.009.
  - The two biggest moves are the findings the round-1 teacher did not help on gold (MCL, Lateral OA), which were
    read as vision gaps on 09-28.
  - Gold-58, report only, points the other way: B′ 0.8943, Z 0.8770, paired −0.017 ± 0.011, P(Z > B′) 0.06. Per
    the standing rules the 691 judges an input change at fixed targets, and n = 58 is noisy. But this is a real
    caveat, and the LB is the final judge.
  - Against Ortho `dread` (0.8440) the ResNet gap narrows from −0.031 to −0.015.
- **N (+ 7,140 NYU pseudo-knees, 6 epochs = B′'s optimizer steps), 92 min: null.** 691 0.8164, paired +0.0035 ±
  0.0051, P 0.75, 6/12 up (Contusion +0.041, Effusion +0.017; Baker's −0.018, Fracture −0.018). Gold 0.878, report
  only. Per its rule, no more NYU image work before Friday's LB read.
- **What Z does not answer:** Z brings the 224 px ResNet to 0.42 mm/px, Ortho's density. Our students already see
  0.42 (Ortho, CoAtNet at 336 → 384) or 0.36 (Stage B). So the gain may be resolution, which the students already
  have, not the crop. The pre-registered arm **R** (ResNet34 at 336, whole image, the same 0.42 mm/px, no crop)
  separates the two and is free on the Mac. Proposed rule, NOT yet pre-registered: Z − R > 0 paired at P ≥ 0.95
  → the crop adds beyond detail → port zoom to a week-2 student. Otherwise zoom stops. The owner decides.
- Also: NYU batch1 completed 22:35 (137,029,024,396 bytes). It is parked.

**2026-09-29 ~10:30 — owner Q: "have we considered free threads with 3.14t Python?"** Not before. On the evidence, no.
Free-threading only speeds Python threads that do CPU work at the same time, and no stage here is bound by that.
- **Kaggle training** is GPU-bound: data wait is 0.1–0.2% of a step (speed bench 09-26).
- **Kaggle inference:** our DICOM decode already runs in separate processes (`infer_windows --workers`, DataLoader
  workers), so the GIL does not bind it. The public stack decodes in threads, but its decoders are C code, and
  swapping the interpreter inside a pinned, internet-off image means shipping 3.14t plus free-threaded builds of
  torch / numpy / pydicom / opencv / timm as a dataset. That is a large risk on a run already 8–9 h of its 9. The
  cheaper runtime room is Open item 4: cut the 20 DINOv2 members (~5% of the vote).
- **The Mac** has the only visible slack: data wait is 7–12% of a step (B′ 9.1%, Z 6.7%, teacher fold 1 12.0%),
  because `TW_WORKERS=0`. macOS workers each pickle a 32 GB copy of the memmap. A plain prefetch thread would recover
  most of it under the normal GIL, since numpy's large copies release it. It is worth ~10% of free Mac time and 0 AUC.

None of it moves AUC, and the binding constraints are the labels, the GPU quota and the 9 h limit. Parked.

**2026-09-29 ~12:00 — R read → zoom STOPS (rule pre-registered 08:37:10, `fbcae28`).** R = B′ with `--res 336` only,
181.7 min on the M5.
- **THE RULE:** Z − R **+0.0070 ± 0.0045** (1.5σ), **P 0.943**, which is under 0.95. So Z's gain was mostly detail,
  which the 336/384 px students already have, and zoom stops for this competition. It is a near-miss. The rule was
  written to refuse exactly a 1.5σ read, and it is not a second chance.
- **Reported:** R − B′ +0.0089 ± 0.0046, P 0.979, 10/12 up. Resolution alone buys over half of Z's +0.0159. The 691:
  B′ 0.8129, R 0.8218, Z 0.8288.
- **Z − R per finding:** MCL **+0.052** (0.680 → 0.732), the one clearly crop-specific gain; Effusion +0.025, ACL
  +0.012. PF OA −0.013, Synovitis −0.010, Lateral Meniscus / Fracture ±0.001. Per-finding picking is closed.
- Gold-58, report only: B′ 0.894, R 0.894, Z 0.877.

**Owner's cascade idea, simulated on the saved OOFs (no training): B′ for every knee, Z's scores only for the most
unsure.** Unsure = max over findings of p(1 − p) under B′, fixed in advance, compared with random picking.
- Zooming the 50% most unsure keeps **72%** of Z's gain (random picking: 49%). 30% keeps 35% (random: 29%).
- A cascade of two models costs 1 + 2q passes: at q = 50% that equals zoom everywhere (2.0). It only saves compute
  if one model can run with and without its crops, which was never measured (Z's weights were not saved).
- Moot anyway once zoom stopped. The crop cost is paid per knee, not per finding. Each finding already has its own
  attention head, which learns which views to trust.

**2026-09-29 ~13:45 — the CoAtNet student trains on ALL the knees (owner: "yes train on them").**
- **Why:** its start, the public CoAtNet, trained on all 4,349 non-gold knees plus gold ×3. So for this model the 691
  was never a clean check, and the LB is its only judge. Holding the 691 back bought nothing and cost 24% of the
  training knees: 4,349 vs 3,502, i.e. the 691 plus 156 more fold-0 knees never used before.
- **What was built:** `build_backbone_ab_kernel.py --all-data` (`b90ab08`). Train = every non-gold study with complete
  targets; val = the 58 gold studies (disjoint, report only); no 691 pairing; slug suffix `-all`. Default builds are
  unchanged but for the inert branch.
- **CPU smoke** `rsna-knee-c6-train-coatnet-student-r1-all` v1, read 13:30: targets sha `75757be8`, partition 691/691,
  train 4,349, val = the 58 gold exactly, train ∩ gold 0 (the 681 non-gold of the 691 are in train), init strict 486
  tensors `20d45e07`, epoch 0 ran, 23.3 min, no traceback.
- **Friday entry 1 swapped** to it, pinned to `b90ab08` in its own worktree. Entry 2 (Ortho) stays at `b5331ff` with its
  holdout. Both were test-built ~13:40 and pass every flag check.
- **Cost:** ~7.3 GPU-h (was ~6), done ~Sat 00:40. (A)'s core is now the `-all` kernel. The Ortho student keeps the 691
  because for it the 691 is still clean.

**2026-10-03 ~19:15 — owner Q: "could XGBoost help us?"** No. Estimated LB gain ~0, with real downside; nothing built.
XGBoost (boosted decision trees) learns from a table of columns, so the question is which columns. All three were
already measured here, or cannot be fitted honestly:
- **Metadata columns** (series types, scanner): `eda_04_metadata_baseline.py` used sklearn's boosted trees and read
  **0.471** macro on gold-58, i.e. chance. The site prior read **−0.0057** in the only shippable setup (archive
  IMPROVEMENTS §3u).
- **The models' 12 outputs as columns (a stacker):** a fitted cross-finding stacker (logistic, 5-fold on the 4,407)
  read **+0.0033 on report labels and −0.0132 on gold**, positive in 1% of draws (archive IMPROVEMENTS §3r, 08-13). It
  learns the reports' co-occurrence (effusion mentioned → synovitis mentioned), not the images'. Trees fit that
  artifact harder than a logistic does. Closed in `DECISIONS.md`; this does not reopen it.
- **Deep features as columns** (trees in place of the attention head): the features are in-sample for every study the
  model trained on, and trees are weak on dense 1,024-dim embeddings. Not measured; expected ≤ 0.
- **Why it cannot be fixed by tuning:** a stacker needs (a) labels graded from images and (b) out-of-fold predictions.
  We have 58 image-graded knees (9–35 positives per finding), and the stack's members were trained on all data, so
  they have no out-of-fold predictions. AUROC is also rank-only per finding, so a stacker gains only by reordering
  knees with information the models lack, and the metadata has none.
- **What would reopen it:** a few thousand image-graded labels, or a K-fold family with a label source independent of
  the reports. Neither exists. The blend stays an unfitted equal rank-mean; the measured lever is seeds plus the
  reversed pass (`docs/REVIEW-2026-10-04.md` §2.1, +0.003–0.006 per family).
