# PLAN — RSNA Knee, goal 0.97 macro AUROC

Finals **2026-10-22 16:59 PDT**. Branch `labels-small-backbone` (the only work branch; `main` is the 10-03
portfolio snapshot). **This page is the current state.** [`docs/LOG.md`](docs/LOG.md) is the dated archive of every
run and the rule written before it: read it only for a past result. Last updated 2026-10-17.

## Where the score is lost

0.97 is the mean of twelve AUROCs. Eight findings are near 0.966; **four hold the macro down**, and 0.97 needs them
near 0.95:

| finding | our five folds, LB probe (image-graded) | gold-58: ours / OAI-trained CoAtNet | why |
|---|--:|--:|---|
| Synovitis | 0.884 | 0.853 / 0.828 | labels: radiologists mark it in 27/58 gold knees, reports mention it in 14–22 % |
| Lateral OA | 0.896 | 0.822 / 0.770 | vision: the report has it (readers 0.872 on gold), the model doesn't see it |
| PF OA | 0.896 | 0.852 / 0.840 | vision |
| Lateral Meniscus | 0.932 | 0.894 / 0.883 | both |

Every model in the blend learns from reports and **makes the same mistakes on these four** (within-class ρ 0.89–0.955
between ours and the OAI-trained CoAtNet). Improving a model on what the base already gets right moves nothing.
A gain on one of the four big enough for 0.97 (+0.03 to +0.07) **is** visible in a single-finding probe (submission
score = (5.5 + AUROC) / 12); a +0.002 on the macro is not (one score ±0.008). **Judge weak-four work on its finding.**

## Backwards from 0.97 (10-17; math in the LOG)

- **0.97 − 0.958 = +0.144 AUROC summed over the twelve.** The eight strong findings sit ~0.97–0.98 in the base
  (estimate) and cannot carry it: **the four must average ~0.95–0.96, +0.036 each.**
- **The 0.957/0.958 notebook is the base, not the route.** All its members learn from report labels; their errors
  are shared (ρ 0.86–0.89, a public analysis; 0.78–0.955 our own measurements) and that family tops out ≈ 0.960.
- **What a member must be to add +0.036 on a finding at 0.91** (two-score normal model, best non-negative weight):
  ρ 0.8+ → it must already be ≥ 0.946 alone (impossible on report labels: K, Raptor, our folds all tie at ~0.90);
  ρ 0.5 → ≥ 0.925; ρ 0.3 → ≥ 0.894. **Only labels the reports didn't produce can give ρ ≤ 0.5.**
- **The route that fits:** K's knee-centred crops trained on **OAI MOAKS image-graded labels** for the four (plus
  Medial Meniscus / Medial OA / Effusion as supplements), on this Mac. If it reaches 0.90 at ρ 0.5 on each of the four
  (assumed, unmeasured) the four gain ~+0.025 each → ~0.966; supplements (other OAI-graded findings, K + teacher at
  small weight) +0.002–0.004 → ~0.97. Every number in that line is to be measured.

## Where the submissions are

| | LB | |
|---|--:|---|
| free public base (heliosli, 10-15) | 0.958 | the 0.9576 notebook with per-finding Raptor weights (Lat OA 0.2, PF OA 0.6, Syn 0.8, Baker's 0.35) |
| ours: 0.957 recipe + our five teacher folds | 0.957 | `57241351`; with R6 + R7 in one decode (B) the same, `57244451`: **R6 + R7 is the ConvNeXt half** |
| **D: R6 + R7 + our five consensus folds, no Raptor** | **0.954** | `57244659`, **the OAI-free final** |

## The plan (owner 10-16)

1. **K, the weak-four model** (cache built 10-16, 4,407 studies, 0 errors; **folds 0–2 tie the teacher on the four** (mean
   −0.0055 / +0.0005 / −0.0030, ρ 0.78–0.91); **its Lateral OA probe (fold 0) read 0.533 = 0.896, the same tick as
   the five-fold model: FAIL by the owner's +0.03 bar, folds 3–4 not launched, K stands at folds 0–2** (10-17 16:36;
   ~16 GPU-h of the week left; LOG 10-16 / 10-17). The knee joint and patella found by the SKM-TEA segmenter; four
   regions cut from the original series in the canonical frame: lateral sagittal and lateral coronal (Lateral OA,
   Lateral Meniscus), axial through the patella (PF OA), central sagittal around the front (Synovitis, PF OA); 8
   slices each at 256 px, trained at 384 on the four only. CPU kernels build the cache (`build_kcache_kernel.py`), then
   five folds on the GPU (`--recipe k`). **Fold-0 rule:** stop and look if the mean over the four is > 0.03 below
   `runs_cnxs_teach1_f0` on the same studies, or < 16 epochs. (W4, `--recipe w4`, is built but set aside: same labels,
   same mistakes.)
1b. **KDC: K on decoupled targets** (10-18; LOG): Lateral OA / PF OA from the report-only labels, masked so they no
   longer track the other compartments (φ 0.35 → 0.000). **Fold 0 running from 12:20, done ~17:30**; rules in the LOG
   (decoupling ρ ≤ 0.60, not broken ≥ 0.75, then a Lateral OA probe on the owner's go). Closed beside it: read-time
   erasure (≤ ~0.01) and the cell screen (xgboost on location-fixed cells: 0.68–0.78, fail).
2. **Measure in the target currency: probes of K only** (owner 10-16: "whatever attacks the weakest parts first, and
   only we do that"; no probes of public models, W4 is not trained). K counts on a finding only if its single-finding
   LB probe beats our five-fold model's 10-12 probe there (the CONSENSUS folds, 0.945; Synovitis is consensus fold 0 alone;
   the teacher folds were never probed per finding) (Synovitis 0.884, Lateral OA 0.896, PF OA 0.896, Lateral
   Meniscus 0.932) by **≥ +0.03**. Probes need the owner's go for the slots.
3. **If K does not move the four**, the remaining lever is labels graded from images: OAI's MOAKS reads (2,399 knees;
   1,659 graded for synovitis; table in `~/oai_data/`, never in this repo). Review 10-17: the only label step that
   ever paid on the LB came from OAI (+0.006, the OAI-trained teacher); Raptor used OAI for PF OA, Lateral OA and
   Synovitis only, **never the menisci** (Lateral Meniscus 21 % positive, double-read agreement 0.939). **No rent
   needed:** the trainer runs on this Mac (`--mps`; ~15 h for a full all-data model here, estimated), 705 GB free.
   **Owner's call, first step only a look:** sign in to NDA, build the package for the 2,400 knees' baseline MRIs and
   read its size (~200–400 GB is an estimate) before committing. Images stay on this Mac (DUC §3 / §8).
4. **Finals by 10-22:** main = the 0.958 base with R6 + R7 + our members, per finding where they passed step 2;
   OAI-free = D. Built on our private dataset copies, without `--read`, and with the vendored-MONAI tree check
   (sha256 `1dcf1dabb83ab80eec50a5c0abfc407b3ba785bf9f30ae6c0e25da124ef87b05`, 194 files of heliosli's
   `cloud_code/training/vendor_runtime/monai060`) added to the R6 / R7 worker's path.

## Closed — do not re-propose (detail in the LOG)

- A bigger backbone (ViT-L lost by 0.011); denser slices (Stage C); centre crops / zoom as a standalone model; an
  nnU-Net crop; a stronger LLM re-reading the reports (report tables cap ~0.90 on gold); Qwen labels; a per-site prior.
- Training on the 58 gold studies (~+0.0005); R7's mirror pass at inference (0.949 = 0.949).
- Blend-weight tuning on the public board; consistency rules between findings (flat or negative); distilling public
  models into ours; fine-tuning with R7's trainer (needs a 5090); fixed half-crops of our cache; the public small CNN.
- OAI images "to add knowledge": the OAI-trained CoAtNet already has it and is no better on the four. A knee-centred
  compartment CNN: R6 / R7 already read knee-anchored compartment views (`bliverrigpdoge`'s crops: +0.001).
- "Traumatic cartilage loss counts as OA" (a 6-knee artefact); Synovitis from Effusion (0.70 on gold).
- **Cartilage / meniscus measurements from the SKM-TEA segmenter** (10-16: Lateral OA 0.654, PF OA 0.619 alone;
  +0.00008 on the macro).

## Rules

- **Ask before spending GPU hours and before a batch of submissions.** A go covers what it named.
- **OAI data never enter this repo, GitHub or Kaggle** (`~/oai_data/` only, mode 700; DUC §3).
- No Claude attribution in this repo's commits. Commit and push after every step.
- Kernels are generated from committed code: commit first, build into a scratch folder, push.

## The repo

| path | what |
|---|---|
| `fusion/train_windows.py` | the trainer: ConvNeXt-small, per-finding attention over 3-slice windows of the Stage B cache; v3 / W4 flags |
| `fusion/infer_windows.py` | the same model at test time (what our submission stage runs) |
| `fusion/score_oof.py`, `fusion/score_gold.py` | paired scoring on held-out studies; gold-58 (report only) |
| `fusion/runs_*/` | held-out predictions of our trained folds (teacher, consensus), the teacher's, and two scoring controls |
| `pipeline/` | DICOM → the Stage B slice cache (seven series slots, 96 slices, 384 px, left knees mirrored; also run at test time); `kcache.py`: K's knee-centred region crops |
| `labeling/` | the training target tables; the OAI MOAKS → twelve-finding mapping (unused unless step 3) |
| `notebooks/build_backbone_ab_kernel.py` | a training kernel (`--recipe v3` / `w4`) |
| `notebooks/build_submit_kernel.py` | our model as a submission stage (sha-checked weights, a reproduction gate, a deadline) |
| `notebooks/build_public_blend_kernel.py`, `build_r6r7_kernel.py`, `build_r6r7_bundle.py` | the public notebook as stage 1, ours as stage 2; R6 + R7 in one decode |
| `notebooks/build_weights_sha_kernel.py`, `submit_at_reset.py`, `offkaggle.py` | weights hashes; a timed submitter; training off Kaggle |
| `notebooks/*_fixture.py` | offline tests (no GPU) |
| `provenance/` | pinned inputs and the site folds |

## How to run

`K=.venv/bin/kaggle`, `PY=.venv/bin/python`, `C=rsna-knee-abnormality-detection`. Images live only on Kaggle; `data/`
holds tables (git-ignored).

```bash
# the K cache: five CPU shards (smoke first: --smoke 10), then one K fold (folds 1..4: --folds k --tag k-f<k>)
$PY notebooks/build_kcache_kernel.py --shard <0..4> --out <dir> && $K kernels push -p <dir>
$PY notebooks/build_backbone_ab_kernel.py --arch convnext_small --corpus stage_k --folds 0 --n-train 0 --recipe k \
    --time-budget-h 10 --targets data/student/targets_teacher_v3.csv \
    --targets-from raahimnawaz/rsna-knee-student-targets --tag k --out <dir>
$K kernels push -p <dir>                       # then read `fit-budget: … -> 16 epoch(s)` in its first minutes

# results (small files only) and the paired fold-0 check
$K kernels output raahimnawaz/<kernel> -p fusion/runs_<name> --file-pattern '.*\.(json|csv|txt|log)$'
$PY fusion/score_oof.py fusion/runs_cnxs_teach1_f0/runs_convnext_small/fold0 fusion/runs_<name>/runs_convnext_small/fold0/avg3

# weights sha256; our model as a submission stage; inside the public blend (see --help for the options)
$PY notebooks/build_weights_sha_kernel.py --kernels raahimnawaz/<training kernel> --out <dir> && $K kernels push -p <dir>
$PY notebooks/build_submit_kernel.py --ours-only --weights-kernel raahimnawaz/<k1>[,<k2>] --weights-sha <hex>[,<hex>] --tag <t> --desc "<d>" --out <dir>
$PY -I notebooks/build_r6r7_kernel.py $PWD data/public_notebooks/heliosli-0-9576-16a9d953 <out> --tag <t> --desc "<d>"
# read in its log before submitting: `sha256 ok` · `GATE PASSED (n/n)` · 0 tracebacks
$K competitions submit $C -k raahimnawaz/<kernel> -v <n> -f submission.csv -m "<what, against what, and the rule>"

# offline tests
$PY notebooks/w4_fixture.py; $PY notebooks/mirror_fixture.py --out <dir>
$PY notebooks/canon_fixture.py --out <dir> && $PY notebooks/infer_parity_fixture.py --fixture <dir>
```
