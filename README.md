# RSNA Knee Abnormality Detection

Twelve-label knee-MRI classification, macro-AUROC. Final submission **2026-10-22**.

> ## ▶ START HERE: [`RUNBOOK.md`](RUNBOOK.md)
>
> **The plan changed on 2026-09-07 and this README predates it.** `RUNBOOK.md` is the
> operational handoff — current state, the exact next command, and what to do with its answer.
> `PLAN-C6.md` is the reasoning behind it.
>
> Two things below are now stale: the efficiency track is **no longer** the target (the goal is
> the best model, not a placing), and the local corpus was **deleted** — `data/` does not exist,
> and this repo is the only copy of the project. Everything runs on Kaggle now.

**Board pulled live 2026-09-07 — 3,289 teams.** Top **0.954**, prize line (10th) **0.950**,
ours **0.914** at rank **1446**. The table below is the 09-06 pull, kept for comparison.

**Board pulled live 2026-09-06 — 3,243 teams.**

| | |
|---|---|
| **best submitted** | **0.914** — **rank ~1389** |
| **free public notebook** | **0.9360** — **506 teams sit on exactly this**; 877 are ≥ it |
| **rank 100** | 0.9410 |
| **prize line — 10th place** | **0.9500** (was read as 0.940; it moved +0.016 in 25 days) |
| **top** | 0.9540 |
| **efficiency track** | 3 prizes, $18,000 — **now the primary target**, see `PIPELINE.md` |

# ▶️ NEXT SESSION: START HERE

**State on 2026-09-06.** Three things were measured and they redirect the project.

1. **The label stage was re-audited on 09-06 and `DATA.md` rewritten.** The "6.68 : 1 over-call"
   was a binarisation threshold — the same file reads 29.2:1 or 1.00:1 depending on the cut,
   with macro-AUROC 0.8873 throughout. What is real, on the 30 studies that are both gold and
   hand-read: **graded output +0.086** (the incumbent already banks it) and **reading skill
   +0.055, CI [+0.005, +0.111]**. The job is **six labels wide**, and **Synovitis is an
   imputation problem, not an extraction one** — 12.4% mention rate.
   → **`DATA.md`**, reproduce with `python extractor/audit_label_stage.py`
2. **We sample 18 slices, all at `depth 0.5`.** The frontier takes 64 across 6–94% of each stack,
   which is where the lateral meniscus and collateral ligaments are. Ours read 0.6422 and 0.6433;
   ACL, midline, reads 0.8689. → **`MODEL.md`**
3. **Our NIfTI pixel path costs 0.085** against the same config on DICOMs, 14.1σ. **Abandoned.**
   → **`PIPELINE.md`** Stage P. ⛔ **Amended 2026-09-23:** the files are not the cause (the fork's
   weights score the same on them). The port saw one centre window per series. See `DECISIONS.md` D3
   and `RUNBOOK.md` §2b.

**And `§3p` was restored as D4:** our own `targets.csv` converts to LB **~0.938** against a 0.940
prize line, while the model reproduces those same labels at only **0.8486**. We are further from
the labels we have than they are from truth — so **Stage L is cheap and parallel, not the lever.**
The 0.05 is in `MODEL.md`.

**And the plan was reassessed on 09-06 against the live board.** The prize line is **0.950**, not
0.940. This plan's own best case — `MODEL.md`'s "~0.90 gold-58" — converts to LB ~0.939, **rank
~125**. The accuracy track is not reachable on the evidence. The **efficiency track** (3 prizes,
$18k) is: a 0.936 model run in 12 minutes scores better than a 0.950 model run in 30. Both tracks
share the pipeline and diverge only at the two final submissions.

**Do next:** `PIPELINE.md` **S0** — bank 0.936 (one day, one submission, and it is the efficiency
entry ticket), re-read the efficiency formula on the competition page, and post the MRNet/OAI
question that has been drafted and unsent for 27 days.

| read | for |
|---|---|
| **`DATA.md`** | **the data plan** — the per-label map, the two-tier instrument, the paired gate, the schema |
| **`MODEL.md`** | the architecture — 64 slices, 140 mm, per-label attention |
| **`PIPELINE.md`** | stages, gates, fallbacks, budget |
| `DECISIONS.md` | what is settled. **Consult it, do not extend it** |
| `REFERENCE.md` | external ground truth — §2 is the official label criteria, the most useful document here |
| `FINDINGS.md` | measured facts about the data |
| `fusion/archive/`, `docs/archive/` | provenance. Not maintained, do not read in full |

---

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
python extractor/run_extract.py    # writes data/pseudo_labels.csv
```

**Run `eda_01` before `eda_03`** — `eda_03` reads `data/lang_guess.csv` and `eda_01` is the only
thing that writes it.

Once the tables below exist, `python extractor/audit_label_stage.py` reproduces every number in
`DATA.md` in about a minute and writes nothing. **Run it before trusting that document** — it is
what retracted two of the three findings the plan previously rested on.

### Regenerating derived data — `data/` is gitignored, so this is the only record

Competition data is licensed and does not belong in git, so **none of the tables the plan depends
on survive a fresh clone.** Rebuild in this order — ~2 minutes, no GPU:

```bash
# 1. targets: the public LLM labels (§2f, §2i-c)
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

# 2. scanner fingerprints -- no Kaggle run needed, the probe author published the headers
mkdir -p data/external
kaggle kernels output zhukovoleksiy/rsna-metadata-probe -p data/external
mv data/external/headers.parquet data/external/dicom_headers_zhukovoleksiy.parquet
python pipeline/site_fingerprint.py    # expect 265 fingerprints, top 20 = 45.5%

# 3. both fold sets. Build BOTH: the gap between them is the site-leakage number (§2j)
python fusion/folds.py --out data/folds.csv
python fusion/folds.py --group-by site --out data/folds_site.csv
```

`site_fingerprint.py` warns if the count leaves 200–340. **Take it seriously** — §2j records a
version that silently put 24% of the corpus in one group while reporting a healthy count.

**On a fresh clone, re-key the hand labels before trusting any evaluation.**
`labeling/model_labels.csv` joins on `item_id`, assigned *positionally* after a shuffle, not bound
to `StudyInstanceUID`. A different `lingua` version reshuffles the strata and those 86 labels
silently describe the wrong studies — hence the exact pin in `requirements.txt`.

```bash
python labeling/sample_for_labeling.py
python labeling/rekey_labels.py --check    # fails loudly if the mapping drifted
```

## Training

Two machines, one pipeline. Pixels never leave Kaggle; the laptop only sees vectors — except the
slot-pixel caches, which are built locally so the backbone can be fine-tuned.

```
Kaggle   kaggle_01 audit  ->  kaggle_02 cache  ->  publish Dataset   [~2.4 GB]
M5       download  ->  fusion/folds.py  ->  fusion/train.py          [MPS, minutes/experiment]
Kaggle   upload fold*.pt  ->  kaggle_03_submit.py  ->  submission.csv
```

```bash
python fusion/dataset.py                              # self-test: shapes, masks, degenerate studies
python notebooks/kaggle_02_dinov2_cache.py --self-test # scheduler, resume, spawn pool. No DICOMs
python fusion/train.py --synthetic                    # whole loop on random features. No cache
python fusion/train.py --features data/features
```

`--synthetic` is not a toy — it emits the exact shapes, dtypes, masks and edge cases the real cache
produces. On random features it scores macro **0.518** on the 58 gold studies. Chance is the
correct answer there; anything above it means a leak.

### Building the pixel cache locally

> **The local NIfTI corpus was deleted on 2026-09-07** — 19,859 `.nii`, 458.4 GB, stored
> uncompressed. Nothing here reads it any more: D3 abandoned the NIfTI path and the tile caches
> under `data/tiles336*` are already built. The steps below still work, but step 1 is now a
> ~178 GB download, not a no-op. Before paying it, read S5 in PIPELINE.md — the plan is to take
> a public preprocessed pixel dataset instead. `pipeline/validate_nifti.py` re-checks coverage
> against `data/train_series.csv`, which was kept.


```bash
# 1. one NIfTI per series, ~178 GB zipped / ~386 GB extracted. Delete each zip as you go
python -m kaggle datasets download -d davidadekanmi/rsna-knee-nifti-part1 -p data/nifti --unzip

# 2. prove the repackaging matches the DICOMs BEFORE building anything (5 checks)
python pipeline/validate_nifti.py --geometry data/series_geometry.csv --thumbs data/series_thumbs.npz

# 3. build. IMG_SIZE feeds the fingerprint, so 224 and 518 caches cannot be confused
IMG_SIZE=224 caffeinate -dimsu python pipeline/build_cache_local.py --validated --out data/features_224
```

⚠️ **`box_mm` is load-bearing and has silently defaulted to the full 160 mm field once already**
(`DECISIONS.md` §R). Any new slot cache must record `box_mm` in its manifest and it must be
checked against `data/external/pilkwang_weights/manifest.json` before a run is funded.

**The NIfTI mirror is incomplete.** Parts 1–12 and 16–18 exist; 13–15 return 403. That is **81.7%
of studies** and **47 of 58 gold**. Re-check periodically — parts have appeared twice.

**Do not containerise the training.** Docker Desktop on macOS has no Metal passthrough; a
dockerised run trains on CPU. Reproducibility comes from the `requirements.txt` pins plus the
preprocessing fingerprint.

## Submitting

⚠️ **This is a CODE competition: submit the KERNEL, never the file.** The in-notebook
`submission.csv` is 3 rows off a dummy test set; Kaggle re-runs the kernel on the hidden 1,322
studies (~2 h of the 30 h weekly quota).

```bash
kaggle competitions submit rsna-knee-abnormality-detection -k <kernel> -v <n> -f submission.csv -m "..."
```

**Before submitting a blend, grep the kernel log for the arm count.** It must name every family you
expect. A guard that refuses an arm produces a valid-looking file that is just the banked path, and
submitting it burns a submission for nothing — this has happened.

## Working rules

1. **Read `DECISIONS.md` first.** It exists so closed routes are not re-derived and confounded ones
   are not trusted.
2. **Pre-register the gate before the run**, and do not renegotiate it after seeing the number.
   The exception is a proven-miscalibrated instrument, which is a reason to re-measure regardless
   of which way the result went.
3. **Name the question a measurement can answer before believing it.** Every large error in this
   project's history is a real number pointed at the wrong question — see the closing section of
   `DECISIONS.md`.
4. **Record a negative result at the text that still recommends the route**, not only at the
   bottom of the log. Append-only notes are how a corrected claim keeps steering the next session.
5. **Commit and push after each substantive step.** Sessions restart; unpushed work is invisible
   to the next one.
