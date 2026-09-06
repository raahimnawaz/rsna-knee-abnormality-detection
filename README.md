# RSNA Knee Abnormality Detection

Twelve-label knee-MRI classification, macro-AUROC. Final submission **2026-10-22**.

| | |
|---|---|
| **best submitted** | **0.914** (2026-08-18) — **rank ~1374 of 3,239 as of 2026-09-06** |
| **free public notebook** | **0.936** — 8+ independent teams sit on exactly this score |
| **public frontier** | **0.941**, source public (`dreaddevelopment`, rank 92) |
| **prize line** | 0.940 ≈ rank 100 |
| **baseline** | 0.7229 ± 0.0048, site-grouped report-OOF, n=2,612 |

⛔ **WE ARE 0.022 BELOW THE FREE PUBLIC CEILING. SHIP BEFORE YOU BUILD.** See §S.

---

## Where this stands

Every arm that has ever scored for this project was **downloaded**. Every arm trained here landed
at 0.69–0.77 against a downloaded band of 0.847–0.852.

On 2026-09-06 that gap got a mechanical explanation: **every locally trained model was trained at
0.476 mm/px because `slot_cache`'s protocol slots had `box_mm=None`, while the fork's 20 members
are unanimously `crop_mm=130.0` → 0.387 mm/px.** Read `DECISIONS.md` §R before trusting any
conclusion drawn from a locally trained run — a fair number of them do not survive it.

**In flight:** `fusion/runs_port_repro` — the first training run in this project's history at the
fork's actual config, on the 130 mm cache `data/tiles336_pilk` built 2026-08-19. Fold 0, 30 epochs.

```bash
tail -f fusion/runs_port_repro.log
.venv/bin/python fusion/train_port.py --cache data/tiles336_pilk --tag pilkwang \
  --backbone pilkwang --epochs 30 --run-folds 0 --out fusion/runs_port_repro --resume
```

Fold 0's 691 val studies give SD ≈ 0.008 against a 0.076 gap — about 9σ, so **one fold settles it**
and the other four are not needed to read the answer.

| result | meaning | next |
|---|---|---|
| **≈ 0.84** | the local loop works; the band was the crop | re-measure the arms tagged `CONFOUNDED-160`, starting with OrthoFoundation-L |
| **≈ 0.77** | pixels are genuinely the cause | §3i-4's header-only CPU kernel becomes the critical path |

Either way it is the first uncontaminated number this project has produced from its own training.

## Layout

| path | what it is |
|---|---|
| **`DECISIONS.md`** | **the ledger.** Every settled question, one line, tagged `CLOSED` / `CONFOUNDED-160`. Read before proposing anything |
| `REFERENCE.md` | external ground truth — host rulings, label criteria, forum claims, literature. Every claim carries a source and a read-date |
| `FINDINGS.md` | measured facts about the data — label coverage, languages, series structure |
| `ARCHITECTURES.md` | model/architecture notes |
| `COMPETITION_RULES.txt` | verbatim |
| `pipeline/` | preprocessing, caches, site fingerprints. **Shared with the submission notebook** |
| `fusion/` | training, scoring, arms, audits |
| `extractor/`, `labeling/` | report → label work (retired as a target source, §2f) |
| `notebooks/` | Kaggle kernels — build from the `.py`, never hand-edit the notebook |
| `docs/archive/` | `PLAN.md`, `IMPROVEMENTS.md`, the pre-cut README. Provenance only — **not maintained, do not read in full** |

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
