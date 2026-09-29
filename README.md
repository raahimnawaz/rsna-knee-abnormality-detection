# RSNA Knee Abnormality Detection

Twelve-label knee-MRI classification, macro-AUROC. **The goal is 0.965 AUC** (owner, 2026-09-27): a valid,
pre-registered improvement over the free public 0.943 stack. The prize line and the efficiency track do not
matter. Entry and team-merger deadline **2026-10-15**, final submission **2026-10-22**.

> ## ▶ START HERE: [`RUNBOOK.md`](RUNBOOK.md)
>
> The RUNBOOK is the only place that tracks state: scores, what is running, the next command, and the
> rule each pending result is judged by. It is one page by design. This README holds only what does not
> change from week to week, so it carries no scores. **Where the two disagree, the RUNBOOK wins** -- and
> fix this file.

The local DICOM/NIfTI corpus was deleted on 2026-09-07. Ortho and CoAtNet train and infer on Kaggle; small
CNNs train on the Mac (`--mps`) from the Stage A cache pulled back into `data/stage_a/` (~32 GB, gitignored).

| read | for |
|---|---|
| **`RUNBOOK.md`** | state, the next command, and what to do with its answer |
| `docs/LOG.md` | the detail behind the RUNBOOK since 09-28: read results, diagnoses, owner Q&A (append-only) |
| `DECISIONS.md` | what is settled and what is closed. **Consult it, do not extend it** |
| `MODEL.md` | the architecture; §0 is the public frontier's recipe, read from its code |
| `DATA.md` | labels, the two-tier instrument, Stage 0 / Stage A measurements |
| `PREMORTEM.md` | the known ways this plan fails |
| `REFERENCE.md` | external ground truth -- §2 is the official label criteria |
| `PLAN-C6.md`, `PIPELINE.md`, `FINDINGS.md` | the reasoning behind earlier plans; dated, read for why, not for what next |
| `fusion/archive/`, `docs/archive/` | provenance. Not maintained, do not read in full. `docs/archive/RUNBOOK-2026-09-28.md` is the full RUNBOOK to 09-28; older "RUNBOOK §x" references point there |

---

## Setup

```bash
python -m pip install -r requirements.txt       # .venv is Python 3.12
# auth: new-style KGAT_ token at ~/.kaggle/access_token (legacy kaggle.json no longer works)

# all four CSVs are needed -- eda_01 reads test.csv/test_series.csv too
for f in train.csv train_series.csv test.csv test_series.csv; do
  python -m kaggle competitions download -c rsna-knee-abnormality-detection -f "$f" -p data
done
```

### Regenerating derived data -- `data/` is gitignored, so this is the only record

Competition data is licensed and does not belong in git, so **none of the tables the plan depends
on survive a fresh clone.** Rebuild in this order -- ~2 minutes, no GPU:

```bash
# 1. targets: the public LLM labels (DATA.md §11 closed the choice: steven_v2)
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

**These steps hold their inputs to `provenance/inputs.json`** (`pipeline/provenance.py`). A download
that differs from the record exits and names both hashes instead of silently moving the targets and
folds. `--accept-new` moves the record deliberately; `bench_public_labels.PINNED_VERSIONS` recovers
an old version. `site_fingerprint.py` warns if the count leaves 200-340 -- **take it seriously**, §2j
records a version that silently put 24% of the corpus in one group while reporting a healthy count.

**On a fresh clone, re-key the hand labels before trusting any evaluation.** `labeling/model_labels.csv`
joins on `item_id`, assigned *positionally* after a shuffle, not bound to `StudyInstanceUID`. A
different `lingua` version reshuffles the strata and those 86 labels silently describe the wrong
studies -- hence the exact pin in `requirements.txt`.

```bash
python labeling/sample_for_labeling.py
python labeling/rekey_labels.py --check    # fails loudly if the mapping drifted
```

**To reproduce `DATA.md`'s label audit** (every number in it, ~1 minute, writes nothing):

```bash
python eda_01_labels.py            # writes data/lang_guess.csv   <- eda_03 needs this; run it first
python eda_03_langid.py            # writes data/lang_detected.csv
python extractor/run_extract.py    # writes data/pseudo_labels.csv
python extractor/audit_label_stage.py
```

## The pipeline

Every stage with pixels is a Kaggle kernel, generated by a builder in `notebooks/` that embeds the
repo files it runs (hash-checked on the other side) and records the commit it was built from.
**Commit before building**, or the builder warns that the run is not reproducible.

| stage | code | builder | output |
|---|---|---|---|
| Stage A cache: 64 slices, 2-98%, 140 mm, from DICOMs | `pipeline/build_slots64.py` (`slots64.py` is the spec) | `build_stage_a_kernel.py` | `raahimnawaz/rsna-knee-stage-a-s{0..3}of4` |
| train one encoder on 2.5D windows | `fusion/train_windows.py` | `build_backbone_ab_kernel.py --corpus stage_a` | `runs_<arch>/` with `oof_all.csv`, `last.pt` |
| score, paired, on the 691 fold-0 studies | `fusion/score_oof.py` (locally) | -- | the only number an arm is judged on |
| inference from DICOMs | `fusion/infer_windows.py` | `build_submit_kernel.py` | `submission.csv` |
| the upgraded stack: our CoAtNet student in the public stack's core slot | `notebooks/core_swap_fixture.py` (local test) | `build_submit_kernel.py --core-*` | `submission.csv` |
| AUC vs windows per study, T4 cost | `fusion/window_curve.py` | `build_window_curve_kernel.py` | `curve/n*/oof_all.csv` |

`infer_windows.py --check-against` and `window_curve.py`'s n=54 pass both **re-score studies the OOF
already scored and refuse to continue unless they reproduce it** -- decode, crop, windows, weights,
normalisation and RoPE periods all sit on that one comparison. Both are smoke-tested off Kaggle on
`notebooks/stage_a_fixture.py`'s synthetic DICOMs.

Kaggle limits that shape everything: **30 GPU-h per week**, at most **2 GPU sessions at once**, 12 h
per session, 9 h per submission run. `machine_shape: NvidiaTeslaT4` gives **two** T4s
(`train_windows.py` uses one).

The retired paths -- the DINOv2 feature cache trained on the laptop, the NIfTI mirror -- are in git
history before `cbe737e`; `DECISIONS.md` D3 says why they were dropped.

## Submitting

⚠️ **This is a CODE competition: submit the KERNEL, never the file.** The in-notebook `submission.csv`
is 3 rows off a dummy test set; Kaggle re-runs the kernel on the hidden 1,322 studies.

```bash
python -m kaggle kernels push -p <built dir>              # the commit run, on the 3 visible studies
python -m kaggle kernels status raahimnawaz/<slug>
python -m kaggle competitions submit rsna-knee-abnormality-detection -k raahimnawaz/<slug> -v <n> \
    -f submission.csv -m "..."
```

**Before submitting a blend, grep the commit log for its arm count.** It must name every family you
expect. A guard that refuses an arm produces a valid-looking file that is just the banked path, and
submitting it burns a submission for nothing -- this has happened. `build_submit_kernel.py` prints an
`ARMS:` line and stops the notebook, rather than falling back, if its gate fails. For the upgraded stack, the
`[core] ARMS:` line must name our student and its sha256; the cell before it refuses the run otherwise.

## Working rules

1. **Read `DECISIONS.md` first.** It exists so closed routes are not re-derived and confounded ones
   are not trusted.
2. **Pre-register the gate before the run**, and do not renegotiate it after seeing the number. The
   exception is a proven-miscalibrated instrument, which is a reason to re-measure regardless of which
   way the result went.
3. **Name the question a measurement can answer before believing it.** Every large error in this
   project's history is a real number pointed at the wrong question -- see the closing section of
   `DECISIONS.md`.
4. **Record a negative result at the text that still recommends the route**, not only at the bottom
   of the log. Append-only notes are how a corrected claim keeps steering the next session.
5. **Commit and push after each substantive step.** Sessions restart; unpushed work is invisible to
   the next one.
