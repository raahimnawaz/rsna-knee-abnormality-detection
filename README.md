# RSNA Knee Abnormality Detection

Twelve-label knee-MRI classification, macro-AUROC. Final submission **2026-10-22**.

## Read these in order

| doc | what it holds |
|---|---|
| **`PLAN.md`** | strategy, architecture, timeline, efficiency-track maths |
| **`FINDINGS.md`** | measured facts about the data — label coverage, languages, series structure |
| **`IMPROVEMENTS.md`** | **running friction log.** Open decisions, ranked weaknesses, resolved-bug provenance. Read before touching the extractor. |
| `labeling/README.md` | hand-labelling workflow |

## The three facts that shape everything

1. **58 of 4,407 training studies carry labels (1.3%).** The rest must be derived from the
   free-text radiology reports. The extractor is the critical path, not a side quest.
2. **Reports span 9 languages, 61% non-English** (English, Spanish, Turkish, Croatian, Greek,
   German, Bulgarian, Dutch, French). English-only clinical NLP is useless here.
3. **Gold labels are not a function of the report text** — they look like independent expert
   image reads. Careful full-text reading agrees with gold on 84.7% of labels, but ~2/3 of
   that gap is one-directional threshold error and is recoverable. See `IMPROVEMENTS.md` §2b.

## Layout

```
data/                  competition CSVs + derived label tables      (gitignored)
extractor/             rule-based report -> 12-label extractor
  rule_extractor.py      the extractor (clause-scoped, 9 languages)
  run_extract.py         run over corpus + evaluate on gold
  diagnose.py            bootstrap CIs + targeted failure diagnostics
  verify_claim.py        stress-tests the "labels aren't report-derived" claim
labeling/              hand-labelling workflow
  glossary.json          multilingual terms for 12 findings + negation/uncertainty/severity
  build_labeler.py       generates the offline labelling UI
  model_labels.csv       labels produced so far (86/303, all 30 blind gold done)
  eval_model_labels.py   vs gold and vs rule extractor
  error_direction.py     splits disagreement into calibration vs true divergence
eda_01_labels.py       label coverage, prevalence, co-occurrence
eda_02_langs.py        language identification + series structure
eda_03_langid.py       lingua-based language detection (supersedes the heuristic)
```

## Setup

```bash
python -m pip install kaggle pandas lingua-language-detector
# auth: new-style KGAT_ token at ~/.kaggle/access_token (legacy kaggle.json no longer works)
python -m kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data
python -m kaggle competitions download -c rsna-knee-abnormality-detection -f train_series.csv -p data
python eda_03_langid.py            # writes data/lang_detected.csv
python extractor/run_extract.py    # writes data/pseudo_labels.csv
```

Whole text pipeline runs in ~22 s on CPU. No GPU involved, and none needed until the LLM
extractor and the vision model — see `IMPROVEMENTS.md` §1.1.

## Status

- [x] Data logistics, language ID, series structure
- [x] Rule extractor v1 — macro AUC **0.775** on gold, 95% CI [0.72, 0.81]
- [x] Hand-labelling UI + all 30 blind gold studies labelled
- [ ] LLM extractor (method B) — host undecided, see `IMPROVEMENTS.md` §1.1
- [ ] Preprocessing cache / vision baseline — needs Kaggle or cloud, 570 GB will not fit locally
