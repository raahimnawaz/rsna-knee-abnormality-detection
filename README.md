# RSNA Knee Abnormality Detection

Twelve-label knee-MRI classification, macro-AUROC. Final submission **2026-10-22**.

## Read these in order

| doc | what it holds |
|---|---|
| **`PLAN.md`** | strategy, architecture, timeline, efficiency-track maths |
| **`FINDINGS.md`** | measured facts about the data — label coverage, languages, series structure |
| **`IMPROVEMENTS.md`** | **running friction log.** Open decisions, ranked weaknesses, resolved-bug provenance. Read before touching the extractor. |
| `labeling/README.md` | hand-labelling workflow |

## The four facts that shape everything

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
   earned rather than assumed. Measured in `PLAN.md` §7.1.

## Does our extractor actually beat the free one?

Yes, on both references. This is the week-2 decision gate (`PLAN.md` §7), and it is
reproducible with `python extractor/compare_methods.py`:

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
  compare_methods.py     every label source scored head-to-head. the decision gate
  metrics.py             auc / bal_acc, shared. separate module so importing it is safe
  diagnose.py            bootstrap CIs + targeted failure diagnostics
  verify_claim.py        stress-tests the "labels aren't report-derived" claim
labeling/              hand-labelling workflow
  glossary.json          multilingual terms for 12 findings + negation/uncertainty/severity
  compartment_patch.py   adds _side_* / _compartment_struct. IMPROVEMENTS.md 2.2
  build_labeler.py       generates the offline labelling UI
  model_labels.csv       labels produced so far (86/303, all 30 blind gold done)
  item_id_map.csv        item_id -> StudyInstanceUID. the durable fingerprint, no report text
  rekey_labels.py        binds labels to StudyInstanceUID; --check catches sample drift
  eval_model_labels.py   vs gold and vs rule extractor
  error_direction.py     splits disagreement into calibration vs true divergence
notebooks/             Kaggle-side only; the 570 GB of pixels never come local
  kaggle_01_dicom_audit.py    did (0020,0060) Laterality survive? + decode cost per syntax
  kaggle_02_dinov2_cache.py   frozen DINOv2 -> ~800 MB of per-slice features, shardable
eda_01_labels.py       label coverage, prevalence, co-occurrence
eda_02_langs.py        language identification + series structure
eda_03_langid.py       lingua-based language detection (supersedes the heuristic)
eda_04_metadata_baseline.py   series metadata alone scores 0.471 on gold. do not retest
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
python extractor/run_extract.py    # writes data/pseudo_labels.csv
```

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

The images are 570 GB and stay on Kaggle. Two scripts run there, in this order — run the audit
first, because if laterality is unrecoverable then four of the twelve labels are unreliable and
that changes what the feature cache is worth building against.

```
notebooks/kaggle_01_dicom_audit.py    -> /kaggle/working/dicom_audit.json
notebooks/kaggle_02_dinov2_cache.py   -> /kaggle/working/features  (publish as a Dataset)
```

The cache is the reason local work is **no longer text-only**. Frozen DINOv2 embeddings are
~800 MB for the whole corpus, so the §3.3 fusion head — slice transformer, attention pool,
series-type embedding, series attention — trains on a laptop in minutes per experiment. The
backbone is the same one everyone else has; the fusion layer, the labels, and external data are
where the edge has to come from.

`kaggle_02` shards via `SHARD` / `N_SHARDS` and skips finished studies on restart, because
decode is the bottleneck and this will not complete in one session.

## Status

- [x] Data logistics, language ID, series structure
- [x] Rule extractor v1 — macro AUC **0.777** on gold, 95% CI [0.74, 0.82]
- [x] Compartment attribution (`IMPROVEMENTS.md` §2.2) — ~1,000 studies off the flat 0.45.
      Invisible in gold macro (±0.038 CI); justified on corpus evidence, per §0
- [x] Hand-labelling UI + all 30 blind gold studies labelled
- [x] **Our labels beat the public weak labels** on gold and on hand labels — the moat is real
- [x] Series-metadata shortcut tested and rejected (0.471, below chance)
- [ ] Hand-labelling — 86/303 done; the remaining 217 are the only validation set we will have
- [ ] **First LB submission** — fork a public DINOv2 baseline. Nothing else is measurable until
      this exists
- [ ] Laterality confirmed (`kaggle_01`) — four side-specific labels depend on it
- [ ] DINOv2 feature cache built and published (`kaggle_02`)
- [ ] LLM extractor (method B) — host undecided (`IMPROVEMENTS.md` §1.1); ~$44 batched on the
      API, or ~30–90 min a pass on a 5090
- [ ] External data — MRNet, OAI, fastMRI+ cover ~6 of 12 labels with real expert image reads
