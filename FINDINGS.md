# Phase-1 EDA findings — 2026-08-06

Measured from `train.csv` (4,407 rows) and `train_series.csv` (24,371 rows).
Scripts: `eda_01_labels.py`, `eda_02_langs.py`.

---

## 1. Label supervision is essentially absent

| | |
|---|---|
| Training studies | **4,407** |
| Studies with gold labels | **58 (1.3%)** — all 12 labels present, no partials |
| Studies with reports | 4,407 (100%) |
| Test studies (real) | ~1,300 |

**58 labelled studies.** Not "a small subset" in any ordinary sense. You cannot train a
vision model on these, and you can barely validate an extractor with them.

### 1.1 The gold set is enriched, not a random sample

Positive rate among the 58: ACL 41%, MCL 16%, Medial Meniscus 45%, Lateral Meniscus 40%,
Medial OA 26%, Lateral OA 19%, PF OA 36%, Effusion 60%, Synovitis 47%, Baker's 21%,
Contusion 33%, Fracture 31%.

These are far too high and far too even to be natural prevalence — the 58 were clearly
curated to cover every finding. Consequences:

- **Do not estimate corpus or test prevalence from the gold set.**
- **Do not calibrate to it.** (AUC doesn't care about calibration, but any thresholding or
  prevalence-matching heuristic built on these numbers will be wrong.)
- Rarest gold positive is MCL at **n=9**. Per-label extractor validation on this set has
  enormous error bars.

---

## 2. Reports are 9 languages, and gold labels don't cover them

Detected with **lingua** (`eda_03_langid.py` → `data/lang_detected.csv`), median confidence
1.00 in every language, only 4 studies below 0.60.

| Language | Studies | % | Gold-labelled |
|---|---:|---:|---:|
| English | 1,720 | 39.0% | 28 |
| Spanish | 682 | 15.5% | 10 |
| Turkish | 546 | 12.4% | 6 |
| Croatian | 406 | 9.2% | 4 |
| Greek | 321 | 7.3% | 3 |
| German | 262 | 5.9% | 2 |
| **Bulgarian** | 220 | 5.0% | 3 |
| Dutch | 154 | 3.5% | 2 |
| French | 96 | 2.2% | **0** |

Report length: median 977 chars, p05 205, p95 2,453, max 4,743.

**61% of reports are not English, and every non-English language has ≤10 gold-labelled
studies.** French has zero. There is no way to measure extraction quality across most of
this corpus using the provided labels.

### 2.1 Superseded: the first-pass heuristic was wrong for 13% of studies

An initial stopword/medical-term fingerprint disagreed with lingua on **571 / 4,407 (13.0%)**.
Three errors mattered, and they are worth recording because they are easy to repeat:

- **263 Spanish reports were filed as Dutch.** The token `de` was in both probe lists, and
  Spanish medical prose is dense with it. Dutch was inflated 154 → 413; Spanish deflated
  682 → 384.
- **The Cyrillic block is Bulgarian, not Russian.** Confirmed by content — `МР находка`,
  `ставен излив` (joint effusion), `кръстна връзка` (cruciate ligament), `б.о.`
  (unremarkable). A Russian glossary written against it fired on only 4 of 16 keys.
- **A 53-study "Latin" bucket was terse English** — telegraphic reports ("ACL normal. MCL
  normal. Medial meniscus tear.") too function-word-poor for lingua to place until Latin
  was removed from the candidate set.

Portuguese and Italian do not exist in this corpus; both were misdetections.

### 2.2 Greek reports use the MICRO SIGN, not Greek mu

Greek text is written with **µ (U+00B5)** rather than **μ (U+03BC)** — `µηνίσκου`, `µε`,
`σήµα` — a Greek-Windows encoding artifact. Any glossary or regex using the real Greek
letter silently matches nothing. The labeler normalises with NFKD (which folds the two)
plus accent-stripping and final-sigma folding, all index-preserving. This lifted Greek
highlight coverage from 9.2 to 20.0 marks/report and from 10/16 to 16/16 finding keys.

**Assume this class of bug exists elsewhere.** Any downstream extractor must normalise
identically or it will silently under-detect Greek.

---

## 3. Series structure

24,371 series / 4,407 studies → median **5** per study (mean 5.53, range 3–14).

### 3.1 `Fluid_Sensitive` and `Fat_Suppression` are perfectly redundant

|  | Fat_Sup=0 | Fat_Sup=1 |
|---|---:|---:|
| **Fluid_Sensitive=0** | 10,361 | 0 |
| **Fluid_Sensitive=1** | 0 | 14,010 |

Zero off-diagonal. Collapse to a single binary flag → **6 series types** (3 planes × FS/nonFS),
not 12. Simplifies the series-type embedding.

### 3.2 Missing series is the norm, not an edge case

Fraction of studies containing at least one series of each type:

| Series type | Coverage |
|---|---:|
| Axial FS | **100.0%** |
| Sagittal nonFS | 96.8% |
| Coronal FS | 96.4% |
| Sagittal FS | 94.2% |
| Coronal nonFS | 77.3% |
| **Axial nonFS** | **19.4%** |

**87.2% of studies are missing at least one of the six types.** Series-dropout augmentation
and a pooling layer robust to variable series sets are mandatory, not defensive polish.

Axial FS is present in 100% of studies — the one series you can always rely on.

---

## 4. Label correlation structure (n=58 — wide CIs, treat as directional)

Two clinically coherent axes emerge, anti-correlated with each other:

- **Degenerative:** Medial OA ↔ Baker's 0.48, Medial OA ↔ Medial Meniscus 0.42,
  Lateral Meniscus ↔ Lateral OA 0.42, Lateral Meniscus ↔ Medial OA 0.41,
  Lateral OA ↔ Baker's 0.40, Effusion ↔ Synovitis 0.40
- **Acute trauma:** ACL ↔ Contusion 0.38, Contusion ↔ Fracture 0.33, ACL ↔ Fracture 0.27
- **Between axes:** Medial OA ↔ Contusion −0.33, PF OA ↔ Contusion −0.30,
  ACL ↔ PF OA −0.27, ACL ↔ Medial OA −0.26

This is two distinct patient populations (acute injury vs degenerative disease) sharing one
label space. Worth exploiting via multi-task structure and a correlation-aware output
stacker — but **n=58, so ±0.25 error bars.** Re-measure on the pseudo-labelled corpus once
the extractor exists.

---

## 5. What this changes

1. **The report extractor is not a component of the solution — it is the solution.**
   4,349 of 4,407 training labels must be manufactured. Extractor quality is a hard ceiling
   on everything downstream.
2. **Build your own validation set.** 58 studies (9 positives for MCL) cannot validate a
   12-label multilingual extractor. Hand-labelling a stratified 250–400 reports across all
   9 languages is the single highest-leverage manual task available, and most teams won't
   do it. Machine-translate to read them; the labels are what matter, not your fluency.
3. **English-only clinical NLP is off the table.** No NegEx, no CheXbert, no PubMedBERT-EN.
   Needs a multilingual instruction-tuned LLM run offline, cross-checked against a second
   method. Report per-language agreement, never just overall.
4. **4,407 studies is small for a 12-label 3D vision task.** External data is explicitly
   permitted and may be decisive:
   - **OAI (Osteoarthritis Initiative)** — thousands of knee MRIs with expert OA gradings.
     Directly relevant to Medial OA / Lateral OA / PF OA.
   - **MRNet (Stanford)** — 1,370 knee MRIs, labels for ACL tear, meniscal tear, abnormality.
   - **fastMRI+ knee** — annotations over the fastMRI knee corpus.
   - **KneeMRI (Rijeka)** — ACL injury grades; likely the same source as the Croatian reports.
   Pretraining on these, then fine-tuning on the pseudo-labelled competition corpus, is a
   strong play given how label-poor this competition is.
5. **Series handling:** 6 types, not 12. Axial FS always available. 87% of studies miss
   something — train with series dropout from day one.

## 6. DICOM findings — 2026-08-07

From `notebooks/kaggle_01_dicom_audit.py` (200 studies) and
`notebooks/kaggle_01b_patients_laterality.py` (all 4,407). Raw output in `data/dicom_audit.json`
and `data/laterality_check.json`.

### 6.1 There is no JPEG 2000. There is no transfer-syntax mix at all.

| syntax | files | decode |
|---|---:|---:|
| Explicit VR Little Endian | **200 / 200** | **3.1 ms/slice** |

`PLAN.md` budgeted for "uncompressed, JPEG Lossless, JPEG 2000, Implicit VR" and ranked decode
throughput as the largest efficiency lever. Both are retired: `dicomsdl`/`pylibjpeg`/GDCM buy
nothing here, because their advantage is JPEG 2000 and there isn't any.

> **CORRECTION 2026-08-08 — the conclusion drawn from this table was wrong.** It read "~700k
> slices at 3.1 ms = ~36 min single-threaded, against ~4 h of GPU, so the cache build is
> GPU-bound." Four cache-build attempts say otherwise. **This benchmark timed decode, n=6, on
> files it had already opened** — it never measured the open. On the 570 GB mount that open is
> the dominant cost, and there are ~700k of them. The 224 proving shard ran **9 h against a 2.7 h
> estimate**. The build is **I/O-latency-bound**; see `PLAN.md` §6.3.1.
>
> The ~19 ms/open figure quoted in `pipeline/` and `notebooks/` is an **inference from those
> failed runs, not a recorded measurement** — nothing in `data/dicom_audit.json` or
> `data/laterality_check.json` contains it, and `kaggle_01b` (which the code comments cite) does
> not time opens. `kaggle_02`'s PROBE at 25/100/400 series is the instrument that will replace it
> with a real number. **Record that number here when the next shard runs.**

### 6.2 Laterality survives on half the corpus, and the obvious fallback is wrong

| | studies |
|---|---:|
| `(0020,0060) Laterality` present **and non-empty** | **2,203 / 4,407 (50.0%)** |
| present but **empty** (must not be read as a side) | 2,204 |
| usable `ImagePositionPatient` | 4,407 (100%) |

Testing the geometry rule against the 2,203 studies that have both:

| boundary | agreement with tag |
|---|---:|
| `x < 0` → R (the obvious guess) | **89.3%** |
| `x < -62` → R | **97.7%** |

The errors at a zero boundary are lopsided — 98.9% correct on R-tagged studies but only 79.5% on
L-tagged — and every disagreement sits at |x| ≈ 10. The scanned knee is placed at **isocentre**
rather than the patient being centred, so R knees cluster at x ≈ −150 and L knees at ≈ +14, and
the two modes straddle −62, not 0.

Cross-validated (fit 80% / test 20%, ×20): **97.32% ± 0.72%**, threshold stable at **−62.4 ± 4.5**.
It should transfer — the untagged half has a near-identical x distribution (below −62 / between /
above: 51.2/8.9/40.0 tagged vs 53.8/9.9/36.3 untagged).

**Adopted:** tag first, geometry second, and the source is stored per series in the feature cache
so the geometry-derived subset stays identifiable and can be re-measured against the model.

### 6.3 There is no patient linkage in this dataset

`(0010,0020) PatientID` is present in every DICOM — and is **unique per study**: 4,407 distinct
IDs for 4,407 studies, zero patients with more than one. It is de-identified per study.

Nor is shared report text a patient proxy. Those groups are **templates**: the largest is 37
studies sharing one Turkish boilerplate normal report (*"Diz eklemi içi sıvı miktarı normal.
Çapraz ve yan bağlar normal…"*) — 37 different people with identical text.

So there is nothing to group folds by, and grouping on report text actively hurt: it forced fold
sizes to 664–1,077 to prevent a leak that cannot occur, since the model consumes images and the
report is the target's source rather than an input. Ungrouped stratified folds come out 881–882.

### 6.4 Other tags

83 distinct tags survive. `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`,
`SliceThickness`, `InstanceNumber`, `Rows`/`Columns` are all 200/200 — slice ordering by
projected `ImagePositionPatient` is safe. `BodyPartExamined` is 189/200 and **dirty**: values
include `ADRENAL`, `LIVER`, `LSPINE`, `ANKLE`. Do not route on it.

---

### 6.5 The in-plane axes are canonical per plane — which way is medial `MEASURED 2026-08-10`

**Look here before writing any code that says "the medial side of the image".** Measured over
the 396-series geometry sample by projecting each direction cosine onto the nearest signed LPS
axis. The nearest axis is **unanimous, 132/132, for every plane and every axis**:

| plane | col index + | row index + | slice normal | median obliquity | p90 | max |
|---|---|---|---|---:|---:|---:|
| Axial | +x (Left) | +y (Posterior) | +z (Superior) | 4.1–4.9° | ≤15.1° | 41.8° |
| Coronal | +x (Left) | −z (Inferior) | +y (Posterior) | 4.4–8.2° | ≤19.3° | 40.2° |
| Sagittal | +y (Posterior) | −z (Inferior) | −x (Right) | 2.4–7.3° | ≤16.0° | 25.0° |

`kaggle_01c`'s log reports **"distinct IOP rows: 374"** under a comment reading *"if this is ~3
the protocol is clean per plane"*. That number is float obliquity, **not** a mixture of
conventions, and reading it as one would rule out fixed anatomical crops for no reason.

Composed with `canonicalise` mirroring left knees onto `CANONICAL_SIDE = 'R'`, and with a right
knee's medial side facing the midline at +x, the three rules that follow are:

- **increasing column index is MEDIAL** on axial and coronal
- **anterior is LOW ROW index** on axial
- **anterior is LOW COLUMN index** on sagittal

Verified visually as well as arithmetically — a montage of built tiles shows the patella
anterior on axial and sagittal and the condyles above the plateau on coronal. Do the visual
check again if you change the reader: an axis table is exactly the kind of claim this project
has got wrong twice from pure reasoning (K16's first verdict, K18's docstring).

**Sagittal is the exception, and it is why K16 matters.** There medial/lateral is the *slice*
axis, not an in-plane one: the normal is −x, so ascending spatial order runs medial → lateral
for a canonical right knee. That holds only for a volume known to be in ascending order, and a
third of the NIfTI series are not (K16, §2n). `pipeline/slot_cache.py` refuses the sagittal
anatomical slabs while that bit is missing.

**Re-measure if the corpus grows.** n=396, and it is the entire licence for detector-free crops.

---
