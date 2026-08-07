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
