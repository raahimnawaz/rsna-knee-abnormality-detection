# Extractor: friction log & improvement backlog

Running record of known weaknesses, open decisions, and things to re-check. Updated as work
proceeds. **Read this before touching the extractor** — most of it is already diagnosed.

**Status 2026-08-07:** rule-based extractor running over all 4,407 reports.
Macro AUC **0.777** on the 58 gold studies, 95% CI **[0.74, 0.82]** (±0.038).
Outputs: `data/pseudo_labels.csv` (soft targets), `extract_states.csv`, `extract_evidence.csv`.

Latest change is §2.2 (compartment attribution) + R10. It moved ~1,000 studies off the flat
0.45 `weak` score but is **invisible in gold macro AUC** (0.775 → 0.777, against a ±0.038 CI).
That is the expected shape of a real improvement at n=58, not a disappointment — read §0.

---

## 0. The measurement problem — read first

Every number below rests on **58 studies**. Bootstrap 95% CIs per label:

| label | AUC | 95% CI | width |
|---|---:|---|---:|
| ACL | 0.908 | [0.80, 0.99] | 0.18 |
| Lateral Meniscus | 0.858 | [0.76, 0.94] | 0.17 |
| Baker's | 0.850 | [0.71, 0.95] | 0.24 |
| Medial Meniscus | 0.819 | [0.73, 0.92] | 0.20 |
| MCL | 0.813 | [0.64, 0.98] | 0.34 |
| Fracture | 0.768 | [0.62, 0.88] | 0.26 |
| PF OA | 0.760 | [0.63, 0.88] | 0.26 |
| Effusion | 0.743 | [0.50, 0.71]\* | 0.22 |
| Contusion | 0.741 | [0.61, 0.83] | 0.22 |
| Medial OA | 0.715 | [0.54, 0.84] | 0.31 |
| Lateral OA | 0.714 | [0.51, 0.86] | 0.35 |
| **Synovitis** | **0.607** | **[0.48, 0.73]** | 0.24 |

\* Effusion CI measured pre-fix; re-bootstrap after any further change.

**Consequences that keep biting:**

- **Differences below ~0.15 AUC per label are not resolvable.** Do not tune on these.
- Synovitis, Lateral OA and Medial OA have CIs that touch or nearly touch **0.50** — they
  are not demonstrably better than random on this evidence.
- The macro CI is ±0.038, so the round-1 gain (0.757 → 0.775) is **not statistically
  significant**. It was justified by corpus-level evidence instead — see §3.
- **The 58 gold studies are enriched** (positive rates 16–60%), not sampled. Precision and
  recall at threshold 0.5 are therefore not meaningful in absolute terms; only ranking is.

**→ The hand-labelled set (`labeling/`, 303 items) is the fix for all of this.** Until it
exists, prefer corpus-level evidence (per-language positive rates, state distributions) over
gold AUC when deciding whether a change is right.

---

## 1. Open decisions — need a call

### 1.1 Where does the LLM extractor run? **BLOCKING for method B**
The plan calls for a second, independent extractor to cross-check the rules. Options:

| | Pro | Con |
|---|---|---|
| **Local (GTX 980 Ti, 6 GB, Maxwell sm_52)** | free, private | 6 GB fits only ~7B at 4-bit; no tensor cores, no bf16; PyTorch Maxwell support is deprecated. 4,407 reports would take many hours |
| **Hosted API** | best multilingual quality, fast | costs money; reports leave the machine (de-identified competition data, but check rules) |
| **Kaggle / Colab GPU** | free T4/P100, allowed | session limits; needs the corpus uploaded as a private dataset |

Training-side work, so the no-internet rule does **not** apply. Recommendation: Kaggle GPU
notebook — free, no data leaves Kaggle, and the corpus is already there.

### 1.2 What does "not mentioned" mean?
Currently scored 0.08 vs 0.03 for explicit negation. Radiologists mostly report positives, so
silence ≈ absence — but **not always**, and it varies by institution template. The 4-state
hand-labelling is designed to measure exactly this. Re-tune both constants afterwards.

### 1.3 Soft-target constants are guesses
`pos 0.95 / hedged 0.65 / weak 0.45 / neg 0.03 / absent 0.08`. Chosen by reasoning, never
fitted. Once hand-labels exist, fit them (isotonic or simple grid) against observed positive
rates per state.

---

## 2. Known weaknesses, worst first

### 2.1 Synovitis is barely working — AUC 0.607, recall 0.37 `HIGH`
Gold positive rate **47%**; extractor fires **8.3%**. Per-language positive rate:

```
german 0.8 | bulgarian 1.4 | dutch 3.2 | croatian 3.9 | turkish 4.2
greek 12.5 | english 13.4 | spanish ~15 | french ~9
```

The word is essentially **not used** in German/Bulgarian/Dutch/Croatian reports. Synovitis on
non-contrast MRI is usually *inferred* from the effusion–synovitis complex or synovial
thickening, not named. A term-matching approach cannot recover it.

Options: (a) let the LLM infer it; (b) accept it and let the vision model learn it from the
Effusion↔Synovitis correlation (φ = 0.40 in `FINDINGS.md` §4); (c) derive it as a function of
effusion volume + synovial-thickening terms. **Probably (a) + (b).**

### 2.2 OA compartment attribution `PARTLY FIXED 2026-08-07` — was `HIGH`
1,415 studies landed in the `weak` state for Medial OA and 1,582 for Lateral OA — i.e. OA was
mentioned but **no compartment named**, so all three labels got a flat 0.45. A flat score
across ~1,500 studies contributes nothing to ranking, which is why both sat near 0.71.

**§2c's diagnosis was wrong, and the way it was wrong is the point.** It read this as a
*scope* problem and proposed propagating the compartment from an enclosing section header.
Measured against the corpus, there is usually nothing to propagate: in 6 of 9 languages the
formal phrase (`medial compartment`) appears **nowhere in the report** — Dutch 0.0%,
Turkish 0.8%, German 7.9%, Greek 8.2%, Spanish 10.3% of weak studies. It was §3's pattern
yet again — **the vocabulary was wrong, not the logic.** The compartment is named in the
*same clause*, through anatomy the glossary did not know:

```
german     'Knorpelirregularitäten an der medialen Femurcondyle'      condyle, not compartment
turkish    'Medial tibiofemoral eklem düzeyinde'                      word order flipped
croatian   'degenerativne promjene FT zgloba'                         abbreviated
dutch      'Mediaal femorotibiaal gewrichtscompartiment'              different inflection
turkish    'Lateral eklem aralığında kıkırdakta %50'den fazla kayıp'  joint space
```

**Fix** (`labeling/compartment_patch.py`, `rule_extractor._near`): a compartment is a
laterality adjective (`_side_medial` / `_side_lateral`) within **25 characters** of a
structure belonging to exactly one compartment (`_compartment_struct`: condyle, tibial
plateau, joint space, the femorotibial joint, the compartment itself).

- **Menisci are deliberately excluded** from `_compartment_struct`. The medial meniscus does
  sit in the medial compartment, but a *degenerative meniscal tear is not OA* and
  `_OA_generic` contains `degenerativ` — including it would fire Medial OA on every
  degenerative medial meniscal tear in the corpus.
- **The 25-char window was chosen from the gap distribution and by reading, before looking at
  gold.** Over the 299 studies the rule newly resolves the gap is 3 at the median and 22 at
  p90, then a long tail to 188 that is ~80% false positives — `Patellofemoral compartment
  cartilage: … medial patellar facet` (the medial *patellar* facet is in the patellofemoral
  compartment, not the medial tibiofemoral one), impaction fractures, meniscal degeneration.
  Bare clause co-occurrence is too loose; the side has to actually modify the structure.

| | before | after |
|---|---:|---:|
| Medial OA `weak` | 1,415 | **989** |
| Lateral OA `weak` | 1,582 | **1,176** |
| PF OA `weak` | 1,126 | **933** |

~1,000 studies moved off the flat 0.45. **Gold cannot certify this and was not used to
justify it:** macro went 0.775 → 0.777 AUC / 0.744 → 0.749 bal-acc, and Medial OA's own CI is
[0.611, 0.902]. Per §0 the justification is corpus-level. On the 83 hand labels — the largest
reference — macro bal-acc went 0.850 → **0.862**.

**Still open:** Spanish barely moved (+2 of 213), but see §2.11 — that is a true absence plus
a different bug, not a compartment gap. Self-training the ~1,000 remaining `weak` studies off
the vision model's compartment predictions (§5) is still the endgame.

### 2.3 Bulgarian Baker's may now be over-corrected `MED`
Went 62.7% → **2.3%** after requiring `киста~поплитеал` in one clause. But `поплитеал` appears
in **90/220** Bulgarian reports. The two words are probably often in *different* clauses.
Fix: widen the conjunction scope from clause to sentence-window, then re-measure.

### 2.4 Greek Contusion over-fires — 48.3% vs ~18% elsewhere `MED`
The Greek Contusion list contains `οίδημα` (oedema), which matches **any** oedema — soft
tissue, subcutaneous, muscle — not just bone marrow. Require co-occurrence with a bone/marrow
term: `οίδημα~μυελ`, `οίδημα~οστ`.

### 2.5 Bulgarian PF OA 65.9% and Medial Meniscus 60.0% `MED`
Both far above every other language. Likely `хрущял` (cartilage, in `_OA_generic`) plus any
patella mention → PF OA. Needs the same treatment as 2.4.

### 2.6 Clause splitting is naive `MED`
`re.split(r"[.;\n]+|\s[-–—•]\s")`. Breaks on the Bulgarian abbreviation `б.о.`, on decimals,
and on any list that spans a full stop. Negation scope is clause-local, so a bad split
silently inverts findings. Worth a proper sentence segmenter.

### 2.7 Negation is clause-level with no directionality `MED`
"No tear of the ACL; the MCL is torn" in one clause negates both. Real fix is scope-limited
negation (cue → following N tokens, stopping at contrastive markers like *but/however/ancak/
но/αλλά*). Currently handled only by luck of punctuation.

### 2.8 Laterality of the KNEE is not used at all `LOW for text, HIGH for vision`
The extractor never reads left/right. Irrelevant for text labels, but the vision pipeline
**must** canonicalise handedness or Medial/Lateral labels are meaningless — see `PLAN.md`
§3.2. Confirm `(0020,0060) Laterality` survived the 86-tag allowlist.

### 2.9 Two sources of truth for terms `LOW`
`glossary.json` now serves both the labeler (highlighting) and the extractor (classification),
and the two want different things — `μηνίσκ` in both meniscus lists was right for
highlighting and fatal for extraction. Currently resolved by routing unqualified terms to
`_meniscus_generic`. If this bites again, split into two files.

### 2.10 Unvalidated glossaries for 8 of 9 languages `MED`
Terms came from domain knowledge plus inspection of a handful of reports, not a clinical
lexicon. Croatian and Bulgarian have already been caught using entirely different words than
guessed (§3). **Assume the others have similar gaps.** The per-language positive-rate table in
`run_extract.py` is the detector — a row that is flat or wildly off the others is a bug.

### 2.11 Spanish `_OA_generic` is too loose — the per-language table is flagging it `MED`
After the §2.2 fix, Spanish Medial OA sits at **5.1%** positive against 13–35% in every other
language, and Lateral OA at 4.5%. That is the §3 detector firing. It is *not* a compartment
gap. Of the 210 Spanish studies still `weak`, the OA clause contains:

```
cartílago 146   femorotibial 74   compartiment 44   menisc 40   troclea 36   cóndilo 2
```

Reading them, two distinct faults:

1. **`cartílago` alone is in `_OA_generic`.** It matches every normal report —
   *"Cartílagos de los compartimentos femorotibiales y de la tróclea femoral **sin
   alteraciones**"*. Same class as R6 (`μηνίσκ` in both meniscus lists).
2. **`pinzamiento` alone is in `_OA_generic`.** It means joint-space narrowing in an OA
   context, but the corpus's most frequent use is *"**Pinzamiento** de la almohadilla grasa
   de Hoffa"* — Hoffa fat-pad **impingement**, an unrelated finding.

Neither inflates Spanish *positives* (it is the lowest language); both inflate the `weak`
bucket with clauses that are not OA at all. Fixing it moves those studies `weak` (0.45) →
`absent` (0.08), which is a ranking gain even though no new positive appears. Needs its own
measurement because `_OA_generic` feeds all three OA labels.

Also unaddressed: Spanish genuinely names both compartments together (*"compartimentos
femorotibiales"*, plural, no side) where other languages name one. A both-compartments rule
would be a separate small win.

### 2.12 Greek `χόνδρ` has the R5 prefix bug, unmeasured `LOW`
The `^` word-start marker added for English `chondral` (R10) applies to Greek `χόνδρ` too: it
fires inside `οστεοχονδρινο` and `ενχόνδρωμα` (an enchondroma is a benign bone tumour, not
OA) **20 times out of 64**. Real, but small, and word-start may cost more than it saves on
Greek's heavily prefixed compounds. Measure before changing. Dutch `artrose` inside
`gonartrose`, German `arthrose` inside `gonarthrose`/`retropatellararthrose` and Bulgarian
`артроза` inside `гонартроза` were all checked and are **correct** — that is the substring
matching working as designed.

---

---

## 2b. FINDING: the gold labels are **not** derived from the reports `CRITICAL`

Verified directly against `train.csv` with no merges — this is not a pipeline artefact.

- One gold report states "Medial meniscus tear" in plain text; its Medial Meniscus label is **0**.
  The same study is labelled Lateral OA = 1, Synovitis = 1, Contusion = 1 with **no** textual
  support for any of them ("Cartilages normal", no mention of synovitis or contusion).
- Across the 4 terse-template gold reports, "Medial meniscus tear" appears in 3, with labels
  1, 1, **0**. "Synovitis" is labelled 1 twice where the word never appears.

The dataset description only ever said reports are provided so you "may wish to derive"
labels — it never claimed the gold came from them. The gold looks like **independent expert
image review**, and the private test labels are presumably produced the same way.

**Consequences:**
1. Report extraction has a **hard ceiling well below AUC 1.0**. The rule extractor's 0.775
   may be closer to that ceiling than to a failure.
2. This is a **noisy-label weak-supervision** problem, not a text-parsing problem. Past a
   point, further extractor polish buys nothing — invest in the vision model and in
   noise-robust training instead.
3. Do not chase perfect report agreement. Chase *calibrated* soft labels.

### 2b-i CORRECTION — the first version of this finding was overstated

Two flaws in the original evidence, both caught by re-checking:

1. **I read truncated reports.** 34 of 58 gold reports exceed the 1,100-char dump cut, so
   "no textual support" was unsafe for most of them. *(The specific case cited — the
   `Patella type 3 Wiberg` report — is 443 chars total against 420 read, so that one
   survives: full text says "Medial meniscus tear" and "Cartilages normal" while the labels
   are Medial Meniscus 0, Lateral OA 1, Synovitis 1, Contusion 1.)*
2. **My contradiction probe used naive regex.** It flagged a report as lacking medial-meniscus
   evidence when the text plainly read *"Extensive complete tearing of the body, posterior
   horn, and posterior root medial meniscus"* — it only wanted the literal string
   "medial meniscus tear". Exactly the failure mode §2.10 warns about.

Tests that could NOT be run: no two gold studies share identical report text, so the
strongest test — identical text mapping to different labels — is unavailable. 177/4,407
studies do share a report with another study, but none among the gold.

### 2b-ii Measured ceiling, n=31 gold — and how much of it is recoverable

Careful full-text reading agrees with gold on **84.7%** of labels (57 errors / 372 cells).
But **the direction of those errors matters more than the count**:

| label | I said yes, gold no | I said no, gold yes | reading |
|---|---:|---:|---|
| **Effusion** | **10** | **0** | pure threshold — gold ignores small/mild/minimal effusion |
| **Synovitis** | 2 | **8** | gold infers it from images; report rarely names it |
| **Fracture** | **6** | 2 | gold does not count osteochondral impaction / insufficiency / subchondral fracture |
| **Lateral OA** | 4 | 1 | I over-call |
| Lateral Meniscus | 1 | 3 | I under-call |
| PF OA | 4 | 3 | genuinely mixed |
| Contusion | 2 | 2 | genuinely mixed |
| ACL / MCL / Medial Meniscus / Medial OA / Baker's | 1/2/1/2/2 | 0/0/1/0/0 | few errors, mixed |

**37 of 57 errors sit in one-directional labels** — those are *calibration*, not divergence,
and a threshold move recovers them. Only **20 errors** are genuinely mixed-direction.

> **So the report-only ceiling is nearer ~95% than ~85%.** The earlier claim that "further
> extractor polish buys nothing" was **wrong**. Calibrating Effusion severity and the
> Fracture definition alone addresses 16 of 57 errors.

Caveat: ~95% assumes an oracle per-label threshold, so it is an upper bound; the realistic
gain is smaller. And n=31 with 3–17 positives per label means every per-label count here has
a wide interval. Treat the *directions* as signal and the *magnitudes* as provisional.

**What still stands:** gold is not a function of report text. Genuine contradictions exist
(the `Wiberg` case), Synovitis is inferred rather than read, and gold applies severity
thresholds the text does not state. It remains a noisy-label problem — just a considerably
less noisy one than the first pass suggested.

### 2b-iii Structural quirk: some studies carry more than one report
`it0116` contains two concatenated MRI reports under a single StudyInstanceUID (flagged in
the data itself as `[BILATERAL NOTE: two reports filed under this study]`). Any extractor
that assumes one report = one exam will mis-scope negation and merge findings across two
sittings. Count how many studies are affected before training on report-derived labels.

---

## 2c. Model-vs-rule disagreements (n=62 labelled) → two concrete extractor bugs

Mean agreement 0.883. The outliers are diagnostic:

| label | model says pos | rule says pos | agreement | reading |
|---|---:|---:|---:|---|
| **PF OA** | 29 | 13 | 0.645 | rule finds **less than half** |
| **Medial OA** | 26 | 11 | 0.726 | rule finds **less than half** |
| **Lateral OA** | 17 | 6 | 0.790 | rule finds **a third** |
| **Contusion** | 10 | 22 | 0.774 | rule finds **twice too many** |

- **OA under-detection** is the `weak` state from §2.2 — when OA is named without a
  compartment, the rule assigns a flat 0.45 that never crosses threshold. Reading the
  surrounding context resolves the compartment most of the time. ~~Fix: propagate compartment
  from the enclosing section header (`MEDIAL COMPARTMENT:` / `Mediales Kompartiment:` /
  `compartimento femorotibial medial`) rather than requiring it in the same clause.~~
  **Superseded 2026-08-07 — that fix was aimed at the wrong thing.** Those headers barely
  exist outside English and Dutch; the compartment is in the same clause, named by anatomy
  the glossary lacked. See §2.2 for what was actually wrong and what was done.
- **Contusion over-detection**: the rule counts *any* marrow oedema. Gold and careful reading
  both distinguish **traumatic** contusion from **degenerative/reactive** subchondral oedema
  and from oedema **adjacent to a fracture** (gold labels that Contusion = 0). Fix: require a
  trauma context term, or exclude when `subchondral`/`reactive`/`degenerative` qualifies it.

Intra-rater consistency: **100%** on the 2 duplicate pairs seen so far (small, but clean).

---

## 3. Resolved (kept for provenance — these are the failure *patterns* to watch for)

| # | Issue | Root cause | Fix |
|---|---|---|---|
| R1 | 263 Spanish reports filed as Dutch | `de` in both probe lists | replaced heuristic with `lingua` |
| R2 | Cyrillic glossary fired 4/16 keys | corpus is **Bulgarian**, not Russian | rewrote in Bulgarian |
| R3 | Greek matched nothing | reports use **µ U+00B5**, not **μ U+03BC** | NFKD normalisation, index-preserving |
| R4 | `цялост` negated every Bulgarian tear | appears in *preserved* AND *disrupted integrity* | removed bare cue |
| R5 | `normal` matched inside `abnormal` | substring cue matching | cues require word-start; stems still substring |
| R6 | Greek menisci 100% identical | `μηνίσκ` in both side lists | removed; routed to `_meniscus_generic` |
| R7 | Croatian OA compartments 0.0% | corpus says `femorotibijaln`/`kompartm`, never `odjeljak` | co-occurrence terms |
| R8 | Bulgarian Baker's 62.7% | bare `киста` matches any cyst; `Бейкер` appears 0/220 | `киста~поплитеал` (see 2.3) |
| R9 | Effusion AUC 0.604 | severity ignored; 32% of hits qualified trace/minimal | `minimal` cue class → downgrade to hedged. **0.604 → 0.743** |
| R10 | English `chondral` fired 1,019× non-word-initially | substring stem is a suffix of `subchondral` (846) and `osteochondral` (170), so subchondral oedema / insufficiency fracture / bone island all read as cartilage pathology | `^` word-start marker → `^chondral`. Checked all 9 languages; English is the only one that needed it (see §2.12) |

**The recurring pattern:** guessed vocabulary is wrong far more often than the logic is. Every
single one of R2/R6/R7/R8 was found by the per-language positive-rate table, not by gold AUC —
and §2.2 is the same story a fifth time, with the extra twist that the *diagnosis* in §2c had
already committed to a logic fix (section-header scope) that the corpus does not support.
Check the vocabulary against the corpus before designing the rule. Keep that table in every run.

R10 was found neither way: it surfaced from **auditing what a change newly fires on**, by
reading 25 clauses. Worth repeating after any rule that moves a few hundred studies — the
per-language table is too coarse to see a fault that is spread evenly across one language.

---

## 4. To check when the hand-labels land

1. Re-bootstrap all CIs on 303 items instead of 58 — expect widths to roughly halve.
2. **Per-language** extractor agreement. This is the number that does not currently exist and
   matters most.
3. Human-vs-gold agreement on the 30 blind gold studies → are our label *definitions* right?
4. Intra-rater agreement on the 20 duplicate pairs → the ceiling on everything.
5. Refit the soft-target constants (§1.3) against observed positive rates per state.
6. Re-measure whether "not mentioned" ≈ "negated" (§1.2), per language and per institution.

## 5. Later, once the vision model exists

- Self-training: use compartment predictions to disambiguate the ~1,500 unattributed OA
  studies (§2.2), then retrain.
- Disagreement mining: studies where rules and LLM disagree are the highest-value candidates
  for additional hand-labelling.
- Check whether pseudo-label noise is correlated with language — if so, weight the loss by
  per-language extractor reliability rather than uniformly.
