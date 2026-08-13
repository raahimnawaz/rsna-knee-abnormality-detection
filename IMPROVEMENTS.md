# Friction log & improvement backlog

Running record of known weaknesses, open decisions, and things to re-check. Updated as work
proceeds. **Read this before touching the extractor** — most of it is already diagnosed.

§0–§5 are the **extractor** track and are the bulk of this file. §6 is the **Kaggle-side
pipeline** track, added 2026-08-08: same format, same purpose, different failure modes. Read it
before touching `pipeline/preprocess.py` or either cache notebook — every entry in it has cost or
would have cost a GPU session.

> **STATUS 2026-08-10 — read §2f before anything else in this file.** The rule extractor is
> **last of six** label sources on gold-58: 0.777 against a free public 0.893, losing on 12/12
> labels and reducing a rank-mean it is added to. The extractor track is **closed as a source
> of training targets**; §1.1, which blocked it, is closed with it. Most of §2.1–§2.12 below
> describes bugs in a component that is no longer on the critical path — kept for provenance
> and because the *patterns* still apply, but do not spend a day on any of it.

**Status 2026-08-07 (superseded, kept for provenance):** rule-based extractor running over all
4,407 reports. Macro AUC **0.777** on the 58 gold studies, 95% CI **[0.74, 0.82]** (±0.038).
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

### 1.1 Where does the LLM extractor run? `CLOSED 2026-08-10` — it does not have to run at all

**The answer is `kaggle datasets download`.** Four LLM-read label tables are published as free
public Datasets, and the best of them scores **0.893** on gold-58 against our rules' 0.777 —
see §2f, which also shows ours losing on 12/12 labels and *subtracting* from a rank-mean. So
method B is not a build; it is a download, and the decision this entry was blocking on turned
out to be the wrong question. **`data/pseudo_labels.csv` is retired as a training target.**

What the rule extractor is still for: it is the only source here whose per-clause evidence is
inspectable (`extract_evidence.csv`), so it stays as a **disagreement detector** — where it and
the LLM readers diverge is where a report is genuinely ambiguous, which is a per-study
confidence signal worth testing. That is a hypothesis, not a result; the one fusion test run so
far (§2f) says it does not help as a *label*.

The original options are kept below because the reasoning was sound and the blocker was real;
what it missed was that the corpus is shared and someone else would pay the cost first.

| | Pro | Con |
|---|---|---|
| **Local (GTX 980 Ti, 6 GB, Maxwell sm_52)** | free, private | 6 GB fits only ~7B at 4-bit; no tensor cores, no bf16; PyTorch Maxwell support is deprecated. 4,407 reports would take many hours |
| **Hosted API** | best multilingual quality, fast | costs money; reports leave the machine (de-identified competition data, but check rules) |
| **Kaggle / Colab GPU** | free T4/P100, allowed | session limits; needs the corpus uploaded as a private dataset |

| | Pro | Con |
|---|---|---|
| **Local (GTX 980 Ti, 6 GB, Maxwell sm_52)** | free, private | 6 GB fits only ~7B at 4-bit; no tensor cores, no bf16; PyTorch Maxwell support is deprecated. 4,407 reports would take many hours |
| **Hosted API** | best multilingual quality, fast | costs money; reports leave the machine (de-identified competition data, but check rules) |
| **Kaggle / Colab GPU** | free T4/P100, allowed | session limits; needs the corpus uploaded as a private dataset |

Training-side work, so the no-internet rule does **not** apply. Recommendation: Kaggle GPU
notebook — free, no data leaves Kaggle, and the corpus is already there.

### 1.2 What does "not mentioned" mean? `MEASURED 2026-08-09` — was open
**Silence is 2.3× more likely to hide a true positive than an explicit negation is**, and the
scores say 1.3×. Measured against gold: `absent` → **0.167** [0.132, 0.209], `neg` → **0.073**
[0.029, 0.173]. The direction of the guess was right and the size was not.

The reason this matters more than the other four constants: **`absent` is 52.4% of the target
matrix** — 365 of 696 gold cells, and 62.1% across the full corpus. It is not a corner case, it
is the modal training target, and it was set to less than half its measured value.

It is also the constant that most clearly **cannot be one number**. Per label, `absent` runs
from 0.031 (ACL) to 0.372 (Synovitis). That spread is the §2.1 ceiling stated in target terms:
87.6% of reports never mention synovitis and 37% of those knees have it, so a flat 0.08 asserts
"no synovitis" on 43 of 58 gold studies while 16 are positive. Nothing downstream can recover
from a target that wrong; §2.1's option (b) — let the vision model learn it off Effusion — is
now the *only* option, because the alternative is training against noise labelled as certainty.

### 1.3 Soft-target constants are guesses `FITTED 2026-08-09` — `extractor/calibrate_states.py`
`pos 0.95 / hedged 0.65 / weak 0.45 / neg 0.03 / absent 0.08`, chosen by reasoning. Measured
P(gold=1 | state) over the 58 gold studies, 696 cells:

| state | cells | share | P(gold=1) | 95% CI | SCORE | delta |
|---|---:|---:|---:|---|---:|---:|
| pos | 182 | 26.1% | **0.747** | [0.679, 0.805] | 0.95 | −0.203 |
| hedged | 52 | 7.5% | **0.558** | [0.423, 0.684] | 0.65 | −0.092 |
| weak | 42 | 6.0% | **0.238** | [0.135, 0.385] | 0.45 | −0.212 |
| neg | 55 | 7.9% | **0.073** | [0.029, 0.173] | 0.03 | +0.043 |
| absent | 365 | 52.4% | **0.167** | [0.132, 0.209] | 0.08 | +0.087 |

**The ladder is monotone in the right direction — that is the extractor's state machine passing
an independent test — but every rung is in the wrong place.** It is compressed at both ends and
stretched in the middle: the two confident states are too extreme, and the two uncertain states
(`weak`, `absent`) are pushed toward the negative rail when the data puts them well inside it.

`pos` at 0.747 is the one to read carefully, because **it is not extractor error.** Gold is an
independent *image* read. When a report says "ACL tear", the image reader agrees three times in
four; the last quarter is genuine report-vs-image disagreement and no extractor work removes
it. 0.95 claims a certainty the modality does not have. This is the quantitative form of README
fact 3 — gold labels are not a function of the report text — and it caps what §7.2 can show.

**Fit on gold, not on the hand labels.** §1.3 originally said to fit these once hand-labels
exist. That works for `pos`/`hedged`/`weak` and is wrong for `absent`: the hand labels are read
from the *report*, so P(hand=1 | absent) measures extractor recall — "did a careful human
reading the same text also see nothing". The training target's job is to predict the **image**,
and only gold is an independent image read. The two questions diverge exactly where the answer
matters, which is why `pos` comes out at 0.747 rather than near 1.

**Per-label tables are shrunk, not fitted.** Per-label `absent` cells run 4–50 (Effusion has 4),
so raw per-label rates are anecdote. `calibrate_states.py` shrinks each toward the pooled rate
with a Beta prior whose strength is chosen by leave-one-out log loss. Treat the per-label column
as directional and the pooled column as the result.

> **CORRECTION 2026-08-09.** This said the search "picks **m = 20 pseudo-counts**", which is the
> value for the POOLED fit. The cross-fitted tables each run their own search and they do not
> agree: `{'0': 20.0, '1': 50.0, '2': 20.0, '3': 50.0, '4': 50.0}`. So the five folds of the §1.3a
> arm differed in shrinkage strength — an uncontrolled nuisance across the very folds that
> comparison reads as one arm. It does not overturn §1.3a (the effect is ordered by
> `absent_raise`, which is a per-label quantity, and three of five folds share m), but the next
> run of that experiment should pin `--m` explicitly so the arms differ by one thing.

**Using this cannot be allowed to burn gold.** Fitting on all 58 and training on the result
would make the pooled-OOF gold macro a fitted number rather than a held-out one — the exact
failure §0 and §C5 of the research notes warn about. So the tables are **cross-fitted**: one per
fold, each fitted only on gold *outside* that fold. Across folds they are stable (`pos`
0.707–0.779, `absent` 0.151–0.184), which is itself evidence the pooled fit is not noise.
`fusion/train.py --calibrated-targets` consumes them and refuses a non-cross-fitted file. It is
**opt-in**, so the §7.2 label A/B keeps comparing one thing at a time.

### 1.3a The calibration was TESTED and it LOST. `MEASURED 2026-08-09`

`0.743 → 0.699` on the 224 cache, 37 gold studies, identical folds and seed. **The recalibrated
targets are worse than the guessed ones**, and the mechanism is measurable rather than inferred:

> **BASELINE CORRECTION 2026-08-09 (later).** The `0.743` arm is **no longer reproducible** and
> the current default-target baseline on the same cache is **0.719**. Two things happened after
> it was measured: K17 destroyed the run's artefacts, so old-vs-new OOF cannot be diffed; and the
> `_nifti_axes` fix deleted 43 mis-reformatted cache entries, leaving those studies with fewer
> series. Two identical re-runs on the post-deletion cache give **0.719 and 0.719** — every
> per-label AUC agreeing to three decimals — so **training is fully deterministic on MPS** and
> the 0.024 is attributable to the deleted entries, not to run noise. Removing corrupt series
> lowered the score because it removed *data*, not because the corruption helped.
>
> That determinism is worth more than the correction itself: **every A/B in this file measures a
> real difference, with no run-to-run floor underneath it.** The ±0.038 bootstrap CI is still
> the sampling limit on n=37 and still governs whether a difference *generalises* — but it is
> not masking nondeterminism. Re-read the ladder comparison as `0.719-equivalent → 0.699`; the
> sign and the mechanism below are unaffected, since both arms shared folds, seed and cache.

| label | absent 0.08 → | share `absent` | ΔAUC |
|---|---:|---:|---:|
| Synovitis | 0.307 | 87.6% | **−0.182** |
| Lateral OA | 0.139 | 55.8% | −0.110 |
| Fracture | 0.173 | 79.6% | −0.099 |
| PF OA | 0.211 | 45.1% | −0.087 |
| Lateral Meniscus | 0.212 | 59.5% | −0.076 |
| Contusion | 0.151 | 56.9% | −0.057 |
| Effusion | 0.223 | 17.2% | −0.006 |
| MCL | 0.091 | 94.5% | +0.000 |
| Medial OA | 0.160 | 53.1% | +0.000 |
| ACL | 0.084 | 82.7% | +0.009 |
| Medial Meniscus | 0.194 | 45.8% | +0.031 |
| Baker's | 0.107 | 67.7% | **+0.043** |

```
corr(absent_raise × absent_share, ΔAUC) = −0.776
corr(absent_raise,                ΔAUC) = −0.630
```

**How much a label's `absent` target was raised predicts how much AUC it lost.** Synovitis was
raised furthest across the largest share of the corpus and lost most; Baker's was raised least
and gained. A random effect would not order itself that way, which is what makes this more than
the n=37 noise floor — the *macro* delta of −0.044 on its own would not be readable.

**Why, and this is the part worth keeping.** §1.3 fitted P(gold=1 | state) and treated it as the
right training target. But the `absent` bucket is **heterogeneous** — it mixes true positives and
true negatives — and assigning it its mean teaches the model the mean instead of the
discrimination. The extractor's 0.08 is badly calibrated and strongly *separating*; macro AUC is
a ranking metric and only rewards the separation. **Better calibration, worse ranking.**

So §1.3's measurement stands and its conclusion does not. P(gold=1 | absent) = 0.167 against a
0.08 target is still a fact; `pos` = 0.747 is still real report-vs-image disagreement. What is
falsified is that correcting those improves rank ordering.

**The research notes were righter than the fix.** They said `NOT_MENTIONED = MASKED, not 0`.
Masking was judged too aggressive here — 94.5% of MCL cells would vanish — and re-targeting was
taken as the moderate middle. It is not a middle: masking removes the heterogeneous bucket from
the loss, re-targeting actively trains toward its average. Opposite treatments.

**Next experiment, now well-posed:** mask `absent` from the loss rather than re-target it, on the
labels where it dominates. If the mechanism above holds it should beat both arms. Run it against
the 0.743 baseline on the fuller cache — at n=37 it cannot be resolved.

`--calibrated-targets` stays in `fusion/train.py` as the harness that produced this, not as a
recommended setting. **Do not enable it.**

Still open: whether masking beats the guessed ladder.

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

### 2.8 Laterality of the KNEE is not used at all `LOW for text, PARTLY FIXED for vision`
The extractor never reads left/right. Irrelevant for text labels, but the vision pipeline
**must** canonicalise handedness or Medial/Lateral labels are meaningless — see `PLAN.md`
§3.2. Confirm `(0020,0060) Laterality` survived the 86-tag allowlist.

> **CORRECTION 2026-08-09.** This item read as satisfied by `canonicalise()`. It was satisfied
> for **half the corpus**: axial and coronal only. Sagittal carries medial/lateral on the slice
> axis, which nothing in the pipeline reversed — 40.5% of series, 43.0% of them left knees.
> See **K18**. Do not treat this item as closed until `SAGITTAL_LR_SLICE_FLIP` is on.

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

## 2d. The vision model's ceiling is resolution, not labels `MEASURED 2026-08-09`

> **TESTED 2026-08-10 — the mechanism holds, the magnitude does not, and the title is wrong.**
> A 1,000-study cache was built at 518 (3.88 h) and trained against the *same* 1,000 studies at
> 224, identical folds and seed. Result: **+0.013 macro**, a third of the bottom of the predicted
> +0.03–0.07 range and well inside the ±0.038 CI. Direction was right where it was called —
> Lateral Meniscus **+0.067** (n=18) and Fracture **+0.051** (n=13), the two labels named in
> advance as the sharpest tests — and the predicted group averaged +0.034 against −0.008 for the
> rest. Restricted to labels the CI can see (n_pos ≥ 12) that narrows to +0.024 vs +0.006.
>
> So resolution is real, concentrated where predicted, and **too small to build a plan around**.
>
> **The finding that replaces it:** the same run measured 1,000 → 2,649 studies at 224 as
> **+0.024**. *Data is worth about twice resolution*, and the corpus sits at 60% of 4,407. That
> inverts the ranking this section argued for and promotes §3.4's external data.
>
> What survives untouched is the part below that rules out the **labels** as the ceiling —
> Spearman −0.17 (p=0.60) is a property of the label/vision relationship, not of resolution.
> What was wrong was inferring from "not labels" that it must be resolution, and then reordering
> the whole route on an untested inference. The 518 rebuild is now explicitly **not** planned
> (README "Where this goes next").

The obvious reading of a 0.719 macro is that the pseudo-labels cap it: the model is trained on
extractor output, so it cannot be better than the extractor. **That is measurable, and it is
false.**

Per-label, extractor-vs-gold against vision-OOF-vs-gold, same 12 labels:

| label | extractor | vision | Δ |
|---|---:|---:|---:|
| Lateral Meniscus | 0.858 | 0.526 | **−0.332** |
| Fracture | 0.768 | 0.494 | **−0.274** |
| ACL | 0.908 | 0.702 | −0.206 |
| Medial Meniscus | 0.819 | 0.634 | −0.185 |
| Contusion | 0.741 | 0.603 | −0.138 |
| MCL | 0.813 | 0.694 | −0.119 |
| PF OA | 0.760 | 0.767 | +0.007 |
| Baker's | 0.850 | 0.919 | +0.069 |
| Synovitis | 0.607 | 0.685 | +0.078 |
| Lateral OA | 0.714 | 0.824 | +0.110 |
| Effusion | 0.743 | 0.863 | +0.120 |
| Medial OA | 0.715 | 0.913 | **+0.198** |

**Spearman between the two columns is −0.17 (p=0.60).** Label quality does not predict vision
performance at all. Lateral Meniscus has the second-*cleanest* labels in the set and vision sits
at chance; Synovitis has the *worst* and vision beats it by +0.078. A label ceiling would bind
only if the noise were correlated with image appearance, and over 4,349 studies it plainly is
not — the model exceeds its own training labels on six of twelve.

**What the split tracks instead is the physical size of the finding.** The six vision wins are
centimetre-scale and high-contrast: osteophytes and joint-space narrowing, bulk fluid, a
popliteal cyst. The six losses are millimetre-scale structural lines: a meniscal tear, a
non-displaced fracture, ligament fibre discontinuity. *(The size reading is anatomical
interpretation; the correlation above is not.)*

**And the 224 cache resolves 0.71 mm/px.** `normalise_and_resample` resamples to
`TARGET_MM` = 0.35 and centre-fits to `round(FOV_MM/TARGET_MM)` = **457 px** — the full 160 mm
knee, correctly — and then `imagenet_normalise` interpolates that straight down to `IMG_SIZE`:

| IMG_SIZE | mm/px over the 160 mm FOV | tokens/slice |
|---:|---:|---:|
| 224 | **0.714** — the 0.35 mm resample is discarded | 261 |
| 518 | 0.309 ≈ as designed | 1,374 |

At 0.71 mm/px a meniscal tear line or a non-displaced fracture line is about one pixel. That is a
resolution-limited signature, and it says the 518 rebuild is a **larger lever than the ordering
fixes** — K16 and K18 target the four medial/lateral labels, resolution targets six including
the two sitting at chance.

**Consequence for sequencing.** The ordering fixes require a cache rebuild anyway. Rebuild at
518 rather than 224 and the two land in one job instead of two. Before committing the ~16 h,
`--limit` keeps all gold, so a 518 subset build (~5 h) compared against the *same* subset of the
existing 224 cache (free) isolates resolution from corpus size and answers it the same day.

---

## 2e. The architecture cannot fine-tune, and that is the gap `MEASURED 2026-08-10`

Read from `pilkwang`'s notebook directly (`kaggle kernels pull`), not from its description:

```python
UNFREEZE_LAST = 6          # trainable transformer blocks, from the output end
LR_BACKBONE   = 8e-6       # the encoder is adapted, not retrained
LR_HEAD       = 1e-3
```

**It fine-tunes the last six encoder blocks. Our design makes that structurally impossible.**
Caching frozen DINOv2 embeddings once and training a head on the vectors means the encoder can
never adapt — at any resolution, under any head, with any labels. A self-supervised
natural-image ViT is far out of distribution on knee MRI, and the late blocks are exactly where
the adaptation would happen.

That single difference explains the results this project has been getting. Resolution moved
+0.013 because frozen features are frozen at every resolution (§2d). The labels measured out as
not-the-ceiling (§2d). Everything lands near 0.70 regardless of what changes downstream, because
the one thing that would move it is the one thing the architecture forbids.

**Three claims in README §6 were also wrong**, all from reading the notebook's description
rather than its code:

| claimed | actual |
|---|---|
| ensembles DINOv2 **and** EfficientNet | one backbone, DINOv2 only |
| across 224/336 | **correct — I was wrong to deny this.** `RUNS = [{"img":224},{"img":336}]`; `CACHE_IMG` is the cache size, not the only run |
| "our backbone is already the same checkpoint theirs is" | theirs is DINOv2 **small**; ours is **base** with registers |

So it beats us with a *smaller* backbone. We are not behind on capacity. We are behind on being
able to train the thing.

> **AND THE FORK AS ATTACHED DOES NOT TRAIN AT ALL.** `main()` calls `find_weights()`, and when
> `pilkwang/rsna-knee-weights` is mounted it takes `infer_from_package()` and returns —
> 20 pre-trained members, rank-meaned, 74 seconds. The 0.891 is *inference from published
> weights*, not a training run. So the first attempt at Phase 0 — swapping the label table —
> changed nothing, because no training happened for the labels to enter. To test labels the
> weights package must be **detached** so the training path executes, which costs a full ~8 h
> Kaggle run per arm against a 30 h weekly quota. That is the argument for doing the local
> training port *first*: two arms locally is ~4 h and free.

It also already ships what our Phase 3 listed as differentiators: `SlotHead` is per-diagnosis
attention over slot embeddings with per-target priors (`SLOT_PRIOR_STRENGTH = 0.55`) — the
per-pathology query tokens — plus confidence-weighted targets (`W = 0.25 + 0.75 * conf`),
multi-window TTA and rank-mean ensembling.

### Why this is good news: the compute is small, and the advantage is ours

`N_SLOT = 6` (`SAG_FLUID_FS`, `COR_FLUID_FS`, `AX_FLUID_FS`, `SAG_FLUID_NOFS`, `COR_T1`,
`SAG_T1`), `GROUP = 3` slices stacked as RGB. So a study is **6 encoder inputs**, not the ~155
slices ours embeds. The fork gets more out of ~20x less encoder work by adapting the weights
instead of pooling many frozen slices.

Estimated on the M5 from our own measured 9.9 img/s for base@518: DINOv2-small at 336 is ~0.26x
the parameters and ~0.42x the tokens, so ~90 img/s forward, and ~36 img/s training with six
blocks open. 4,407 studies × 6 slots = 26,442 images → **~12 min/epoch, ~2 h for the fork's
`EPOCHS = 10`.**

Against Kaggle's `TIME_BUDGET = 8 h`, a 30 h weekly quota and a GPU lottery that refuses four
draws in five, **that is a 10–20x iteration advantage on the architecture that actually works.**
This — not 518 — is the asymmetry worth having, and it is the one the local pixel route was
always positioned to deliver.

The pixel cache it needs is **~9 GB** (26,442 slot images, 3 × 336² uint8), against 458 GB of
NIfTI. Once it is built the NIfTI mirror is deletable.

---

## 2f. The moat is inverted: our extractor is last of six `MEASURED 2026-08-10`

Reproduce with `python extractor/bench_public_labels.py --download`. Every row is the same 58
gold studies and the same macro AUROC as §0, so these numbers sit directly against our 0.777.

| label source | macro AUROC | SE |
|---|---:|---:|
| `stevenleehans/llm_labels_v4_blend` | **0.893** | 0.015 |
| `stevenleehans/llm_labels_full` | 0.878 | 0.016 |
| `pilkwang/report_labels_v2` | 0.866 | 0.016 |
| `lixin73/labels_llm_gpt56sol` | 0.835 | 0.018 |
| `pilkwang/report_labels_v1` | 0.813 | 0.019 |
| **ours (rules)** | **0.777** | 0.021 |

**+0.116 against the best, ~4.5× the combined SE.** This is not a §0 problem — it is the one
extractor result in this project that comfortably clears the noise floor, and it points the
wrong way. Per label it is worse: **0/12**. Synovitis 0.607 vs 0.790, Medial OA 0.764 vs 0.932,
MCL 0.813 vs 0.976, Medial Meniscus 0.819 vs 0.954.

**And it is not additive.** Rank-mean of the two best public readers scores 0.890; adding ours
takes it *down* to 0.887. All five public plus ours is 0.882 against 0.885 without. There is no
combination in which the rule extractor pays for itself.

> Caveat kept next to the result: `v4_blend` is described by its author as a blend and may have
> been selected on these same 58. The unblended reads settle it anyway — `llm_labels_full`
> (0.878) and `report_labels_v2` (0.866) are single LLM passes and still clear us by ~0.10.

**How this happened.** §1.1 has stood open since 2026-08-07 as "Where does the LLM extractor
run? **BLOCKING for method B**". While it was blocked the field published four LLM-read label
tables as free Kaggle Datasets. The blocker was real and the answer turned out to be that we
never had to run one: `kaggle datasets download` is the whole of method B. Roughly five days
went into `rule_extractor.py`, `glossary.json`, compartment attribution (§2.2), the soft-target
ladder (§1.3) and the hand-labelling UI, to reach 0.777 against a free 0.893.

That is README §9's failure pattern — *a belief that was an inference rather than a
measurement, left unexamined because it was load-bearing* — running for the third time. §9.1
retired "local work is text-only". §2e retired "the extractor caps the vision model". This
retires **"the moat is real"**, which README asserted from a comparison against `nekkon`'s
week-one binary CSV and never re-pointed, even after §5 flagged it as stale on 2026-08-09.

### Why §2d measured labels as *not* the ceiling and was still wrong

§2d found Spearman −0.17 (p=0.60) between per-label extractor AUC and per-label vision AUC and
concluded the labels were not the binding constraint. Re-run as a direct A/B instead — same
`features_224` cache, same folds, same seed, only `--labels` swapped to `steven_v4`:

| targets | gold-37 macro |
|---|---:|
| ours (`pseudo_labels.csv`) | 0.719 |
| `steven_v4` (+0.116 at the target level) | **0.744** |

**+0.025 of vision AUC bought with +0.116 of label AUC**, and inside the ±0.038 CI. That is not
evidence the labels do not matter. It is what a *second* binding constraint looks like: a frozen
encoder cannot exploit a better target, so per-label label quality does not propagate to
per-label vision quality and the §2d correlation vanishes **even when labels matter**. §2e is
correct and incomplete — trainability and supervision bind at the same time, and neither is
visible while the other holds.

The practical consequence is an ordering one: swap the labels **before** the training port, so
the port is validated against the targets we will actually ship rather than against 0.777.

---

## 2g. The large-n instrument works — for one class of question `MEASURED 2026-08-10`

Run before committing to the plan that depends on it, on the `features_224` cache that already
exists. `fusion/train.py` now writes `oof_all.csv` (every study, not just the 37 gold);
`fusion/instrument_test.py` scores it.

| instrument | n | macro | bootstrap SD |
|---|---:|---:|---:|
| gold-37 (image-read, current) | 37 | 0.744 | **±0.031** |
| report-derived OOF (proposed) | 2,612 | 0.771 | **±0.0046** |

**6.7× tighter**, and the two land 0.026 apart while measuring different references. That is the
result the plan needed and it holds: a local instrument that can resolve ~0.01 exists, costs
one CSV, and needs no new cache.

### But it does not license label-source A/Bs, and the first attempt at one was confounded

The test tried to go further: score arm A (trained on our labels) and arm B (trained on
`steven_v4`) against `lixin_gpt56` as a *neutral* third reader neither trained on. It returned
+0.039 at 6.0σ — and the "neutral" reader is not neutral. **`steven_v4` predicts `lixin` at
AUC 0.9998.** Arm B was scored against a near-copy of its own training targets. **The 6.0σ is
void; do not quote it.**

Pairwise mean |r| over the twelve labels, n=4,406:

| | steven_v4 | steven_full | pilkwang_v2 | pilkwang_v1 | lixin | ours |
|---|---:|---:|---:|---:|---:|---:|
| **steven_v4** | 1.000 | 0.936 | 0.918 | 0.726 | **0.947** | 0.685 |
| **steven_full** | 0.936 | 1.000 | 0.885 | 0.641 | 0.795 | 0.592 |
| **pilkwang_v2** | 0.918 | 0.885 | 1.000 | 0.700 | 0.866 | 0.657 |
| **lixin** | 0.947 | 0.795 | 0.866 | 0.715 | 1.000 | 0.706 |
| **ours** | 0.685 | 0.592 | 0.657 | 0.695 | 0.706 | 1.000 |

**Two consequences, and one of them cancels planned work.**

**The public readers are near-duplicates of each other** (0.87–0.95 among the good ones). There
is almost no diversity to ensemble, which retro-explains §2f: the rank-mean of all five (0.885)
is *worse* than `steven_v4` alone (0.893). So the per-label accuracy-weighted fusion the 0.903
writeup recommends — and which Phase 0 step 1 was going to build — is **not worth building
here**. It pays when readers are independent. These are not. **Ship `steven_v4` and move on.**

**Our extractor is the only genuinely independent source in the table** (0.59–0.71 against
everything else, where the public readers sit at 0.87–0.95 against each other). That is the one
interesting property it has, and §2f already shows the diversity is *error* rather than signal —
it is uncorrelated because it is wrong in its own direction, and adding it to a rank-mean loses
0.005. This is the strongest available support for the §1.1 disagreement-detector framing and
the strongest available argument against ever using it as a label.

### What the instrument is therefore for

- **Valid:** architecture, hyperparameters, pooling, slice count, augmentation, the training
  port's reproduction gate — anything compared **at fixed targets**, where both arms stand in
  the same relation to the reference.
- **Not valid:** comparing label *sources*. The reference is itself a label source, so the arm
  whose targets resemble it wins by construction. Gold-58 remains the only arbiter there, it is
  noisy, and at the target level it is also already decisive (0.777 vs 0.893, §2f) — so this
  limitation costs nothing we currently need.

---

## 2h. The port is affordable — measured, not inferred `MEASURED 2026-08-10`

§2e's "~12 min/epoch, ~2 h for `EPOCHS=10`" was scaled from our 9.9 img/s for base@518 by
parameter and token counts. `pipeline/bench_port.py` and `pipeline/bench_cache_build.py` run it.

**A. Training** — dinov2-small @336, last 6 blocks open (10.7M of 21.8M params trainable),
batch 8 studies × 6 slots = 48 images, MPS:

| | measured |
|---|---:|
| step | 1,683 ms |
| throughput | **28.5 img/s** (train) · 74 img/s (inference) |
| epoch, 4,407 studies × 6 slots | **15.4 min** |
| 10 epochs, one fold | **2.6 h** |
| 5 folds × 10 epochs | **12.9 h** |

**Estimate was 1.29× optimistic — inside the ~3× gate, so the port proceeds.** For scale, the
public 0.903 five-fold B3 run took 12.4 h, so 12.9 h locally is the same order for a full
ensemble, and **2.6 h is the single-fold iteration unit**. Against Kaggle's 8 h cap, 30 h weekly
quota and four-in-five GPU refusals, the asymmetry §2e claimed is now measured rather than
argued.

**B. Cache build** — the surprise, and it is in our favour:

| | |
|---|---:|
| per study, single process | 0.22 s median (4.8 series read) |
| **full corpus, single process** | **~16 min** |
| the 3,599 studies with NIfTI on disk | ~13 min |

**Two orders of magnitude cheaper than any cache this project has built**, and the reason is
structural rather than lucky: the frozen cache was slow because it ran DINOv2 over ~155 slices
per study, so its cost was *encoder* time. The slot cache stores **pixels** — 6 slots × 3 slices
= 18 per study — so its cost is a NIfTI read and an in-plane resample off a local SSD. The
encoder work moves into training, where it belongs, because that is the only place it can be
adapted.

So the entry price for the architecture that can actually fine-tune is **~16 minutes**, against
the 21 h unsharded attempt, the 9 h serial-curve run and four failed Kaggle sessions spent on
the architecture that cannot. That comparison is the cost of §2e having gone unexamined.

**One thing to carry into the build:** slot coverage is uneven, as `FINDINGS.md` §3.2 predicted —
over 20 sampled studies, Axial-FS hit 20/20 and **Axial non-FS hit 4/20**. Masked slots are the
normal case, not an edge case, and `fusion/model.py` already treats series dropout as a real
augmentation for exactly this reason.

---

## 2i. Three forum posts, read 2026-08-10 — two change decisions made the same day

Rule 4 ("check what is free before building it") applied to the discussion forum rather than to
Datasets. It paid twice.

### 2i-a. Site leakage is measured at **0.053**, and it lands on our instrument

`zhukovoleksiy/rsna-metadata-probe`. DICOM headers only, **no pixels**, targets report-derived,
`HistGradientBoosting`:

| | macro |
|---|---:|
| random 5-fold | **0.6516** |
| GroupKFold on scanner fingerprint | **0.5981** |
| **gap = site memorisation** | **0.0534** |
| series composition alone, no DICOM reads | 0.5954 |

The fingerprint is `Manufacturer | ManufacturerModelName | SoftwareVersions | ImagingFrequency |
ReceiveCoilName` — **265 distinct values, top 20 covering 45.5% of studies.**

**This is the concern §2g left open, now quantified by someone else for free — and it is a
problem for us specifically, because `data/folds.csv` is ungrouped.** Our report-OOF instrument
(§2g, macro 0.771) is scored under random folds, so some unknown part of it is the model
recognising a scanner rather than a knee. Three consequences, in order of severity:

1. **The reproduction gate is the real casualty.** Comparing our ungrouped OOF against the
   fork's published OOF is not like-for-like, and the gate would pass on a number inflated by
   an amount of the same order as the differences we intend to measure.
2. **Fixed-target A/Bs mostly survive**, because both arms inflate together — but not
   perfectly: a change that helps the model exploit scanner cues would score well spuriously.
3. **The 6.7× variance claim is untouched.** That is arithmetic on n, not on fold policy.

**So the header pass is back on the critical path — for a different reason than before, and this
time a measured one.** It was demoted earlier the same day because the *slice-direction* bit it
was carrying is not needed before the reproduction gate (the fork takes `GROUP=3` slices around
each series centre, and reversing a volume does not move its centre). Site fingerprinting is a
separate justification and it does bind. The author notes 265 fingerprints is finer than
institution, so 0.053 is an **upper bound**.

### 2i-b. `eda_04`'s "0.471 on gold, do not retest" was another 37-study artefact

`README.md` records the series-metadata shortcut as **rejected at 0.471, below chance**. The
probe above gets **0.5954 from series composition alone** — the same four columns already in
`train_series.csv`, no DICOM reads. Both can be true (different reference, different features,
and ours was n=58) but the instruction that followed — *"do not retest"* — was drawn from the
instrument §2g has now retired. **This is the third finding this file has had to reopen because
gold-37 could not see it.** Metadata is not a shortcut worth chasing; but "do not retest"
should not have been written from n=58 at all.

### 2i-c. Ship `steven_v2`, not `steven_v4_blend`

`stevenleehans` published the derivation of their own label sets, and our measurements reproduce
their published numbers **exactly**:

| key | their post | `bench_public_labels.py` |
|---|---:|---:|
| v1 / `full` | 0.8780 | **0.8780** |
| v2 (Synovitis repaired) | 0.8873 | **0.8873** |
| Synovitis column, v1 → v2 | 0.678 → 0.790 | **0.678 → 0.790** |

And v2 differs from v1 in **that one column and no other** — every other per-label AUC is
identical, which is the signature of the targeted single-column repair they describe.

The mechanism is worth knowing on its own: they ask the reader for an explicit *"the report does
not address this"* answer mapped to 0.5, and **25.4% of all cells come back undecided** —
Synovitis **83.7%**, ACL 8.3%. Filling only the undecided Synovitis cells from the **Effusion**
field (never overriding an explicit statement) moves that column 0.678 → 0.790. A field that is
not about synovitis predicts synovitis better than the synovitis field does, because
radiologists report effusion readily and synovitis rarely, and the two co-occur.

**And generalising it lost**: the blanket twelve-label learned imputation (v3) scored **0.8805**,
*below* v2's 0.8873. Their explanation is the one to keep — silence means different things per
finding. Gold-positive rate when the report is **silent** vs when it **speaks**: Baker's
**0.03 vs 0.44**, Medial OA **0.00 vs 0.36**, Synovitis **0.34 vs 0.76**. For Baker's the silence
*is* the label and overwriting it destroys information; for Synovitis the silence is genuinely
uninformative. They also report that the *cheating* version — choosing which findings to impute
using gold — scored 0.8845, still below the disciplined 0.8873.

**`v4_blend` has no published derivation.** It measures 0.8927, which is +0.0054 over v2 — far
below the ~0.02 that author states is resolvable on 58 studies, and below our own §0 floor. So
the choice is between a documented, reproduced 0.8873 and an undocumented 0.8927 whose margin is
unmeasurable. **Take v2.** This is also the "code you could explain in an interview" standing
decision applied to a dependency rather than to our own source.

One line from that post belongs in this file verbatim, because we measured it independently on
the same day (§2f): *"A better key is not automatically a better model. We swapped these labels
in and got no gain on the first attempt."* Ours bought **+0.025 of vision AUC for +0.116 of
label AUC** — the same shape, and §2f explains why.

*(The third post, `maximolorenzoylosada/4407-studies-and-58-labels`, contains nothing this repo
does not already have — its prevalence figures match `eda_01` exactly and its language split is a
keyword heuristic that `eda_03`'s lingua-based ID supersedes. Recorded so it is not re-read.)*

---

## 2j. Our own site leakage: **+0.024** `MEASURED 2026-08-10`

Phase 0 step 2b, done. `pipeline/site_fingerprint.py` + `fusion/folds.py --group-by site`.
Same targets (`steven_v2`), same `features_224` cache, same seed — **the folds are the only
difference**:

| folds | n | report-OOF macro | bootstrap SD |
|---|---:|---:|---:|
| ungrouped (`data/folds.csv`) | 2,612 | 0.7468 | ±0.0045 |
| **site-grouped (`data/folds_site.csv`)** | 2,612 | **0.7229** | ±0.0048 |
| **our site leakage** | | **+0.0239** | ~5σ |

**About half the metadata-only probe's 0.0534** (§2i-a), which is the right shape: our model
reads pixels through a frozen encoder, and those features carry less scanner identity than raw
headers do. But 0.024 is **larger than the resolution effect (+0.013) this project dismissed as
unmeasurable, and the same size as the entire label-swap gain (+0.025)**. Every OOF number
produced here before today was inflated by roughly this much.

**The honest baseline going forward is 0.7229 ± 0.0048, site-grouped.** Ungrouped numbers stay
usable for A/Bs where both arms share the fold policy, but nothing gets compared against an
external score except under site-grouped folds.

Note the gold-37 macro moves the *other* way — 0.7389 ungrouped vs 0.7465 grouped. With 5–19
positives per label that is noise, and it is a good example of why §2g retired it: the
37-study instrument cannot see a 0.024 effect and here it reports the wrong sign.

### The bug that would have made this look fine

The first version of `site_fingerprint.py` produced **215 groups, largest 1,077 studies** — 24%
of the corpus in one group, which would have forced a quarter of the data into a single fold
while the fold report printed a healthy-looking group count. Cause: the parquet's string columns
are pyarrow-backed, so `.astype(str)` leaves `pd.NA` intact, `pd.NA + "|"` is `pd.NA`, one
missing field nulls the whole fingerprint, and **`pd.factorize` maps NaN to −1** — collecting
every such study into one bucket. Fixed by forcing each field through a real `str` first, plus
two asserts.

This is the failure mode that module's own docstring warns about, committed inside the module.
After the fix: **265 fingerprints, top 20 covering 45.5%** — an *exact* match to the published
probe, which is what a correct reproduction looks like and what the first version did not have.

Two things made it visible, and both are cheap to repeat: **the fold report prints
`max N studies` per group**, and there was a published number to reproduce. A grouping guard
with neither is untestable.

---

## 2k. Pipeline audit — normalisation, hashing, dtype, split unit `AUDITED 2026-08-10`

Six questions asked of the code rather than of memory. Three clean, three with a caveat worth
carrying.

### Is the intensity normalisation a leakage path? **No.**

`pipeline/preprocess.py::normalise_and_resample` takes **per-volume** robust percentiles —
`np.percentile(vol, [0.5, 99.5])` over *that series only* — then scales to [0,1] and clamps.
`imagenet_normalise` applies fixed ImageNet constants. **No statistic is ever computed across
studies**, so there is nothing for train to learn about test. This is also the right choice on
its merits: MRI has no HU standard, so a fixed window would be meaningless.

Two caveats that are not train/test leakage but are worth knowing:

- **`LATERALITY_X_THRESHOLD = -62.0` is a fitted constant** — fit on the tagged half of train,
  cross-validated at 97.32% ± 0.72% with the threshold itself stable at −62.4 ± 4.5. Fitting on
  train and applying at test is legitimate, but it is the one number in the preprocessing that
  came from data rather than from physics, and it is baked into the fingerprint.
- **Per-volume normalisation does not remove scanner signature.** Residual contrast and texture
  remain scanner-specific, and §2j measured what that is worth: **+0.024**. Not leakage between
  splits — leakage between *sites*, which is why `folds_site.csv` now exists.

### Is the preprocessing config hashed? **Yes, with two known holes — both already in the code.**

`_fingerprint()` → SHA-256 over `{model, img_size, slices, target_mm, fov_mm, embed_dim,
canonical_side, norm, stack, lat_x_threshold}`, truncated to 12 hex and written into every
cache manifest (`preprocess_version: cdaee5e66c6b` for `features_224`).

- **It hashes the DESCRIPTION, not the implementation.** `norm` is the literal string
  `"pct_0.5_99.5"`. When the implementation moved from `torch.quantile` to `np.percentile` — a
  forced change, `torch.quantile` raises above 2²⁴ elements and the corpus has 768×768 series —
  the fingerprint could not have detected it. It happened to be a no-op, and nothing was cached
  yet. **A hash over a description cannot police the code it describes.**
- `SLICES_PER_SERIES_TRAIN` is excluded on purpose (it describes the head, not the cache) and is
  stamped into each checkpoint instead — K19, §6.2. `SAGITTAL_LR_SLICE_FLIP` is hashed only when
  enabled, so leaving it off is a genuine no-op.

### What is the dtype?

| | dtype | why |
|---|---|---|
| `feats` | **float16** `(94, 1536)` | post-LayerNorm embeddings, well inside fp16 range; halves a 2.2 GB cache |
| `series_idx` | int16 | ≤ 24,371 series |
| `plane`, `fluid_sensitive`, `laterality` | int8 | 3–6 categories |
| **slot pixel cache (to build)** | **uint8** | 3 × 336² per slot image, ~9 GB; what the fork uses |

The uint8 choice is a real decision, not a default: after percentile normalisation to [0,1] it
gives 256 levels across the 0.5–99.5 range, ~0.4% intensity resolution. The public 0.891 fork
quantises the same way, so it is evidently adequate for these findings — but it is the kind of
constant that should be in the fingerprint of the new cache, and it was never in the old one.

### Are the images split by patient or by study? **By study, and there is no alternative.**

Re-confirmed against the header parquet, which is a stronger source than the earlier
`kaggle_01b` pass: **4,407 studies, 4,407 distinct `PatientID`s, zero patients with more than
one study.** The IDs are de-identified per study, so patient linkage does not exist in this
dataset and `fusion/folds.py`'s module docstring is correct.

**But the host has now stated that bilateral studies exist** — both knees occasionally scanned
under one `StudyInstanceUID`, with the report text or DICOM metadata adjusted so participants
can disambiguate (`REFERENCE.md` §1.4). The labels are for **one** knee. Nothing in this
pipeline currently detects a two-knee study, and `canonicalise()` would mirror both into the
same handedness. Related to §2b-iii. Unmeasured; needs a count before it is worth fixing.

### Has the text branch been exploited? **No, and half of it is structurally impossible.**

`test.csv` has no `Report` column, so a text branch has nothing to read at inference and can
never be part of the model. Text can only ever produce **targets**.

What is *not* impossible and has never been tried: **text as auxiliary supervision.** The image
model can be trained to predict report-derived quantities beyond the twelve bits — an auxiliary
head, or a report-embedding alignment objective in the ConVIRT / GLoRIA family. `PLAN.md` §2.1
explicitly asked for "structured attributes, not just the 12 bits" and
`extractor/run_extract.py` has been emitting them to `data/extract_states.csv` since week one.
**Nothing has ever consumed that file as a training signal.** It is the one asset the retired
extractor track produced that is not superseded by the public labels, because the public tables
ship twelve numbers and nothing else.

---

## 2l. The in-plane axes are canonical per plane — this is what licenses fixed crops `MEASURED 2026-08-10`

README's step 4 says to design **anatomical crop slots in from the start** — medial, lateral,
patellofemoral, intercondylar notch — and justifies skipping a detector with "volumes are
already in mm space". That is only half the licence. Millimetres tell you how far a box is from
the image centre; they do not tell you which way is *medial*. A fixed box needs the in-plane
axes to mean the same thing in every study, and the geometry kernel's own log looked like
evidence that they do not: **"distinct IOP rows: 374"** out of 396, under a comment reading
*"if this is ~3 the protocol is clean per plane"*.

Measured over those 396 series, by projecting each direction cosine onto the nearest signed LPS
axis. The 374 is float obliquity, not a mixture of conventions — the nearest axis is
**unanimous, 132/132, for every plane and every axis**:

| plane | col index + | row index + | normal | median obliquity | p90 | max |
|---|---|---|---|---:|---:|---:|
| Axial | +x (Left) | +y (Posterior) | +z | 4.1–4.9° | ≤15.1° | 41.8° |
| Coronal | +x (Left) | −z (Inferior) | +y | 4.4–8.2° | ≤19.3° | 40.2° |
| Sagittal | +y (Posterior) | −z (Inferior) | −x | 2.4–7.3° | ≤16.0° | 25.0° |

Composed with `canonicalise` mirroring left knees onto `CANONICAL_SIDE = 'R'`, and with a right
knee's medial side facing the midline at +x, this gives the three rules the crop boxes in
`pipeline/slot_cache.py` are written against: **increasing column index is medial** on axial and
coronal, **anterior is low row index** on axial, **anterior is low column index** on sagittal.

Verified visually as well as arithmetically, because an axis table is exactly the kind of claim
this project has got wrong twice from reasoning (K16's first verdict, K18's docstring). A
montage of the built tiles shows the patella anterior on axial and sagittal and the condyles
above the plateau on coronal, for three studies across all six protocol slots.

**Sagittal is the exception and it is why K16 blocks.** There medial/lateral is the *slice* axis:
the normal is −x, so ascending spatial order runs medial → lateral for a canonical right knee.
That is a statement about a volume known to be in ascending order, and a third of them are not.
`slot_cache.py` therefore refuses `sag_med` / `sag_lat` without the direction bit rather than
building a coin flip — the same choice `canonicalise` makes for unknown laterality.

**Re-measure this if the corpus grows.** It is a 396-series sample and it is the entire licence
for detector-free crops; it is cheap and it is load-bearing.

---

## 2m. The stratified thumbnails were already paid for and never downloaded `2026-08-10`

`validate_nifti.py` check 4b was still printing **"100.0% forward, n=48, series types covered:
1/6"** — the stale all-`Axial_0` verdict K16 records as wrong. The header of that file still
carried it as a result. But `kaggle_01c` had already been re-run with the shuffle fix on
2026-08-09 at 22:39, *after* the local `series_geometry.csv` was downloaded at 11:39, and only
the CSV was ever pulled down. One `kaggle kernels output` retrieved the stratified
`series_thumbs.npz` that run had produced and left sitting there.

With it installed, check 4 covers **6/6 series types** for the first time (winner `as-is`, best
for 98% of series, median r = 1.0000 — so the transpose is confirmed for coronal and sagittal,
not just axial), and check 4b reproduces K16's number locally: **66.7% forward, 33.3% reversed,
n=51**.

This is **rule 4 — "check what is free before building it" — paying for itself a fourth time**,
and this instance is the cheapest and the most embarrassing of the four: the artifact was not
merely public, it was ours, generated by our own kernel, sitting in our own Kaggle account for a
day while the local check reported a verdict we had already documented as false. The pattern to
take from it: **when a fix is applied to a Kaggle script, the download is part of the fix.**
A re-run whose output is never pulled leaves the local instrument reporting the pre-fix answer,
with no symptom other than a stale caveat nobody re-reads.

---

## 2n. K16 is not a header rule — the converter's order is unpredictable `MEASURED 2026-08-10`

`PLAN.md` 9 Phase 0 step 2 budgets the direction bit as "2–3 header reads per series (~50k
opens, ~20 min)", with a full-header pass (~700k opens, 3.7 h) as the fallback. Both rest on an
assumption nobody wrote down: that the NIfTI converter sorted the slices by *something in the
DICOM headers*, so the bit is recoverable by reproducing that sort.

`notebooks/kaggle_01d_slice_direction.py` tested it over all 24,371 series, exporting three
candidate sort keys and scoring each against the 51 series whose direction the thumbnails
already settle:

| rule | agrees with ground truth | \|rho\| median |
|---|---:|---:|
| InstanceNumber ascends in projection | **56.9%** | 1.000 |
| sorted-filename order ascends | **60.8%** | 0.314 |
| SliceLocation ascends | **56.9%** | 1.000 |

Chance is ~50%. **No rule was adopted**, against a bar of 51/51 — a rule that really is the
converter's sort key reproduces every case, and at n=51 a genuinely 90%-accurate rule clears
that bar by luck only 0.5% of the time.

Three things make this a settled negative rather than a weak measurement:

- **It is not a sampling limit.** `inst` and `loc` both return |rho| = 1.000, i.e. perfectly
  monotone in projection over the sampled slices. Reading all 700k headers instead of 146k would
  export the same signs. **The 3.7 h fallback is dead, and it would have bought nothing.**
- **It is not noise in the ground truth.** Restricting to the most confident half of the
  thumbnail calls (margin ≥ 0.30, n=42) moves nothing: 59.5% / 59.5% / 57.1%.
- **The marginals look like a match and are a coincidence.** `inst` is 38.1% descending against
  a true 33.3% reversed, which is close enough to be tempting. Per series it is 56.9%. Worth
  remembering as a pattern: **a matching rate is not a matching assignment**, and only the
  join tells them apart.

The `file` rule's |rho| of 0.314 is a second, free finding: filenames in this corpus carry no
spatial order at all, so any future code that reaches for `sorted(glob("*.dcm"))` as a slice
order is wrong.

**The replacement is to stop inferring and measure** — `notebooks/kaggle_01e_direction_measure.py`
ships the spatially first/middle/last thumbnail per series and
`resolve_slice_direction.py --measured` reads the bit off directly, exactly as check 4b does for
51 series. Scoped to **sagittal only** (9,864 of 24,371 series), because medial/lateral is the
slice axis only there — axial and coronal are already served by `canonicalise`'s in-plane
mirror. That scoping is what makes a full-header pass affordable: ~296k opens, ~20 min at 01d's
measured 122 opens/s, against ~95 min for the whole corpus.

**None of this blocks the protocol slots or the reproduction gate.** The six protocol tiles sit
at depth 0.5, where a reversal maps the middle slice to itself, so the cache built on
2026-08-10 is direction-invariant to within the channel order of one 3-slice group. K16 gates
`sag_med` / `sag_lat` — the divergence — and nothing before it.

### K16 IS NOW RESOLVED, by measurement `2026-08-10, same evening`

`kaggle_01e` ran over all **9,864 sagittal series** (full header pass + 3 pixel decodes each,
8 workers) and `resolve_slice_direction.py --measured` read the bit off for the **8,048** that
have NIfTI on disk. `data/slice_direction_resolved.csv`.

| | |
|---|---|
| reversed | **50.4%** (forward 49.6%) |
| in-plane layout `as-is` | **97.9%**, median r = 1.0000 |
| margin \|r_first − r_last\| | median 0.388; only **3.6%** under 0.05 |

**Cross-validated 21/21 against the 01c thumbnails** — a genuinely independent instrument
(64×64 float16 from one kernel run against 32×32 uint8 from another), and still 100% when
restricted to series where both sides are confident. **That is the same 100% bar the three
header rules failed at 56.9 / 60.8 / 56.9%**, which is the point: the bar was not too strict,
the rules were wrong.

Note sagittal is ~50/50 reversed against the 8/21 = 38% forward the small stratified sample
suggested. Both are consistent — n=21 carries a ±10.6% binomial SE — but it is a reminder that
the 66.7%-overall figure is a corpus-wide *average over planes*, not a per-plane rate.

> **HAZARD, for whoever builds next: the two caches will disagree until protocol is rebuilt.**
> `sag_med` / `sag_lat` are only meaningful with `SAGITTAL_LR=1`, because without the handedness
> flip "slice 25% is medial" holds for right knees and is exactly inverted for the 43% that are
> left. But `data/tiles336/tiles_protocol.npy` was built with `SAGITTAL_LR=0` and no direction
> bit. Each cache records both flags in its manifest (`sagittal_lr_slice_flip`,
> `direction_bits`), so the disagreement is *visible* — but nothing yet *enforces* it.
> **Rebuild the protocol tiles with the same flags before any run that consumes both**; it is
> 21 minutes. The fold-0 gate result is unaffected and stands, for the depth-0.5 reason above.

---

## 2o. The three numbers near 0.89 are three different quantities `CLARIFIED 2026-08-10`

Written because the question "why is the baseline 0.7229 and not the 0.89 we had?" is the right
question, the docs contain **three** unrelated numbers near 0.89, and conflating any of them with
the baseline silently invalidates every comparison made against it.

| number | what it measures | reference | scale |
|---|---|---|---|
| **0.893 / 0.887** (§2f) | **a LABEL TABLE**, not a model. `steven_v4` / `steven_v2` as predictors of gold. `steven_v2` **is** `data/targets.csv` — our training data | gold-58, image-read | local, n=58 |
| **0.891** (§2e) | the `pilkwang` fork — and it is **inference from published weights, not a training run** | competition test set | **leaderboard** |
| **0.7229** (§2j) | **our model**, out-of-fold | `lixin_gpt56`, held out | local, n=2,612 |

They differ on three axes at once — *what* is scored (labels vs a model), *against what*
(report-derived vs gold image-read), and *on what scale* (local OOF vs leaderboard). Only the
third is a model score computed on our pipeline, which is why it is the baseline.

**Local and LB are not the same scale and the conversion is known** (README "five facts" §5):
OOF 0.632 → LB 0.664, and OOF 0.8544 → LB 0.903. Local reads ~0.03–0.05 *below* LB, so 0.7229 is
worth roughly **LB 0.76** — and site-grouping deliberately costs us a further 0.024 (§2j) that no
external score pays. The gap to the fork is real and is about **0.13**, not the 0.17 the raw
numbers suggest.

### The bug this question found, before it could produce a wrong number

**`0.7229` is NOT "OOF scored against the training targets".** `instrument_test.py::evaluate`
scores against **`lixin73/labels_llm_gpt56sol` — a third label source the model never trained on
— over NON-GOLD studies only**, binarised at 0.5.

`fusion/train_port.py` as first written scored against `data/targets.csv`, i.e. `steven_v2`, its
own training source, and printed the result directly beneath `baseline to beat: 0.7229`. That is
upward-biased — a model is rewarded for reproducing one reader's idiosyncrasies rather than the
signal — and placing it under that line makes it read as a comparison when it is a category
error. Fixed before the first fold finished; **`fusion/score_oof.py` is now the single
definition** and `train_port` calls it.

Two consequences worth keeping:

- **The baseline's own `oof_all.csv` was never kept.** `fusion/runs*/` hold `oof_gold.csv` only.
  So *any* single number placed against 0.7229 is unpaired and on a different study set — the
  frozen cache covers 2,650 studies, the tile cache 3,599, and a one-fold port run ~700.
  `score_oof.py` restricts every arm to the studies they **share** when given more than one, and
  says so loudly when given one. The honest A/B needs the frozen arm re-run under `folds_site`
  (~32 min).
- **Generalised: a metric is a triple — predictions, reference, population.** This project has
  now been bitten at all three. Site leakage was the *population* (§2j, +0.024). Scoring an arm
  against a near-copy of its own targets was the *reference* (§2g's retracted Q3, +0.039). And
  this was the reference again, one level up. **When quoting a macro, quote all three or it is
  not a number anyone can reuse.**

---

## 2p. The port is memory-bound, not compute-bound — the M5 has 17.2 GB `MEASURED 2026-08-11`

> ### !! THE TIMING EVIDENCE BELOW IS CONFOUNDED — see §2v `2026-08-12`
>
> **The machine was ASLEEP for roughly 2 of this run's 3.6 hours.** `pmset -g log` over the run
> window (~22:06–01:42) sums to **≈7,005 s** of Maintenance, Idle, Clamshell and **Thermal
> Emergency** sleep, and every epoch below is ≈10 min of compute plus however long the machine
> was out: epoch 7 = 15.4 min against 5.0 min of sleep, epoch 9 = 22.2 against 12.6, epoch 10 =
> **79.8 against ~54**. Epochs 1–5 are clean at 9.5–10.7 min, and the first sleep in the window
> lands at 22:59 — right where the times start moving.
>
> **`img/s` is computed against wall clock**, so sleep depresses it exactly as thrash would. The
> "collapsed by ~10×" reading cannot distinguish the two, and the conclusion drawn from it is
> therefore not established by this evidence.
>
> **What survives:** the swap figure (24.47/25.6 GB) is a direct memory reading and still says the
> working set does not fit. **What does not survive:** that memory pressure is what made the run
> slow, and the "1.2–3.5 img/s" collapse as its signature. Both need re-measuring under
> `caffeinate -i`. This is the fifth instance of the §2s error class — an instrument entangled
> with an uncontrolled variable — and the variable here was the laptop being asleep.

Fold 0 ran. It trained, the loss fell cleanly from 0.4523 to 0.2877, and **it took 3.6 h against
a 2.6 h budget** — with the cost arriving in a shape neither §2e's estimate nor §2h's measurement
could see:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **10** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| min | 10.7 | 9.6 | 10.3 | 9.7 | 9.5 | 17.8 | 15.4 | 32.9 | 22.2 | **79.8** |
| img/s | ~27 | ~30 | ~29 | ~29 | ~29 | | | | | **1.2–3.5** |

Throughput did not drift. It **collapsed by ~10×**, and the machine state says why:

    RAM 17.2 GB total   ·   tile cache 7.31 GB, randomly accessed
    swap 24.47 GB of 25.6 GB used   ·   33.5M pageins

**This is the number that was missing from the plan: the M5 has 17.2 GB, not the 32–64 GB the
"~9 GB cache is affordable" reasoning in §2e quietly assumed.** A 7.31 GB working set read in
random study order does not fit alongside the model, two loader workers and the OS, so the page
cache loses and every batch starts paying disk.

**Why the benchmark could not have caught it, and this is the reusable part.**
`pipeline/bench_port.py` measures throughput on **synthetic tensors**, and says so in its own
docstring: *"Synthetic tensors on purpose: this isolates compute from I/O, and the cache exists
precisely so that training never touches NIfTI."* That was the right call for the question it was
asked — is the *architecture* affordable — and its 28.5 img/s reproduced exactly for five epochs.
But **a benchmark that deliberately isolates compute cannot predict wall-clock for a workload
that turns out to be memory-bound**, and §2h's "2.6 h/fold, gate passed at 1.29×" was read as a
schedule when it was only ever a compute figure. Same family as README rule 6: the number was
correct and answered a different question than the one being asked of it.

**Honest confound:** the degradation begins at epoch 6, which is when the K16 resolver was run
concurrently — thousands of NIfTI reads that evict exactly the pages training needs. So part of
this is self-inflicted and a clean fold on an idle machine will do better than 3.6 h. It will not
do 2.6 h at 17.2 GB with a 7.31 GB random-access cache, because swap at 24.5/25.6 GB is
structural, not contention.

**What follows, in order of payoff:**

1. **Do not run anything heavy against `data/nifti` while a fold is training.** Free, and it is
   the largest single factor here.
2. **Hand the loader `uint8` and normalise on-device.** `SlotDataset.__getitem__` currently
   returns float32: 8.1 MB per study against 2.03 MB, a **4× reduction** in everything the
   workers hold and prefetch.
3. **Shuffle with locality.** Random study order touches the whole 7.31 GB every epoch. Shuffling
   within contiguous chunks keeps the working set to a slice of the memmap and costs almost
   nothing in randomness at 2,871 studies.
4. **Do NOT build the anatomical tiles as a second full-size cache and train on both at once.**
   That is another ~7 GB, i.e. 14.6 GB of tiles on a 17.2 GB machine. Build it, but train one tag
   at a time until 2 and 3 are in.

Cross-check against the alternative before over-reacting: Kaggle is 8 h per run, a 30 h weekly
quota, and a GPU lottery that refuses four draws in five. **3.6 h locally and unlimited still
wins** — the asymmetry in §2e survives, it is just 3–4× rather than 10–20×.

---

## 2q. Fold 0 of the port: +0.0094 at 1.0σ — the port RUNS, it has not yet WON `MEASURED 2026-08-11`

The first end-to-end fine-tuned fold. Scored through `fusion/score_oof.py`, i.e. against
`lixin_gpt56` over non-gold studies, which is the only thing that sits on 2j's scale.

| | n | macro | ±SD |
|---|--:|--:|--:|
| **port, fold 0** (dinov2-small@336, `UNFREEZE_LAST=6`) | 681 | **0.7323** | ±0.0086 |
| baseline (§2j, frozen cache, 5-fold pooled) | 2,612 | 0.7229 | ±0.0048 |
| delta | | **+0.0094** | **1.0σ** |

**Read this as "no demonstrated improvement yet", not as a win.** One sigma is what the
instrument returns when it cannot tell two things apart, and §0's own rule — nothing below the
instrument's resolution — applies to results we like as much as to ones we do not.

Three separate reasons the number cannot carry more weight than that:

1. **It is unpaired.** The baseline's `oof_all.csv` was never kept (§2o), so this compares
   different study sets — 681 single-fold studies against 2,612 pooled — and difficulty differs
   between them. The frozen arm must be re-run under `folds_site` to settle it (~32 min).
2. **One fold has ±0.0086, not ±0.0048.** The 6.7× instrument advantage in §2g is a *five-fold
   pooled* property. A single fold is roughly 2× noisier, which is most of why +0.0094 lands at
   1.0σ rather than 2σ.
3. **The gold cross-check is empty.** Fold 0's OOF covers **10 gold studies**, 1–6 positives per
   label. `summary.json`'s `macro_gold` of 0.757 is not a measurement and the per-label deltas
   against §2d's frozen numbers (Fracture +0.256, Contusion −0.228) are noise on n=4. The check
   that §0 item 2 asks for — do the two instruments agree? — **cannot be run on one fold**, and
   that is a structural fact about running folds one at a time, not a result.

**What this does establish, which is not nothing.** The architecture §2e said was required now
exists and trains: 10.9M of 22.0M parameters open, loss 0.4523 → 0.2877 against a floor of
0.2040 (§2p), on the honest site-grouped folds, in our own code rather than the fork's. Every
number above is the first one this project has produced from a model that *could* adapt its
encoder.

**Next, in the order that makes the result interpretable:**

1. **Re-run the frozen arm under `folds_site`** and score both through `score_oof.py`, which
   pairs them on shared studies. Until then no delta here means anything. ~32 min.
2. **Step 5, the reproduction gate.** We do not yet know the port is a faithful reconstruction,
   and a +0.009 from an unfaithful port is uninterpretable in either direction.
3. Only then the anatomical slots, which are the divergence aimed at the failing labels.

**Do not conclude "fine-tuning does not help" from this.** That is the §2d error in a new
costume — reading a single under-powered comparison as a verdict on a mechanism. §2e's diagnosis
rests on the fork scoring 0.891 while every frozen-cache arm here sits near 0.70, and one
unpaired fold at 1.0σ does not touch it.

### The paired A/B, run the same day — and it is the number that counts

The frozen arm was re-run under `folds_site` (`fusion/runs_baseline`; it reproduces §2j's
gold-37 **0.7465** exactly, which is what says it is the right control). Both arms scored through
`score_oof.py`, restricted to the **493 studies they share**:

| | macro |
|---|--:|
| frozen cache (`runs_baseline`) | 0.7052 |
| **fine-tuned port** (`runs_port`) | **0.7223** |
| **paired delta** | **+0.0171 ± 0.0088 → 1.9σ, P(delta>0) = 0.978** |

**And the pairing had to be used, not just set up.** The first version of `score_oof.py` reported
`hypot(sA, sB)` = ±0.0138 and called it conservative. It is conservative to the point of being
the wrong test: both arms are scored on identical studies by construction, so most of the
macro's variance is *which studies were drawn*, and that component is common and cancels. Same
delta, **1.2σ unpaired against 1.9σ paired**. Fixed; the paired bootstrap is now what the script
prints.

Note also that neither arm's own number resembles the recorded 0.7229 — the frozen control
scores **0.7052** on these 493 studies. That is a study-set effect, and it is the concrete
demonstration of why §2o's unpaired warning mattered: comparing the port's 0.7323 against 0.7229
was misleading in *both* directions at once.

**The per-label pattern is weak evidence FOR §2e's mechanism, and should be quoted as weak.**

| §2d's split | mean Δ |
|---|--:|
| mm-scale findings (predicted to move) | **+0.0347** |
| cm-scale findings (already working) | −0.0005 |

The three largest gains are **ACL +0.100, Contusion +0.093, Fracture +0.076** — precisely the
findings §2d measured at or near chance and attributed to a frozen natural-image ViT being
unable to resolve millimetre-scale lines. That is the predicted direction on the predicted
labels.

But the sign test is **6/12, which is chance**, and two members of the predicted set move the
wrong way (MCL −0.050, Medial Meniscus −0.017) while Baker's (+0.064, cm-scale) is the
fourth-largest gain. **So: consistent with the mechanism, not a demonstration of it.** The macro
rides on three or four labels, which is exactly the situation where a sign pattern is worth more
than a mean and this one is only half there.

---

## 2r. Code review of the step-4 body — 15 findings; the A-cluster is now FIXED `REVIEWED 2026-08-11`

> **UPDATE 2026-08-11, same day.** A1, A3, A4 and B's `n_oof` are **repaired and verified** —
> see the "As fixed" block at the end of this section. The rest of the findings stand as recorded.

Recall-oriented review of `git diff 1049bcf..HEAD -- '*.py'` — the 1,683 new lines that are
`slot_cache.py`, `train_port.py`, `score_oof.py`, `resolve_slice_direction.py`,
`kaggle_01e_direction_measure.py`. **Everything below was RECORDED, not repaired, when first
written.** Fold 0's
0.7323 and the paired +0.0171 (§2q) are *unaffected* — they ran on protocol tiles at depth 0.5,
which is the one configuration none of the medial/lateral findings touch.

**The headline: the medial/lateral safety story for the anatomical slabs is written in three
docstrings and the README, and implemented in none of them.** A1, A3 and A4 are one gap seen from
three angles. The anatomical build is the next thing PLAN.md says to run, so this is the cluster
to close first — it is the difference between "the divergence didn't help" and "the divergence
was inverted for 43% of studies and we read the result as a verdict."

### A. The medial/lateral cluster — fix before `--slots anatomical`

- **A1. The K16 refusal is global, not per-series.** `slot_cache.py:247`. `gated` fires only on
  `not direction`, i.e. the CSV missing or empty. One resolved series opens the gate for all of
  them; every series without a bit then hits `canonicalise(..., slice_direction=None)`, and
  `preprocess.py:444` requires `slice_direction is not None` before it will apply the left-knee
  sagittal flip. `load_direction()` also drops `direction == "unknown"` rows, so those look
  identical to "never measured". For a left knee with no bit, **`sag_med` is the lateral
  compartment and `sag_lat` is medial** — per study, silent, on the axis four labels depend on.
  The module docstring says it REFUSES to do exactly this; it refuses only the empty-file case.
  Gate per series, or write `has_sag_med=False` for series without a bit.
- **A3. `assert_caches_compatible()` is never called.** `slot_cache.py:154`. Two hits in the whole
  repo: the definition, and the README sentence claiming it raises. Both now annotated in place.
- **A4. Nothing checks `SAGITTAL_LR=1` at build time.** `slot_cache.py:258`. The documented
  invocation is `SAGITTAL_LR=1 python pipeline/slot_cache.py --slots anatomical`; the module
  `setdefault`s `TARGET_MM` and not this. Forget the prefix and the manifest records
  `sagittal_lr_slice_flip: false`, which **matches `tiles_protocol`** — so even a working A3
  would call them compatible. The flag that changes the pixels is the one with no guard.
- **A8. First-matching-series-only drops tiles that exist.** `slot_cache.py:300`. `m.iloc[0]`
  then `continue` if that UID isn't on disk — but a study is admitted if *any* of its series is
  present, and the corpus downloads incrementally. A study whose second Coronal FS series is the
  downloaded one gets `has_cor_fs=False` permanently. Iterate `m.SeriesInstanceUID` until a path
  resolves.

### B. Correctness

- **B2. `NameError` on the summary write when the neutral reference is absent.**
  `train_port.py:419`. `ref` is bound only inside `if NEUTRAL.exists():`; the `else:` branch
  prints "scoring skipped" and falls straight through to `len(ref)`. Lands *after* all folds have
  trained (~13 h) and after `fold*.pt` + `oof_all.csv` are written, but before `summary.json`.
  The most likely error path on a fresh checkout is the one that destroys the run record.
- **B5. One bad DICOM kills the whole `01e` Kaggle run.**
  `kaggle_01e_direction_measure.py:128`. `probe()` guards `glob`, `dcmread`, IOP and IPP — and
  leaves `read_pixels(f)` bare. It raises → propagates out of `ex.map` at line 159 → `main()`
  unwinds → `direction_index.csv` and `direction_thumbs.npz` (written only at 169–170, after the
  loop) never appear. ~29,592 decodes of third-party pixel data, zero output, session spent.
- **B6. The paired bootstrap is not reproducible.** `score_oof.py:144`. `ids` is built by
  iterating `restrict`, a `set` of UID strings, so PYTHONHASHSEED randomisation reorders
  `yy`/`pA`/`pB` per process while `default_rng(0)` draws the same indices. `d0` reproduces; the
  **±0.0088, the σ and the P(δ>0) beside it do not.** In the file whose entire premise is that a
  number quoted next to 0.7229 "is reproduced exactly". One `sorted()` closes it.
- **B7. `n_oof` records the wrong quantity.** `train_port.py:419`. `len(ref)` is the reference
  label table — **4,407 rows** — not the OOF count. The existing `runs_port/summary.json` still
  reads `n_oof: 691` (correct, 691 oof rows) because it predates b3387e8, so every future summary
  will silently contradict the ones already on disk. Want `len(oof)`, or the `n` `score_run`
  already returns.
- **B9. Augmentation is duplicated across DataLoader workers.** `train_port.py:106`. `self.rng`
  is built once in `__init__`, so both forked workers inherit the same generator state and the
  n-th sample each handles gets byte-identical slot-dropout and gamma/gain draws.
  `persistent_workers=True` keeps them locked in step for all 10 epochs. `torch.manual_seed`
  doesn't reach a numpy generator. Seed per worker via `worker_init_fn`.
- **B15. `targets.loc[uids]` can `KeyError` before epoch 1.** `train_port.py:105`. `usable`
  filters on the tile store only; a study in `folds_site.csv` + the cache but not in
  `targets.csv` raises at dataset construction — after the 7.3 GB memmap and timm weights load.
  Intersect `usable` with `targets.index` too.
- **B10. `unfreeze_last` inverts past the block count.** `train_port.py:199`.
  `blocks[len(blocks) - unfreeze_last:]` goes negative: on 12 blocks, `--unfreeze-last 13` gives
  `blocks[-1:]` = **one** block, `20` gives eight. Asking for more unfreezes fewer. Needs
  `max(0, ...)`.
- **B11. `crop_box` can hand `tile_from` a non-square crop.** `slot_cache.py:206`. The clamp is
  per-axis, so an off-grid box shrinks on one side only and `F.interpolate` (227) then stretches
  it anisotropically to 336×336 — the exact failure the `box_mm` docstring forbids ("the backbone
  has never seen a knee stretched 2:1"). Latent: all six current ANATOMICAL boxes fit inside 457
  px. But this module exists to have slots added to it, and nothing asserts the invariant.
- **B12. OneCycleLR makes `LR_BACKBONE` a peak, not the fork's constant.** `train_port.py:279`.
  Defaults (`div_factor=25`, `final_div_factor=1e4`) start the backbone at 3.2e-7, touch 8e-6 at
  25% of training, end at 3.2e-11. Line 60 says changing these constants voids the reproduction
  gate; the file documents its slot-split divergence at length and this one not at all. **If the
  step-5 gate misses, look here before looking at the slot split.**
- **B13. The bootstrap's bare `except` can print `nan` as a result.** `score_oof.py:162`. If
  `mac()` raises on all 2,000 resamples, `ds` is empty → `.std()` is `nan` → the line prints
  `+0.0171 ±nan → nan sigma, P(δ>0) = nan` in the shape of a finding.

### C. Cleanup

- **C14. `--workers` is dead.** `slot_cache.py:243/362`. Parsed, passed into `build()`, never
  read. `--workers 8` on the 458 GB pass silently does nothing. Wire it to a thread pool (the
  NIfTI read is I/O bound — `01e` already assumes that at `WORKERS=8`) or delete the flag.
- Minor, not itemised above: `ser[ser.StudyInstanceUID == st]` rescans a ~24k-row frame once per
  study (`slot_cache.py:294`, O(N·M) — use `groupby`); `train.csv` is read twice in
  `train_port.main()` (390, 407); `import json as _json` shadows the module-level import
  (`slot_cache.py:168`); `first` is assigned and only `del`'d in `score_oof.main()` (124/178);
  `n_tiles / (len(studies) * len(slots))` is a `ZeroDivisionError` when no NIfTI is on disk
  (`slot_cache.py:350`); `--limit 0` means "no limit" via a falsy check (`slot_cache.py:270`);
  and `resolve_slice_direction.py` writes a column named `rho` that holds |r_first − r_last| on
  the `--measured` path and a header-rule correlation on the rule path — same name, same file,
  two quantities.

### As fixed `2026-08-11`

The A-cluster and B's `n_oof` are repaired in `pipeline/slot_cache.py` and
`fusion/train_port.py`. The one refusal became **four guards**, because A1/A3/A4 are one gap seen
from three angles and no single check covers it:

| | guard | fires when |
|---|---|---|
| 1 | the bit exists at all | `slice_direction_resolved.csv` missing/empty **and** a slot needs it (the old check, kept — it is the right check for "the resolver has not been run") |
| 2 | `SAGITTAL_LR=1` required at build time | any `needs_direction` slot would be built with the flip off (**A4**) |
| 3 | cache compatibility | a new cache would be written beside an incompatible manifest — checked **before the first NIfTI read**, not after 21 min (**A3**) |
| 4 | **per-series** K16 refusal | a specific series has no bit → its direction-dependent slots get `False` in the mask (**A1**) |

Guard 4 is the repair. Coverage of the bit is **partial by construction** — the corpus downloads
incrementally and `--measured` writes a row only for a series with NIfTI on disk, thumbnails
present and a matching slice count — so the refusal has to be per series or it is either useless
or permanent. An unresolved series now costs a tile, never a wrong one. `load_direction()`'s
habit of dropping `unknown` rows is safe *only* under this per-series form, and now says so.

`assert_caches_compatible` takes a `(label, dict)` pair as well as a path, so guard 3 can check a
manifest that does not exist yet; the compat fields moved into `cache_compat_fields()` so the
manifest and the check cannot drift.

**Verified, not assumed:**

- Guard 2 refuses `--slots anatomical` with the flip off.
- Guard 3 caught **the live hazard** against the real manifest: `sagittal_lr_slice_flip: True vs
  False`, and `preprocess_version: 086ab411e129 vs 2eddb3ec68d0` — the fingerprint already
  encodes the flag, so it is double-caught.
- Guard 4, forcing partial coverage (every other bit dropped, 30 studies): `sag_med`/`sag_lat`
  fell to **13/30** while **`sag_pf` stayed at 100%** from the same volume — slice-axis slots
  gated, the sagittal in-plane box untouched. **34 tiles the old code would have built as coin
  flips.**
- `preprocess.py --self-test` passes under `SAGITTAL_LR=1`; protocol build path unchanged.

`n_oof` is now `len(oof)`, which fixes both halves of B at once — the wrong count *and* the
`NameError` on the missing-reference path, which fired only after every fold had trained.

**Still open and deliberately not fixed here:** A8 (first-matching-series-only), C14 (dead
`--workers`), the OneCycleLR divergence (a decision about what the gate tests, not a bug — and see
**2s**, which questions whether that gate exists), `score_oof`'s set-iteration non-determinism,
the per-worker RNG, and the C-list minor items.

**The hazard itself is still live.** These guards mean a wrong build now *refuses*; they do not
rebuild anything. `data/tiles336` is still `SAGITTAL_LR=0` and the ~21 min protocol rebuild is
still required before anything consumes both caches. Guard 3 will now stop you rather than let it
through.

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

---

## 6. Kaggle-side pipeline — friction log `ADDED 2026-08-08`

The extractor track is measured in thousandths of AUC. This one is measured in **lost sessions**:
a 9 h cap, a weekly GPU quota, and four cache-build attempts that produced nothing. None of them
failed at the modelling. Recorded in the same format as §3 because the failure *patterns* repeat.

### 6.1 Resolved (provenance)

| # | Issue | Root cause | Fix |
|---|---|---|---|
| K1 | Full-corpus @518 run killed at **21 h** | unsharded, no resume — a cancellation cost everything | `SHARD`/`N_SHARDS`; per-study `.npz` + tmp-and-rename, so a killed session loses one study |
| K2 | Two relaunches died ~1 h in with `no kernel image is available for execution on the device` | Kaggle drew a **P100 (compute 6.0)**; its PyTorch needs ≥ 7.0, and `torch.cuda.is_available()` is **True** on one. `accelerator` in kernel-metadata.json does not reliably override the draw | `pick_device()` in the first seconds of `main()`; cache build exits, submission degrades |
| K3 | Guard v1 **passed a P100 through** | inferred support from `get_arch_list()` inside a bare `try/except` that returned `"cuda"` on any failure — it **failed open** | capability read first, no fallthrough |
| K4 | Guard v2 also passed a P100 | probe kernel did not raise — CUDA errors are reported **asynchronously** and a tiny op did not surface it | probe kept, but `torch.cuda.synchronize()` added and **nothing depends on it** |
| K5 | Guard v3 could not see a GPU too **NEW** | the arch-list check was deleted, leaving only a `major < 7` floor, which is one-sided | arch list restored **two-sided**: below the compiled minimum is fatal, above it warns (PTX JIT is real, and hard-failing it strands working sessions) |
| K6 | Guard v3 condemned **working T4s** | `get_device_capability()` and `get_device_name()` shared one `try`, so a transient NVML failure on the *name* zeroed a capability already read — reported as "compute 0.0 is below the 7.0 minimum" | separate `try` blocks |
| K7 | 224 shard ran **9 h on the serial curve** against a 2.7 h estimate | `ProcessPoolExecutor` **forked** a process that already held a CUDA context (timm `.to('cuda')` runs first) | spawn context; `_decode_task` top-level, `main()` behind `__name__` |
| K8 | The spawn fix made every worker **re-walk the 570 GB mount** | spawn re-imports the module in each child, and the module body globs `/kaggle/input/**/pipeline/preprocess.py` — `**` scandirs ~29k series dirs | parent caches the answer in `RSNA_PREPROCESS_DIR`; bounded-depth patterns before the recursive fallback |
| K9 | 8 spawn workers × torch's default intra-op threads = **~32 compute threads on ~4 cores** | "workers are blocked on I/O" is only mostly true — `normalise_and_resample` runs `torch.quantile` and `F.interpolate` over the whole volume | pool `initializer` pins each child to one thread; `N_WORKERS` capped by `cpu_count` |
| K10 | The preprocessing-parity assert — the project's headline defence — **never ran** | `kaggle_03` looked for `manifest.json`; `kaggle_02` writes `_shard*.json` and `fusion/train.py` wrote neither. `exists()` was always False, so it fell through to a `print` | `train.py` stamps `manifest.json` beside `fold*.pt`; a missing manifest is now **fatal** |
| K11 | The CPU fallback produced **no submission at all** | "a slow submission beats none" — but `to_csv` sat *after* a loop that cannot finish in 9 h at 518px on CPU | all-0.5 placeholder written before the backbone loads, rewritten every 100 studies |
| K12 | Inference fed the head **fp32** | cache is stored fp16 and `dataset.py` upcasts per batch, so the head only ever trained on fp16-quantised vectors. `PREPROCESS_VERSION` hashes constants, not dtypes | `.half().float()` round-trip in `embed_series` |
| K13 | `--self-test` covered **neither** of K7's or K8's mechanisms | probe thresholds started at 25 series and the synthetic corpus holds 21; `self_test` always passed `_SerialPool`, so the spawn pool was never constructed | thresholds are a constant the test lowers; a pool test asserts fan-out, thread pinning, and no re-glob |
| K14 | **The backbone cannot run at `IMG_SIZE=224` at all.** The kernel sets it (`os.environ.setdefault("IMG_SIZE", "224")`) and `forward_features` raises `AssertionError: Input height (224) doesn't match model (518)` on the **first series** — past the GPU guard, past the corpus walk, past the weights download | `vit_base_patch14_reg4_dinov2.lvd142m` is 518-native (1,369 position tokens) and timm will not interpolate the position embedding unless asked. `create_model(MODEL, pretrained=True, num_classes=0)` has carried no size argument since `ab5be8a` | `dynamic_img_size=True` in **both** `kaggle_02` and `kaggle_03`, plus a one-slice smoke of the real `embed()` immediately after the model is built. Reproduced and fixed locally on timm 1.0.28, 2026-08-08 |
| K15 | **`normalise_and_resample` raises on any large series.** `torch.quantile` has a hard ceiling at 2**24 (16,777,216) elements and a 32-slice series at 768×768 is 18,874,368. The corpus contains 768×768 series, so this is not a corner case — it is every large series | `torch.quantile()` is documented as limited but the limit is not in its signature, and the call sits in the **shared** path, so `kaggle_02` would have hit it mid-build after hours of GPU. The five failed attempts all died on the GPU lottery or mount latency before reaching a series this big | `np.percentile` — same statistic, same linear interpolation, agrees to 6.3e-8, no ceiling. Found 2026-08-09 on the **first real run** of `build_cache_local.py`, on the 12th study |
| K16 | **A third of NIfTI series are stored back-to-front**, and nothing in the file says which. Measured on a stratified sample: 66.7% forward overall — Axial 12/12, Coronal 14/18, **Sagittal 8/21** | The affine carries no direction cosines (see §9.1's correction), so `load_series_nifti` cannot know. `load_series` always sorts ascending by IPP projection, so train and test disagree on ~1/3 of series — in the axis medial/lateral depends on, and invisible to `PREPROCESS_VERSION` | **OPEN, and the route changed 2026-08-10 — see §2n before touching this.** ~~Needs a per-series direction bit exported from the DICOMs (`PLAN.md` §9 Phase 0 step 2).~~ **The header-rule route is CLOSED**: InstanceNumber 56.9%, filename 60.8%, SliceLocation 56.9% against ~50% chance over the 51 series the thumbnails settle, and a full-header pass would export identical signs because `inst`/`loc` are already at \|rho\| = 1.000. **Do not re-attempt it.** The live route MEASURES the bit (`kaggle_01e_direction_measure.py` + `resolve_slice_direction.py --measured`), sagittal only. **K18 composes with this** — the sagittal handedness fix is an XOR against this bit and cannot be enabled without it. Not predictable from plane: a plane rule is ~72% accurate. The first verdict said 100% forward because every thumbnail in that sample was Axial_0; the stratified npz that corrects it had been sitting undownloaded in our own kernel output (§2m). **Scope note:** this gates `sag_med`/`sag_lat` only — protocol tiles sit at depth 0.5, where reversal maps the middle slice to itself |

| K17 | **A `--synthetic` smoke run overwrote the real result directory**, and the guard that should have caught it passed. `fusion/runs/` held the fold checkpoints, pooled OOF and summary of the 0.743 run; a smoke run replaced all three at 15:55 on 2026-08-09. The directory is gitignored, so nothing survived | `--out` defaulted to `fusion/runs` regardless of `--synthetic`, and the smoke command in the file's own docstring inherits that default. The second half is worse: synthetic mode left `cache_meta` as `None` and therefore wrote **no** manifest, so the genuine `manifest.json` from the previous run stayed in place and vouched for the random-tensor weights. `assert_matches()` reads only `preprocess_version`, which still said `cdaee5e66c6b`, so a Dataset bundled from that directory would have passed every guard and submitted noise | `--out` resolves **after** parsing to `fusion/runs_synthetic` under `--synthetic`; synthetic mode writes a self-marking manifest so a collision fails closed; `assert_matches()` refuses `synthetic: true` **before** the version check. Verified against all three manifests on disk — the archived clobbered directory still passes the version check, which is the hole demonstrated rather than argued |
| K18 | **`canonicalise()` never corrected handedness for sagittal series.** Medial/lateral is the image x-axis for axial and coronal but the **slice axis** for sagittal, and `vol[:, :, ::-1]` is the only reversal in the whole pipeline — nothing ever touched axis 0. Exposure: sagittal is the largest plane at **9,864 of 24,371 series (40.5%)**, and **1,894 of 4,407 studies (43.0%)** resolve to left knees | The docstring argued that flipping a sagittal series "would mirror the knee front-to-back for no gain" — true of the *in-plane* axis, since sagittal's image x-axis is anterior-posterior, and it simply never considered the slice axis. `spatial_order` sorts ascending along the slice normal; for sagittal that normal is the patient's left-right axis, on which medial is +x for a right knee and −x for a left one, so one sort yields lateral→medial for one knee and medial→lateral for the other. `FusionHead.slice_pos` is a **learned per-index** embedding, so slice 5 meant lateral in one study and medial in the next | **OPEN — code written and gated off.** `canonicalise(..., slice_direction=)` reverses the slice axis for sagittal left knees, as an **XOR against the K16 bit** so the two corrections compose instead of colliding. Behind `SAGITTAL_LR_SLICE_FLIP`: **one** switch consulted by both readers, because `load_series` knows its direction and `load_series_nifti` does not, and a per-call decision would canonicalise the test set but not the training cache. Off is a byte-level no-op — the fingerprint is still `cdaee5e66c6b`, so the existing cache is not invalidated by a no-op. Needs K16's per-series bit to turn on; `preprocess.py --self-test` asserts the axis table both ways |

| K19 | **The slice count reached inference through the cache manifest, where nothing could check it.** `kaggle_03` read `slices_per_series_train` from the manifest with a silent fallback to its own constant. `SLICES_PER_SERIES_TRAIN` is deliberately outside `PREPROCESS_VERSION` — the cache always stores 32 slices and the head samples a subset, so the count changes no cached feature value — which means `assert_matches()` was structurally blind to it | The exclusion from the fingerprint is right; reading it from the *cache* manifest was not. The number describes the **head**, not the cache. Feeding the wrong count is a silent scorer rather than a crash: `slice_pos` is a learned per-index embedding, so the head's positions are simply read at the wrong indices and every guard still passes | `fusion/train.py` stamps `n_slices_train` into each checkpoint, taken from the validation `StudyDataset` that actually fed the head rather than from the constant. `load_heads()` reads it from there, **exits** if folds disagree (they were not one run and must not be ensembled), and exits if it contradicts the manifest. A checkpoint from before 2026-08-09 warns and falls back. All three branches exercised against crafted checkpoints |

**The recurring pattern, and it is not the extractor's.** There, guessed *vocabulary* was wrong
far more often than the logic. Here, three of fourteen entries (K3, K4, K5) are the same guard
**failing open** — and a guard that fails open is worse than no guard, because it looks like
protection and you stop checking. The tell each time was that the fix was never *observed* to
work; it was reasoned to work and then shipped. K13 is the same fault one level up: both
mechanisms that cost sessions were unreachable from the self-test the file's docstring calls "the
gate that says the pipeline is correct".

**K15 is the same bill again, and it is worth stating plainly.** K14 was the one line that
decides whether the backbone runs, never executed because the self-test injected a fake
`embed_fn`. K15 is the one line that decides whether a *large volume* normalises, never executed
because nothing had ever fed this pipeline a real 768×768 series. Both sat in shared,
parity-critical code. Both were found within minutes of finally running the real thing on real
data. The lesson is not "write more self-tests" — it is that **synthetic fixtures reproduce the
shapes you thought of**, and the corpus contains shapes you did not.

**K14 is K13's bill arriving.** The self-test injects `embed_fn`, so in thirteen rounds of fixing
this file the real backbone was never once constructed — and the one line that decides whether it
can run at the configured resolution went unexecuted from `ab5be8a` to now. Every K-entry above it
is scheduling, I/O and device selection: the parts we wrote. K14 sat in the part we assumed.

**It also puts a question mark on K7, which should be settled and not argued.** K7 attributes the
9 h run to the forked CUDA context. But `IMG_SIZE` only became env-overridable in `1195587`
(2026-08-07 16:52 PDT), and the 518 path costs ~5× the 224 path — so a run that *intended* 224 and
silently got 518 would also look like the serial curve. The two explanations are not exclusive and
the evidence is gone. **The PROBE line on the next real shard distinguishes them**; until it
prints, treat the spawn fix as unconfirmed rather than proven.

**So: for anything on this track, the test is not "does it look right" but "what does it print on
the box".** `--self-test` before pushing a Dataset version; the PROBE line before letting a
session run for hours.

### 6.2 Open

- **The device guard is the first fix validated in the wild — 2026-08-08.** Attempt 5
  (`raahimnawaz/rsna-knee-cache-224-s0`) drew a Tesla P100, and `pick_device()` refused: kernel
  status ERROR, `refusing to start on 'unusable'`, no DICOM opened and no weights pulled. This is
  precisely the failure that cost ~1 h twice before. Two details worth keeping:
  - Torch emitted its sm_60 `UserWarning` *before* our line printed, so the log's first screenful
    looks like a crash. It isn't — read down to `WARNING: ... Re-run for a different GPU.`
  - The guard fired at **t=243 s** of session wall-clock, which reads badly against the "costs
    seconds" claim in the README. It isn't the guard: `main()` calls `pick_device()` before timm
    and before the corpus walk. The 243 s is Kaggle's container boot plus the 570 GB mount, and
    it is the floor on *any* re-roll. A bad draw costs ~4 min of session, not ~0.
- **The other twelve fixes still have not touched real DICOMs.** Everything on the decode path —
  the spawn pool, the pinned threads, the PROBE, per-study resume — remains self-tested only. The
  224 shard-0 proving run (`PLAN.md` §9.1) is still the only thing that converts them from
  reasoning into measurement, which is exactly the failure mode of K3–K5. Attempt 5 never reached
  them.
- **The ~19 ms/open cost model is not measured.** It is inferred from the failed runs and
  mis-attributed to `kaggle_01b`, which does not time opens (`FINDINGS.md` §6.1). The PROBE
  replaces it with a real number on the next shard — record it.
- ~~**`SLICES_PER_SERIES_TRAIN` is deliberately excluded from the fingerprint**, so a train/serve
  mismatch on the slice count is not caught by `assert_matches()`. It travels in the manifest as
  data and `kaggle_03` reads it from there; if that indirection ever breaks, nothing raises.~~
  **CLOSED 2026-08-09 — K19.** The exclusion is correct and stays: the count changes no cached
  feature value, so it does not belong in a *cache* fingerprint. The error was reading it from
  the cache manifest at all. It describes the **head**, so it is now stamped into each
  checkpoint from the Dataset that fed it, and `kaggle_03` reads it from there.
- **Two device pickers became one** (`pick_device(allow_mps=...)`), but the `"unusable"` sentinel
  still leaks into callers, each of which special-cases it differently — exit in `kaggle_02`, CPU
  in `kaggle_03` and `fusion/train.py`. That is deliberate (the right answer genuinely differs per
  caller), but it is the kind of shape that drifts.

---

## 2s. Step 5 has no operand, and the instrument does not cover Phase 1 `ANALYSED 2026-08-11`

Written while fold 0 of the gate arm was training. Nothing here is a new measurement — it is
four numbers already in this repo, read against each other for the first time. That is the point:
each was recorded correctly and none of them was ever joined to the others.

### a. The reproduction gate has nothing to reproduce against

README Phase 0 step 5 reads *"if a local run with **its** labels does not land near its published
score, the port is wrong."* `REFERENCE.md` 3.1 records `pilkwang`'s local column as **`—`**. It
publishes no CV, no OOF, no gold number. Its **0.891 is a leaderboard score from
`infer_from_package()`** — 20 pre-trained members, rank-meaned, two resolutions, multi-window
TTA, 74 s, no training (2e). **The gate as specified cannot be run: there is no number to land
near.**

Converting does not rescue it. Two `(local, LB)` anchors exist — `0.632 -> 0.664` and
`0.8544 -> 0.903` — giving "local reads 0.03–0.05 below LB" (2o). That band is **0.02 wide before**
the unmeasured 20-member ensemble gain is added. Any pass margin is smaller than the slop.

### b. The obvious comparison is confounded, and it is 2i's voided result a second time

`score_oof.py` scores against `lixin_gpt56`. The two arms do not train on the same source:

| | ↔ `lixin` |
|---|--:|
| steven family (`runs_port` trains on `steven_v2`) | **0.947** |
| `pilkwang_v2` (the gate arm) | **0.866** |

Scoring both through `lixin` **rewards the arm whose training source resembles the key**. That is
exactly what voided the 6.0σ in 2i — *"Arm B was scored against a near-copy of its own training
targets."* Here it runs against the gate arm rather than for it, and it is the same defect.

**This is the fourth instance of one error class**, and the pattern is the finding:

| | the entanglement |
|---|---|
| 2d | "labels are not the ceiling", measured while trainability was the binding constraint |
| 2i | arm B scored against a near-copy of its own targets — 6.0σ **void** |
| 2o | scored against `targets.csv`, the model's own training source — upward-biased, live bug |
| **2s** | two differently-supervised arms scored through a reference correlated 0.947 / 0.866 |

Every one was caught *after* the run and recorded honestly. **Nothing in the plan asks, before a
run launches, whether the reference is neutral to both arms.** Catching-afterwards is the only
mechanism this project has, and it has now cost four measurements.

### c. The instrument built in step 2 is valid for Phase 0 and not for Phase 1

The report-OOF instrument is excellent and is **valid at fixed targets only** — the reference is
itself a label source. Now read the queue: the **severity-thresholded label read** (`REFERENCE.md`
2.1, "best untested idea on the board") is a label-source change. Phase 1 is label-source changes.
The shared-ceiling thesis says the remaining 0.04 *is* a label problem.

**The best instrument this project has cannot arbitrate its most promising remaining idea.** Step 2
solved the half of the measurement problem Phase 0 needed and the wrong half for what follows.
This was never stated; 2g and 2o each record the fixed-targets caveat, and the plan then queues
work that violates it.

### d. So Phase 1 costs ~5x what the plan prices it at

Gold is the only instrument neutral to a label-source change — it is an expert **image** read, not
derived from reports, so it is external to every candidate source. But gold is 58 studies and
**fold 0's validation set holds 10 of them**:

    fold 0: 10   fold 1: 9   fold 2: 10   fold 3: 9   fold 4: 9

So a label-source experiment needs **all five folds (~18 h) to see gold at all**, and then lands
with the +-0.031 CI that 2g built the report instrument to escape — against differences of ~0.021.
The plan prices label experiments as cheap. They are the most expensive kind here, and the
arbiter is barely sharp enough to read them.

### e. Step 5 depends on step 6, not the other way round

README's table reads `6. One submission for the CV<->LB mapping | blocked on 5`. **It is
backwards.** The reproduction target exists only on the leaderboard, because that is the only
place `pilkwang`'s number exists. "Did we reproduce the fork" is a submission question. Phase 0
has been ordered around reaching a trustworthy local reproduction of a number that was never
local.

### f. What the gate arm should actually be measured against

No absolute target exists, so the target is **a predicted delta**, anchored on the one quantity
measured for both sources on a common instrument (2f, gold-58):

| training source | gold-58 macro |
|---|--:|
| `steven_v2` = `data/targets.csv` (`runs_port`) | **0.8873** |
| `pilkwang/report_labels_v2` (gate arm) | **0.866 +- 0.016** |

Gap **-0.021**, and label-quality gaps transfer at less than 1:1 through 2,871 training studies.

> **Target: the gate arm lands 0 to 0.021 BELOW `runs_port`, paired through `score_oof.py`.**

| paired delta (gate - port) | reading |
|---|---|
| -0.021 .. 0 | faithful port + the known label gap — **pass** |
| ~0 or positive | pass *despite* the `lixin` handicap — **strong** |
| < -0.04 | exceeds label gap **and** handicap — the port likely amplifies label noise |
| clearly positive | contradicts a bench clearing its SE by 4.5σ — suspect port or scoring |

Read it **asymmetrically**: b's confound handicaps the gate arm, so a negative delta is ambiguous
(worse labels, or the confound) while a positive one is clean. This is a *plausibility band*, not
the reproduction gate the README describes, and it should not be called one.

### g. One candidate referee, out of 2i's own table

As a **referee** rather than a label, the rule extractor is the most neutral instrument here:
**0.685** against steven, **0.657** against `pilkwang_v2` — against `lixin`'s lopsided
0.947 / 0.866. For a paired A/B where only the delta matters, **a symmetric-but-noisy referee
beats a sharp-but-lopsided one**, because its bias cancels and `lixin`'s does not. 2f killed it as
a *label source* — 0/12 labels, negative in a rank-mean — and that says nothing about its use as a
yardstick. Untested. It is a scoring change, not a training run, so it costs minutes.

**Do not read this as rehabilitating the extractor.** It is 0.777 and wrong in its own direction;
the claim is only that its wrongness is *balanced* between these two arms, which is the single
property a referee needs and `lixin` lacks.

---

## 2t. Six external concerns, fact-checked `ANSWERED 2026-08-11`

Raised in `REFERENCE.md` (end of file, pushed as 873a394). Answered here in the order they were
raised, each with a **status** and, where the claim is checkable, **the check**. Two are adopted
outright, one is a correction to the concern itself, and one is answered by evidence already in
this repo that nobody had pointed at the question.

### 1. Submit now, beside step 5 — **ADOPTED, and independently arrived at**

> *"The reason to submit now isn't measurement, it's risk retirement... Step 6 should run beside
> step 5, not behind it."*

**Agreed, and §2s-e reached the same conclusion from the other direction** — the reproduction
target only exists on the leaderboard, so step 6 is what step 5 was trying to be. Two independent
routes to "unblock 6" is the strongest signal in this document.

The risk-retirement half is the part §2s missed and it is the better argument: `kaggle_03_submit.py`
has **never executed against a real test DICOM**. The 9 h cap, no-internet, weights-as-a-Dataset,
degenerate series, a study missing a plane — none of these are discoverable locally, all are
schedule risk rather than score risk, and none competes with the local instrument for anything.

The CV↔LB point is also correct and sharper than it looks: the conversion is an interpolation
between **two** public anchors (`0.632→0.664`, `0.8544→0.903`) and it is load-bearing for the
claim that our gap is 0.19 rather than 0.22. **Two foreign data points should not carry a
headline number.** Submitting replaces both with one of ours.

### 2. Rules risk on external data — **CORRECTED: already read, and the baseline is clear**

> *"The same question applies to the public label tables you've already adopted as
> data/targets.csv... the compliance question isn't sitting beside your plan, it's sitting
> underneath your baseline. Read the page today."*

**The page was read on 2026-08-10** and the finding is in `REFERENCE.md` §1.2, which cites
clause **3.6.b**: *"Public sharing on Kaggle is permitted and deemed OSI-licensed → **using the
public LLM label tables is fine.**"* `data/targets.csv` is `steven_v2`, a public Kaggle Dataset,
free and equally accessible — the paradigm case under §2.6.a. **Fold 0 is not built on an
unresolved compliance question.**

What *is* open is narrower and it is not a reading task: **§1.3, external MRI datasets**
(MRNet, fastMRI+, OAI, SKM-TEA). All are free but all sit behind a click-through research
agreement, so "equally accessible at no cost" is genuinely arguable. The host was asked twice,
replied to the thread, and **answered only the LLM question**. That gates a Phase 2 lever and it
is blocked on a *host answer*, not on us. One forum post, costs nothing — and it is now the
oldest unactioned item on the board.

**Check:** `grep -n "3.6.b" REFERENCE.md` and clause 6 of `COMPETITION_RULES.txt`.

### 3. No compute budget — **ADOPTED. Here is one, from measured numbers**

The concern is correct and the gap was real. Budget, all figures measured rather than estimated:

| | measured |
|---|--:|
| deadline | **2026-10-22 — 72 days, 10.3 weeks** |
| one fold, port @336 | **3.6 h** (§2p, memory-bound, not the 2.6 h budgeted) |
| one 5-fold experiment | **~18 h** |
| Kaggle quota | 30 h/week, and the GPU lottery refuses ~4 draws in 5 |
| M5 realistically available | ~40 h/week (it is also the daily-driver laptop) |

**So the whole remaining project is ~20 five-fold experiments, and that is if nothing else runs.**
Phase 1 as written — rank-means across seeds, then resolutions, then backbones — spends that
budget several times over. **Any plan item that does not name its cost in folds is not a plan
item.** This is the constraint the phase list never had.

### 4. Promote the efficiency track to co-primary — **ADOPTED as a live decision, not yet decided**

The argument is strong and the arithmetic favours it: **$18,000 across three efficiency prizes
against $5,000 for 10th** in a field of 908, three places instead of ten, a far thinner field,
and it rewards exactly what this project has been good at — measurement discipline, cheap decode,
knowing what the pipeline costs. §6.2 concluded *accuracy dominates the efficiency formula*, which
is true **of the formula** and says nothing about which track to enter. Those were conflated.

**It is not free**: the efficiency track still needs a competitive score, so it is a constraint
added on top of accuracy work, not a substitute for it. **Decision criterion, to be applied after
the first submission:** if our measured LB lands below ~0.87, the accuracy track is out of reach
inside the compute budget in §3 and efficiency becomes primary. If it lands above, run both.

### 5. The boring hypothesis — **ENGAGED, and the repo already has evidence against it**

> *"The boring hypothesis — the top teams simply ensemble more and bigger models over more data —
> is the one that's usually correct on Kaggle, and the plan doesn't really argue against it."*

Fair, and the plan should have argued it. **It can, from `REFERENCE.md` 3.1, and the argument
mostly goes the other way:**

**Both public anchors bracketing the gap are ALREADY ensembles.** `pilkwang` 0.891 is a
**20-member rank-mean**; Yash B3 0.903 is a **mean of 5 fold sigmoids**. So the 0.04 to the 0.942
top is a gap *between ensembles*, not between a single model and an ensemble. Ensembling 5 → 20
members buys perhaps +0.005–0.010 on a task like this, and it is strongly diminishing — it does
not plausibly produce +0.04 on top of a stack that is already ensembled.

That does not prove the interesting hypothesis; more data and bigger backbones remain live and
both are things money buys. But it does mean the boring hypothesis **cannot be assumed** here,
and the specific "just ensemble more" form of it is weakly contradicted by numbers already in
this file.

**Distinguishing evidence, decided now as asked:** our own first submission gives a single-model
LB point. Placed against `pilkwang`'s 20-member 0.891 — same architecture family, same slot
scheme, ours single-model — the single→ensemble delta stops being a guess. **That is one more
reason step 6 runs now** (§1). If the delta is large, boring wins and the answer is compute. If
small, the 0.04 is something else and the label/crop bets are the right ones.

### 6. Fix §2r-B6 first — **DONE AND VERIFIED, and the concern overstated the damage**

> *"A paired sigma that isn't reproducible run-to-run because it iterates a set undermines every
> number your plan is now built on, including the +0.0171."*

Fixed (`sorted()` at `score_oof.py:144`). **But the scope was measured before fixing, and it is
narrower than the concern says.** Four runs at different `PYTHONHASHSEED`, pre-fix:

    +0.0171 +-0.0091 -> 1.9 sigma, P = 0.969
    +0.0171 +-0.0087 -> 2.0 sigma, P = 0.978
    +0.0171 +-0.0088 -> 1.9 sigma, P = 0.980
    +0.0171 +-0.0086 -> 2.0 sigma, P = 0.978

**The delta is identical every time.** It is computed on all studies at once (`base =
arange(len(yy))`), so it is order-invariant by construction; only the *resamples* moved. What
wobbled was the third decimal of the SD — 1.9–2.0σ, P 0.969–0.980. **§2q's +0.0171 and its 1.9σ
were never in danger.** Post-fix, all four seeds give `+0.0171 +-0.0088 -> 1.9 sigma, P = 0.977`,
which is the number §2q already published.

`score()` itself was never affected: it iterates `oof.index` (deterministic CSV order) and uses
`restrict` only as a membership test. So **0.7229 ± 0.0048 and every per-arm figure were always
reproducible.** Worth stating plainly, because "undermines every number" would otherwise sit in
the record as true.

**Check:** `for s in 1 2 3 4; do PYTHONHASHSEED=$s python fusion/score_oof.py fusion/runs_baseline fusion/runs_port | grep PAIRED; done`

---

## 2u. The gate arm died at epoch 7 and left nothing — the loop had no checkpoint `2026-08-12`

The label-swap arm (§2s-f) ran 7 of 10 epochs over ~3 h and was killed. **`fusion/runs_gate/` was
empty afterwards.** No partial result, no weights, no OOF — the whole run was unrecoverable.

**Cause of the kill: undetermined, and stated as undetermined.** No Python traceback reached the
log, and `log show --predicate 'eventMessage CONTAINS "jetsam"'` over the window found nothing, so
an OS memory kill and an external stop cannot be distinguished from this side. §2p makes a memory
kill plausible — the previous fold 0 drove swap to 24.47/25.6 GB — but plausible is not measured
and this is not being recorded as an OOM.

**The finding is not the kill. It is that a kill cost everything.** `train_port.py` wrote
`fold{f}.pt` only when `run_fold` RETURNED, and `oof_all.csv`/`summary.json` only after every fold.
So any death inside a fold — OOM, a closed laptop, a stray Ctrl-C, a 9 h Kaggle cap — discarded the
entire fold. Against §2t-3's budget of **~20 five-fold experiments for the whole remaining
project**, an unrecoverable long run is the expensive class of bug, and this one had been sitting
in the file since step 4 was built.

**The arm was healthy when it died**, which is worth recording because the raw loss invited the
opposite reading. Normalised to each arm's own span — the two differ, `runs_port` has floor 0.2040
/ prior 0.4640 while the fork's 5-level labels give floor **0.2916** / prior 0.4525 — it tracked
the completed run to within two points the whole way:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `runs_port` % of span | 4% | 16% | 21% | 28% | 35% | 42% | 51% |
| gate arm % of span | 2% | 13% | 20% | 26% | 33% | 41% | **50%** |

**Raw losses across these two arms are not comparable and never were** — the floor differs by
0.088. A gate arm finishing near 0.35 against `runs_port`'s 0.2877 would look far worse and would
be roughly the same performance. That trap is now live for whoever reads the next run.

### Fixed

Per-epoch checkpointing in `run_fold`, plus `--resume`:

- One `fold{N}_last.pt` per epoch, **overwritten** each time — 175 MB measured (it carries AdamW's
  two moments as well as the state_dict), so the cost is 175 MB of disk and ~1 s per epoch.
- Written to `.pt.tmp` and **renamed**, because a kill *during* the save would otherwise leave a
  truncated checkpoint that fails to load — exactly when it is needed most. Rename is atomic
  within a filesystem.
- `--resume` is **opt-in** and guarded by `_fold_cfg()`, which must match exactly.

**The config guard is the part that matters, and it is a §2s defence.** It includes the labels
path, so resuming a `targets.csv` run into a `report_labels_v2` run **refuses** instead of
silently producing an arm trained on two label sources with nothing in the output to say so —
the provenance error class, arriving through the back door of a convenience feature. It also
includes `epochs`, because `OneCycleLR`'s `total_steps = epochs x len(tr_dl)`: resuming across an
epoch-count change would restore a scheduler state that no longer means anything.

### Verified — and it took three attempts, two of which were bad tests

Worth writing down in full, because the two failures were both *testing* errors of the kind §2r-A3
is about: a guard nobody exercises, and then a test that does not exercise it either.

1. **Attempt 1 — `--epochs 3` against an `--epochs 2` checkpoint. Correctly REFUSED.** The test
   was wrong, not the guard: `epochs` is in `_fold_cfg` precisely because `OneCycleLR`'s
   `total_steps = epochs x len(tr_dl)`.
2. **Attempt 2 — hand-rewound `st['epoch']` to 1 without rewinding `st['sched']`.** Crashed:
   `ValueError: Tried to step 16 times. The specified number of total steps is 15`. That is an
   inconsistent checkpoint the real code **cannot** produce — a genuine death saves `epoch` and
   `sched` together. **And the grep filter on that test was `"ep [0-9]+/|RESUMED"`, which
   swallowed the traceback**, so the run looked silently empty rather than failed. Two errors in
   one test.
3. **Attempt 3 — a real `kill -9` mid-run, which is the only version that means anything.**

```
    ep 1/3  loss 0.5205  16.3 min          <- killed here
  checkpoint holds epoch 1 of 3 | n_train 40
  RESUMED from fold0_last.pt at epoch 1/3
    ep 2/3  loss 0.4519  0.7 min
    ep 3/3  loss 0.4348  15.3 min
```

**Attempt 2's crash paid for itself**, because the `ValueError` exposed a real gap the guard was
missing: `total_steps` depends on `len(tr_dl)`, and the corpus downloads incrementally, so
resuming after more studies land would restore a scheduler built for a different total. It would
surface as that same error **at the end of the run** — loud, but hours late. `n_train` is now in
`_fold_cfg`, so it fails in the first second instead.

**A resumed run is NOT bit-identical to an unbroken one, and must not be quoted as one.** Against
the uninterrupted baseline the resumed epochs differ — 0.4519 / 0.4348 against 0.4486 / 0.4387 —
because `SlotDataset.rng` and the shuffle order restart rather than being checkpointed. Resume
**recovers** a run; it does not **reproduce** it. For salvaging 3 h that is the right trade, but a
resumed fold should not be A/B'd against an unbroken one at the third decimal.

Also verified: `--resume` against a checkpoint written under **different labels** prints both
configs and exits — the provenance defence, working.

---

## 2v. The laptop sleeps, and it has been inside every measurement `MEASURED 2026-08-12`

The gate arm (§2u) did not crash and was not killed by anyone. **The machine went to sleep.**

    kill / pkill in shell history ....... none
    crash report / JetsamEvent ......... none          <- so not an OOM kill either
    log show, 01:00-01:06 .............. 0 lines       <- the machine was not awake
    pmset -g log ....................... 2026-08-12 01:04:37
                                         Entering Sleep state due to 'Maintenance Sleep', 1025 secs

01:04:37 is the minute the training log stops. During that run the machine slept ~20 times,
including a **Clamshell Sleep at 22:23** and six thermal emergencies. The log holds **16 Thermal
Emergency sleeps** overall.

### This was not new, and that is the finding

The same thing happened to the run in §2p, and nobody looked. `runs_port` fold 0 ran ~22:06-01:42
on 08-10/11; sleeps inside that window sum to **≈7,005 s — about 2 of its 3.6 hours**:

| epoch | wall clock | sleep in window | ≈10 min compute + sleep |
|---|--:|--:|--:|
| 1–5 | 9.5–10.7 min | none — sleeping starts 22:59 | clean |
| 7 | 15.4 | 5.0 | 15.0 |
| 9 | 22.2 | 12.6 | 22.6 |
| 10 | **79.8** | **~54** | ~64+ |

Epoch boundaries here are inferred from durations rather than logged timestamps, so this is a
strong correlation and not a proof. It is more than enough to disqualify the timing evidence:
**every slow epoch is explained by sleep without invoking memory at all.**

### Consequences

- **§2p's headline is not established by its evidence.** Annotated in place. The swap reading
  stands; "memory pressure is what made it slow" does not, and `bench_port.py`'s exoneration was
  argued from the same contaminated timings.
- **Every wall-clock figure taken on this machine before 2026-08-12 is suspect**, including the
  3.6 h/fold that §2t-3's compute budget is built on. The budget's *shape* is right — the machine
  is the scarce resource — but the number needs re-deriving under `caffeinate`.
- **`caffeinate -i` is now mandatory for any long run.** It prevents idle and maintenance sleep.
  It **cannot** prevent Thermal Emergency Sleep, which is hardware protection, and 16 of those
  says the box is thermally saturated under sustained MPS load.
- **If the real ceiling is thermal rather than memory, the tuning advice inverts.** A *smaller*
  sustained load (batch, workers) could finish sooner by not tripping emergencies. Untested, and
  it should be tested before anyone spends 18 h on folds 1-4.
- **§2u's checkpointing is not a nicety.** On a machine that sleeps ~20 times per run, resume is
  the only reason a multi-hour job ever finishes.

### The pattern, stated plainly

Five instances now: §2d, §2i, §2o, §2s, and this. **An instrument entangled with something
uncontrolled.** Four of the five were caught only after the measurement was quoted. The one
mechanism that would have caught this one is not a rule about references — it is that nobody
asked what the machine was doing while it was being measured. `pmset -g log` costs one second and
has been available the whole time.

---

## 2w. Course change: stop out-building the fork, start out-ensembling it `DECIDED 2026-08-12`

Prompted by a direct challenge — *"is our solution viable? why aren't we using the .89 repo?"* The
answer is that there is no good reason, and the arithmetic has said so for a while.

> **This table is the stale 08-10 read. E1 ran on 08-12 and every line moved against us — see
> §2x for the live board.** 10th is **0.934**, top **0.946**, and our 0.891 is **rank 400 of
> 1,276**. The course change below is correct and gets *stronger*; only its numbers were wrong.

| | LB |
|---|--:|
| our own pipeline, best estimate (§5 conversion) | **~0.76** |
| **the fork, already submitted 2026-08-09** | **0.891** — rank 230/908 → **now 400/1,276 (§2x)** |
| best visible public (see caveat below) | 0.903 — *writeup, not downloadable; the best kernel is 0.899* |
| **10th place — the last prize** | ~~0.926~~ → **0.934 (§2x)** |
| top | ~~0.942~~ → **0.946 (§2x)** |

**Our own pipeline is 0.13 BELOW a thing we already have banked.** Phase 0 spent its whole budget
rebuilding, in our own code, an architecture that is free to download, and reached 0.7229 with one
fold worth +0.0171. At that increment the port needs ~8 more equivalent wins to reach its own
starting line.

**README rule 6 already says "The fork is the base, not a reference."** It was written and then
not followed; every step since has treated it as a reference. This section is the correction.

### Where the "own code" standing decision is right, and where it is not

*"If a component cannot be explained from first principles it does not ship, no matter what it
scores"* is a good **research** value and a bad **competition** value, and the plan never split
them. It holds for what we intend to CHANGE — you cannot modify what you do not understand. It
does not hold for what we intend to KEEP. Rebuilding `SlotHead` bought understanding of a
component we were never going to alter.

**The local trainer itself remains justified** and that argument is unchanged: the fork cannot be
fine-tuned on Kaggle for under ~8 h/arm against a 30 h quota, so testing *our* ideas needs a local
loop (§2e). What was wrong is treating the local trainer's SCORE as the project's score.

### The public field, surveyed live 2026-08-12 — and REFERENCE was stale

`kaggle kernels list --competition rsna-knee-abnormality-detection --sort-by voteCount`:

| kernel | votes | last run |
|---|--:|---|
| `pilkwang/rsna-knee-baseline-v1` | **295** | 08-08 |
| `prvsiyan/rsna-knee-read-the-report-then-the-knee` | **192** | 08-11 |
| `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` | 131 | 08-11 |
| `romanrozen/rsna-knee-data-structure-eda-baseline` | 91 | 08-10 |
| `aadigupta7686/0-899-let-me-cook` | 79 | **08-12** |
| `sakhawathossen/rsna-knee-enhanced-ensemble` | 72 | 08-10 |
| `romantamrazov/rsna-knee-dinosaur-v2` | 69 | **08-12** |

Three corrections to `REFERENCE.md` 3.1, which was read 08-10 and has moved:

- **`0.899 let me cook` is `aadigupta7686`, not `prvsiyan`.** 3.1 attributes it to prvsiyan.
- **No `Yash Bishnoi` / B3 kernel exists publicly.** Searches for `yash`, `b3` and `efficientnet`
  return only fuzzy matches. So **the 0.903 B3 line is a writeup, not a downloadable kernel** —
  reproducing it is a TRAINING JOB, not a download. That was the assumption the first version of
  this course change rested on, and it is wrong.
- **`prvsiyan` is a second strong lineage with a public notebook**, heavily forked already
  (`hyunseop1`, `bang1850`, `gengsr` all carry copies). 3.1 lists it at 0.899.

**All LB figures here come from `REFERENCE.md` and are two days stale. Re-verify against the live
leaderboard before any of them is load-bearing** — see [[check-whats-free-first]]; this is the
second time in three days that a belief about the outside world expired.

### The plan

**0.891 is the FLOOR. Every number is a delta on top of it.**

1. **Verify the live leaderboard and the two lineages' actual scores.** Minutes. Everything below
   depends on numbers that are currently two days old.
2. **Rank-mean `pilkwang` + `prvsiyan`.** Two published notebooks, different lineages, both with a
   working inference path. This is the cheapest +0.01–0.02 available and it is a Kaggle run, not a
   training job. **Cost: 1 submission.**
   > **§2x amends this: benchmark it against 0.899, not 0.891.** The free plateau is
   > `aadigupta7686` at **0.899** (183 teams on it), so submitting *someone else's better kernel*
   > is the zero-work competitor this step has to beat. A rank-mean landing 0.90–0.91 is rank
   > ~75–130 — real, but it books a +0.01 that is only +0.002 over the free alternative.
3. **Submit it.** Also retires the §2t-1 risk — `kaggle_03_submit.py` has still never run against
   a real test DICOM — and yields the single-model LB point that tests the boring hypothesis
   (§2t-5).
4. **Make the port EARN a slot.** At ~0.76 it would *hurt* a 0.891 rank-mean. Its bar is: does
   adding it to the ensemble improve a held-out score? Until it clears that, it gets no more
   compute. This is the decision the plan has never made.
5. **Then spend the differentiators where they compound** — severity-thresholded labels and
   anatomical crops, measured as deltas on the ensemble rather than on our own 0.72 pipeline. The
   label bet still has no valid instrument (§2s); solve that first or it is 18 h for a number
   nobody can read.
6. **Treat efficiency as co-primary (§2t-4).** $18,000 over three places, a thinner field, and
   `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` is a Kaggle-staff kernel that makes the
   target measurable. In the accuracy track 0.891 is rank 230.

### What we keep

Not wasted, just not score: the site-grouped folds (+0.024 of leakage most public teams still
carry), the report-OOF instrument at fixed targets, K16 resolved by measurement, the four guards,
resume, and — as of today — the knowledge that this laptop was asleep inside every timing ever
taken here (§2v). **That is the apparatus for judging an ensemble honestly. It was never going to
BE the ensemble.**

---

## 2x. E1 has RUN: the board is worse than §2w assumed, in four places `MEASURED 2026-08-12 17:22 UTC`

Full 1,276-team leaderboard pulled with `kaggle competitions leaderboard -d`, plus a live kernel
list. **Every correction runs in the pessimistic direction**, which is the argument for having run
it before spending a submission on the §2w plan.

| | §2w believed (read 08-10) | **live 08-12** | |
|---|--:|--:|---|
| teams | 908 | **1,276** | +40% in three days |
| top | 0.942 | **0.946** | |
| 10th — the last prize | 0.926 | **0.934** | cutoff **+0.008 in three days** |
| our banked 0.891 | rank 230 | **rank 400** | **−170 places for doing nothing** |
| gap to the prize | 0.035 | **0.043** | |

### The finding §2w did not anticipate: 0.891 is a commodity, and it is not even the free one

Score plateaus with ≥15 teams — a plateau *is* a shared public kernel, since independent training
runs do not agree to three decimals:

| plateau | teams | best rank | what it is |
|--:|--:|--:|---|
| 0.900 | 15 | 129 | lightly-tuned `0.899` |
| **0.899** | **183** | 144 | `aadigupta7686/0-899-let-me-cook`, unmodified |
| 0.897 | 22 | 335 | |
| **0.891** | **73** | **400** | the `pilkwang` fork, unmodified — **this is us** |
| 0.500 | 102 | 1163 | degenerate submissions |

**73 teams hold our exact score.** We are not on a floor we built; we are in a queue, and the
queue is 183 teams deep one plateau above us. **The free plateau is 0.899, not 0.891** — the
better public kernel is a download, worth ~256 places for one submission and no work.

**Consequence for §2w step 2.** "Rank-mean `pilkwang` + `prvsiyan`, the cheapest +0.01–0.02
available" now has a competitor it must beat: *just submit the 0.899 kernel*. If the rank-mean
lands at 0.90–0.91 it is rank ~75–130 — real, and still ~0.025 short of a prize. Neither is
wrong; both are cheap; but the ensemble must be benchmarked against **0.899**, not 0.891, or it
will book a +0.01 that was +0.002 over the free alternative.

### What this does to the strategy — it sharpens §2w rather than replacing it

**The entire public field is compressed into 0.891–0.900.** 326 teams sit at or above 0.899;
only 33 clear 0.92 and only 10 clear 0.934. So the real target was never our 0.043 to 10th — it
is the **~0.035 between the free public plateau and the prize**, and that is the bar every idea
on this board has to clear.

That is the §2w "unexplained 0.04" restated with better evidence, and it now has a *mechanism
visible in the distribution*: the public teams pile into a 0.01-wide band because they share the
report-derived label tables, and the top ten are 0.035 clear of the band on something unpublished.
**The two differentiators are therefore not garnish on an ensemble — they are the whole game**:
the severity-thresholded label read (`REFERENCE.md` §2.1) and the anatomical crops. What changes
is only where they get measured: as deltas on ~0.899, never on our own 0.72 pipeline.

### And the clock is visible now

The cutoff moved **+0.008 in three days** while the field grew 40%. Do not model that as linear to
2026-10-22 — early-competition boards move fastest — but the direction is not in doubt, and
**0.934 is a lower bound on what a prize costs**, not a target. Any plan that lands at 0.91 in
October lands outside the money.

**Standing rule, third time this project has needed it ([[check-whats-free-first]]): a leaderboard
number older than a few days is not evidence. Re-run E1 before any submission decision.**

---

## 2y. The fork ships its OOF, so the port's slot question is answered for free `MEASURED 2026-08-12`

`pilkwang/rsna-knee-weights` contains **`oof.npz` (368 KB) and `manifest.json`** alongside the 20
`.pt` files. `oof.npz` is honest out-of-fold predictions for **all 4,407 training studies** on the
same 12 targets in the same order, with a `gold_mask` that sums to exactly **58**. So the 0.891
fork can be scored on our instrument with **no GPU, no Kaggle run and no submission** — and every
comparison against it before today was a number read off a web page.

Imported with `fusion/import_pilkwang_oof.py` → `fusion/runs_pilkwang/oof_all.csv`, which makes it
an ordinary arm for `fusion/score_oof.py`.

### The 20 members are ONE config

`manifest.json`: 20 members = **5 folds × 4 seeds** (2026, 7717, 20260808, 31337), and
**`distinct_configs: 1`** — dinov2-small @336, 12 slices, group 3, `crop_mm` 130, band [0.2, 0.8],
`unfreeze_last: 6`, `cls_mean`, the six `SAG/COR/AX` slots. Mean member holdout AUC **0.8398**,
mean gold-58 **0.8375**.

**0.891 is a seed-and-fold average of a single architecture, not a diverse ensemble.** That is why
`prvsiyan` gained by bolting on RadImageNet ResNet-50 (0.899 → 0.906) and it says the
architectural headroom above the fork is real. It also means **our port is a 21st sample of the
same config** — it was built to reproduce this exact recipe.

### The port does not earn a slot, and the result is clean because it is asymmetric

`fusion/score_oof.py fusion/runs_pilkwang fusion/runs_port`, paired on the 681 shared scorable
studies:

| arm | macro |
|---|--:|
| `runs_pilkwang` | **0.8434 ± 0.0061** |
| `runs_port` | 0.7323 ± 0.0086 |
| **paired delta** | **−0.1111 ± 0.0072 — 15.4σ, 0/12 labels won** |

**The reference leans toward the port and it lost anyway.** `lixin_gpt56` correlates 0.947 with
`steven_v2` (the port's targets) and 0.866 with `pilkwang_v2` (theirs), the §2s asymmetry — so
this measurement is handicapped *in our favour* and still reads −0.111. Only a positive delta
would have been ambiguous. This one is not.

`fusion/blend_test.py` then asks the actual §2w-step-4 question — not "is it better alone" but
"does base+port beat base" — swept over the rank-blend weight:

| w | macro | vs base |
|--:|--:|--:|
| 0.00 | 0.8434 | — |
| 0.05 | 0.8429 | −0.0005 |
| 0.10 | 0.8418 | −0.0016 |
| 0.20 | 0.8380 | −0.0054 |
| 0.50 | 0.8120 | −0.0314 |
| 1.00 | 0.7323 | −0.1111 |

**Monotonic. No weight helps.** And this is *not* the §2f near-duplicate trap: mean rank
correlation between the arms is **0.639** (min 0.478), far below the 0.87–0.95 at which reader
fusion died. **The port had genuine diversity to sell and is simply not good enough to pay for a
slot.** That is a cleaner and more damning result than redundancy would have been.

**§2w step 4 is therefore CLOSED, for free, in minutes.** The port gets no more compute as an
ensemble member. It remains what §2e justified it for: a local loop for testing *our* ideas, one
that the frozen cache structurally could not provide.

### What we hold that DOES compose with the fork

The apparatus, exactly as §2w predicted — and now with a demonstrated use, since the measurement
above is one no submission could have bought:

1. **Site-grouped folds vs theirs.** `prvsiyan`'s `assign_group_balanced_folds(seed=20260809)`
   keeps **normalized-report groups atomic** — a duplicate-report guard, *not* a site guard. We
   measured site leakage at **+0.024 (~5σ)** over 265 scanner fingerprints. Their blend weights
   are OOF-selected on report-grouped folds, so they are fitted on an optimistically biased
   signal. Re-selecting those weights on `data/folds_site.csv` is a **scoring change, no
   training**, and it is the cheapest remaining shot at a delta on 0.906.
2. **A 45× bigger selection set.** `prvsiyan` selects on the **58** image-adjudicated studies and
   says so plainly — *"these tests reuse the same 58 image-adjudicated subjects, so they measure
   estimator stability rather than independent clinical generalization."* Their V34 PCA arm's
   claimed **+0.0273** was chosen that way. Our report-OOF reads ±0.0046 over 2,612. **Testing
   whether their arms survive a real instrument is free, and a *removal* can gain score.**
3. **K16.** Every one of the 20 members carries `rules: {order: 'normal', lat: 'centre'}` — no
   per-series slice-direction handling at all. We resolved the bit by measurement for **8,048
   sagittal series, 50.4% reversed, cross-validated 21/21**. Consistent train/infer, so it is not
   a bug in their pipeline — it is *signal they are leaving on the floor*, and a member trained on
   the resolved direction would be genuinely diverse from all 20 rather than a 21st seed.

---

## 2z. Re-fitting their blend weights on site-grouped folds gains NOTHING `MEASURED 2026-08-12`

§2y proposed this as "the cheapest remaining shot at a delta on 0.906": `prvsiyan` selects its
per-target rank-blend weights on report-grouped folds, we hold site-grouped folds, and §2j
measured site leakage at +0.024. **The hypothesis is wrong. Recorded here rather than left
standing in §2y.**

`fusion/fold_scheme_test.py`, nested selection over 4,349 non-gold studies — pick per-target
weights on the training folds under each scheme, always score on the **site** held-out fold:

| held-out fold | select on SITE | select on REPORT-like |
|--:|--:|--:|
| 0 | 0.8472 | 0.8471 |
| 1 | 0.8430 | 0.8429 |
| 2 | 0.8527 | 0.8528 |
| 3 | 0.8510 | 0.8510 |
| 4 | 0.8505 | 0.8508 |
| **MEAN** | **0.8489** | **0.8489** |

**Delta −0.0000.** The selector that could see scanner structure and the one that could not chose
weights that perform identically on site-held-out data.

And the whole selection is nearly free anyway: all-`ours` scores **0.8482**, per-target selection
**0.8489** (+0.0007), fixed 50/50 **0.8424**, all-`imported` **0.8036**.

### Why — and this is the transferable part

**Site leakage inflates an absolute estimate; it does not necessarily reorder a low-dimensional
selection.** One parameter per target over a 21-point grid, chosen on ~3,500 studies, is a
very low-variance fit: leakage shifts every candidate's score together and leaves the argmax where
it was. §2j's +0.024 is a statement about how good a model *looks*, not about which of two blend
weights *wins*.

So the standing rule needs a qualifier it never had: **"nothing gets compared against an external
score except under site-grouped folds" is right for reporting a score and overkill for choosing
one scalar.** Site-grouping earns its keep where the fitted object is big enough to memorise a
scanner.

**This makes §2y hypothesis 2 the better bet, not a worse one.** The argument above protects
prvsiyan's *per-target weights* precisely because they are low-dimensional and fitted on
thousands of studies. It gives **no protection at all** to their V34 PCA arm, whose claimed
+0.0273 was selected on the **58** image-adjudicated studies — high-variance selection on a tiny
set is the regime where selection actually overfits, and they say themselves it "measures
estimator stability rather than independent clinical generalization." That one is still live and
is now the first thing to test.

**Caveat on this experiment.** The "report-like" arm is a random 5-way split under their seed, not
a reproduction of `assign_group_balanced_folds` (which also balances gold and pseudo-label mass).
It is a fair proxy for *a selector that cannot see site structure* — which is the variable under
test — but it is not their exact scheme. The two arms are `ours`/`imported` from `merge_gain.npz`,
the only two aligned prediction matrices their artifacts expose; `oof.npz` ships the already-merged
result, so **the 20 members cannot be re-weighted individually from anything published.**

---

## 3a. External survey: three rejects, two keeps, and one filter that does most of the work `2026-08-12`

Searched GitHub and the literature for work on this problem shape. **The filter is §2w's: it must be
measurable as a delta on the ~0.906 ensemble we are building on, without out-training a field
whose floor already beats our trainer by 0.11 (§2y).** Most of what turned up fails it, and the
failures are recorded here so they are not re-proposed.

### THE FILTER: macro-AUROC is invariant to per-label monotone transforms

Worth stating once, in this file, because it silently kills a whole class of attractive ideas.
AUC depends only on the *order* of scores within a label. So **any per-label recalibration,
threshold, prevalence/prior correction, or temperature applied to finished predictions is worth
exactly zero.** `pilkwang`'s own notebook says the same thing — *"the scores for label i, so
calibration and thresholds are worth nothing."*

A re-ranking is different from a calibration and only the former can move the metric: it needs a
**new per-study signal**, not a new function of the existing score.

### REJECT — Gold Loss Correction / noise-transition-matrix methods

[Hendrycks et al. GLC](https://arxiv.org/pdf/2111.14932) and the transition-matrix family estimate
label noise from a small trusted clean set — which maps temptingly onto our **58 gold vs 4,407
report-derived**, and onto §2b's "one-directional threshold error".

**Rejected as a post-hoc method by the filter above**: a per-label transition correction applied to
finished predictions is a monotone transform and cannot change macro-AUROC. GLC works by
correcting the *loss during training*. Applying it means retraining, which is the activity §2y
just measured us losing at. Keep the framing — the report→gold map really is a noisy-label
problem with a trusted anchor set — and discard the method.

### REJECT — CoPAS, despite being the closest published match

[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/) · [code, Apache 2.0](https://github.com/zqiuak/CoPAS).
*Twelve* knee abnormalities, multi-sequence, multi-plane — the nearest thing in the literature to
this competition's task. It still fails on four counts:

- **Different twelve.** Theirs are MENI/ACL/CART/PCL/MCL/LCL/EFFU/CONT/PLICA/CYST/IFP/PR. Ours
  split meniscus and OA by compartment and include Synovitis, Baker's and Fracture. Maybe five
  overlap, none exactly.
- **Weaker.** Average AUC **0.812** internal, against 0.906 we can download today.
- **No pretrained weights** — the repo ships code and 50 sample studies. Using it is a
  from-scratch training job.
- **Needs 5 specific sequences** (PDW sag/cor/ax + cor T1 + sag T2). Our metadata resolves only
  3 planes × 2, since `Fluid_Sensitive ≡ Fat_Suppression` over all 24,371 series.

**Keep one idea from it.** Its ablation reports that cross-plane attention beats concatenation and
max-pooling for multi-sequence fusion. All 20 `pilkwang` members aggregate their six slots with
plain `cls_mean` (`manifest.json`), so slot fusion is a named weak point in the thing we are
building on. That is a hypothesis about *their* architecture, not a plan — acting on it is a
training job, so it stays parked behind everything below.

### REJECT (for now) — MRI foundation models

[Triad](https://pmc.ncbi.nlm.nih.gov/articles/PMC11952655/),
[MRI-CORE](https://arxiv.org/html/2506.12186v1) (DINOv2-pretrained on 116,806 volumes),
[RadFM](https://www.nature.com/articles/s41467-025-62385-7). A genuinely diverse encoder is
exactly what pays here — `prvsiyan` gained 0.899 → 0.906 by adding RadImageNet ResNet-50. But
these ship *encoders*, so using one means training a head and probably fine-tuning: a multi-week
bet on the M5, in direct competition with the activity §2y measured us losing at by 0.11.
**Revisit only if a submission-measured experiment shows encoder diversity is where the remaining
0.03 lives.**

### KEEP 1 — the submission budget changes the instrument problem, which is the project's #1 blocker

The standing blocker since 08-10: *"the instrument does not cover Phase 1"* — report-OOF is valid
at fixed targets only, and gold-58 reads at ±0.031 against effects of ~0.021. Every label-source
idea has been stuck behind it.

> **NARROWED BY §3c (same day): this is true of the `pilkwang` lineage ONLY.** `prvsiyan`
> trains its added arms in-notebook against an ~8.7 h limit, so a submission there costs hours,
> not seconds. Read the paragraph below as being about `infer_from_package()` kernels.

**It is no longer binding, and the reason is arithmetic we never did.** A submission is
`infer_from_package()` — **74 seconds** of GPU (§2e). The 30 h quota was never the constraint on
*inference-only* experiments; at 5 submissions/day that is **~6 minutes of quota per day for five
independent leaderboard reads.** "The leaderboard is the instrument" was adopted and retracted the
same day on 08-10 as unaffordable — **that retraction was correct for training runs and wrong for
re-ranking experiments**, and nobody separated the two.

So: **post-hoc re-ranking ideas can now be measured on the real target, five a day, for minutes of
quota.** That is the unlock. It does not extend to anything that trains.

### KEEP 2 — the ensemble-selection-overfitting literature indicts `prvsiyan`'s PCA arm specifically

The [low-data ensembling work](https://arxiv.org/pdf/2010.06866) and the
[cross-validation guide](https://pmc.ncbi.nlm.nih.gov/articles/PMC10388213/) both name the failure
directly: *with small validation sets the ensembling process itself overfits*, and the standard fix
is bootstrapped greedy selection rather than argmax on the raw set.

`prvsiyan` selects on **58** image-adjudicated studies and says so — *"these tests reuse the same 58
image-adjudicated subjects, so they measure estimator stability rather than independent clinical
generalization."* Their V34 PCA arm's claimed **+0.0273** was chosen that way.

§2z showed low-dimensional selection is robust to a bad fold scheme; **that argument gives their
PCA arm no cover at all**, because 58 studies is the high-variance regime it excludes. This is
free to test on OOF we already hold locally, needs no GPU, and — unusually — a **removal** can gain
score. **First action.**

### The severity re-rank, restated correctly

Still the best idea on the board (`REFERENCE.md` §2.1), and the literature supports the mechanism:
[VisualCheXbert](https://www.researchgate.net/publication/350736257) lifts weighted F1 0.55 → 0.73
by training the labeller against *image* labels, and
[uncertainty-adjusted LLM extraction](https://arxiv.org/pdf/2510.05664) is the same move.

Two corrections to how this project has been framing it:

1. **It must be a re-rank on a new per-study feature, not a relabel-and-recalibrate.** A graded
   severity score blended into the ensemble's ranking changes order; a monotone repair of the
   existing score cannot.
2. **It cannot be scored against `lixin_gpt56` or any report-derived reference.** Those encode
   P(mention), which is precisely the quantity a severity re-rank is trying to move away from —
   scoring it there would *punish* a correct result. Gold-58 and the leaderboard are the only
   valid arbiters, and KEEP 1 just made the second one affordable.

---

## 3b. What a gain selected on 58 studies is actually worth `MEASURED 2026-08-12`

§3a made `prvsiyan`'s V34 PCA arm the first action, on the argument that its **+0.0273** was
selected on 58 studies and therefore probably overfit. **The measurement refines that and partly
contradicts it. Read this before acting on §3a.**

Their per-arm predictions are not published — their artifact datasets are the blank entries in
`kernel-metadata.json` — so the arm cannot be scored directly. What *can* be measured exactly is
the **procedure**: `fusion/selection_optimism.py` selects a per-target rank-blend weight on a
random n-study subset and scores that choice on the studies it never saw. 4,349 non-gold studies,
`ours`/`imported` as stand-in arms.

| n | mode | CLAIMED | REALIZED | optimism | P(realized ≤ 0) |
|--:|---|--:|--:|--:|--:|
| **58** | argmax | **+0.0137** | **−0.0034** | +0.0170 | **92%** |
| **58** | bagged | +0.0092 | −0.0015 | +0.0107 | **72%** |
| 150 | argmax | +0.0068 | −0.0008 | +0.0076 | 59% |
| 400 | argmax | +0.0043 | +0.0008 | +0.0035 | 18% |
| 1000 | argmax | +0.0030 | +0.0018 | +0.0012 | 0% |

**A weight chosen on 58 studies reports a gain and delivers a loss.** Not a small gain — a loss,
in 92% of draws. And **bagging does not rescue it**: resampling within 58 studies cuts the
optimism from +0.0170 to +0.0107 and still lands negative, because *resampling does not create
information*. That is `prvsiyan`'s own procedure — "99.4% of 500 partitions" is 500
re-partitions of the same 58 subjects — and their own sentence is the correct description of it:
it measures **estimator stability, not generalization**.

### The correction to §3a, which called for removing their PCA arm

Under this null, how often does selection on 58 studies *claim* at least their +0.0273?

| threshold | argmax | bagged |
|---|--:|--:|
| ≥ +0.0100 | 69% | 35% |
| ≥ +0.0200 | 14% | 7% |
| **≥ +0.0273** | **4%** | **2%** |

**So their number is a 2–4% tail event, not a typical noise draw. It is probably not pure noise,
and §3a's "a removal can gain score" was too confident.** The defensible statement is narrower and
still useful:

> Their +0.0273 is likely measuring *something*, but the CLAIMED magnitude carries almost no
> information about the REALIZED one. Expected realized value of any 58-selected choice here is
> ≈0. Removing the arm is a coin flip, not an edge.

**Action, revised: do not remove it and do not trust it — submit it both ways.** This is exactly
the class of question §3a's KEEP 1 made affordable: two inference-only runs, ~74 s of GPU each,
against a 5/day allowance. **The leaderboard is the arbiter for this, and it now costs minutes.**

### The rule that generalises, and it lands on our own plan

**Never select on gold-58.** The crossover where selection starts reliably paying is **n ≈ 400**
(+0.0008 realized, 18% failure); at n = 1000 it is +0.0018 at 0%. Gold-58 is an order of magnitude
short and always will be — it is 58 studies because that is all the competition adjudicated.

This is a direct hit on **our** plan, not just theirs. The standing assumption has been that gold
is "the only neutral arbiter" for label-source changes, and §2s/§2w both route the severity bet
through it. **Gold-58 can *evaluate* a fixed decision at ±0.031. It cannot *choose* one.** Any
severity-read variant selected on gold-58 — which threshold, which prompt, which blend weight —
inherits the −0.0034 above.

So the severity work needs its selection done on the leaderboard or not at all, and that is
affordable now (§3a KEEP 1). **Settle this before building any of it**, which is what the
long-standing "the instrument does not cover Phase 1" note has been asking for since 08-10.

**Caveat, stated because it bounds the claim.** The stand-in arms have little true gain available
(§2z: +0.0007 at full data), so this is a **null model** — it says what selection on 58 studies
manufactures when there is nearly nothing to find. That is the right null for "is this noise?"
and it is *not* an estimate of their arm's true effect. Their arm may be genuinely good; the
point is that 58 studies cannot tell them, or us, which.

---

## 3c. Setting up the V52 submission: three blockers, and a correction to §3a `2026-08-12`

Went to fork `prvsiyan` V52 and submit it. **It is not the download §2w assumed.** Findings, in
the order they bite:

### 1. §3a's "submissions are ~free" is TRUE OF THE `pilkwang` LINEAGE ONLY — correction

§3a KEEP 1 argued the leaderboard is now an affordable instrument because a submission is
`infer_from_package()` at **74 seconds**. **That is right for `pilkwang`/`aadigupta` and wrong for
`prvsiyan`, and the difference decides how the remaining 71 days get spent.**

| | submission cost |
|---|---|
| `pilkwang` / `aadigupta` **with the weights package attached** | short-circuits to `infer_from_package()` — **~74 s** |
| `prvsiyan` | trains its added arms in-notebook: `_V49_TOTAL_LIMIT_SECONDS = 8.70 × 3600`, B3 runtime limit `9 × 60 × 60`, 66 sites of training machinery |

So the instrument unlock is real but **narrower than stated**: five cheap leaderboard reads a day
exist **on the pilkwang lineage**, where members are pre-trained and the notebook only infers.
On `prvsiyan` a submission is a ~9 h GPU job — **~3 per week against a 30 h quota**, not 5 a day.
§3a has been amended in place. Post-hoc re-ranking experiments should therefore be built on the
**pilkwang** inference path, which is also where our `oof.npz` work already lives.

### 2. `prvsiyan`'s 0.906 depends on a PRIVATE dataset we cannot attach

Its V49 five-fold EfficientNet-B3 arm reads `/kaggle/input/rsna-knee-b3-v47-folds-0-3`.
That dataset **does not exist publicly** — dataset search returns nothing, direct access returns
**403**, and `prvsiyan`'s public datasets are ESC-50 and some Pokémon-TCG weights. Their five blank
`dataset_sources` entries in `kernel-metadata.json` are private attachments.

The notebook is explicit about what happens without it: *"the independently completed public .899
family remains the fallback until every pinned artifact and output check passes"*, with
`_V49_BASELINE = /kaggle/working/submission_public_0899.csv`. **So a fork that cannot see their
private B3 weights degrades toward 0.899, not 0.906.**

**The V52 arm itself is reachable, though**, and that is the part worth having: it is an *official
RadImageNet ResNet-50 **frozen** slice encoder*, loaded strictly from the **public**
`marwanmath/resnet-50-radimagenet-marwan`. Frozen means no pretrained-arm dependency — but the
512-dim head on top of it is fitted in-notebook under the grouped cross-fit protocol, which is
where the ~9 h goes.

### 3. The API will not serve a specific version, and their V52 gate is a 58-study gate

`kaggle kernels pull <owner>/<kernel>/52` returns **403** — arbitrary versions of another user's
kernel are not downloadable, so **pinning V52 has to be done by hand in the Kaggle UI**
(version history → Copy & Edit from V52). A plain fork gets the newest version (~V64), whose
score their own header calls *"diagnostic until Kaggle returns a completed official score"*.

And the branch itself is gated on exactly the thing §3b just measured: *"V49 is preserved unless
**exact 58-label outer-fold evidence** supports this branch."* **Their 0.906-vs-0.899 decision is
made on 58 studies**, which §3b shows delivers −0.0034 on average and is negative in 92% of draws.
That does not mean 0.906 is fake — it was a completed Kaggle row — but it does mean the *branch
selection* rests on the weakest instrument in this competition, and a fork may well gate the other
way on the same 58 studies.

### Revised submission plan

1. **`aadigupta`-style pilkwang + TTA knobs — minutes of quota.** Banks a score above our stale
   0.891, and establishes the cheap, repeatable inference path that every post-hoc re-ranking
   experiment will use. **This is the one to run first**, and it is nearly free.
2. **One `prvsiyan` fork — ~9 h, a considered spend.** Its purpose is to measure *what a fork
   actually scores without their private B3*, which nobody can predict from the notebook text.
   Pin V52 by hand if pinning at all.
3. **Do not build the re-ranking programme on `prvsiyan`.** Build it on the pilkwang inference
   path, where a submission is 74 s and `oof.npz` gives a matching local instrument.

---

## 3d. The 0.899 kernel's advertised changes do not touch the scored path `MEASURED 2026-08-12`

Built our own submission (`notebooks/build_tta_kernel.py`, pushed as
`raahimnawaz/rsna-knee-per-target-tta-pooling`) by porting `aadigupta7686/0-899-let-me-cook`'s
delta onto `pilkwang`. Reading the code rather than the description changed what the port *is*.

**`0-899-let-me-cook` advertises itself as "Vertical Flip Test-Time Augmentation (TTA) and
increased CACHE_SLICES (15)". Neither reaches a submitted score.**

| advertised | reality |
|---|---|
| vertical-flip TTA | lives in `predict()`, called **only** at the three in-notebook TRAINING sites (val, gold AUC, post-training test). The scored path is `infer_from_package()` → `predict_member()` (line 2137), which never calls it. |
| `CACHE_SLICES` 15 | **`N_GROUP_MAX = 1` is identical in both notebooks**, so `CACHE_SLICES = GROUP * N_GROUP` is identical too. Not a difference at all. |

**The entire real delta is per-target pooling over TTA windows inside `predict_member`**: a
logit-mean base (`TTA_POOL` `"prob"` → `"logit"`), overridden to **max** for Fracture, Contusion,
both menisci and Baker's, and **top-2 mean** for ACL and MCL. Diffuse findings — the three OA
labels, Effusion, Synovitis — are deliberately left on the mean.

**That is a prior, not a hack, and it is why this is worth porting.** A focal finding appears in
*some* TTA windows, so a mean over windows dilutes its evidence and a max does not; a diffuse
finding is present in all of them and the mean is the better estimator. It is the same reasoning
as the anatomical crops, one level cheaper — and it costs a config change on an inference-only
notebook instead of a training run.

**Built onto `pilkwang`, not forked from 0.899**, because the 0.899 copy is encoding-corrupted:
**1,216 mojibake sequences and zero Greek/Cyrillic**, against pilkwang's 922 Greek + 858 Cyrillic
(§2x). Dead code on the scored path — targets come from the mounted LLM table — but there is no
reason to inherit it. `build_tta_kernel.py` **refuses to write** if its anchors do not appear
exactly once or if the encoding check fails, so the patch cannot silently rot when pilkwang
publishes a new version.

This is the third time on this project that reading a competitor's *code* contradicted its
*description* (§2f the extractor, §2x the leaderboard survey, §3d here). **The rule is now
three-for-three and should be treated as a default, not a precaution.**

---

## 3e. 0.899 SCORED — the code reading was right, and the "cheap instrument" claim was wrong twice `2026-08-12`

`raahimnawaz/rsna-knee-per-target-tta-pooling`, submission **55465252**: **public LB 0.899**,
against our banked **0.891**. **+0.008, and it exactly matches `aadigupta7686`'s score.**

### This is a clean experimental confirmation of §3d, not just agreement

§3d claimed from reading the code that `0-899-let-me-cook`'s two advertised features — vertical-flip
TTA and `CACHE_SLICES 15` — **never touch the scored path**, and that the entire delta is
per-target TTA pooling in `predict_member`.

**We shipped only the pooling. We omitted both advertised features. We reproduced their score to
three decimals.** That is a positive experiment, not an argument: had the vflip or the cache depth
contributed anything, we would have landed below them. **Fourth time reading a competitor's code
beat reading their description — and the first time it was confirmed on the leaderboard.**

Also banked: our copy has pilkwang's 922 Greek + 858 Cyrillic intact against the 0.899 kernel's
1,216 mojibake sequences, so we hold the same score on undamaged source.

### The first same-model CV↔LB anchor this project has had

§2t-1 noted the conversion was interpolated from **two foreign anchors**. Now there is a real one:

| model | our instrument (report-OOF, `lixin`, non-gold) | public LB |
|---|--:|--:|
| `pilkwang` 20-member | **0.8434** | **0.891** |
| + per-target TTA pooling | *(not yet scored locally)* | **0.899** |

**These are the same weights measured two ways, not two quality levels** — the local number reads
lower because it scores against a *report-derived* source while the LB scores against expert
*image* labels. The ~0.048 gap is the report→image discrepancy, measured, and it is the same
quantity `REFERENCE.md` §2.1's severity thesis is about. It means a local +0.01 is not an LB
+0.01, and nothing here has established the slope.

### CORRECTION, second one on the same claim: a submission is NOT 74 seconds

§3a KEEP 1 argued the leaderboard is affordable at "74 s per submission"; §3c narrowed that to the
pilkwang lineage. **Both were wrong.** This submission was PENDING for **over 99 minutes**.

The 74 s in §2e is the **20 members' forward passes**. A submission additionally decodes the
**entire hidden test set from DICOM** — invisible in a public-sample run, which is why the local
run took 376 s on 3 studies.

**Corrected budget: order ~2 h per submission**, so roughly **15 runs a week against a 30 h
quota** — not 5/day-for-free, and not prvsiyan's ~3/week either. Still enough to make the LB the
selection instrument §3b showed gold-58 cannot be, but re-ranking experiments must be batched.

**Three corrections to one claim in one day.** Each time a number measured on a small or partial
configuration (74 s of forward passes, 3 public studies) was extrapolated to the full run without
checking. Same shape as §2v's sleeping laptop.

### Standing

Top **0.946**, 10th **0.935** (was 0.934 this morning — the cutoff is still climbing). Our 0.899
leaves the 0.891 plateau for the 0.899 one; ties break by earliest submission, so we enter at its
bottom, ~rank 326 of ~1,280, up from 400. **The free public ceiling is now reached. Everything
from here has to be something the field does not already have.**

---

## 3f. Opening the black box: where the 0.891 ensemble fails, and on whom `MEASURED 2026-08-12`

With `oof.npz` local (§2y), the fork is auditable for the first time. `fusion/error_analysis.py`
over 4,349 non-gold studies. **Several standing beliefs come out changed.**

### Where the macro actually loses

| label | prev | AUC | | label | prev | AUC |
|---|--:|--:|---|---|--:|--:|
| **Lateral Meniscus** | 0.148 | **0.767** | | Effusion | 0.590 | 0.855 |
| MCL | 0.146 | 0.817 | | Medial OA | 0.359 | 0.872 |
| PF OA | 0.451 | 0.822 | | Synovitis | 0.123 | 0.886 |
| Lateral OA | 0.257 | 0.829 | | Baker's | 0.246 | 0.887 |
| Medial Meniscus | 0.393 | 0.834 | | Fracture | 0.064 | **0.898** |

**The lateral compartment is systematically harder than the medial** — Meniscus 0.834 → 0.767
(−0.067), OA 0.872 → 0.829 (−0.043).

**And that is a property of the TASK, not a pilkwang defect.** An independent study reports
medial/lateral meniscus AUCs of **0.834 / 0.746**; pilkwang gets **0.834 / 0.767**. Nearly
identical, arrived at independently. The literature localises it further — misclassification is
highest at the **posterior horn of the lateral meniscus**. **So "fix Lateral Meniscus" is
attacking a known-hard problem, and the anatomical crops are the intervention with the right
shape** (a specific horn of a specific compartment), not a generic capacity increase.

### LATERALITY: no disparity — but the question is not fully closed

All 20 members run `rules: {lat: 'centre'}`, no mirroring, while medial/lateral **swap sides**
between knees. Testing compartment labels against the eight non-compartment ones as a control:

**mean gap compartment +0.0027, control +0.0041, DIFFERENCE −0.0014.** Nothing. Left knees
(n=1,663) are predicted as well as right (n=2,686) on exactly the labels where sides swap.

**Read this precisely.** It shows left knees are not *neglected*; it does **not** show mirroring
would not help. A model trained on mixed L/R learns both orientations at the cost of splitting
capacity, and would still score equally on each — the disparity test cannot see that. The
circumstantial hint is that the *rarer* compartment is where performance collapses, which is what
a capacity split would predict, but prevalence confounds it. **Settling it needs a trained arm,
not an audit** — park it behind the cheap work.

### Subgroup disparities, largest first

| dimension | biggest gaps |
|---|---|
| **Manufacturer** | Synovitis 0.848 Siemens vs **0.914** GE (3.2σ) · Effusion 0.841 vs 0.892 (3.4σ) · Fracture 0.897 vs 0.946 · control mean **−0.0286** |
| **Sex** | ACL **M 0.865 / F 0.820** (+0.044, 2.3σ) · Medial OA M 0.852 / F 0.889 (−0.037, 3.1σ) · Baker's +0.030 (2.1σ) · compartment vs control **−0.0213** |
| **Field strength** | **Lateral Meniscus 1.5T 0.748 / 3T 0.801** (−0.054, 2.7σ) · Effusion 1.5T 0.872 / 3T 0.830 (+0.043, 3.5σ) · Contusion −0.036 |

**The weakest label is weakest exactly where physics predicts**: Lateral Meniscus at 1.5T is
**0.748**. A small structure at lower SNR. That is the same axis as §2d's resolution finding
(224→518 = +0.013) and points at the same fix.

### HARMONISING SITE AWAY COSTS 0.013–0.032, AND THAT INVERTS THE STANDARD ADVICE

The harmonisation literature names scanner manufacturer as *the* dominant site effect and offers
ComBat as the remedy. `fusion/harmonise_test.py` applies the parameter-free version — within-group
rank normalisation, which cannot overfit:

| grouping | groups | macro | delta | σ |
|---|--:|--:|--:|--:|
| site_id | 42 | 0.8149 | **−0.0319** | 18.2 |
| ManufacturerModelName | 27 | 0.8209 | −0.0260 | 17.8 |
| Manufacturer × field | 17 | 0.8273 | −0.0196 | 15.2 |
| Manufacturer | 11 | 0.8337 | −0.0132 | 13.3 |

**Why: case mix genuinely differs by scanner.** Medial OA prevalence is **0.479 / 0.353 / 0.338**
across the three largest manufacturers; PF OA 0.579 / 0.454 / 0.405; Lateral OA 0.364 / 0.256 /
0.232. The between-group score difference is **signal** — the model rightly scores OA higher at
OA-heavy sites — and removing it destroys real ordering.

### This reinterprets §2j's site leakage, and it matters for every fold decision

§2j measured **+0.024** of "site leakage" and the project concluded the model memorises scanner
signatures. **The more likely mechanism is that it learns site-level PREVALENCE** — and that is
legitimately available whenever the test split is by *study*, which is how this competition splits.

So the two fold schemes answer different questions, and the project has been conflating them:

* **site-grouped** → *"how would this do at a NEW hospital?"* Right for reporting generalisation.
* **random study-level** → *"how would this do on held-out studies from THESE hospitals?"* This is
  what the leaderboard asks.

**§2j's "nothing gets compared against an external score except under site-grouped folds" is the
right rule for honesty and the wrong rule for predicting this leaderboard.** Both belong; they
have to be labelled. (§2z already found the scheme makes no difference to low-dimensional
*selection* — this is about the *estimate*, which is a separate thing.)

### Consequence: add site signal rather than remove it — small, real, and the sign is the finding

`fusion/site_prior_test.py`, random study-level folds, per-site per-label prevalence shrunk to the
global rate (empirical Bayes, strength K), blended into the ranking at weight w:

| w | 0.05 | **0.10** | 0.20 | 0.30 |
|---|--:|--:|--:|--:|
| delta | +0.0017 | **+0.0023** | −0.0000 | −0.0084 |

Stable across K = 10, 25, 50, 100 (+0.0020 to +0.0023 at w = 0.10) and **unimodal in w** — noise
does not produce a smooth curve across four independent shrinkage settings. Modest, and one
quarter of what §3d's TTA pooling paid, but it is **free, post-hoc, and points opposite to the
published remedy**. Per §3b, n = 4,349 puts selection optimism near +0.0005, so most of it should
survive — but it is 16 (K, w) pairs scored on the studies they were chosen on, so **the
leaderboard confirms it or it does not count.**

### What this changes about what to build

1. **Lateral Meniscus at 1.5T (0.748) is the single clearest target**, it is known-hard in the
   literature, and it is localised to the posterior horn. The **anatomical crops** are the
   intervention shaped for it. This is the strongest case yet for building them.
2. **Site prior is a free +0.002** — batch it into the next submission rather than spending a
   ~2 h run on it alone (§3e).
3. **Do not harmonise.** Measured, costly, and contrary to the literature's default.
4. **Sex disparities are real** (ACL 2.3σ, Medial OA 3.1σ) but a *per-study covariate cannot be
   added post-hoc as a monotone transform* — same §3a filter. It is a training-time feature or a
   group-conditional prior like the site one; test it the same way before believing it.

---

## 3g. The fingerprint does not check the pixels — the cheapest crop route was never closed `MEASURED 2026-08-12`

**`PLAN.md` §9b asserted that no crop route can reuse the fork's frozen members. That assertion
was inferred, never read, and it is wrong.** It cost nothing yet only because it was written the
same day it was refuted.

### What §9b said, and where it came from

> every pilkwang member **verifies a fingerprint on its pixel contract** (`crop_mm 130`,
> `img 336`, band 0.2–0.8) — feeding a different crop breaks it by design, so no crop route can
> reuse their frozen members.

That is a claim about `fingerprint()` in `pilkwang/rsna-knee-baseline-v1`. The kernel was pulled
and read (2,493 lines, 21 code cells). The function is at line 1867:

```python
g = torch.Generator().manual_seed(seed)
imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size),
                     generator=g, dtype=torch.uint8).to(dev)
```

**The input is generated from a seed. It is a synthetic bag of random bytes, and no image ever
reaches it.** `check_fingerprint` is called once per member at load time (line 2102,
`check_fingerprint(model, dev, IMG, ck["fingerprint"], ...)`) and never again. It is a
weight-and-architecture identity check — *"a set of weights carries the answer it gave to a
question with no data in it"* — and its own docstring closes the question outright:

> *"This checks that the model computes what it computed when it was fitted. **It cannot check
> that the pixels reaching it are the right pixels**; `read_slot` and the header pass answer to
> their own tests."*

**So the guard is real but it guards something else.** `img_size` is an argument to `fingerprint`,
so changing *resolution* trips it. `CROP_MM` and `SLICE_BAND` are not arguments to anything the
fingerprint touches, so changing *those* does not. **There is no guard between us and a re-cropped
input to the twenty frozen members.**

What remains is a genuine risk of a different kind: the members were fitted at 130 mm, so a
tighter crop is a domain shift and may simply degrade them. That is an empirical question with a
cheap paired local answer. It is not an impossibility, and §9b filed it as one.

### F2-cheap: crops as extra TTA windows, with no training run at all

The route §9b closed is the one that costs least and is most likely to pay:

* **Additive, not substitutive.** The 130 mm view stays in the TTA pool and the tighter crop is an
  *additional* window. No member is ever asked to predict from an out-of-distribution input alone;
  the pool is a strict superset of the one scoring 0.899 today.
* **The pooling rule already exists and is already banked.** §3d's per-target pooling — max for
  Fracture, Contusion, both menisci and Baker's; top-2 mean for ACL and MCL; mean for the diffuse
  labels — is exactly the estimator a crop window wants. A focal finding is present in the tight
  crop and diluted in the wide one, which is the same argument that paid **+0.008**.
* **§3d said this in as many words and nobody noticed:** the pooling prior is *"the same reasoning
  as the anatomical crops, one level cheaper — and it costs a config change on an inference-only
  notebook instead of a training run."* It was more literally true than it read.
* **The fork prices the gain in its own comments** (line 818): pitch is `CROP_MM / P`, so 130 mm at
  336 px is **0.387 mm** against the **0.5 mm** a 1 mm tear needs. A 90 mm crop at 336 px is
  **0.268 mm**. `RUNS` is annotated *"Resolution is the axis under test"* — their phrase.

### How to test it without spending a submission

`pilkwang/rsna-knee-weights` is **1.54 GB, CC0-1.0**, 20 × ~89 MB checkpoints, and downloads in
about ninety seconds. The A/B is then entirely local:

1. Run **each member on its own held-out fold only** — `manifest.json` carries the fold and seed
   per member. This is not optional: a member run over its *training* studies reports on memorised
   cases, which survive a domain shift that novel ones do not, and would bias the result toward
   **"the crop makes no difference"** — a false negative, and the expensive direction to be wrong
   in here.
2. Score crop-130 against crop-130 + crop-90 through `fusion/score_oof.py`, paired.
3. **The reference is neutral by construction** — same weights, same studies, same targets, one
   config value apart. Per §2s this pre-check is mandatory and it has never before been this easy
   to satisfy. Four measurements on this project were lost to an entangled instrument; this arm
   cannot have that problem.

### The transferable lesson, and it is not the one already on file

The standing rule from §2f/§2x/§3d is **"read a competitor's code, never its description"**, and
it is three-for-three. **This is a fourth instance with a twist: the description that misled us
was our own.** §9b was written from this repo's earlier prose summary of the fork rather than from
the fork, by a session that had read the code days before and was compressing it. A second-hand
summary of a first-hand reading decays exactly like a competitor's marketing copy does.

**Consequence — mark derived claims as derived.** Every external assertion in `REFERENCE.md`
carries a source and a read-date, and that discipline works. Internal assertions carry nothing, so
an inference and a measurement look identical three days later. The `CLOSED ROUTES` table added in
`0b31437` inherits this: it is headed *"Each was measured, not argued"*, and while most rows were,
`CoPAS / foundation models / Gold Loss Correction` (surveyed), `post-hoc calibration` (an argument
from AUC invariance) and `a C++ port` (an extrapolation) were argued. **A route closed by argument
reopens when its premise moves. A route closed by measurement does not.** The two belong in the
same table only if the table says which is which.

---

## 3h. Step 0 groundwork: the slots ARE reconstructable, and the shipped second arm is not worth blending `MEASURED 2026-08-12`

Three results from the weights package, all before a single GPU forward pass. The first corrects a
standing claim, the second closes a route, the third hands the gate its reference.

### 3h-1. The fork's six slots are reconstructable offline — the standing claim was scoped wrong

This repo and the project memory both record that they are not:

> `Fluid_Sensitive` and `Fat_Suppression` are byte-identical over all 24,371 series, so the fork's
> six slots (`SAG_FLUID_FS … SAG_T1`) are **not reconstructable** from the competition metadata.

**True of the competition metadata, and false of what is on disk.** `annotate()` never reads
`Fluid_Sensitive`. It recovers both axes from seven raw header fields — `SeriesDescription`,
`SequenceName`, `ScanOptions`, `ScanningSequence`, `RepetitionTime`, `EchoTime`, `PixelSpacing` —
and **all seven are columns of `data/external/dicom_headers_zhukovoleksiy.parquet`**, which has sat
on disk since 08-10. The old note named this as a *fallback* if the reproduction gate missed; it is
the primary path and it is sufficient. A claim scoped to one source was carried as if it were
scoped to the machine.

`fusion/slot_assign_pilkwang.py` transcribes their `annotate` and `pick_slots` verbatim — regexes,
the exact-token match on `ScanOptions` (GE writes `SAT_GEMS`, so a substring test on `SAT` fires on
non-fat-sat series), the `np.where` weighting cascade, the thickest-stack tie-break. Over all
24,371 series, plane known for 100%:

| | fatsat False | fatsat True |
|---|--:|--:|
| PD | 2,724 | **10,922** |
| T1 | **5,299** | 84 |
| T2 | 2,188 | 2,926 |
| GRE | 226 | 0 |
| UNK | 0 | 2 |

**20,130 slots over all 4,407 studies**, and the fill rates match their own description — *"the
fat-suppressed fluid-sensitive series exist for nearly every study; the T1 and the non-suppressed
fluid-sensitive series are scarcer, which is what the presence mask is for"*:

| slot | filled | |
|---|--:|--:|
| `AX_FLUID_FS` | 4,343 | 98.5% |
| `COR_FLUID_FS` | 4,210 | 95.5% |
| `SAG_FLUID_FS` | 4,119 | 93.5% |
| `COR_T1` | 2,827 | 64.1% |
| `SAG_FLUID_NOFS` | 2,760 | 62.6% |
| `SAG_T1` | 1,871 | 42.5% |

Mean 4.57 of 6 slots per study; none below 2. **This is a shape check, not a proof** — it says the
recovery is not wildly wrong, and only the gate says it is right. But it was worth ten minutes to
learn here rather than from a member scoring 0.6.

**Pixel coverage:** 16,417 of the 20,130 assigned slots have NIfTI on disk (81.6%), covering
**3,599 studies, of which 3,593 hold every slot they were assigned**. Ample for a gate.

### 3h-2. `merge_gain.npz` — a second arm nobody knew shipped, and it does not earn a slot

The weights dataset carries **three** files, not two. Beside `oof.npz` and `manifest.json` sits
`merge_gain.npz`: `ids`, `y`, `gold_mask`, and **two separate OOF matrices — `ours` and
`imported`**. It is the record of an experiment they ran and shipped the evidence for. Scored
against their own `y`, non-gold n = 4,349:

| arm | macro | gold-58 |
|---|--:|--:|
| `ours` | **0.8492** | 0.8447 |
| `imported` | 0.7932 | 0.8084 |
| rank-mean 50/50 | 0.8382 | 0.8437 |

**A 50/50 merge costs 0.011.** Swept properly, the way `blend_test.py` swept our port:

| w | 0.00 | 0.05 | **0.10** | 0.15 | 0.20 | 0.30 | 0.50 |
|---|--:|--:|--:|--:|--:|--:|--:|
| delta | — | +0.0005 | **+0.0007** | +0.0005 | +0.0000 | −0.0021 | −0.0110 |

**Peak +0.0007, and `imported` loses 0/12 labels** — every one, by 0.016 to 0.083. Per §3b,
selection optimism at n = 4,349 is already ~+0.0005, so the peak is indistinguishable from the
cost of having looked. **Not worth a slot.**

Note the shape: mean rank correlation `ours` vs `imported` is **0.752** — genuinely diverse, and
uniformly weaker, so no weight pays. That is the same result §2y found for our port at 0.639, and
the second time on this project that a diverse arm has failed to buy a slot on strength alone.
**Diversity is necessary and it is not sufficient, and this is now measured twice.**

### 3h-3. The gate has its reference, and it is not the one `score_oof.py` uses

`oof.npz['y_derived']` is byte-identical to `merge_gain.npz['y']`, so **their exact label table is
on disk**. The manifest also carries **per-member `fold`, `holdout` and `annot`** — holdout AUCs
run 0.8305 to 0.8600 (mean 0.8398), gold-58 AUCs run 0.746 to 0.889.

That per-member spread is worth pausing on: **the same twenty members, on 58 studies, span 0.143
of AUC.** §3b said gold-58 cannot select; this is the same fact seen from the fork's own side.

So the gate scores **against their `y`, on each member's own holdout fold, targeting that member's
recorded number** — deliberately *not* through `lixin_gpt56`, which `score_oof.py` uses everywhere
else. The two references have different jobs: the gate must **match their measurement** to detect
reconstruction error, while the crop A/B needs one **neutral to both arms**. Using the project's
default here would have made a faithful path look broken.

Also confirmed from the manifest config, since it nearly read as a refutation: `slices: 12`,
`group: 3`, so `window_starts(12, 3)` gives **10 TTA windows**, not the 1 that `N_GROUP_MAX = 1`
implies. That cap governs *training-time cache planning*; `adopt_config_globals` overrides
`CACHE_SLICES` from the member config at inference. Had it been 1 window, §3d's per-target pooling
would be a mathematical no-op and the +0.008 would need another explanation. It is not, and it
does not.

---

## 3i. The gate MISSES, K16 is validated for the first time, and §2y's third differentiator is void `MEASURED 2026-08-12`

The pixel path was rebuilt (`fusion/pilkwang_pixels.py`) and run through all 20 members
(`fusion/pilkwang_gate.py`, n = 60 studies, 270 slots). **It does not reproduce them.** Recorded
here in full because the negative result is more useful than the positive one would have been.

### 3i-1. The three readouts

| readout | K16 **on** | K16 **off** (ablation) | reading |
|---|--:|--:|---|
| partition (target 20% each) | 20.0 / 20.0 / 25.0 / 21.7 / 13.3 | 16.7 / 21.7 / 25.0 / 21.7 / 15.0 | both ≈ uniform, χ² p ≈ 0.7 |
| margin, median | **0.0177** | 0.0165 | identifiable |
| margin > residual | **63.3%** | 55.0% | |
| **residual, mean** | **0.0168** | 0.0185 | |
| **residual, median** | **0.0134** | 0.0164 | |

**The partition is clean and the residual is not, which is the case the gate's docstring named in
advance as "recognisable but not reproduced".** The four members that held a study out never saw
it and the other sixteen trained on it, so the memorisation gap identifies the fold on pixels that
are merely approximately right. Only the residual is sensitive to being exactly right.

**Correction to the threshold, which was mine and was too lenient.** §3h-3 proposed the fork's
0.0165 self-consistency as the bar. That is the distance between two of their own *training runs*.
We run the *same weights*: with identical pixels the residual would sit at fingerprint level,
**~1e-5**. We are three orders above that. Passing the 0.0165 bar would not have meant what it was
said to mean, and this is the second time in one day that a benchmark had to be re-scoped after
being chosen (see §3h-3's reference note).

### 3i-2. K16's first real test, and it passes

`data/slice_direction_resolved.csv` was cross-validated 21/21 against the 01c thumbnails, but it
had never been tested against anything that *cares* — the fold-0 gate sat at depth 0.5, where a
reversal maps the middle slice to itself. The ablation is its first, with a **predicted sign**:
if the measured bit is right and slice order is what the residual is made of, applying it must
lower the residual.

**It does, on all three readouts at once** — residual mean 0.0185 → 0.0168, median 0.0164 →
0.0134 (−18%), and margin and partition both improve. Three quantities that could have moved
independently move together in the predicted direction. **K16 is now measured, cross-validated,
and load-bearing.**

### 3i-3. Why order is the mechanism, and why the direction bit cannot finish the job

A one-member sensitivity test — not a search for the order that minimises the residual, which
would fit the instrument to the thing it tests:

| perturbation of the 12-slice stack | mean \|Δ\| |
|---|--:|
| whole stack reversed | 0.0186 |
| adjacent pairs swapped | 0.0289 |
| random permutation | **0.0501** |
| *(the gate residual)* | *0.0168* |
| *(within-fold member vs member)* | *0.0495* |

**A scrambled stack is worth as much as a different member** (0.0501 vs 0.0495). Slice order is
load-bearing for these models, which also explains why the fork spends up to 90 minutes
(`ORDER_BUDGET_S = 5400`) sorting slices before it decodes anything.

**But the arithmetic says direction bits cannot close this.** K16 covers sagittal and buys 0.0017.
Coronal is the only other plane with a known reversal rate (`validate_nifti` 14/18 forward, so
~22%), and coronal slots are 34% of ours, giving an upper bound of about **0.0014** — even if
measured perfectly. Against a residual of 0.0168, the two together are a tenth of the problem.

**The residual is 0.0168 / 0.0501 ≈ 34% of fully scrambled**, which is what a third of series being
*permuted* rather than merely *reversed* looks like. That is mechanically plausible: interleaved
and multi-echo acquisitions do not number slices in spatial order, which is exactly why the fork
sorts by the projection instead of by `InstanceNumber`, and why §2n measured `InstanceNumber`
tracking true direction at only 56.9%. **K16 answered "is the stack reversed". It never asked "is
the stack sorted at all", and that is now the open question.**

**It cannot be answered with what is on disk.** `data/direction_thumbs.npz` holds **first / mid /
last only** — three anchors per series, 29,592 entries over ~9,900 series. Enough for a direction,
never for a permutation. Per-slice `ImagePositionPatient` exists only in the DICOMs.

### 3i-4. The fix is one Kaggle CPU kernel, and it is not GPU quota

Export, per series, the permutation that sorts slices by `k = p · (r_x × r_y)` — the fork's own
`order_slices` key. Header-only reads with `stop_before_pixels`, the same pass the fork runs at the
start of every submission inside its 90-minute budget with 32 threads. The output is a few MB of
integers and it is reusable forever. **CPU kernels do not draw on the 30 h GPU quota**, so this
costs schedule and not budget. Per §2m, the download is part of the fix.

Until it exists, **any local measurement on the frozen members carries a ~0.017 per-prediction
reconstruction error**, and the crop A/B — which is paired, so much of it cancels — still cannot be
transferred to a submission with confidence, because the submission runs *their* DICOM path.

### 3i-5. §2y's third differentiator is void, and this one is a real loss

§2y item 3 reads the members' `rules: {order: 'normal', lat: 'centre'}` as **"no per-series
slice-direction handling at all … signal they are leaving on the floor."**

`order_slices` under `normal` computes the signed through-plane projection per slice and sorts by
it. **That is a full geometric ordering.** `dominant_axis` is the legacy fallback; `centre` is
likewise their better laterality rule against legacy `corner_x`. **`normal` and `centre` are the
names of their good rules, and this repo read them as the absence of rules.**

So K16 is not an edge over the fork. They solve ordering properly from geometry we do not have;
K16 is *our* repair for a problem *we* created by converting to NIfTI. The consequence is
strategic and unwelcome: §2y listed three things that compose with the fork — item 1 closed at
−0.0000 (§2z), item 2 is partly spent, and **item 3 was never real. Our edge over the fork is
thinner than this repo believes**, and what is left is the two bets that were always the whole
game: the crops and the severity labels.
