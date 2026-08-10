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
| K16 | **A third of NIfTI series are stored back-to-front**, and nothing in the file says which. Measured on a stratified sample: 66.7% forward overall — Axial 12/12, Coronal 14/18, **Sagittal 8/21** | The affine carries no direction cosines (see §9.1's correction), so `load_series_nifti` cannot know. `load_series` always sorts ascending by IPP projection, so train and test disagree on ~1/3 of series — in the axis medial/lateral depends on, and invisible to `PREPROCESS_VERSION` | **OPEN.** Needs a per-series direction bit exported from the DICOMs (`PLAN.md` §9 Phase 0 step 2). **K18 composes with this** — the sagittal handedness fix is an XOR against this bit and cannot be enabled without it. Not predictable from plane: a plane rule is ~72% accurate. The first verdict said 100% forward because every thumbnail in that sample was Axial_0 |

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
