# The data plan

**4,349 of 4,407 training labels must be manufactured from report text. The extractor is not a
component of the solution, it is the solution — and the ceiling it sets is the only ceiling the
rest of the field also has.**

Every number below is reproduced by `python extractor/audit_label_stage.py`, section numbers
match. Run it before trusting any of this; it takes about a minute and writes nothing.

---

## 0. What the 2026-09-06 audit retracted `MEASURED 2026-09-06`

Two of the three numbers this document was built on do not survive their own definition. They
are recorded here rather than deleted, because the route they motivated is still the right one
and the next session needs to know which parts were load-bearing.

**⛔ RETRACTED — "the labels over-call gold 6.68 : 1."** That ratio is a *binarisation
threshold*, not a property of the labels. The same unchanged file reads (§1):

| cut | 0.5 | **0.60** | 0.75 | 0.85 | 0.95 | prevalence-matched |
|---|--:|--:|--:|--:|--:|--:|
| over-call | 29.2:1 | **6.63:1** | 2.50:1 | 0.71:1 | 0.09:1 | **1.00:1** |
| cell agreement | 0.609 | **0.792** | 0.809 | 0.838 | 0.776 | **0.848** |
| **macro AUROC** | **0.8873** | **0.8873** | **0.8873** | **0.8873** | **0.8873** | **0.8873** |

D1's 6.68:1 is the 0.60 row and nothing else. **The Stage L gate "over-call < 2:1" was
satisfiable by editing a constant, and could not have moved the score. Deleted.**

**⛔ RETRACTED — "headroom is 79.0% → 84.7% cell agreement."** Same artifact from the other
side: 79.0% is the 0.60 row, and at the prevalence-matched cut the public labels already read
**84.8%**, at or above the 84.7% that §2b called the careful-human ceiling. There is no
headroom in that currency — and cell agreement is not the currency the competition reads.

**✅ STANDS — the mechanism.** Ten of twelve criteria carry a severity or acuity cut-point and
every public extractor detects mentions. That is still true, still unexploited by the field, and
§3 below now measures what it is worth instead of asserting it.

---

## 1. The two effects that are real, and how big they are `§3`

Thirty studies are both gold-labelled and hand-read (`labeling/model_labels.csv` ∩ the 58).
That overlap is the only place where a careful human reader, the public reader, and image-derived
truth can all be compared on the same studies. Scoring the public labels **binarised at their own
gold prevalence** — the kindest binary version of them — separates two effects that the old
framing had fused:

| | macro on the 30 | vs binary baseline | 95% CI | P(>0) |
|---|--:|--:|---|--:|
| public, binarised to match | 0.7830 | — | | |
| **human hand read, binary** | 0.8385 | **reading skill +0.055** | [+0.005, +0.111] | 0.985 |
| **public, as shipped (soft)** | 0.8689 | **graded output +0.086** | [+0.042, +0.130] | 1.000 |

**Read this carefully, because it is the whole plan:**

- **Graded output is worth more than reading skill** — and the incumbent already banks it. The
  public labels beat our hand read *only* because they are soft and our export was binarised.
  Any extractor of ours that emits bits instead of grades starts 0.086 behind for free.
- **Reading skill is real but modest: +0.055, and its lower bound is +0.005.** That is the
  honest headroom over the shipped incumbent — 0.887 → ~0.94 if it transfers — and it is one
  bad week away from zero. Size the effort accordingly.
- The two overlap (both partly encode "how sure and how severe"), so **do not add them.**
  +0.055 over `targets.csv` is the expectation; the sum is an optimistic ceiling.

⚠️ **n=30, and the hand labels were written by the same person who will write the extractor
prompt.** "Reading skill" therefore includes "this reader's idiosyncrasies", and the hand set is
an instrument, never a gold standard. §4 is built around that limitation rather than ignoring it.

---

## 2. The per-label map — this is the granular part `§2, §6, §7`

The job is **six labels wide, and each of the six fails for a different measured reason.** A
single "severity prompt" rewrite is the wrong shape for the problem.

**Arithmetic first (§2).** Macro is 0.8873. The top six average 0.948 and have nowhere to go, so
reaching macro 0.93 requires the bottom six to move **0.827 → 0.912**. That is the entire target.

**Addressability (§6).** `mention` is the fraction of all 4,407 reports whose text discusses the
finding at all; it bounds every label from above, because no extractor can rank what the report
never says. `graded` is the fraction of those mentions carrying a severity word in context.

### The six that need work

| # | label | AUROC | mention | graded | acuity | measured | why it fails | the lever |
|--:|---|--:|--:|--:|--:|--:|---|---|
| 1 | **Effusion** | 0.853 | **0.832** | **0.652** | 0.023 | 0.066 | criterion is *moderate or large*; **49% of mentions are explicitly trace/small** and are scored positive today | **severity. The cleanest case on the board** |
| 2 | **Lateral OA** | 0.830 | 0.623 | 0.640 | 0.045 | 0.103 | two failures stacked: compartment attribution (the glossary cannot separate medial from lateral) **and** the ≥1 cm / >50%-thickness cut | compartment first, then extent |
| 3 | **Contusion** | 0.831 | 0.431 | 0.467 | 0.183 | 0.060 | edema-vs-discrete-fracture-line is *per lesion*, not per study (§2.2) | schema, not severity |
| 4 | **Fracture** | 0.793 | 0.205 | 0.360 | **0.377** | 0.044 | criterion is *acute*; acuity is stated whenever the finding is — the highest acuity rate of the twelve | acuity extraction |
| 5 | **Lat. Meniscus** | 0.862 | 0.702 | 0.282 | 0.026 | 0.046 | criterion is signal *contacting the surface on ≥2 images*; reports rarely say either | hardest. Expect the least |
| 6 | **Synovitis** | 0.790 | **0.124** | 0.823 | 0.073 | 0.097 | **the report does not discuss it.** Public fills undecided cells from Effusion; vs our hand read it agrees on **0.542** of cells with **45.8% over-call** — the worst cell in the matrix | **not an extraction problem.** See §5 |

**Effort ranks by headroom × addressability: Effusion, Lateral OA, then Fracture and Contusion,
then Lateral Meniscus. Synovitis leaves the prompt entirely.**

### The six to freeze

ACL 0.993 · MCL 0.964 · Medial Meniscus 0.954 · Baker's 0.946 · Medial OA 0.931 · PF OA 0.901.

**Do not spend prompt budget here, and guard them.** A rewrite that lifts Effusion while costing
ACL 0.02 is a net loss the macro will hide. PF OA at 0.901 is the one stretch candidate — it is
well addressed (mention 0.811, graded 0.704, and 20.7% of mentions carry an actual measurement) —
take it only if it comes free with the OA compartment work.

### Two defects to fix before any of the above

- **⚠️ Glossary bug: 'moderate' is in the MINIMAL cue list for German (`mäßig`) and French
  (`modér`).** Moderate is *above* the cut for Effusion and Baker's, so the cue set inverts the
  criterion for 358 studies (8.1%). `§6` flags it on every run.
- **Turkish and Croatian are the weakest languages** — public-vs-hand cell agreement 0.788 and
  0.778 against English's 0.928, over-call 21–22% (§7, small n, directional). `REFERENCE.md` §3
  pre-registers the mechanism: **Turkish negates after the term** (`efüzyon izlenmedi`), which a
  left-window negation detector silently inverts. 952 studies across the two.

Also known and small: **99 reports (2.2%) use a bilateral word**, and the host confirms labels are
for one knee with disambiguation in the report or the DICOM metadata (`REFERENCE.md` §1.4).
Nothing handles it. One schema field, not a workstream.

---

## 3. The instrument — two tiers, because there is no third `§4, §5`

This is where the previous plan would have failed, and it is the part to get right.

**gold-58 cannot be iterated against.** The marginal bootstrap SD of one macro on 58 studies is
**0.0170** — a 95% CI **0.068 wide** on a target that asks for +0.043. Iterating a prompt until
that number crosses a line is selection on 58 examples, which `DECISIONS.md` §3b already priced
at *claims +0.0137, delivers −0.0034*.

**Tested and closed: there is no large-n substitute.** Scoring the image model's honest OOF over
the 4,349 non-gold studies against seven candidate label sets ranks them at Spearman **0.429
(p=0.337)** versus their gold-58 order — and it puts `pilkwang_v2` near the top because it was
*trained* on those labels. It measures agreement with its own teacher. **Not a gate (§5).**

So the instrument is two-tier, and the tiers do different jobs:

| tier | what it is | n | measures | used for |
|---|---|--:|---|---|
| **iteration** | the hand set, `labeling/model_labels.csv` extended to all 303 | 303 studies, 9 languages | *did the extractor read the report correctly* | every prompt revision, freely |
| **arbiter** | gold-58, **paired** | 58 studies | *does the target rank like image truth* | at most 3 pre-registered candidates, once each |

**Pairing is what makes 58 usable at all.** Comparing a candidate to the incumbent on the same
studies drops the bootstrap SD from 0.0170 to **0.0042 — 4.1× more power** — and makes **+0.008**
the smallest delta an honest gate can name (§4).

### The pre-registered Stage L gate

Written before the run. Not renegotiable after seeing the number.

```
PRIMARY   macro-AUROC of targets_sev.csv vs data/targets.csv,
          paired bootstrap over the 58 gold studies, 4000 resamples, seed 20260906
          PASS  point estimate >= +0.010  AND  95% CI excludes 0
GUARD     no individual label may fall more than 0.02 below the incumbent
SECONDARY McNemar over discordant cells against the 303-study hand read
          (a 2:1 win needs >= 40 discordant cells to clear p<0.05 -- §4)
BUDGET    at most 3 candidates ever touch gold-58. All other iteration is hand-set only.
```

`+0.010` is chosen because the detectable floor is `+0.008`; anything smaller is a coin flip
dressed as a result. §1's expectation is `+0.055`, so a real win should clear this comfortably —
and if it does not, that is the answer.

**FALLBACK.** Three candidates spent without a pass → keep `data/targets.csv`, write one line in
`DECISIONS.md`, and go. **Stage L ends 2026-09-15 whether or not it has won.** Do not spend a
second week; `PIPELINE.md` already says so and it was right.

---

## 4. Extraction schema — grade the finding, then apply the cut

One structured object per (study, label). **Extract the graded quantity the criterion names,
then threshold it.** Never emit a bare probability, and never grade diagnostic certainty —
that is a different axis, and it is the one `rule_extractor.py` grades today (`REFERENCE.md` §2.1).

    "definite trace effusion"  -> certain, and BELOW the cut  -> should be ~0.05, is ~0.9
    "possible large effusion"  -> uncertain, and ABOVE the cut -> should be ~0.6

| label | extract | positive iff |
|---|---|---|
| ACL | `grade ∈ {none, signal-change, low-partial, high-partial, complete}`, `fibres_pct` | high-partial+ or >50% |
| MCL | `grade`, **`acuity ∈ {acute, chronic, remote, unstated}`** | high-grade+ **and** acute |
| Med/Lat Meniscus | `signal ∈ {none, intrasubstance, surface-contact}`, `morphology ∈ {normal, truncated, diminutive, displaced}`, `n_images` | surface-contact on ≥2, or morphology ≠ normal |
| Med/Lat/PF OA | `thickness_pct`, `extent_cm`, `grade`, **`compartment`** | ≥1 cm **and** >50% thickness |
| Effusion | **`size ∈ {none, trace, small, moderate, large}`** | moderate or large |
| Baker's | `size`, `location_typical` | moderate or large |
| Synovitis | `inflammation`, **`thickening`** | both present |
| Contusion | `marrow_edema`, `discrete_fracture_line` (**per lesion**) | edema present |
| Fracture | `cortical_break`, **`acuity`** | acute |

**Three fields are mandatory on every object, and two of them are new:**

| field | why |
|---|---|
| `p_above` ∈ [0,1] | the target. **P(above the official cut-point)**, carrying uncertainty *about the graded value* — so "small-to-moderate effusion" lands near 0.5 because it straddles the cut, not because the reader was unsure a mention existed. This is §1's +0.086 and it is not optional |
| **`mentioned` ∈ {0,1}** | was the finding addressed at all. §6 says this is 0.124 for Synovitis and 0.205 for Fracture. Without it, silence and denial are the same value and §5 cannot be run later without re-billing the whole corpus |
| `grade_raw` | the graded fields above, verbatim. Richer than 12 bits, enables an auxiliary severity-regression head, and lets a threshold be re-derived without a second API pass |

Plus `side ∈ {left, right, bilateral, unstated}` for the 99 bilateral reports.

⚠️ **The reports are 9 languages and 61% are not English.** Grade the finding, never the phrasing,
and normalise identically to `FINDINGS.md` §2.2 — Greek is written with the **micro sign µ**, not
Greek mu, and any regex using the real letter matches nothing.

---

## 5. Synovitis is a separate workstream, and it is not a prompt

Mentioned in **12.4%** of reports. Public labels fill the undecided cells from the Effusion field
— crude, and it works (0.790 on gold-58) because Effusion ↔ Synovitis correlate at 0.40
(`FINDINGS.md` §4). Against our hand read it agrees on 0.542 of cells with 45.8% over-call.

**No extractor can fix this, because the information is not in the text.** The lever is
imputation, and `REFERENCE.md` §4.1 already reports which kind works in the closest studied
analogue (CheXpert): **U-Ignore — masking the cell in the loss — is reported ineffective**;
**U-SelfTrained** — train, then impute the masked cells from the model's own predictions — is the
one label repair that measurably worked, and is what `stevenleehans` did to Synovitis by another
name.

**This is a `MODEL.md` decision, not a `DATA.md` one.** All Stage L owes it is the `mentioned`
mask, which makes all three options testable later at zero extra API cost. Ship the mask; leave
the choice to Stage M.

---

## 6. Cost, and why iteration happens on 303 studies

Measured from `data/llm_requests.jsonl`: 4,407 requests, ~1,180 input tokens each (**5.2 M
total**), system block 3,121 chars. The graded schema roughly doubles output over the old
five-state one.

| | full corpus, 4,407 | the 303-study hand set |
|---|--:|--:|
| Claude Opus 5, Batch API | **~$60–150** | **~$5–10** |
| Claude Sonnet 5, Batch API | ~$25–60 | ~$2–4 |

**Iterate on the 303. Run the full 4,407 only for a candidate that has already cleared the hand
set.** Three full passes is $450; three hand-set passes plus one full pass is $180 — and the
hand set is the better instrument anyway (§3), so the cheap path is also the correct one.

Four notes on the batch itself, none of which were true when `llm_requests.jsonl` was built on
2026-08-17:

- **Thinking is on by default on Claude Opus 5** (adaptive). That file sets `max_tokens: 16000`
  with no `thinking` key, which now means adaptive thinking billed as output — the difference
  between the low and high ends of the table above. Set `output_config: {effort: "medium"}`
  deliberately rather than inheriting it.
- **Use structured outputs** (`output_config.format`) so a malformed object is impossible rather
  than caught downstream. The schema in §4 is exactly what it is for.
- **Put the official criteria verbatim in the system block.** `REFERENCE.md` §2 is the cut-points
  themselves — ">50% of fibres", "≥1 cm and >50% thickness", "moderate or large", "acute" — and
  a reader given the actual criteria is the entire hypothesis. It also pushes the system block
  past the minimum cacheable prefix, which the current 870-token block sits under.
- `fallbacks` is rejected on the Batch API. Handle refusals by re-queuing, not by parameter.

**Host ruling: this is permitted.** `REFERENCE.md` §1.1 — submitting report text to a commercial
LLM for label extraction is explicitly *not* prohibited private sharing, governed by §2.6.b.

---

## 7. Stage L, day by day `Sep 7 – 15`

Runs **in parallel with Stage P**, which is Kaggle GPU and shares nothing with this. Both idle is
the failure of 2026-09-06.

| | day | work | artifact | done when |
|---|---|---|---|---|
| **L0** | Sep 7, ~1 h | fix the German/French `moderate` bug. **Re-export the hand labels with all four states preserved** (`+ / ? / − / —`) — the current CSV is binarised and throws away §1's +0.086. `rekey_labels.py --check` | `glossary.json` v2, `model_labels_states.csv` | `--check` passes, `§6` prints no warning |
| **L1** | Sep 7–9, ~5 h manual | finish the remaining **217 of 303**. For the six weak labels record a **grade**, not a bit; binary is fine for the frozen six | the 303-study iteration instrument | 303 rows, intra-rater checked on the 20 duplicate pairs |
| **L2** | Sep 8–11 | build the graded extractor to §4. Criteria verbatim in the system block, structured outputs, `mentioned` mask. Iterate on the hand set | `data/severity_v1.jsonl` | a candidate clears the hand set |
| **L3** | Sep 11–13 | full-corpus pass for winners only, then the §3 gate. **≤3 candidates on gold-58** | `data/targets_sev.csv` | gate passes, or three are spent |
| **L4** | Sep 13–14 | emit and verify the `mentioned` mask across all 12 labels; hand §5 to `MODEL.md` | mask columns | Synovitis mask ≈ 0.12, Fracture ≈ 0.21 |
| **L5** | Sep 15 | version and freeze | `targets_sev.csv` + schema hash | every downstream run records which label version it trained on |

**Versioning is mandatory.** `§R` happened because a geometry constant was written nowhere. Do
not repeat it with labels: `targets_sev.csv` carries a schema hash, and any training run records
it.

---

## 8. Closed by this audit — do not re-derive `§ numbers are audit sections`

| route | verdict | § |
|---|---|---|
| the over-call ratio as a gate | a threshold, not a property; AUROC invariant across it | 1 |
| "79.0% → 84.7% cell agreement" as the headroom | same artifact; prevalence-matched reads 84.8% already | 1 |
| the image-model OOF as a large-n label gate | ρ=0.429 p=0.337, and it ranks its own teacher first | 5 |
| iterating any prompt against gold-58 | marginal CI is 0.068 wide; use the paired delta and the hand set | 4 |
| a single severity prompt as the fix | six labels, six different causes; Synovitis is not extraction at all | 2, 6 |
| exporting our labels as bits | costs 0.086 macro, measured. Grade or do not bother | 3 |

**And the one that stands:** a model taught the severity distinction is learning something no
public arm was taught, because every public arm trains on the same mention-based tables. It is
worth about **+0.055** on the labels' own AUROC, with a lower bound near zero. That is smaller
than this document claimed yesterday, it is still the only lever the rest of the board does not
already have, and it is bounded — so run it to the bound and not past it.
