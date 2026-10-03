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

### ⛔ And the thing this stage is NOT — `§3p`, restored 2026-09-06

**`§3p` (measured 2026-08-13, valid, and dropped from `DECISIONS.md` in the 09-06 rewrite) says
the labels are not the binding constraint, and nothing since has overturned it.**

| on the same gold studies | gold | → LB at §3v's +0.039 |
|---|--:|--:|
| **`steven_v2` — literally our `data/targets.csv`** — used as a predictor | **0.8985** | **~0.938** |
| pilkwang 20-member OOF | 0.8516 | ~0.891 |
| our model reproducing **its own training labels**, n=4,407 | **0.8486** | |

**A model that perfectly fit the labels we already have would score ~0.938 — at the prize line of
0.940.** The teacher→gold gap is 0.10; the model→teacher gap is 0.15. **We are further from the
labels we have than those labels are from truth.**

So Stage L raises a ceiling the model is currently **0.05 below**. That is worth doing — it is
cheap, it is CPU/API, it runs on a different machine than Stage P, and the ceiling will bind once
Stage M lands. **It is not the highest-impact stage, and `PIPELINE.md` was wrong to label it so.**
`MODEL.md`'s slice coverage and per-label attention are where the 0.05 is.

**One correction to §3p's companion.** `§3x` refuted the severity argument on the grounds that
"if severity confusion were binding, the teacher would be weakest on the threshold labels." That
is a cross-label correlation and it does not carry: teacher AUROC against the fraction of mentions
that are explicitly sub-threshold reads **ρ = −0.38 (p = 0.22)**, and against mention rate
**ρ = +0.49 (p = 0.11)** — both the right direction for their respective stories, **neither
significant at n = 12 labels.** Severity is therefore *unprioritised by §3p's structural argument*,
not refuted by §3x's. Same practical conclusion, honest reasoning, and the third re-derivation of
this argument stops here.

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

---

## 9. C6 Stage 0 — the data, measured `2026-09-23`

Kernel `raahimnawaz/rsna-knee-c6-stage0` v2, competition DICOMs, headers only. **4,407 studies,
24,371 series, 0 errors**; true slice spacing from a 400-series sample (`sample_n` 400, image
positions projected on the slice normal). Smoke-tested first against a fixture with known geometry
(`RUNBOOK.md` §2.1) and on 20 studies on Kaggle (§2.2). Outputs are in `data/stage0/`, which is
gitignored; the tables below are the record. The rules applied are `RUNBOOK.md` §2.4's, fixed
before the run. **No AUC is produced or quoted here.**

**A · Anisotropy — isotropic 3D is contraindicated.** `r_median_overall` **12.2** against a bar of 4.

| plane | in-plane mm | true spacing mm | `SliceThickness` mm | gap mm | r median | r p10 | r p90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Sagittal | 0.3125 | **3.30** | 3.0 | 0.30 | 11.9 | 5.8 | 17.6 |
| Coronal | 0.3125 | **3.60** | 3.0 | 0.60 | 11.8 | 7.2 | 19.5 |
| Axial | 0.3125 | **3.90** | 3.0 | 0.90 | 12.8 | 5.4 | 20.8 |

There is an **inter-slice gap** in every plane (10–30% of the slab). `SpacingBetweenSlices` agrees
with the measured spacing (3.3 / 3.6 / 3.9); `SliceThickness` does not. No code on the training or
scoring path uses `SliceThickness` as a sampling interval (checked 09-23; only the audit scripts
read it). A 3D arm, if ever built, uses anisotropic `(1,3,3)` kernels — Stage E.

**B · Stack depth — `slots64` survives.**

| plane | series | p10 | median | p90 | max |
|---|--:|--:|--:|--:|--:|
| Sagittal | 9,864 | 21 | **30** | 36 | 320 |
| Coronal | 8,609 | 18 | 30 | 36 | 80 |
| Axial | 5,898 | 22 | 32 | 44 | 254 |

Sagittal median 30 meets the ≥ 30 bar exactly, so 18 slices over 6–94% is a real sub-sample and the
spec stands. At the p10 (21 slices) SAG-1 takes nearly every slice of the stack — dense, not padded.

**C · Two series per plane — both second slots stay.**

| | measured | bar | verdict |
|---|--:|--:|---|
| ≥ 2 sagittal series | **97.3%** | 60% | keep `sag_2` |
| ≥ 2 coronal series | **74.5%** | 60% | keep `cor_2` |
| ≥ 3 sagittal / coronal | 21.8% / 16.2% | — | the studies the 09-23 `cor_2` fix touches |
| ≥ 1 sagittal / coronal | 100% / 100% | — | |

Composition by plane × fluid-sensitivity (studies with ≥ 1 series): Sagittal fluid 94.2%, non-fluid
96.8%; Coronal fluid 96.4%, non-fluid 77.3%; Axial fluid **100%**, non-fluid 19.5%. So the AX slot
is almost always a fluid-sensitive series, and COR-2 falls back to a second fluid series in ~23% of
studies.

**What it means.** All three of the `slots64` assumptions that were made without evidence hold.
This is also the frontier's geometry (`MODEL.md` §0), so Stage 0 now supports it with our own
measurement rather than theirs.

---

## 10. Stage A — the 64-slice cache, built `2026-09-23`

`pipeline/build_slots64.py` via `notebooks/build_stage_a_kernel.py`: four CPU kernels
`raahimnawaz/rsna-knee-stage-a-s{0..3}of4`, each writing `stage_a/{vols.npy, masks.npy, ids.npy,
slots.csv, manifest.json}` (~8 GB of `vols.npy` per shard). Spec `c6-64slice`, span **2–98%**, 140 mm
physical crop (padded, not clamped), 336 px, 2–98th percentile window per series. Built from commit
`de29d23`. Manifests and `slots.csv` are in `data/stage_a/s*/` (gitignored); the numbers below are
the record.

| | shard 0 | shard 1 | shard 2 | shard 3 | all |
|---|--:|--:|--:|--:|--:|
| studies | 1,102 | 1,102 | 1,102 | 1,101 | **4,407** |
| study errors | 0 | 0 | 0 | 0 | **0** |
| decode failures | 0 | 0 | 0 | 0 | **0** |
| series ordered by geometry | 5,220 | 5,183 | 5,197 | 5,192 | **20,792 (100%)** |
| wall, 4 workers | 12.0 min | 11.1 | 10.8 | 14.6 | |

**GATE A — passed.**
- *Manifest records geometry:* `fov_mm` 140, `depth_range` [0.02, 0.98], per-series spacing,
  `n_available` and `source_step` in `slots.csv`. ✅
- *Slot fill ≥ Stage 0's prediction:* sag_1 1.000, sag_2 **0.973**, cor_1 1.000, cor_2 **0.745**, ax 1.000
  — **equal to §9's independent header count to the third decimal** (97.3%, 74.5%). ✅
- *Edges are anatomy, not wrap or air:* the 20-study render shows knee at both ends of every slot and
  no wrap. Measured over 1,200 studies, the share of tissue pixels (>25/255) is thinner at the ends:

  | slot | first | mid | last |
  |---|--:|--:|--:|
  | sag_1 / sag_2 | 0.50 / 0.49 | 0.83 / 0.81 | 0.52 / 0.51 |
  | cor_1 / cor_2 | **0.22 / 0.19** | 0.81 / 0.81 | 0.75 / 0.73 |
  | ax | 0.52 | 0.60 | 0.66 |

  The anterior-most coronal slice is mostly air around the patella. Kept at 2–98% because the
  frontier's 2–98% corpus beat its 6–94% one on LB (0.928 vs 0.924); the attention head can ignore
  thin windows. ✅ with that caveat.

**Two corrections, measured here.** (1) No series in the corpus has anisotropic pixels (0 of 20,792),
so the frontier's single-spacing crop, which would stretch such a scan, never fires on this data.
The difference that is live is padding: **6.2%** of series (8% of axial/coronal) have a field of view
under 140 mm on their short side (§9's table), where the frontier's clamp silently zooms in and
this cache pads. (2) A window's three channels span, in the source stack, a median of **1.6** slices
(sag_1), 2.0 (sag_2), 2.5 (cor_1), 3.9 (cor_2) and 2.6 (ax) — consecutive *sampled* slices, the
fork's and the frontier's convention (`MODEL.md` §3).

### 10c. Stage B — the 96-slice, 384 px, canonical cache, built `2026-09-25`

`build_slots64.py --spec c7 --canon` via `build_stage_a_kernel.py --stage b`: eight CPU kernels
`raahimnawaz/rsna-knee-stage-b-s{0..7}of8` (~7.8 GB of `vols.npy` each), built from commit `62fa3fe`,
6.8–8.3 min per shard, 3.21 s/study/worker. **GATE B passed** (all 8 manifests): one spec (`c7-96slice-384`),
every canonical, one span (2–98%); **all 4,407 train studies exactly once; 0 study errors, 0 decode
failures**; fill sag_1 1.000, sag_2 **0.973**, cor_1 1.000, cor_2 **0.745**, ax_1 1.000 — Stage 0's count to
the third decimal — and the new slots sag_3 **0.218**, ax_2 **0.248**. Side for every study: R 2,295 / L 2,112,
from the tag on **2,203** (FINDINGS §6.2's tagged count, exactly) and from the −62 mm rule on 2,204. Operations
applied: only the left-knee mirror (coronal `yH` 3,627, axial `zH` 2,587, and the sagittal slice order);
**no scanner in the corpus stores reversed or transposed rows/columns**, so canonicalisation here is laterality
alone. Real-knee check on the 20-study smoke: corrected right and left knees agree better as built than
mirrored in all three planes (§10b).

### 10b. What Stage A throws away — measured `2026-09-24`, from `slots.csv`, Stage 0 and the header probe

| | measured | what it means |
|---|---|---|
| **slices** | a series holds a median **30** (axial 32, p90 48; 3D series up to ~186); we keep 18 / 14 / 12 / 8 / 12, i.e. a median **60% / 47% / 40% / 27% / 38%** of each series. **Every** series loses slices. | a 3 mm-slice lesion can sit between kept slices; a window's channels are 1.6–3.9 source slices apart, not adjacent |
| **resolution** | pixel spacing median **0.321 mm** → **437 px** natively across 140 mm; stored at 336. **76%** of series are downsampled, **50%** by more than 1.3× | small, localised findings (menisci, MCL, fracture) lose the most; §3x on the old path: 224→518 = +0.013 |
| **series** | a study has a median 5 series; the 5 slots take ≤2 sagittal, ≤2 coronal, 1 axial. **44%** of studies have at least one series never used — **3,579 of 24,371 (15%)** | unknown value: some are 3D reconstructions of a kept series, some are other contrasts |
| **laterality** | `build_slots64.py` does **not** canonicalise left/right (the retired `preprocess.py` did; the public stack's `normalise_laterality` does). Side is recoverable: tag on 50%, the IPP-x rule at 97.3% (FINDINGS §6.2) | sagittal slices run patient-left → patient-right, so a slot's first slices are the LATERAL compartment on a left knee and the MEDIAL one on a right knee; coronal/axial are mirrored. `WindowMIL` has no slot or depth input, so today the model must tell compartments apart from content alone |

**The laterality diagnostic is inconclusive.** On the 484 of the 691 with a known side (tag or
SeriesDescription token; R 240 / L 244), `c6_ortho` differs by side a lot on some labels (MCL 0.720 R vs
0.844 L, Fracture 0.766 vs 0.927, per-label SE 0.03–0.09), but the mean R−L gap is no larger on the
side-specific labels (−0.027) than on the others (−0.038). That is the signature of site/case mix (side
availability is itself site-dependent, FINDINGS §7), not of orientation. Only an A/B can answer it.

**Why MCL and the OA labels are the weakest — measured 2026-09-25, not argued.** On the 681 scorable
fold-0 studies, `c6_ortho` reads MCL 0.761, Lateral OA 0.748, PF OA 0.785 against 0.835–0.898 elsewhere.
Two popular explanations do NOT hold here: (1) *label noise on the instrument* — `steven_v2` (the training
labels) predicts the `gpt56sol` reference at **0.979 (MCL), 0.988 (Lat OA), 0.998 (PF OA)**, as clean as
any label (0.93–1.00; corr with model AUC across labels **+0.00**); (2) *class scarcity* — reference
prevalence is MCL 14.6%, Lat OA **25.7%**, PF OA 45.1%, against Fracture 6.4% (which reads 0.874); corr of
prevalence with model AUC **+0.09**. What is left is vision: the model recovers the smallest share of the
report signal on exactly these three (model ÷ label agreement **0.78 / 0.76 / 0.79**, others 0.85–0.91).
All three are thin or fine-textured — a 2–3 mm ligament at the image edge, 2–3 mm cartilage and marginal
osteophytes — i.e. smaller than one ViT token (16 px = 6.7 mm at 336 px, 5.8 mm at 384). Consistent with
it: the CNN-stage CoAtNet beat Ortho on Lateral OA and PF OA in the A/B (`RUNBOOK.md` §2c). Against the
58 gold studies (reported only, SE 0.04–0.09) the REPORTS are weakest on Lateral OA (0.830), which caps
what any report-trained model can show on the leaderboard for that label.

**The model has reached its labels — measured 2026-09-25 on gold-58 (reported only, never selected on).**
Against the radiologist's 58, `steven_v2` (the training labels) reads **0.8873**; `c6_ortho` reads **0.8777**,
**−0.0096 ± 0.0158** — level within noise, where `ab_ortho` / `ab_coatnet` were −0.030 / −0.029 on 09-23.
It already beats the labels on **Effusion 0.942 vs 0.853, Contusion 0.907 vs 0.831, Medial OA 0.944 vs
0.931** (findings reports under-describe) and trails on ACL, MCL, PF OA, Lateral Meniscus (per-label SE
0.04–0.09). This is the condition the 36th-place competitor names (Kaggle discussion, 09-25) for switching
effort to labels: a model whose predictions sit closer to the radiologist than the labels it learned from
carries information the labels lack, which is what teacher→student pseudo-labelling recovers. `D4`'s
"labels are not the binding constraint" was measured at 0.76; it predicted the ceiling would bind later,
and this is later. ⚠️ **The 691 instrument cannot see that gain:** it is scored against `gpt56sol`, another
LLM reading of the same reports (0.98–1.00 agreement with `steven_v2`), so a model that moves from the
reports toward the images reads as a LOSS there. A pseudo-label run needs gold-58 as its diagnostic and
the leaderboard as its judge ([[measure-in-the-target-currency]]).

None of the four is measured as a loss yet; each is a gap between what the scanner recorded and what
the model sees. The public stack's newest members move the same way (`raptor-knee-native384dense`,
a 96-slice "Global96" CoAtNet, D4's depth zones inside each slot).

---

## 11. Public label tables against gold-58 `2026-09-23`

`python extractor/bench_public_labels.py --download` (extended 09-23 with four new sources, a paired
read against the training targets, a split-half test of per-label picking, and a gold-coverage guard).
Gold-58 has 9–35 positives per label (MCL 9, Lateral OA 11, Baker's 12). Every third-party read is a
**ceiling**: a prompt tuned on these rows leaves no trace.

| source | gold-58 macro | paired vs `steven_v2` | P(better) | note |
|---|--:|--:|--:|---|
| `jev_hybrid` | 0.9025 | +0.0152 ± 0.0063 | 0.996 | **per-target pick made on gold-58**, blend weight on gold-74 (its README) |
| `steven_v4` | 0.8927 | +0.0054 ± 0.0043 | 0.908 | a blend, possibly selected on gold (author) |
| **`steven_v2`** | **0.8873** | — | — | **our training targets**; matches the author's published 0.8873 |
| `steven_full` | 0.8780 | −0.0094 | 0.004 | |
| `jev_text` | 0.8745 | −0.0129 ± 0.0110 | 0.128 | JEV's text-only half |
| `pilkwang_v2` | 0.8658 | −0.0216 | 0.011 | |
| `hosseini_jev` | 0.8554 | −0.0319 | 0.000 | class probabilities, scored P(1) + ½ P(not said) |
| `lixin_gpt56` | 0.8352 | −0.0522 | 0.000 | the OOF scoring reference |
| `pilkwang_v1` | 0.8125 | −0.0749 | 0.000 | |
| `dread_soft` | — | — | — | **carries 0 of the 58 gold rows** — the frontier held gold out; not scorable here |

**The per-label pick does not transfer.** Choosing, per finding, the best of `steven_v2`, `steven_v4`,
`pilkwang_v2`, `jev_text`, `hosseini_jev` on a random 29 gold studies and scoring the composite on the
other 29 against `steven_v2` on the same 29: **−0.0002 ± 0.0090, P(better) 0.527**, 1,996 splits.
The picks are unstable (Fracture goes to `jev_text` 76% of the time, Lateral OA 60%; nothing else
above 68%). `jev_hybrid`'s +0.015 is the selection, not the labels — `never-select-on-a-tiny-set`,
measured rather than warned about.

**No leak flags** on any source (`leaks_gold`, `gold_rows_differ`). **No table is a copy of the scoring
reference**: mean per-label Spearman with `lixin_gpt56` on non-gold rows is 0.59–0.77 (`pilkwang_v2`
highest at 0.774), so a training-label switch would not win on `score_oof.py` by construction —
but none is worth switching to. **`tuyennguyenthicam/rsna-knee-expert-58` is not a label table**: it
holds `test.csv`, `test_series.csv` and test DICOMs.

**Added 2026-09-26 — the Qwen family, including our own read** (`extractor/bench_qwen_labels.py`, pre-registered in
`RUNBOOK.md`'s Qwen block; public tables from the 09-26 scratch bench):

| source | gold-58 | paired vs `steven_v2` | note |
|---|--:|--:|---|
| `qwen35_9b_v1` rank-mean with `steven_v2` (C2) | 0.8928 | +0.0054 [−0.006, +0.018] | **FAIL** (bar +0.010, CI > 0) |
| `xhaitanya/qwen38b-labels-rsnaknee` | 0.8797 | −0.0076 | public Qwen3-8B |
| **`qwen35_9b_v1`** (ours: Qwen3.5-9B, graded letters, local) | 0.8765 | −0.0108 [−0.030, +0.009] | **FAIL**, frozen-six −0.039 |
| `laymond` Qwen3-8B v2 filled / raw / v1 | 0.8686 / 0.8521 / 0.8295 | −0.019 / −0.035 / −0.058 | |
| `lvthanhxun` Qwen teacher v9 (fused) | 0.7541 | −0.133 | |
| ⛔ `dhiyaneshbabu/rsna-knee-labelled-train`, `yehezkielhaganta/psuedo-labeling-knee` | 1.0000 | — | **GOLD IS IN THE FILE** |

Every Qwen read is a different reader (Spearman with `steven_v2` 0.72 on non-gold rows) and reads Fracture far better
(ours +0.108) while losing the frozen labels (ACL, MCL, Medial OA, Baker's −0.03 to −0.05). The rank-mean keeps the
fracture gain and gives back most of the losses — and still does not clear the gate.

**⛔ SUPERSEDED 2026-09-27 by the leaderboard: `dread_soft` is the training target** — step 5 retrained on it, nothing else
changed, scored LB 0.916 against 0.901 (`RUNBOOK.md`). gold-58 could not read it (it holds gold out); the LB could.
**What `dread_soft` IS, read 09-27** (4,349 rows, 0 gold): a **graded** table with 4–7 fixed levels per finding, not a
probability — Effusion 0.05 (30%) / **0.20 (44%)** / 0.85 (21%) / 0.95 (4%); Baker's 0.05 / 0.20 / 0.85 / 0.95; Medial
Meniscus 0.05 / 0.20 / 0.45 / 0.80 / 0.95. The big step sits AT the official threshold: "present but below it" is 0.20, near
absent — the same 44% "below threshold" share our Qwen read found for Effusion (§12). `steven_v2` orders studies much the same
(Spearman 0.84–0.94 on 9 of 12; Synovitis 0.53, Fracture 0.61, Contusion 0.75) but places them higher and continuously (~30
levels; mean Effusion 0.58 vs 0.33, Baker's 0.47 vs 0.17, Fracture 0.30 vs 0.12). **Hypothesis, not measured:** it trains better
because its largest target gap falls where the radiologists draw their line (borderline = negative), so BCE spends its effort
there — consistent with the biggest gold gain landing on Fracture (0.78 → 0.91), where the two tables disagree most. The LB proved
the table is better, not why. It is the best table measured, not an ideal one: the ideal is image reads, and 58 exist.
~~**Decision:** `steven_v2` stays the training target.~~ ~~Labels were not the binding constraint
(the model reproduces its own training labels at 0.8486, below the labels' own read)~~ — superseded
09-26: Stage B reads gold-58 0.890 against `steven_v2`'s 0.887, so that argument is spent (`DECISIONS.md`
D4). No public table moves them. `DECISIONS.md` CLOSED.

---

## 12. Where the reports and the image reads part — and what each kind of gap needs `2026-09-26`

From the owner's questions on 09-26 ("are the reports reliable, can the adjectives help, how do real
vision models deal with this"), answered from the data, not from argument.

**Three kinds of mismatch, each seen on gold-58** (report paraphrased; gold is the image read):

| report says | gold | `steven_v2` | Qwen (`qwen35_9b_v1`) | kind | what can fix it |
|---|--:|--:|---|---|---|
| medial meniscus signal "without extension to the articular surface" | 0 | 0.82 | 91% "present, below threshold" | **misread severity** | a better reader (Qwen got it) |
| "moderate thinning" of cartilage, full-thickness fissures, compartment not tied down | 0 (Medial OA) | 0.94 | 69% "below threshold" | **borderline judgement** | severity-aware targets |
| "moderate joint effusion" — the criterion's own word | 0 | 0.95 | "meets" | **report ≠ images** | nothing in the text; only the images |
| "osteochondral fracture, 8×12 mm" and, elsewhere, "no fracture is seen" | 0 | 0.95 | "meets" | **contradictory report, and ≠ images** | nothing in the text |

The last two are why a report reader has a ceiling (~0.90 on gold for the best tables; a careful human read
~0.84–0.87, §1): some of the image truth is not in the text at all. Of the 3 English gold reports that say
"moderate/large effusion", the image readers call 1 negative.

**The severity the binary labels throw away — Qwen's most likely grade, share of all 4,407 reports:**

| finding | meets | present but BELOW threshold | explicitly absent | not mentioned |
|---|--:|--:|--:|--:|
| Effusion | 0.20 | **0.44** | 0.30 | 0.06 |
| Contusion | 0.18 | **0.52** | 0.10 | 0.21 |
| Medial OA | 0.12 | **0.39** | 0.19 | 0.30 |
| MCL | 0.01 | **0.33** | 0.49 | 0.17 |
| Medial Meniscus | 0.33 | 0.23 | 0.36 | 0.07 |
| Fracture | 0.03 | 0.08 | 0.33 | 0.55 |

(All twelve, and the full six-letter distribution per study, are in `data/qwen_labels/qwen35_9b_v1/probs.jsonl`.)
Today "trace effusion" and "no effusion" reach the image model as the same kind of negative.

**What each kind of gap needs — candidate experiments, NOT run, NOT pre-registered yet:**
1. **⛔ Mostly ALREADY IN THE TARGETS since 09-27 (measured 09-27):** `dread_soft`'s levels ARE severity grades. Cross-tab
   against our Qwen graded read, Effusion: level 0.05 = "absent" 90%; **0.20 = "present, below threshold" 75%**; 0.85 = "meets"
   65%; 0.95 = "meets" 93%. Medial Meniscus: 0.05 = absent 84%; 0.20 = "not mentioned" 95%; 0.45 = below threshold 82%; 0.80 /
   0.95 = meets 75% / 89%. So training on `dread_soft` already teaches "small vs large" — part of its +0.015 on the LB. A separate
   severity head would mostly repeat it; NOT tested, and not worth a run on its own. LOCATION (item 5) is the untested half.
   Original note: **An auxiliary severity output** (the owner's idea, 09-26): a second head predicts the grade distribution
   (absent / below / borderline / meets) beside the 12 main outputs, so the model learns what small vs large
   looks like — the ordering AUROC rewards. Main targets stay `steven_v2`, so the 691 CAN judge it (fixed-target
   comparison). One run. Grades are Qwen's reading, which lost on gold as a *replacement* target (§11) — as a
   secondary signal they are a different use, untested. "Where" (horn, condyle) is a possible later target.
2. **Teacher → student** for report ≠ image — **✅ the ingredient measured 09-27 (diagnostic, fixed 50/50, nothing tuned):**
   on gold-58 the equal rank-mean of the report label (`steven_v2`) and our image model's OOF (`dread`, never saw gold)
   reads **0.9269** vs the image model 0.9121 (+0.015 [−0.001, +0.031], P 0.96) and the report label 0.8873 (+0.040
   [+0.024, +0.057]). The combined target sits closer to the radiologists than either — what a student would train on. It
   needs OOF image predictions for EVERY training study (k-fold teachers): ~5 × 8 GPU-h with Ortho ViT-L, far less with a
   small CNN (`REFERENCE.md` §3.1b). Original note: out-of-fold image models vote; where confident and in
   disagreement with the label, the label moves. The board's 0.950 single models credit this. Several teacher
   runs — the expensive route. Precondition met: Stage B reads gold 0.890 against its labels' 0.887.
3. **A medically pre-tuned model** (the Ortho idea, for text or images): Google's **MedGemma** — 1.5 4B
   (image+text; its model card lists 3D CT/MRI volumes and an MRI set spanning head, abdomen and **knee**;
   reported MRI macro accuracy 64.7% on 10 conditions) and 27B text. As a report reader it meets the same
   ~0.90 text ceiling and would spend the LAST gold-58 read; as an image model it is a second domain-pretrained
   candidate for the blend. Gated on Hugging Face (the owner must accept the terms — our account is not
   authorised, checked 09-26), under Google's Health AI Developer Foundations terms — admissible under §2.5.a's
   carve-out, but the winner obligation to publish weights must be checked against those terms first.
   **✅ TERMS READ 2026-09-28 (owner's question; `developers.google.com/health-ai-developer-foundations/terms`) — workable, unlike
   OAI; still NOT used.** Anyone may use, modify and Distribute; commercial use is not excluded. Distributing HAI-DEF or a "Model
   Derivative" (§3.1) requires passing on §3.2's use restrictions as enforceable terms, a copy of the agreement, a Notice file and
   marked modifications; extra terms are allowed if they comply. §3.2: no uses in the Prohibited Use Policy, nothing that could make
   Google a medical-device "manufacturer"; clinical use needs regulatory authorization. **The catch:** "Model Derivatives" include any
   model trained on HAI-DEF's Outputs "to perform similarly", distillation and synthetic training data included — so a student
   distilled from a MedGemma teacher inherits these terms (a text-label pass feeding an image model is arguably not "similar", but
   grey). The competition side fits: free click-through (host: non-commercial terms alone do not exclude), and rules §5's carve-out
   lets a winner publish a model built on an incompatibly licensed pretrained model under its own terms instead of CC BY-NC.
   **Why still not used:** as a report reader it meets the same ~0.90 text ceiling (our similar-size Qwen3.5-9B read FAILED); as an
   image model its card's MRI result is 64.7% macro accuracy on 10 conditions, our one VLM attempt (Qwen) failed to learn, OrthoFoundation
   already fills the domain-pretrained slot, and the public stack already uses ~8–9 h of the 9 h runtime.
5. **WHERE, from the reports (the owner's idea, 09-27)** — the reports say where a finding is far more often than the labels
   use: of English reports (1,715 of 4,349, by a crude word test) with a `dread_soft` positive, **86–100% name a sub-location**
   (Medial Meniscus 91% — horn/body/root, of 207; Lateral Meniscus 86% of 92; Medial OA 95% of 173; Baker's 100% of 170;
   Contusion 19/19 and Fracture 15/15, few). ⛔ **`test.csv` carries NO reports** (StudyInstanceUID only), so location can
   only TEACH during training; at test time the model must find it itself. Three forms, cheapest first: (a) extra "where"
   outputs — medial / lateral / patellofemoral, anterior / posterior horn, femur / tibia / patella — from the same LLM pass as
   item 1's severity grades, one run, judged on the 691 like item 1 (the 12 main targets unchanged); (b) nudge the per-finding
   window attention toward the stack region the text names (needs Stage B's canonical orientation; a wrong mapping teaches
   the wrong place); (c) pixel-level crops — needs anatomy landmarks we do not have (segmentation data is external, blocked).
   ⚠️ This repo once built compartment attention for what was a slice-sampling bug (`DECISIONS.md` §U→D2): before (b), check
   where the trained model's attention already goes.
   **Owner, 09-28 ("crop to the report's location, image ~50% smaller"):** not at test time — no reports there — and cropping
   training images from the report would train on views the test never gets. The report-free form IS the Mac CNN test's
   zoom arm (`RUNBOOK`, "CNN TEST — detail vs pretraining"): a centre crop at the same pixel count ≈ 1.5× detail on the joint
   line and notch, where the weak findings live. Order: zoom arm first; only if zoom HELPS, form (a) — report "where" as extra
   training outputs, ignored at test — as the one follow-up run.
7. **"A big proven model makes the labels, ours learns from them" (the owner, 09-27)** — that IS teacher → student; what
   matters is that the teacher (a) is good on images and (b) has NOT seen the knees it labels. Every "proven" public model (the
   0.928 CoAtNet, the stack's members) trained on all ~4,350 training studies — on them it largely echoes the labels it was
   trained on, so its "new labels" are not independent. Two teachers that are: **our own k-fold models** (the plan), and **a
   model trained on fastMRI+'s radiologist image boxes** — it never saw a competition knee, and its signal comes from
   radiologists reading images, not from reports. Limits: coronal-only (to confirm), a different hospital, labels to map
   (meniscus tear without medial / lateral; ACL / MCL graded low vs high, as ours). Bigger is not the lever here: the board's
   0.950 single models say labels were, and small CNNs reach 0.94+.
6. **A "normal knee" baseline from Ortho (the owner's idea, 09-27)** — score each knee by how far it sits from normal knees
   in Ortho's feature space, per slot and depth (anomaly detection, PatchCore/PaDiM-style), as an extra signal. Measured
   first, on `dread_soft`: **1,131 of 4,349 knees (26%) are below threshold on all 12 findings, but only 26 (0.6%) are
   clean** (every finding graded absent — no "trace" anything); gold-58 has none. So "normal" here means "below the
   radiologists' line", which is full of small findings — a distance-from-normal flags trace effusion, which the test grades
   NEGATIVE; the supervised head must still learn the threshold. Frozen Ortho features were weak as classifiers (C-5: 0.771,
   confounded by the 160 mm crop, vs 0.815 fine-tuned). Best hope: rare, visually distinct findings (Fracture, Contusion,
   MCL). No board evidence. Cheapest test: embeddings from the trained `dread` model (~1.5 GPU-h), kNN distance to the 1,131
   per slot/depth, and whether it adds anything to the model's own OOF on the 691 — pre-register before. Ranked BELOW teacher
   → student. Variant: distil Ortho into the small CNN if the CNN falls short — at the cost of the CNN agreeing more with Ortho.
8. **Owner Q 10-03 ("how do we fix the labels, are there features we are not considering") — candidates, NOT run:**
   (a) **Synovitis by the `mentioned` flag:** where the report is silent (87.6%), take the image teacher's prediction as the
   target instead of the effusion copy (`probs.jsonl` holds "not mentioned" for all 4,407). (b) **Compartment for the OA
   findings**, with item 5(a)'s "where" outputs: Lateral OA fails on medial-vs-lateral attribution (§2), and the reports
   name the place in 86–100% of positives. (c) **Other report findings as extra training outputs** (PCL, LCL, cartilage
   grade, osteophytes, loose bodies): more supervision per knee, ignored at test; nowhere in this repo before today.
   (d) **Sex / field strength / scanner as inputs** (old PLAN F3, never run): OA prevalence differs by scanner (FINDINGS
   §7.1), but the site prior died (−0.0057), so low odds; no patient age exists. Judge any of them by a paired
   `--core-alone` LB read against student r1, never on gold-58. On 10-03 the weak findings were Synovitis, PF OA, Lateral
   OA and Lateral Meniscus for both students and the public core (`RUNBOOK.md`).
4. **External expert image labels** — ⛔ *corrected 09-26: MRNet's released labels are report-extracted (only its hidden
   test set is image-read), so of these only OAI/MOAKS and fastMRI+ carry image supervision; `REFERENCE.md` §1.3 has
   the two permission questions and who can answer each.* MRNet (1,370 knee MRIs; ACL, meniscus, abnormal; expert-read test set),
   OAI with MOAKS grades (severity, from images), fastMRI+ (boxes: location) — exactly the supervision the
   reports cannot give. **Blocked on the host's ruling** (`REFERENCE.md` §1.3: asked twice by others, never
   answered; our post drafted 08-17, never sent). The MRNet *model* (2018, AlexNet per plane) is not worth
   borrowing: its ACL 0.965 / meniscus 0.847 are on its own expert-labelled set, and our recipe is stronger.
