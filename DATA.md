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
