# The data — diagnosis and the pipeline that fixes it

**Data sets the ceiling. The model can only be as good as what it is taught, and what it is taught
is currently wrong in a specific, measured, fixable way.**

---

## The diagnosis, ranked by impact

| # | problem | measured | who else has it |
|---|---|---|---|
| **1** | **labels are graded on the wrong axis** | **6.68:1 one-directional over-call vs gold** | **the entire field** |
| 2 | slice coverage — 18 middle slices, not 64 spread | Lateral Meniscus 0.6422, MCL 0.6433 | nobody: the frontier fixed it |
| 3 | pixel source — third-party NIfTI repackaging | 0.085 (§W), 81.7% corpus coverage | nobody: the field reads DICOMs |

2 and 3 are catch-up. **1 is the only one where fixing it puts us ahead of the field**, because
every public arm trains on the same mention-based labels.

---

## 1. The labels report MENTIONS; the competition grades SEVERITY

The official criteria are explicit — *"ambiguous or borderline findings were graded as NEGATIVE to
favour specificity"* — and **10 of 12 labels carry a severity or acuity cut-point**: ACL needs >50%
of fibres, effusion needs *moderate or large*, OA needs ≥1 cm of >50%-thickness cartilage loss, MCL
must be *acute*, meniscus signal must *reach the surface*.

**Measured on the 58 gold studies:**

| | gold | public | over-call |
|---|--:|--:|--:|
| Effusion | 0.603 | 0.879 | 1.46× |
| Synovitis | 0.466 | 0.879 | 1.89× |
| MCL | 0.155 | 0.310 | 2.00× |
| Lateral OA | 0.190 | 0.310 | 1.64× |
| Contusion | 0.328 | 0.534 | 1.63× |
| *…every one of the twelve* | | | **1.17–2.00×** |

    public POSITIVE, gold negative : 127
    public negative, gold POSITIVE :  19
    ratio                          : 6.68 : 1

**Every label over-calls, and the disagreement is 6.7-to-1 one-directional.** That is not noise.
That is a systematic threshold shift, exactly as `REFERENCE.md` §2.1 predicted and §2b measured
from the other side.

### ⛔ Why this is not fixable by re-thresholding

**AUROC is invariant to monotone transforms** (§3a). Re-thresholding the labels to match gold
prevalence changes nothing — not the labels' own AUROC, and not the ranking a model learns from
them. **Anyone who "recalibrates" these labels is doing zero work.**

The labels *are* already soft — 80 distinct values, 52% of mass strictly between 0.05 and 0.95. So
the machinery for graded targets exists and is being used. **It is being used for the wrong
quantity: diagnostic certainty, not severity.** A report saying a tear is *suspected* becomes 0.8;
a report saying *trace effusion* also becomes high, because the extractor is confident the mention
is there.

    "definite trace effusion"    -> certain, and BELOW threshold  -> should be ~0.05, is ~0.9
    "possible large effusion"    -> uncertain, and ABOVE threshold -> should be ~0.6

**Recovering the severity gradient is new ranking information, not a transform of existing
information.** That is why it moves AUROC and re-thresholding cannot.

### ⛔ And why the model cannot denoise its way out

§T found the model *beating* its own labels on 5 of 12 — so denoising is real. But denoising works
on **random** error, which averages out. A 6.68:1 one-directional shift is **systematic**, and a
model will faithfully learn it: it learns to predict *"the report mentions this"*, which is the
target it was given, and is then scored against *"a radiologist graded it above threshold"*.

That is the single cleanest explanation for why the field piles up at 0.936.

### The bounded opportunity, stated honestly

Report-derived labels can never exceed what a careful human gets reading the same reports. §2b
measured that at **84.7% cell agreement** with gold. Our current public labels sit at
**79.0%** (146 disagreements of 696 cells).

**The headroom is 79.0% → 84.7% of cell agreement, and ~2/3 of the gap is the one-directional
over-call.** Not unlimited. But it is the difference between the field's ceiling and ours, and no
amount of architecture buys it.

---

## 2. Extraction schema — grade severity, per label

One structured object per (study, label). **Extract the graded quantity the criterion names, then
apply the threshold.** Never emit a bare probability.

| label | extract | positive iff |
|---|---|---|
| ACL | `grade ∈ {none, signal-change, low-partial, high-partial, complete}`, `fibres_disrupted_pct` | high-partial+ or >50% |
| MCL | `grade`, **`acuity ∈ {acute, chronic, remote}`** | high-grade+ **and acute** |
| Med/Lat Meniscus | `signal ∈ {none, intrasubstance, surface-contact}`, `morphology ∈ {normal, truncated, diminutive, displaced}`, `n_images` | surface-contact on ≥2, or morphology≠normal |
| Med/Lat/PF OA | `cartilage_loss_thickness_pct`, `extent_cm`, `grade` | ≥1 cm **and** >50% thickness |
| Effusion | **`size ∈ {none, trace, small, moderate, large}`** | moderate or large |
| Baker's | `size`, `location_typical` | moderate or large |
| Synovitis | `inflammation`, **`thickening`** | both present |
| Contusion | `marrow_edema`, `discrete_fracture_line` | edema present |
| Fracture | `cortical_break`, **`acuity`** | acute |

Emit the target as **P(above threshold)**, carrying uncertainty *about the graded value* — so
"small-to-moderate effusion" lands near 0.5 because it straddles the cut-point, not because the
extractor was unsure a mention existed.

**Also emit the raw graded fields.** They are a richer auxiliary signal than 12 bits (PLAN §2.1
asked for this and it was never built), and they enable an auxiliary severity-regression head later.

⚠️ **The reports are multilingual** (§0.3). Whatever extracts severity must handle that, and the
schema must be language-independent — grade the finding, never the phrasing.

---

## 3. The pipeline — three independent stages, each with its own gate

The structural point: **the label stage has a free, instant gate.** That is the fast loop this
project has never had.

```
STAGE L — labels                                    CPU/API, minutes per iteration
  4,349 reports
    -> severity extraction (schema above) -> data/severity_v1.jsonl
    -> threshold mapping                  -> data/targets_sev.csv
    -> GATE (free, instant, no training):
         macro-AUROC of targets-as-predictions vs gold-58
         current public = 0.8873   TARGET >= 0.93
         also: over-call ratio, currently 6.68:1  TARGET < 2:1
    -> iterate the prompt until the gate clears. Minutes, not hours.

STAGE P — pixels                                    Kaggle, ~8 h, ONCE
  DICOMs (never NIfTI)
    -> 5 slots / 64 slices, sampled 6-94% of each stack
    -> 140 mm mm-crop via PixelSpacing -> 336 px
    -> publish Dataset `rsna-knee-tiles140`
    -> GATE: DINOv2-small fold 0 >= 0.83 on the 691-study window
             (the same recipe scored 0.7597 on NIfTI)

STAGE M — model                                     Kaggle, ~3 h per experiment
  windows -> backbone -> per-label attention -> 12 logits   (MODEL.md)
    -> GATE: Lateral Meniscus and MCL +0.05 each; macro +0.02 from the head
```

**L and P are independent and must run in parallel** — L is CPU/API and P is Kaggle GPU. Neither
blocks the other, and today both machines sat idle.

**Versioning is mandatory.** `targets_sev.csv` carries a schema hash; any run records which label
version it trained on. §R happened because a geometry constant was never written into a manifest.
Do not repeat it with labels.

---

## 4. Why this is the 0.97 route and architecture is not

- Architecture gets us to the field's frontier. The frontier is **public**, and its 0.9167 gold-58
  was trained on **the same mention-based labels we have**.
- Every public arm inherits the 6.68:1 over-call. It is the shared ceiling.
- **A model taught the severity distinction is learning something no public arm was taught.**

That is a real mechanism, and it is the only one on the table that the rest of the board does not
already have. It also has a bound (§1), so it should be run to that bound and not past it.

**Order: L first — it is free, it is fastest to iterate, and it changes what every later run is
trained on.** Retraining on bad labels is the expensive mistake.
