# Reference — external facts, verbatim where it matters

Things that are **true about the competition and the world**, not about our code. Separate from
the other four documents on purpose:

| doc | holds |
|---|---|
| `README.md` | current state, the ledger, the plan |
| `PLAN.md` | strategy and architecture |
| `FINDINGS.md` | measured facts about **our** data pull |
| `IMPROVEMENTS.md` | friction log — what broke and why |
| **`REFERENCE.md`** | **external ground truth: host statements, official criteria, forum facts, literature** |
| `COMPETITION_RULES.txt` | the rules, verbatim |

**Why this file exists.** README §9 records that this project's most expensive failures were
claims about the world outside the repo — recorded once, then treated as constants. Four have
now been retracted (§9.1, §2e, §2f, §2i-b). A claim in here carries its **source and the date it
was read**, so the next reorder can check it instead of inheriting it.

---

## 1. Host rulings

### 1.1 Commercially hosted LLMs are PERMITTED `HOST STATEMENT, read 2026-08-10`

> "Use of commercially hosted LLMs and other external inference services is permitted, provided
> that the service and method of use otherwise comply with the Competition Rules... submitting
> Competition Data, including report text, to an external LLM or API for inference or other
> computational processing (for example, extracting labels from reports) **will not, by itself,
> be considered prohibited PRIVATE SHARING** of Competition Data outside the Team."
>
> "The PRIVATE SHARING restriction is intended to prohibit sharing Competition Data, code, or
> competition-specific work product with **other participants, teams, or third parties for
> collaboration or competitive use** outside the registered Team."
>
> Governed by **§2.6.b** (External Data and Tools), not §2.4.b. Constraints that do apply:
> "reasonably accessible to all" and "minimal cost". The Host reserves the right to rule a
> particular service prohibitively costly or unfair.

**This reverses a reading recorded in `README.md` earlier the same day**, which took §2.4.b
("not to transmit... the Competition Data to any party not participating") to forbid exactly
this. That reading was wrong, and the error is instructive: **§2.4.b was read without the
host's clarification, and a rule read without its official interpretation is the same class of
mistake as a competitor's notebook read from its description** (§2e). Corrected in place; see
§4.1 below for what it unblocks.

`IMPROVEMENTS.md` §1.1 raised this worry in week one and blocked method B on it. The worry was
reasonable and the answer is that it does not bind.

### 1.2 What the rules themselves say

Full text in `COMPETITION_RULES.txt`. The clauses that settle live questions:

| § | effect |
|---|---|
| 2.2.a | **5 submissions/day**, 2 final. This is what makes leaderboard-driven iteration unaffordable. |
| 2.6.a | External data must be "publicly available and equally accessible to all Participants... at no cost" → **argues for MRNet, but see §1.3: the host was asked twice and has not answered.** |
| 2.5.a | "input data or pretrained models with an **incompatible license**... you do not need to grant an open source license for that data" → explicit carve-out for research-only terms. |
| 1.6 | Winner licence **CC-BY-NC 4.0** — non-commercial, unusually compatible with research-only external data. |
| 3.6.b | Public sharing on Kaggle is permitted and deemed OSI-licensed → **using the public LLM label tables is fine.** |
| 3.4.b | No hand labelling "of the **validation dataset or test data records**" → `labeling/` works on train reports and is unaffected. |
| 1.5 | Prizes to **10th place** ($5,000) + **$18,000 across three efficiency prizes**. `PLAN.md` §6 is a live second route. |

No forum-disclosure requirement for external data appears anywhere in the rules.

### 1.3 External MRI datasets — ASKED TWICE, NOT ANSWERED `read 2026-08-10`

`discussion/733652` asks directly whether MRNet, fastMRI+, OAI and SKM-TEA count as "publicly
available and equally accessible at no cost", given that all are free but all require a
click-through research-use agreement. A second participant re-asked under the host's own LLM
post, noting most such datasets forbid commercial use while the prizes are money. **The host
replied to the thread but answered only the LLM question and left this one open.**

So: §2.6.a and §2.5.a *argue* MRNet is admissible, a participant (`NNMax`) reads it that way,
and no host has confirmed it. **Downgraded from "very likely admissible" to "unresolved".**
It gates a Phase 2 lever, so ask directly rather than assume — that is one forum post and it
costs nothing.

**STATUS 2026-08-17: STILL UNRESOLVED, STILL UNASKED BY US — seven days.** A third-time-of-asking
post is drafted and **not yet sent** (it needs a Kaggle account action, not a repo change). It
cites 2.6.a against 2.5.a and 1.6's CC-BY-NC winner licence and asks for a yes/no. **`IMPROVEMENTS.md`
§3x flags this as the only lever that could supply image-read severity supervision** — MRNet and
OAI/MOAKS carry expert *image* grades for exactly the meniscus/ACL findings §3p shows the model
cannot extract. **The answer takes days to arrive, so send it before starting anything else.**

### 1.4 What the host confirmed about the labels `discussion/733826, read 2026-08-10`

A participant audited 20 gold studies report-only (240 decisions) and put agreement at
**82.5%** — within noise of `IMPROVEMENTS.md` §2b's independently measured 84.7%. Their error
split is the important part: **FP 25 against FN 17**, i.e. report reading **over-calls**. That
is the direction §2.1 below predicts, measured by someone else.

The host's answers, verbatim where they matter:

| question | host |
|---|---|
| Were labels assigned independently from the images, rather than extracted from the reports? | **"Yes"** |
| If image interpretation and report text disagree, is the image-derived label authoritative? | **"Yes."** |
| Do negative labels mean confirmed-absent, or possibly not-annotated? | **"the finding was annotated as absent"** |
| Are the discrepancies annotation issues? | *"Discrepancies are plausible and expected because clinical reports typically involve one signing radiologist who created it for clinical care, and the image-based labels uses **multiple readers with stricter image-based thresholds**."* |

**That last line is §2.1 below, confirmed by the host in their own words**, and it upgrades
`IMPROVEMENTS.md` §2b from an inference to a documented design property of the dataset.

**And one new fact with design consequences:** *"both knees may occasionally be scanned under
one StudyInstanceUID. For the challenge, each bilateral study or bilateral report was
individually reviewed, and the released report text or DICOM metadata was adjusted as needed to
provide sufficient information for participants to disambiguate."* So bilateral studies exist,
the labels are for **one** knee, and the disambiguation lives in the report text or the DICOM
metadata. `FINDINGS.md` §6.2's laterality work and `IMPROVEMENTS.md` §2b-iii's "some studies
carry more than one report" are the same phenomenon seen from two sides — and neither currently
handles the case where one study contains two knees.

---

## 2. The official label criteria `OFFICIAL OVERVIEW, read 2026-08-10`

**The single most useful external document for this project, and it was not read until day
four.** Every study was labelled independently by two subspecialty MSK radiologists with a
third adjudicating.

> **In each case, ambiguous or borderline findings ("on the fence") were graded as NEGATIVE to
> favour specificity.**

| label | positive requires | explicitly NEGATIVE |
|---|---|---|
| **ACL tear** | high-grade partial or full-thickness; complete discontinuity **or >50% of fibres** disrupted | mild signal change, degeneration, thickening **without discontinuity** |
| **MCL tear** | high-grade partial or complete **acute** tear, disrupted fibres + edema | low-grade sprains; **chronic or remote** stress changes |
| **Medial meniscus** | abnormal signal **definitely contacting the surface on ≥2 images**, or morphologic abnormality (truncated / diminutive / displaced fragment) | intrasubstance degeneration **not reaching the surface** |
| **Lateral meniscus** | same criteria | same |
| **Medial OA** | **≥ ~1 cm** area of **>50% thickness** cartilage loss | lesser cartilage loss |
| **Lateral OA** | same | same |
| **PF OA** | same, patella vs femoral trochlea | same |
| **Effusion** | **moderate or large** fluid distending the joint | trace / small |
| **Synovitis** | inflammation **and thickening** of the synovial lining | — |
| **Baker's cyst** | **moderate or large**, characteristic location | small |
| **Contusion** | marrow edema-like signal from impact, **without a discrete fracture line** | — |
| **Acute fracture** | **acute** cortical break or fracture line | — |

### 2.1 Why this matters more than it looks: gold thresholds SEVERITY, reports report MENTIONS

Eight of the twelve carry an explicit severity or acuity threshold. A radiologist writing
"trace effusion", "mild chondromalacia", "intrasubstance degeneration", "low-grade MCL sprain"
or "chronic ACL changes" is describing something the gold labels **negative** — while any
mention-detector, ours or the public LLMs, scores it positive.

**This is very likely the mechanism behind `IMPROVEMENTS.md` §2b**, which measured careful
report reading at 84.7% agreement with gold and found *"~2/3 of that gap is one-directional
threshold error"*. §2b found the symptom from the data; this document states the cause and the
exact cut-points.

The consequence is about **ranking**, which is what the metric reads. If an extractor scores
"trace effusion" and "large effusion" the same, the sub-threshold cases are false positives
ranked level with true ones and the AUC is spent. Our `rule_extractor.py` grades on
**diagnostic certainty** (definite / probable / possible / negated) — a different axis from
severity. "Definite trace effusion" is certain and below threshold; "possible large effusion"
is uncertain and above it. **Grading the wrong axis is worth testing as a contributor to
§1.3a's calibration failure.** Not yet measured — see §4.2.

### 2.2 One hypothesis this generated, TESTED AND NOT SUPPORTED

"Contusion = edema *without a discrete fracture line*" reads like a mutual exclusion, which
would mean a study with a fracture should tend to be Contusion-negative. Measured on gold-58:

| | P(Contusion=1) |
|---|---:|
| Fracture = 1 (n=18) | **0.556** |
| Fracture = 0 (n=40) | 0.225 |

They **co-occur positively**, because the exclusion is per-lesion, not per-study — a fracture
site commonly has marrow edema elsewhere in the knee. The public labels put the same
conditional at 0.723 against gold's 0.556, which is the right direction for an over-call but
~1.4σ at n=18. **Not resolvable, not actionable.** Recorded because the hypothesis was
plausible and someone will have it again.

### 2.3 Anatomy, for reading the plane/slot design

Three compartments — medial, lateral, patellofemoral — and OA is graded separately in each.
Cruciates and menisci are best evaluated on **sagittal and coronal**; patellofemoral cartilage
on **axial**. Fluid-sensitive sequences (PD/T2, usually fat-suppressed) are where most
abnormalities are detected: *"a meniscal tear appears as abnormally increased signal that
reaches the surface of the meniscus on more than one image."* For this competition
"fluid sensitive" means edema/haemorrhage bright **and fat suppressed in some way** — which is
why `Fluid_Sensitive` and `Fat_Suppression` are identical on all 24,371 series.

---

## 3. Corpus facts from the forum, checked against our own data

| claim | source | our check |
|---|---|---|
| 58 of 4,407 labelled, all-or-nothing | multiple | **confirmed**, `eda_01` |
| `Fluid_Sensitive` ≡ `Fat_Suppression`, all 24,371 series | `nekkon` | **confirmed exactly** |
| Every study has all three **planes** | `nekkon` | **confirmed — 4,407/4,407** |
| "no fallback path needed for a fixed-shape input" | `nekkon` | **FALSE for a 6-slot design.** Only **12.8%** of studies carry all six plane×contrast slots; **87.2% miss at least one**. Axial-FS 100%, Axial non-FS **19.4%**, Coronal non-FS 77.3%. True for planes, false for slots — masking is mandatory. Matches `FINDINGS.md` §3.2. |
| Turkish negates **after** the term (`efüzyon izlenmedi`) | `nekkon` | plausible and unverified by us; would silently invert the 2nd-largest language under a left-only window. Relevant only if we build our own reader — `IMPROVEMENTS.md` §2.7 flagged the directionality gap independently |
| gold is enriched — ACL 41% vs ~20% corpus | `nekkon` | **confirmed**, and consistent with `IMPROVEMENTS.md` §0 |
| 7 languages + 428 unplaceable | `nekkon` | ours is finer: **9 languages** via lingua (`eda_03`), which supersedes a stopword detector |
| Site leakage = **0.053** | `zhukovoleksiy/rsna-metadata-probe` | adopted; see `IMPROVEMENTS.md` §2i-a |
| LLM 0.8780 vs regex 0.8136 on gold | `stevenleehans` | **reproduced exactly** by `extractor/bench_public_labels.py` |
| 25.4% of label cells "not addressed"; Synovitis 83.7% | `stevenleehans` | consistent with our per-label results |

### 3.1 Public leaderboard anchors

**VERIFIED LIVE 2026-08-12 17:22 UTC** (E1), by `kaggle competitions leaderboard -d` over the
full 1,276-team board. Supersedes the 08-10 read, which was wrong in four places. **Re-verify
before any of this is load-bearing — the board moved 0.008 at the prize cutoff in three days.**

| | local | LB | read |
|---|---|--:|---|
| public baseline notebook | OOF 0.632 | 0.664 | 08-10 |
| `pilkwang/rsna-knee-baseline-v1` — **our banked submission** | — | **0.891** | live |
| `aadigupta7686/0-899-let-me-cook` (NOT prvsiyan) | — | **0.899** | live |
| Yash Bishnoi B3 — **writeup only, no public kernel** | OOF 0.8544, gold-58 0.8568 | 0.903 | 08-10 |
| **10th place — the last prize** | — | **0.934** | live |
| **leaderboard top** | — | **0.946** | live |

**The board as it actually is, 2026-08-12 — 1,276 teams (was 908 on 08-09, +40% in three days):**

| plateau | teams | best rank | what it is |
|--:|--:|--:|---|
| 0.900 | 15 | 129 | lightly-tuned `0.899` |
| **0.899** | **183** | 144 | `aadigupta7686` run unmodified — **the free plateau** |
| 0.897 | 22 | 335 | |
| **0.891** | **73** | **400** | the `pilkwang` fork unmodified — **this is us** |
| 0.500 | 102 | 1163 | degenerate/all-constant submissions |

Cumulative: **10** teams ≥ 0.934 · 33 ≥ 0.92 · 74 ≥ 0.91 · 105 ≥ 0.906 · 326 ≥ 0.899 · 472 ≥ 0.891.

Four corrections to the 08-10 read, all in the pessimistic direction:

1. **Top is 0.946, not 0.942.**
2. **10th is 0.934, not 0.926.** The prize cutoff rose **+0.008 in three days**.
3. **Our 0.891 is rank 400 of 1,276, not 230 of 908.** It decayed ~170 places in three days
   without us doing anything. **A banked score is not a banked rank.**
4. **0.891 is not a floor, it is a commodity** — 73 teams hold it exactly, and the *free*
   plateau is 0.899 with 183 teams on it. A submission of someone else's better public kernel
   is worth ~256 places and costs one submission and no work.

**The whole public field lives in 0.891–0.900.** So the real gap is not our 0.043 to 10th; it is
the **~0.035 from the free public plateau to the prize**, and that is the number every idea has to
beat. This *strengthens* the standing read below rather than softening it: every public team
trains on the same report-derived label tables and they all pile up in a 0.01-wide band, while the
top ten sit 0.035 clear of it on something not published.

---

## 4. Literature

### 4.1 CheXpert / VisualCheXbert — the same problem, already studied

Chest X-ray labelling from reports is the closest well-studied analogue: silver-standard labels
from text, an explicit *uncertain* class, and image labels that disagree with report labels.

- **Uncertainty handling is not a free choice.** U-Ones and U-Zeros (map uncertain to 1 or 0)
  "yielded minimal improvement". **U-Ignore — masking uncertain labels in the loss — is
  reported as *ineffective***, notably on borderline cases like "minimal cardiac enlargement".
  The strategies that worked were **U-MultiClass** (uncertain as a third class) and
  **U-SelfTrained** (self-predicted soft labels for uncertain cells).
  → **This is evidence against `PLAN.md` Phase 3's "mask `absent` rather than re-target it".**
  It is also support for what `stevenleehans` did to Synovitis, which is U-SelfTrained by
  another name and is the one label repair that measurably worked.
- **VisualCheXbert** ([paper](https://www.researchgate.net/publication/349546850_VisualCheXbert_Addressing_the_Discrepancy_Between_Radiology_Report_Labels_and_Image_Labels))
  attacks exactly §2.1: it trains the labeller to predict **image** labels rather than report
  labels, lifting weighted F1 **0.55 → 0.73**. The technique needs many image-labelled examples
  and we have **58**, so it cannot be copied directly — but it names the right target, and it
  is the strongest external argument that the report→image threshold gap is worth attacking
  rather than accepting.

### 4.2 Achievable per-label ceilings — MRNet

[MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699)
(Stanford, 1,370 exams, expert **image** reads): abnormality **0.937**, ACL **0.965**,
meniscal **0.847**. Our gold-37: ACL 0.702, meniscus 0.634 / 0.526. The split is diagnostic —
gross-appearance findings work, fine local texture at a specific site does not.

MRNet also supervises three of the labels we are worst at, with expert image reads rather than
report text, and §2.6.a/§2.5.a admit it.

### 4.3 Fingerprint biometrics — a different field with our exact failure mode

Suggested as an analogy and it holds better than expected. Latent-fingerprint recognition is
fine ridge-level texture at specific locations, which is destroyed by global pooling — the same
shape as ours, where gross-appearance labels work (Baker's 0.919, Medial OA 0.913) and
localized fine-texture labels sit at chance (Fracture 0.494, Lateral Meniscus 0.526).

What that field converged on:

- **Minutiae patch embedding** ([MinNet, CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Ozturk_MinNet_Minutia_Patch_Embedding_Network_for_Automated_Latent_Fingerprint_Recognition_CVPRW_2022_paper.pdf)):
  crop patches around detected keypoints, embed each patch, aggregate. Not one embedding of the
  whole image.
- **Dual global + local streams**: a global texture representation *plus* local descriptors,
  fused — neither alone is sufficient.
- **Multi-scale patches** at several sizes around the same keypoint.
- **Patch-level attention** ([DeFraudNet](https://arxiv.org/pdf/2002.08214)): learn which
  patches are discriminative end-to-end rather than pooling uniformly.

**Two unrelated fields — fingerprint biometrics and the RSNA-2024 lumbar-spine winner (§4.4) —
independently answer our failure mode the same way: localize first, then embed crops.** That
convergence is worth more than either result alone.

The direct transfer here is cheap because knee anatomy is stereotyped and the compartments are
already named by the labels themselves: **medial, lateral, patellofemoral, and the
intercondylar notch** for the cruciates. Four anatomical crops fed as additional slots, rather
than one whole-FOV image per plane, is the minimal version — and it targets exactly the labels
that are failing, since meniscal tears live at the tibial plateau margins and the ACL lives in
the notch. No detector is needed for a first pass: `FOV_MM`/`TARGET_MM` already put the volume
in millimetre space, so fixed relative crops are available for free.

### 4.4 Localize-then-classify

The RSNA 2024 Lumbar Spine winner used a two-stage
[localize-then-classify pipeline](https://www.rsna.org/news/2024/november/2024-ai-challenge-winners):
3D localisation, then classification on multiview 2.5D crops. The 0.903 recipe here already
does a mm-based crop, which is the same idea in weaker form.

---

## 5. What this file says we should do that we are not yet doing

1. **§1.1 unblocks our own LLM read.** Not because the public labels are bad — they are good —
   but because §2g measured the public readers as near-duplicates (|r| 0.87–0.95), so they
   ensemble poorly. A genuinely independent good reader is worth more than a sixth near-copy,
   and our rule extractor is independent-but-wrong (§2f). Cost is a few dollars of API calls
   over 4,407 short documents.
2. **§2.1 is unexploited by everyone.** No public post discusses severity thresholding, and the
   official criteria are explicit about it. A reader prompted with the **actual grading
   criteria** — ">50% of fibres", "≥1 cm and >50% thickness", "moderate or large", "acute" —
   should place sub-threshold mentions below threshold ones *in rank order*, which is the only
   thing the metric reads. This is the cheapest untested idea on the board and it is grounded
   in §4.1's VisualCheXbert result.
3. **§4.1 contradicts a planned Phase 3 item.** Masking `absent` is reported ineffective in the
   closest studied analogue. Re-rank it below U-SelfTrained-style imputation.




The first is the one I raised last time, and reading §0 has made me more confident rather than less. §0 retracts "the leaderboard is the instrument." Fine — but the plan then over-corrects into deferring the first submission behind step 5. Those are separate claims. The reason to submit now isn't measurement, it's risk retirement: by your own admission kaggle_03_submit.py has never executed against a real test DICOM. Inference-path failures — the 9-hour cap, no internet, weights-as-a-Dataset, degenerate series, a study missing a plane — are only discoverable by submitting, they're schedule risk rather than score risk, and they don't compete with your local instrument for anything. Related: your CV↔LB conversion is currently an interpolation between two public anchors, and it's load-bearing for the claim that your gap is 0.19 rather than 0.22. That's a big inference resting on someone else's two data points. Step 6 should run beside step 5, not behind it.

The second is the rules risk on external data, which I think you've mis-scoped. It's flagged as gating Phase 2's largest lever, and reading a rules page is a twenty-minute task blocking weeks of work — that alone is wrong-way-round. But the deeper issue is that the same question applies to the public label tables you've already adopted as data/targets.csv. Your own note says not to build on the swap until the rules are read, and fold 0 is built on it. So the compliance question isn't sitting beside your plan, it's sitting underneath your baseline. Read the page today.

Third, and this is the one I'd actually change: the plan has phases and exit criteria but no compute budget. Phase 1 asks for rank-means across seeds, then resolutions, then backbones; Phase 2 adds external pretraining. You're doing that at 3.6 hours per fold on a machine with 17.2 GB that thrashes, plus a 30-hour weekly Kaggle quota you mostly lose to the lottery. Eighteen hours for the remaining four folds is already a meaningful fraction of a week. Meanwhile you've correctly identified that the entire spread from a free download to first place is about 0.05, and that most of what lives in there is ensembling and resolution — i.e. the thing money and GPUs buy. That's the structural problem: your plan's endgame requires the resource you have least of.

Which is why I'd argue the efficiency track deserves promotion from a week 9–10 variant to a co-primary target. Same prize money, three places instead of ten, a far thinner field, and it rewards precisely what you've demonstrated you're good at — measurement discipline, cheap decode, knowing what your pipeline actually costs — instead of what you don't have. Your §6.2 concluded accuracy dominates the efficiency formula, which is true, but it doesn't follow that you should chase accuracy in the crowded track.

One more thing worth engaging honestly. Your central bet is that the unexplained 0.04 between the best visible public solution and the top is label quality or anatomical crops. That's plausible and it's the interesting hypothesis. The boring hypothesis — the top teams simply ensemble more and bigger models over more data — is the one that's usually correct on Kaggle, and the plan doesn't really argue against it. Decide now what evidence would distinguish them, because the answer determines which prize you're actually competing for.

Smallest and most urgent item: fix §2r-B6 first. A paired sigma that isn't reproducible run-to-run because it iterates a set undermines every number your plan is now built on, including the +0.0171. And fix the B2 NameError before you spend thirteen hours discovering it.
