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

**⛔ RE-READ 2026-09-26 — the thread, the rules and the datasets' own pages. It is TWO questions, and we asked the wrong one.**
- **May we TRAIN on it?** §2.6.a is a disjunction — "publicly available and equally accessible… at no cost" **or** "satisfies
  the Reasonableness criteria" of §2.6.b, which opens: *"the use of external data and models is acceptable unless
  specifically prohibited by the Host."* No prohibition has been posted; these corpora are free and not geo-restricted.
  On a plain reading: yes (`Mazzutti`, 18th, in the thread — a participant's reading, still not the host's).
- **May we WIN with it?** §2.8.a: winners publish the weights as a **public Kaggle dataset** and *"represent that you have
  the unrestricted right to grant"* the CC-BY-NC 4.0 licence. §2.6.a says external-data permission "does not limit"
  §2.8. Whether weights trained under a research-use agreement can be published that way is **the dataset owner's
  question about its own terms** — a written answer from them is what that representation would rest on. Ask for a
  clarification of the standard terms, and post it in the thread: a private exception for one team argues against
  "equally accessible".
- **Host:** still silent on both (thread re-read 09-26; the last reply, "any verdict on any of this?", is unanswered).
- **MRNet's released labels are NOT image reads.** Its own page: *"labels were obtained through manual extraction from
  clinical reports."* Only its 120-exam hidden test set was read by three MSK radiologists, and it is not released. So
  MRNet is ~1,250 more **report-labelled** knees (abnormal / ACL / meniscal tear, no medial–lateral split; one site, GE),
  not the image-truth supervision `DATA.md` §12 item 4 wanted. Terms sit behind a Redivis sign-in
  (`stanford.redivis.com/datasets/4a2c-4cpkzrn2c`), unread.
- **OAI** is the one with image grades (MOAKS: radiologists scoring the images). Access: a free NDA account; its portal
  permits use *"for human subjects research for scientific and educational purposes"*. Research protocol (3 T, DESS /
  IW TSE), an OA-risk population (few acute injuries), a very large archive. The OAI CORE Knowledgebase
  (`oaiknowledgebase.com`) runs one-on-one office hours — the channel for the terms question.

- **Stanford AIMI's own page** (`aimi.stanford.edu/datasets-commercial-use`): its datasets are shared "under a license that
  is restricted to non-commercial purposes"; commercial use is **$70,000 per dataset per year**. So if a prize
  competition counts as commercial, MRNet fails §2.6.b's "minimal cost" outright.
- **2026-09-26: the owner-terms questions are DRAFTED, not sent** (Gmail drafts in the owner's account, by the owner's
  choice; the host is not being asked): Stanford AIMI (`aimicenter@stanford.edu`, cc the executive director named on
  the commercial-use page) and the NDA Help Desk (`NDAHelp@mail.nih.gov`) for OAI + MOAKS. Each asks: is prize-competition
  training permitted use; may the weights be published on Kaggle under CC BY-NC 4.0; may the answer be posted in the
  thread.
- **⛔ NDA HELP DESK ANSWERED 2026-09-28 — OAI CLOSED for this competition (owner shared the reply).** It restated the Data Use
  Certification, answering none of the three questions directly: (1) use only for scientific investigation, scholarship, teaching or
  research (DUC clause 2) — silent on whether a cash-prize competition qualifies; (2) data may be shared only with researchers listed
  on a non-expired DUC for the same permission group (clause 1) — silent on publishing trained weights, which §2.8.a requires of
  winners under CC BY-NC 4.0 with a warranty of "unrestricted right"; (3) cloud storage is allowed if it meets the NIH security best
  practices and "your institution['s]" IT policy (clause 7) — an institutional frame. With the host's line (§3.1: "substantial
  institutional approval" may be excluded; KneeCoT ruled out as unequal access), a DUC-gated set fails "equally accessible", and a
  winning entry could not honestly warrant the licence. Also against it: DUC approval + a large download + weeks of work vs Oct 22.
  **Verdict: do not apply; the OA / cartilage / synovitis gap is carried by the in-domain image teacher alone.**

- **fastMRI+ — the best-fitting outside data, read 2026-09-27** (`github.com/microsoft/fastmri-plus`, annotations MIT): the
  whole fastMRI knee set reviewed by radiologists — **1,172 volumes, 974 with findings, 16,167 boxes ON THE IMAGES** (Meniscus
  Tear 5,658; cartilage partial 2,985 / full 615; Joint Effusion 1,311; Bone fracture/contusion 1,060; subchondral edema 986;
  periarticular cysts 864; **ACL low-grade 765 / high-grade 678; MCL low–moderate 285 / high 11** — graded as our criteria are).
  Limits (its README): one radiologist, one view at a time, "not clinical ground truth"; fastMRI knee is coronal-only (from
  memory — check on download). Images need fastMRI's data-sharing agreement (online, "academic purposes") — the owner's to
  sign. By §2.6.b ("acceptable unless specifically prohibited") and the host's stated line (registration fine, institutional
  approval not), it leans allowed; the host has not ruled on it (thread 743416 asks).
- **fastMRI ACCESS GRANTED 2026-09-27** (email from fastmri@med.nyu.edu to the owner, links expire in 90 days; the links are
  signed URLs — never paste them into the repo). Needed for fastMRI+: `knee_singlecoil_train` (~72.7 GB) and
  `knee_singlecoil_val` (~14.9 GB) — 973 + 199 = 1,172 volumes, the set fastMRI+ annotated; their h5 files carry a ready
  reconstructed image, so no k-space work. NOT needed: knee multicoil (~550 GB, same volumes, raw), knee DICOMs (~164 GB, ~10k
  clinical studies, unlabelled), brain, prostate. Nothing downloaded until the owner says so (owner, 09-27). **`labels` downloaded 09-27 at the owner's word: it is the PROSTATE set's PI-RADS labels** (`volume_exam_labels.csv`, `t2_` / `dwi_slice_level_labels.csv`), not knee — the knee labels are fastMRI+'s own `Annotations/knee.csv` + `knee_file_list.csv` on GitHub (MIT), so NYU is needed only for the knee IMAGES. The prostate file was deleted at the owner's word; fastMRI+ `knee.csv` + `knee_file_list.csv` downloaded 09-27 to
  `data/external/fastmri/fastmri_plus/` (git-ignored). **Per KNEE, mapped to our findings** (1,173 reviewed, 974 with a box):
  ACL high-grade (our positive) **101 (9%)** · MCL high-grade **4** (useless for MCL) · meniscus tear **663 (57%)**, side
  unknown · cartilage full-thickness 122 · effusion 142 (ungraded) · fracture / contusion **119** · subchondral edema 196 ·
  periarticular cyst 161. A fastMRI+ teacher can speak to ACL, meniscus (any), fracture / contusion, effusion, OA and cysts. Mac has 705 GB free.
  Before any derived copy goes to Kaggle, read the agreement's redistribution terms (a private Kaggle dataset is still a copy on
  a third party's servers).
- **A radiologist labelling TRAINING scans is allowed** (§2.4.b bans hand labels only on "the validation dataset or test data
  records"), **but Competition Data may not reach anyone outside the team** (§7.b) — they must register, accept the rules and
  JOIN the team (max 5) before seeing an image, and by the **entry / merger deadline 10-15**.

**STATUS 2026-09-26: STILL UNASKED BY US — seven weeks.** Re-raised by the owner ("could we just use MRNet"). The same question now also covers Google's MedGemma (gated, Health AI Developer Foundations terms). It is a public forum post from the owner's Kaggle account — not something to send on their behalf — and the answer takes days; the entry deadline is Oct 15. `DATA.md` §12 item 4.

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

### 3.1b Forum intelligence, read 2026-09-27 — what the top of the board says it does

LB re-pulled 16:31 PDT: top **0.959**, 3rd 0.958, 10th 0.956, 25th 0.954, 50th 0.952, 100th 0.947 — the top 25 within 0.005.
Self-reports, not code (`discussion/740610`, `/742926`, `/743416`):
- **Small CNNs score 0.94+.** Tucker Arrants (22nd): *"Start with 224px / 288px and a simple CNN encoder, like a ResNet34
  or EffNetB0. No attention mechanisms, just use simple pooling operations. You can score in the 0.94+ range with this
  alone"*; Tom Aindow (85th) seconds it (*"0.94+ at 224px with small simple model"*). Their method: validate on
  report-derived labels with a noise threshold (~0.003, seeds to tighten), change one thing at a time, then augmentations,
  then backbones. **We never tested a small CNN**: our backbone A/B was Ortho ViT-L vs CoAtNet at 336–384 px.
- **Disagreeing models ensemble better** — Dread Development (157th, the 0.928 CoAtNet's author).
- **Report label + image model beats both**: a 0.940 competitor's gold-57 read — report-only 0.869, E11 OOF 0.858, fixed
  50/50 rank blend 0.908; ours reproduces it (`DATA.md` §12 item 2): 0.887 / 0.912 → **0.927**.
- **External data:** per a participant's summary (host posts not yet read in full), the host **ruled KneeCoT NOT allowed**
  (a formal institutional agreement = unequal access), and said non-commercial terms alone do not exclude a dataset, while
  "straightforward registration" and "substantial institutional approval" may be treated differently — so MRNet
  (click-through) leans allowed, OAI (Data Use Certification) is uncertain. Tucker Arrants (22nd): external data *"even small
  amounts of it, can substantially increase score in this competition"*; the 12th-place team asks about OAI-derived
  pretrained models. Some of the top may be using it.

### 3.2 The public 0.943 stack, taken apart — read from its code `2026-09-27`

`mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` (Apache 2.0), re-pulled 09-27: 24 cells, unchanged since
our Sub B (23/23 code cells verbatim). It is a **chain of rank blends**, each stage mixing into the last:

| stage (cell) | members | how it joins |
|---|---|---|
| DINOv2 (13–15) | 20 DINOv2-small members, 5 folds × 4 (`pilkwang/rsna-knee-weights`), per-finding member weights (Lat. Meniscus 15, Lat. OA 15, Contusion 5, Med. OA 2.5), a legacy soft-pool blend at 0.5 | the base |
| A5 (16–20) | `mattiaangeli/knee-mri-fold-weights` | 0.52 on studies whose series it can read |
| Rad (21) | RadImageNet ResNet-50 heads: E10 + E13 (0.5 each), then a legacy E11 read at 0.20 | α 0.55 (not Baker's or Fracture), then a **fitted linear calibrator** on the three ranks, their differences and **scan-protocol counts** (series per plane, fat-sat), applied at 0.4 on its gated findings |
| CoAt (22) | **public "raptor" CoAtNets** — maxspan-v5 0.60, native384 v8 0.20, native384dense v10 0.10, maxspan-v5 **slice-reversed** 0.10 — blended at 0.6 with **mattiaangeli's four CoAtNets** (resgated, global96, d4-depth-zone SWA3, repairv1; probability mean) at 0.4 | — |
| final (22) | "transformer" chain (DINOv2 → A5 → Rad) vs the CoAt branch | **CoAt 0.70** by default; ACL 0.75, Fracture 0.75, Lat. OA 0.75, Med. Meniscus 0.80, **Lat. Meniscus 1.00** |

So on a typical finding, roughly: the raptor CoAtNets ~42% (maxspan-v5 — dreaddevelopment's 0.928-alone model family — ~25%
by itself), mattiaangeli's CoAtNets ~28%, and everything else ~30%. Inside that 30%, the Rad blends leave the DINOv2 + A5 base
~0.36 of it, so **the 20 DINOv2 members are ~5% of the final vote** (rank mixes are not linear; an estimate from the weights).
Swapping any one piece of the 30% moves the final little; the stack's result rides on the CoAtNets.

**What makes it work:** (1) its core is the strongest public single model (CoAtNet on dense 64-slice input, `dread_soft`-style
labels, LB 0.928 alone); (2) real diversity around it — three architecture families (CoAtNet, ViT-S, ResNet-50 with medical
pretraining), two inputs (44 vs 64 slices, 336 vs 384 px), several label tables, fold ensembles, slice-reversal TTA; (3) every
join is in rank space, which is what AUC scores.

**Where it could be improved:** (1) **its weights look tuned on the public LB** (0.52, 0.55, 0.500001, per-finding overrides,
a fitted calibrator) with no out-of-fold instrument in sight — some of the 0.943 may not survive the private LB; (2) **no member
uses teacher → student label cleaning**, the thing the board's 0.950 single models credit — a ~0.95 member or its CoAtNets retrained
on cleaned labels is the biggest available upgrade; (3) its 30% "transformer" branch is small, older models — the natural slot for
a stronger ViT; (4) **~8–9 h runtime** (20 DINOv2 members and more), so it is out of the efficiency race by construction.
**For us:** our Ortho is a family it lacks (an orthopaedic-MRI ViT-L), but at 0.916 alone it is weaker than the stack's 0.928 core,
and Sub B showed a weaker member at 1/3 costs (0.901 → −0.007). The measured route in is a stronger member, not a new weight.

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

~~MRNet also supervises three of the labels we are worst at, with expert image reads rather than
report text, and §2.6.a/§2.5.a admit it.~~ **⛔ Corrected 2026-09-26:** only MRNet's hidden 120-exam test set is
an expert image read; its released training labels were extracted from clinical reports (§1.3). The ceilings
above stand; MRNet as a supervision source is report-grade, like ours.

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

### 4.5 How similar problems are solved — research 2026-09-27 (owner's ask)

**Past RSNA MRI/CT winners: find it, crop it, classify it.** RSNA 2024 lumbar (MRI) 1st place: stage 1 predicts the slice and the
(x, y) of each level; stage 2 crops tightly around it at 512 px and classifies with ConvNeXt-small / EfficientNet-V2-S + attention
MIL + bi-LSTM over slices; *"convolution models were better than vits"*, convnext-large < base < small, 7–14 epochs, shift / scale /
rotate augmentation, and randomly jittering the stage-1 coordinates was *"crucial"*. RSNA 2023 abdominal trauma (CT) 1st place:
3D segmentation → per-organ crops → 2.5D CNN (CoaT-medium, V2-S) over 96 slices, max over slices. Both had localisation the knee
competition does not provide; for us it would come from anatomy (TotalSegmentator MR: `total_mr` femur, Apache-2.0;
`appendicular_bones_mr` tibia / fibula / patella, free licence for non-commercial use) or fastMRI+'s boxes (coronal only).
**Why it matters here:** our losses are largest on the SMALL structures (menisci −0.11 for the CNN; Stage B's 384 px helped).
⚠️ This repo once built compartment attention for what was a slice-sampling bug (`DECISIONS.md` §U→D2) — crops need a measured
case first. Multi-week build; it would also add segmentation time to the ~9 h submission budget.
**Label noise, the standard tool:** confident learning (Northcutt et al.; `cleanlab`) — k-fold out-of-fold predictions flag the
training labels most likely wrong, then relabel or down-weight; Noisy Student iterates it. That IS our teacher → student design,
with an off-the-shelf implementation to use rather than write.
**Report text as training signal:** image–report contrastive pretraining (CLIP-style; Decipher-MR — GE HealthCare, 22,594 MRI
studies incl. knee — weights NOT released; MG-3D, fine-grained region–phrase alignment). Expensive, unproven here: low priority.
**Clinical reality check:** a 2025 multi-condition knee system (3,121 studies, 23 conditions, radiologists labelling the IMAGES,
3D slice transformer) reached AUC ≥ 0.85 on 8 of 23 conditions; meta-analyses: meniscus tear AUC 0.912 ± 0.053 over 18 studies,
ACL pooled sensitivity / specificity 0.94 / 0.93. A 0.965 MACRO over our twelve (Synovitis, PF OA included) is beyond published
clinical systems.
**Data, ranked for us:** fastMRI+ (have) · OAI MOAKS (large, image grades; NDA data-use certification) · SKM-TEA (155 knees, 16
pathology boxes + segmentations, qDESS, 900 GB, Stanford AIMI research terms — small) · KneeMRI Rijeka (917 sagittal, ACL grades
FROM REPORTS, CC BY-NC-ND — skip) · MRNet (report labels — skip) · Hugging Face: nothing usable (one gated OAI re-upload, likely
against OAI's terms — skip). **Models:** the public 0.928 CoAtNet (CC0, have) · TotalSegmentator MR (localisation) · RadImageNet ·
OrthoFoundation (have) · Decipher-MR (not public) · MedGemma (gated).
Sources: kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution ·
github.com/Nischaydnk/RSNA-2023-1st-place-solution · github.com/wasserth/TotalSegmentator · docs.cleanlab.ai ·
arxiv.org/html/2509.21249v2 · link.springer.com/article/10.1007/s00330-025-12052-8 · pmc.ncbi.nlm.nih.gov/articles/PMC12186967 ·
github.com/StanfordMIMI/skm-tea.

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
