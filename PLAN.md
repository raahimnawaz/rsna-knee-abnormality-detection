# RSNA Knee Abnormality Detection — Plan

**Metric:** macro-averaged AUROC over 12 labels
**Timeline:** started Jul 30 2026 · entry/merger deadline **Oct 15** · final submission **Oct 22 2026**
→ **70 days left as of 2026-08-13.**
**Prizes:** main track 10 places ($9k down to $5k); **efficiency track 3 places ($7k / $6k / $5k)**
**Field, live 2026-08-13:** **1,373 teams**; top **0.946**, 10th/prize **0.935**, ours **0.899**.
**Constraints:** Kaggle notebook, ≤9h, no internet, `submission.csv`; **~15 submissions/week against
a 30 h GPU quota** (§3e). Winners open-source code **and weights**, publish to the forum, record a video.


---

> **How to read this file.** **§9a is the current plan** and §9b–§9e are its items. Everything from §6 down is
> reference — efficiency maths, the disk inventory, the task, the architecture notes, the risk table.
> **Section numbers are deliberately non-sequential**: they are cited across the repo and renumbering
> would break every reference. The 11-week schedule, the 08-07 leaderboard analysis, the E-series and
> F2's crop provenance were removed 2026-08-13; every closed route survives in `README.md`'s table.

---

## 9a. THE CURRENT PLAN — F-series `the live plan; everything else in this file is reference`
**State: 0.899 banked (submission 55465252). Top 0.946, 10th 0.935.** ~~The free public ceiling is
reached — everything from here must be something the field does not already have.~~

> **⛔ THE STRUCK SENTENCE IS FALSE — `IMPROVEMENTS.md` §3l-1, 2026-08-13 pm.** It was surveyed
> 08-12, true then, and dead within a day. The field now runs a **three-family** vote from
> **published weights** (DINOv2 20-member + DINOv3 ViT-S/16 5-fold + RadImageNet ResNet-50 5-fold
> + tonylica); we run **pilkwang alone**. **F6 below is the consequence and it outranks F1–F5.**
> Also from §3l: the instrument that ranked this table (`score_oof.py`) measures a report-derived
> teacher rather than the board's image reads, so **every ordering here was priced in the wrong
> currency** — see §3l's routing rule and `fusion/score_gold.py`.
>
> **⛔ AND §3p RETIRES THE THESIS THIS WHOLE TABLE WAS BUILT ON.** "Every public team shares a
> label ceiling" is **false**: the free tables score **0.898 gold → LB 0.944**, the board top is
> **0.946**, and our model reproduces its own labels at only **0.849**. **Model capacity is the
> binding constraint, not labels.** F4 falls; F6 and Workstream C rise.

**Ordered by expected value, biggest first — this ordering changed on 08-13 pm.** Costs are real:
a submission is **~2 h of the 30 h weekly quota** (§3e), not the 74 s this plan believed twice.

| # | step | cost | status `2026-08-13 pm` |
|---|---|---|---|
| **F6** | **Blend the free diverse arms** — DINOv2 ViT-B/14@336, DINOv3 ViT-S/16, RadImageNet R50 | weights down, `ft_b` scored; **1 submission** | **GO — §3o.** `ft_b` reproduces (fold-resolved OOF **0.8522** vs pilkwang **0.8516** on the same 47 gold), Spearman **0.632** — the port's diversity with none of its weakness. **Equal-weight rank-mean +0.0284, 100% of draws**, ceiling; honest ~+0.015–0.020. **Next: the DINOv3 arm through the same machinery, then submit with F1.** §3b binds — do NOT hunt a better weight on 47 studies. |
| ~~**F1**~~ | ~~Site-prevalence prior at K=10, w=0.10~~ | — | **⛔ DEAD, §3u.** The +0.0023 was fitted and scored on the SAME label source. Built in the form a submission can run, it is **−0.0057, positive in 0/2,000 draws**. The gain splits by *prior source*, not by matching — `lixin_gpt56` helps, `targets.csv` (what we own) hurts — so §3f measured one label table's per-site prevalence, not site prevalence. Best honest cell **+0.0008** against ±0.005 precision. **Do not re-open.** |
| **F3** | **Metadata conditioning** — screen on `data/features_*` | hours, CPU/MPS | **Still unblocked, no longer first.** No GPU, no quota, no pixel path. Sex and field strength are the arms that matter; manufacturer is largely captured by F1's site prior (§3f). **Note §3l-3 re-ranks its targets** — Synovitis and Lateral OA are far worse on gold than §3f believed. See §9c. |
| **F5** | Efficiency track | TBD | **PROMOTED.** $18,000 over three places, thinner field, untouched since §2t-4 — and §3k left behind the capability that makes it measurable: the fork's members run locally with verified fidelity, so member-count, window-count and resolution trade-offs price **offline** instead of at 2 h a submission. `sadamtorres` publishes the break-even: **+0.044 AUC per extra hour**, and their @224 arm is the efficiency play at 0.866/3 h. |
| **F4** | Severity-thresholded label read | — | **⛔ DEPRIORITISED — §3p, AND RE-CONFIRMED AGAINST A FRESH CHALLENGE — §3x `08-17`.** It targets label quality, and **there is no binding label ceiling**: the free public tables score **0.898 gold → LB 0.938** (§3v's corrected offset; §3p said 0.944) against a board top of **0.951**, while our model reproduces its own labels at only **0.849**. **Revisit only if the blend reaches ~0.90 gold.**<br>**§3x tried to revive this and failed, so do not re-run the argument.** The case was: §2b's 84.7% agreement is ~⅔ one-directional, 8/12 criteria carry severity cuts, and the **host confirmed** *"stricter image-based thresholds"*. **§3p's per-label column refutes it** — the model already **beats** the mention labels on every volumetric-threshold label (effusion **+0.141**, medial OA +0.075, Baker's +0.037), and **ACL's mention teacher scores 0.995 against gold**. Better labels help least where the model is bottlenecked. **Only Synovitis survives** (teacher 0.848 *and* blend 0.708, the sole label bad on both axes; ceiling +0.012 macro). |
| **F2** | Anatomical crops at the lateral posterior horn | — | **⛔ CLOSED AS A REPLACEMENT — BOTH INSTRUMENTS (§3k + §3m).** The §3l-4 thread was the last live objection and it ran: pre-registered gold-47 re-read gives **C − A = −0.0038** against report-OOF's −0.0032. **Its own target moved −0.0055.**<br>**⚠️ SCOPE, per §3x `08-17`: what died is crop-INSTEAD-OF-context.** Both §3k and §3m swapped the crop *in place of* the full view on frozen members. §3k's own mechanism — *"the periphery carries ranking signal even for joint-centre findings"*, Spearman(130,90) **0.9303**, the crop reorders and reorders worse — **predicts that full-view PLUS crop as separate slots behaves differently, and that has never been run.** That is the new mechanism the row asks for. §2l's canonical **132/132** axes place the boxes without a detector. **Untested and unpriced — pre-register before building.** |


### 9e. F6 — the free diverse arms `NEW 2026-08-13 pm, IMPROVEMENTS.md §3l-1`

**The argument in three lines.** §2y closed our port as an ensemble member because it was
**weak** (0.7323 vs 0.8434), not because it was redundant — its rank correlation with the fork
was **0.639**, which is real diversity. pilkwang's own 20 members are **5 folds × 4 seeds of ONE
config**, so the ensemble we are running has none. Strong, diverse, independently-trained arms
are now **published weights**, so the member we could not manufacture is a download.

**On disk as of 2026-08-13** (`data/external/`, ~2.1 GB total):

| dir | contents | what it is | author's claim |
|---|---|---|--:|
| `ft_b_dinov2_vitb14_336/` | `ft_f{0..4}_best.pt` | DINOv2 **ViT-B/14 @336**, 5-fold, **backbone+head in each ckpt** (`timm` instantiated `pretrained=False`) | **0.883 solo** |
| `dinov3_vits16_folds/` | `m_f{0..4}.pt` + `timm-1.0.22` wheel | **DINOv3 ViT-S/16**, 5-fold, fine-tuned on this competition | — |
| `radimagenet_r50_heads/` | `rad_head_f{0..4}.pt` + manifest | **RadImageNet ResNet-50 HEADS ONLY** — the backbone is a separate dataset | — |

**Claims are the authors', read from kernel source on 08-13, and are UNVERIFIED here.** The first
job is to reproduce one of them locally, not to blend on faith.

#### What is actually inside the checkpoints `INSPECTED 2026-08-13 pm`

| arm | contents | notes that change the plan |
|---|---|---|
| **ft_b** | `backbone` + `head` OrderedDicts, and **`oof_macro: 0.7222`** | Their own OOF is on **the same scale as our 0.7229 baseline** (§2j) — an unusually direct comparison, and the first time another team's local number has been legible to us at all. Self-contained: `timm` is built `pretrained=False`. |
| **dinov3** | 23.5 M params incl. `enc.vit.*`; `cfg = {backbone: vit_small_patch16_dinov3.lvd1689m, img: 336, pool: 'xcodex', cond: 'token', n_sites: 109}`; `fold` per file | **The encoder is embedded**, so DINOv3's gated licence is not a blocker for *inference*. **⛔ CORRECTION 2026-08-13 late: this arm does NOT do site conditioning.** The state dict's only non-ViT tensor is `enc.tok.weight` at **(7, 384)** — that is 6 slot types + padding, matching `readout.pres_emb.weight (7, 64)`. `n_sites: 109` sits in the config with **no parameter behind it**. `cond:'token'` conditions on **SLOT TYPE**, not site. **F3 is still unbuilt by anyone.** Read the weights, not the config — the config carries dead fields. |
| **radimagenet** | heads only, 3.2 M params, `FoundationQueryHead`, `best_val 0.8167`, frozen official R50 encoder | **Two blockers.** (1) The encoder is a *separate* dataset (`marwanmath/resnet-50-radimagenet-marwan`) and is not pulled. (2) **It ships `fold_sha256` but NOT `folds_v1.csv`**, so its fold assignment cannot be recovered — and without it there is no honest OOF read for this arm. Also note it is **frozen-encoder + head**, the configuration §2e measured as the weak one. **Lowest priority of the three.** |

**Consequence for the order of work: start with `ft_b`.** It is self-contained, it is the strongest
claim (0.883 solo), and it ships a local number on our own scale to reproduce against.

#### All four families, inspected and priced by build cost `2026-08-13 pm`

Definitive attachment list taken from `kernel-metadata.json` (`kaggle kernels pull -m`), not from
the notebook body — that is how `tonylica/rsna2026-models` was finally located.

| arm | build cost | status |
|---|---|---|
| **`ft_b`** | done | **§3o. OOF 0.8522, blend +0.0284.** |
| **`tonylica`** (`rsna2026-models`, 1.3 GB) | **near-free** | **4 folds load STRICT into our existing `pilkwang_model.build_model(pool='cls_mean_focal', prior=True)`** — 233/234 keys already matched, the only extra being `head.slot_prior`. **Same six slots as pilkwang**; pixel path is `pilkwang_pixels` at **`img 224`, `crop_mm 160`, `n_slice 9`**. |
| **RadImageNet** | moderate | encoder now pulled (`ResNet50.pt`, 90 MB). Frozen encoder + head — §2e's weak configuration. Folds via `fold_recover.py`. |
| **DINOv3** | **heavy** | Slot-based like pilkwang, but a much larger transcription: `Net` + `Readout` + `CodexResidualPool` + `_GatedDelta` + `_pad_kv`/`_seg_mean_max`, and a custom encoder doing **token-level SLOT conditioning** (`cond='token'`; *not* site — see §9e's correction). Several hours, and no fingerprint to check it against. |

**⚠️ `tonylica` IS A WEAK ARM AND §2y APPLIES.** Its shipped per-fold gold (`annot`) is
**0.7992 / 0.8068 / 0.7339 / 0.7070, mean ≈ 0.762**, against pilkwang's per-member mean of
**0.8375**. §2y closed our own port at 0.7323 for being too weak *despite* diversity of 0.639.
**Score it before blending it, and be willing to drop it** — the public notebooks include it, but
they never measured whether it earns its slot.

**Its gate:** an honest gold read should land near **0.76** for single members. A large miss means
the pooling is wrong — `cls_mean_focal` computes `topk(k=n/8).mean` as its third part, and the
shape match alone does not prove tonylica used the same third part rather than a plain `amax`.

#### The fold problem, and the way through it `2026-08-13 pm`

**Neither new arm ships a fold split or an OOF.** `folds_v1.csv` is not published anywhere —
searched; the heads dataset carries only its `fold_sha256`, and the public notebook merely
*verifies* that hash, never constructs the split. The DINOv3 checkpoints tag themselves `fold: k`
but ship no study→fold table either.

**This blocks honest local scoring, not the submission.** For inference on the hidden test set the
split is irrelevant — all five folds are averaged. It matters only because an all-member read on
studies a member trained on is biased optimistic, which is the §3k caveat, and it is the whole
reason our numbers are trustworthy.

**§3i's recovery does not transfer as written** — it matched fold-means against pilkwang's
*shipped OOF*, and there is no shipped OOF here. **The replacement uses the same physics in
reverse: for a given study, the fold that HELD IT OUT is the one whose prediction is the outlier**,
because the other four memorised it. Per study, rank the five fold-models by agreement with the
other four and take the minimum. Cheap to test, and it has a free correctness check — the recovered
partition must come out ~20% per fold and χ²-flat against study order, exactly the check §3i
already passed at p ≈ 0.7.

**If recovery fails, the honest fallback is to price these arms on gold-58 only** — 47 studies
with coverage, no fold needed if we accept an all-member read and state that it is biased *toward*
"the new arm adds nothing", i.e. the safe direction. That is a weaker read but not a dishonest one,
and it is still the target currency (§3l-2).

**The rules this arm has to obey**, all of them already paid for elsewhere in this repo:

1. **Rank-mean, never probability-mean.** AUC reads order; averaging sigmoids lets the most
   confident member dominate. Independently stated by `sadamtorres` and consistent with §3d.
2. **Fold-first.** Average seeds/checkpoints *within* a fold, rank that, then average the five
   fold ranks — otherwise correlated members from one fold vote as if independent.
3. **Run each member on its own held-out fold only.** A memorised study survives a domain shift
   that a novel one does not, so member-on-train-study readings are biased optimistic.
4. **Score it on `fusion/score_gold.py`, not `score_oof.py`** (§3l-2). A new *family* is exactly
   the case where the report instrument understates. Gold + 0.046 → LB, ±0.038.
5. **Pre-register the blend rule before any AUC exists** (§3k). Equal-weight rank-mean over
   families is the default; anything fitted is a §3b selection risk.

**Expected size.** `sadamtorres` measured **+0.006** from rank-meaning two arms correlated at
**0.901**. These families differ in backbone, pretraining corpus, resolution and label source, so
they should correlate well below that. This is a ~0.01–0.02 question, which is **above** gold-58's
±0.038 noise floor only in the sense that the *sign* is readable — so gold decides go/no-go and
the board settles the size.

**Cost.** Downloads are done. Local scoring is CPU/MPS hours. Then **one submission (~2 h)**.

**The order of work.**

1. ~~Reproduce **one** family's own OOF locally~~ — **DONE for `ft_b`, partially (§3n).**
   `fusion/ft_b_model.py --check` loads all five strict; `fusion/ft_b_pixels.py --gate` scores
   **0.9015** on 47 gold against a pre-registered bar of 0.837. **PASS, but the gate certifies
   nothing** — an all-member gold read inflates by **+0.1474** on our validated pilkwang path
   (0.8516 honest OOF → 0.9990 all-20), and `ft_b` landing at 0.9015 is either lighter
   memorisation or a degraded pixel path. **Next: the fold-resolved read** — recover `ft_b`'s
   folds by the outlier method below, then compare honest OOF against pilkwang's **0.8516** on
   the same 47 studies. `ft_b` is "loads and runs", NOT "reproduced", until that exists.
2. Score each family alone on gold-58 → four numbers on one scale for the first time.
3. Pre-register the blend, then read the blend on gold-58, paired against pilkwang alone.
4. Only then spend the submission. ~~batched with F1's site prior~~ — **F1 is dead (§3u)**; the
   prior does not survive being built in shippable form.

**What kills this route.** If a family cannot be reproduced locally at fidelity comparable to
§3h's 7e-06 fingerprint match, it cannot be scored honestly, and blending it becomes a leap of
faith priced at 2 h a try. Reproduce first.

### 9b. F2 — dead in its cheap form, damaged in its classic one

**⛔ `IMPROVEMENTS.md` §3k. A (crop 130) 0.8457 · B (crop 90) 0.8340, −0.0117 at 3.9σ · C
(130+90 per target) 0.8425, −0.0031.** n = 592 non-gold, all 20 members, OOF, paired. **Lateral
Meniscus — the label it was aimed at — fell 0.0071.** Spearman(130, 90) = 0.9303, so the crop
really does reorder; it reorders *worse*. **The founding assumption is disproved: the discarded
field of view is NOT irrelevant**, which damages F2-classic too — training removes the domain
shift, not the lost context.

**Do not build a saliency- or detector-guided crop.** Circular for a *miss*: the model is not
looking at the lateral posterior horn, so its own attention will not point there. And the
competition ships no bounding boxes, only study-level labels. Geometry places the region instead
— §2l's in-plane axes are canonical **132/132**.

**⛔ THE LAST OPEN THREAD RAN AND CLOSED IT — §3m, 2026-08-13 pm.** §3l-4 objected that §3k was
scored entirely through the report instrument and off the gold set, against a target whose true
headroom is 0.642 rather than 0.767. Pre-registered, then run on the 47 gold studies with NIfTI
coverage: **C − A = −0.0038** against report-OOF's −0.0032, **B − A = −0.0160** against −0.0117.
Same sign, same size, in the currency the board actually pays in. **Lateral Meniscus — the label
the whole route was shaped for — reads 0.6923 and the crop moved it −0.0055.**

**F2 has no remaining threads.** Reopening needs a new *mechanism*, not a new instrument.

### 9c. F3 — the model is never told anything but pixels


`Model.forward(self, imgs, mask, img_size=None)`. That is the entire input: slot images plus a
mask of which slots exist. Never passed: **sex, age, field strength, manufacturer, laterality,
slice direction.** The competition metadata (`Fluid_Sensitive`, `Fat_Suppression`, plane) is used
*only to route series into slots*, never as a model input — by pilkwang, by prvsiyan, by anyone
public.

What the model can and cannot recover on its own, measured in §3f:

* **laterality — inferable** from pixels (a left knee is a mirrored right knee), and the L/R
  disparity is **−0.0014**, i.e. nil. It has already solved this. *Nothing to add.*
* **scanner — inferable** from texture and FOV convention, and it evidently learns the associated
  case mix; that is why harmonising *costs* 0.013–0.032.
* **sex — barely inferable from a knee, and never given.** ACL runs **M 0.865 / F 0.820** (2.3σ),
  Medial OA M 0.852 / F 0.889 (3.1σ). This is the real gap.
* **field strength — never given**, and Lateral Meniscus is **0.748 at 1.5 T vs 0.801 at 3 T**.
  The model cannot be told "this is a low-SNR acquisition, widen your prior."

**Screen it cheaply before spending GPU.** `data/features_*` are frozen DINOv2 embeddings already
on disk. Fit a head on `[features ‖ metadata]` against one on features alone. If conditioning
gives nothing there, it will give nothing after a fine-tune, and F3 dies for the cost of an
afternoon on the CPU. **Caveat (§3a): a per-study covariate cannot be bolted on post-hoc as a
monotone transform — AUC will not move. It is a training-time feature, or a group-conditional
prior like F1's, and it must be tested as one.**
>
> **Corrections to what this file and `REFERENCE.md` assert about the field**, surveyed live
> 2026-08-12: `0.899 let me cook` is `aadigupta7686`, not `prvsiyan`; and **no public
> `Yash Bishnoi` / B3 kernel exists** — that 0.903 is a writeup, so reproducing it is a training
> job and not a download. Every LB figure in this repo was read 08-10. Re-verify before use.
>
> **The compute budget doubles: a fold is ~1.6 h, not 3.6** (§2v — the laptop was asleep for ~2 h
> of the run that produced 3.6). So ~40 five-fold experiments remain, not ~20. **Always launch
> under `caffeinate -i`.**


---

## 6. Efficiency track

### 6.1 The formula, verified from the competition page

    Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400        [minimize]

`Benchmark` = `sample_submission.csv` score, `maxAUC` = best private-LB AUC.

Note `Benchmark − maxAUC` is **negative** (≈ 0.5 − 0.85 = −0.35), so the first term rewards higher
AUC. The score is not normalized and will come out negative; only the ordering matters. It is an
affine transform of the normalized form `(maxAUC − AUC)/(maxAUC − Benchmark) + t/32400`, so
**both rank identically and both give the same exchange rate.**

`sample_submission.csv` is all-0.5 → every prediction ties → **Benchmark = 0.500 exactly.**
Assuming `maxAUC ≈ 0.85`:

- 1 second of runtime → `1/32400` = **3.09e-5**
- 0.001 macro-AUC → `0.001/0.35` = **2.86e-3**

> **Exchange rate: 0.001 macro-AUC ≈ 93 seconds.** (0.01 AUC ≈ 15.4 min.)
> Robust to `maxAUC`: 108 s per 0.001 at 0.80, 81 s per 0.001 at 0.90.

> **Would a C++ rewrite be worth it at, say, 0.96? No — and §6.1's own exchange rate is why.
> `ANSWERED 2026-08-10`** At 93 seconds per 0.001 AUC, a rewrite has to save *minutes* to be
> worth anything measurable. It cannot: the runtime is dominated by DICOM decode and GPU
> forward passes, and **both are already native code** — the decoder is C, the forward is CUDA.
> Python contributes orchestration only, plausibly under 5% of wall time, so a full port of a
> ~12-minute run saves well under a minute ≈ **under 0.001 AUC equivalent**, against weeks of
> work and a much harder §2.8 winner's-obligation deliverable (training code, inference code,
> weights, and a reproducible environment).
>
> The levers that *are* worth it are all above the language line and are already ranked in §6.3:
> slice count, backbone size, batching, and the decode backend — though note §3.1 measured the
> corpus as **200/200 Explicit VR Little Endian at 3.1 ms/slice**, so even the dicomsdl lever is
> gone. **The efficiency track rewards a strong model with a tuned I/O path, not a rewritten
> one**, and §6.2 below is the argument.

### 6.2 The strategic consequence — accuracy dominates here
**The test set is only ~1,300 studies.** At 3 series × 16 slices that's ~62,000 slices — a lean
pipeline should finish in **5–15 minutes**, i.e. a runtime term of only 0.009–0.028. Even a heavy
30-minute ensemble only costs 0.056.

The entire runtime range you'd plausibly operate in (5 min → 45 min) is worth about **0.074
efficiency units ≈ 0.026 AUC**. Typical ensemble gains are ~0.01 AUC ≈ 0.029 units. So:

- Lean beats heavy, **but only by a modest margin** — this is not a "shrink the model to nothing"
  competition.
- **Do not sacrifice meaningful accuracy for speed.** Dropping from 0.845 to 0.800 costs 0.129
  units; you could never recover that by getting faster, since the whole runtime budget is ~1.0
  and realistic runtimes cost <0.06.
- With only 3 efficiency prizes and a far less contested track, a **strong** model with a
  well-optimized I/O path is the play.

Worked examples (`maxAUC` = 0.85; showing the normalized form for readability — same ranking):

| Submission | AUC | Runtime | Acc. term | Runtime term | **Eff (norm.)** |
|---|---|---|---|---|---|
| Full ensemble + TTA | 0.850 | 5.0 h | 0.000 | 0.556 | 0.556 |
| Ensemble, tuned I/O | 0.850 | 30 min | 0.000 | 0.056 | 0.056 |
| **Single model, tuned I/O** | **0.845** | **12 min** | 0.014 | 0.022 | **0.037** |
| Single model, trimmed hard | 0.835 | 6 min | 0.043 | 0.011 | 0.054 |
| Tiny/fast, accuracy sacrificed | 0.800 | 3 min | 0.143 | 0.006 | 0.149 |

### 6.3 What to optimize, ranked
1. ~~**DICOM decode throughput — free.**~~ ~~**Demoted 2026-08-07.**~~ **Re-ranked 2026-08-08, and
   the demotion was half right.** The *decoder* is still irrelevant: there is no JPEG 2000, every
   file is Explicit VR Little Endian at 3.1 ms/slice, and `dicomsdl`/`pylibjpeg`/GDCM buy nothing.
   But "3.1 ms/slice ⇒ ~36 min single-threaded ⇒ GPU-bound" did not survive contact with the mount.
   That benchmark timed **decode of files already opened**, n=6, inside a 200-study audit. What the
   cache build actually pays is **per-file open latency** on a 570 GB network mount — ~19 ms each,
   ~700k of them, i.e. hours before a single kernel launches. The first 224 shard ran 9 h against a
   2.7 h estimate.

   **So the cache build is I/O-LATENCY-bound, not GPU-bound and not decode-bound.** The lever is
   concurrency of opens (oversubscribed workers, prefetch depth) and file *count* — which is what
   makes slice subsampling and series pruning pay twice. Resolution and slice count still dominate
   the GPU side of the submission notebook, where the test set is ~1,300 studies and the mount is
   read once. `kaggle_02`'s PROBE at 25/100/400 series prints the implied hours for the shard and
   is the instrument that settles this properly — the ~19 ms figure is an inference from failed
   runs, not a recorded measurement (see `FINDINGS.md` §6.1).
2. **Series pruning** using the provided metadata: keep sagittal + coronal + axial fluid-sensitive,
   drop localizers and non-FS duplicates.
3. **Slice subsampling** (16 vs 32) — measure the OOF cost first.
4. **fp16 + `channels_last`**, then ONNX Runtime / TensorRT. Zero AUC cost.
5. **Drop TTA before dropping models** — TTA typically buys ~0.002 for a 2–4× runtime multiplier.
   (And hflip TTA is invalid here anyway, §3.2.)
6. **Only then** shrink backbone / ensemble.

**Accept/reject rule:** take a change only if `Δt > 93000 × ΔAUC` seconds. Instrument the notebook
to log decode / preprocess / GPU / write time separately, then walk the list measuring
`(ΔAUC_oof, Δt)` for each step.

---


---

## 9d. WHAT IS ON DISK AFTER 2026-08-13, and how to pick this up cold

`data/` is gitignored, so a fresh clone has none of this. Regenerating is cheap except where noted.

| path | size | what it is | how to rebuild |
|---|--:|---|---|
| `data/external/pilkwang_weights/` | 1.7 GB | the fork's **20 checkpoints, CC0**, plus `manifest.json` (per-member fold / holdout / annot / fingerprint), `oof.npz` (their shipped OOF, the gate's target) and `merge_gain.npz` (their second arm, closed §3h-2) | `kaggle datasets download pilkwang/rsna-knee-weights --unzip`, ~5 min |
| `data/slots_pilkwang.csv` | 1.5 MB | their six-slot assignment for all 4,407 studies, §3h-1 | `python fusion/slot_assign_pilkwang.py`, seconds |
| `data/tiles336lr/` | 5.5 GB | protocol tiles rebuilt under **`SAGITTAL_LR=1`**, clearing the standing hazard. **Built but never consumed** — F2-classic is what needed it | `SAGITTAL_LR=1 python pipeline/slot_cache.py --slots protocol --out data/tiles336lr`, ~15 min |
| `data/_crop_ab_n600.npz` | | the §3k A/B: all 20 members × 592 studies × both crops, plus the recovered folds | 5.2 h — **do not re-run to check a number** |
| `data/_gate_n60*.npz` | | the §3i gate and its K16 ablation | ~20 min each |
| `data/external/ft_b_dinov2_vitb14_336/` | 1.6 GB | **F6 arm 1** — `ft_f{0..4}_best.pt`, DINOv2 ViT-B/14@336, **backbone+head in each ckpt**, author claims 0.883 solo | `kaggle datasets download sadamtorres/rsna-ft-b --unzip`, ~3 min |
| `data/external/dinov3_vits16_folds/` | 451 MB | **F6 arm 2** — `m_f{0..4}.pt`, DINOv3 ViT-S/16 5-fold, plus a `timm-1.0.22` wheel | `kaggle datasets download mattiaangeli/knee-mri-fold-weights --unzip`, ~1 min |
| `data/external/radimagenet_r50_heads/` | 61 MB | **F6 arm 3 — HEADS ONLY.** `rad_head_f{0..4}.pt` + manifest; the ResNet-50 backbone is a **separate** dataset and is not yet pulled | `kaggle datasets download mattiaangeli/rsna-knee-radimagenet-foldsv1-heads --unzip`, seconds |

**The four files that are the session's actual output**, in dependency order:

1. `fusion/slot_assign_pilkwang.py` — their slot logic, transcribed. Needed by everything below.
2. `fusion/pilkwang_model.py` — their architecture. **`--check` fingerprints all 20 in ~2 min and
   is the first thing to run if anything ever looks wrong**, because it separates a model problem
   from a pixel problem.
3. `fusion/pilkwang_pixels.py` — their `read_slot` against our NIfTI, with `crop_mm` and
   `centre_mm` as parameters.
4. `fusion/pilkwang_gate.py` / `crop_neutrality_test.py` / `crop_ab.py` — the three measurements.

**What this capability is good for now that the crop is dead.** Any *inference-side* question
about the frozen members can be answered locally, paired, in hours instead of at 2 h of quota per
point: window counts, slice counts, resolution, pooling rules, member subsets. That is the
**efficiency track's** entire question (F5), and it is the reason F5 is promoted above.

**What it is NOT good for.** Anything needing per-study precision — the residual is 0.0168 per
prediction (§3i) and the mechanism is slice order, roughly a third of series appearing *permuted*
rather than merely reversed. Fixing that needs **one Kaggle CPU kernel** exporting the per-series
sort permutation by `k = p · (r_x × r_y)`; header-only reads, a few MB out, and CPU kernels do not
draw on the 30 h GPU quota. Deferred, not cancelled (§3j).

**The one process discipline that paid tonight.** `crop_ab.py` fixed its per-target pooling rule
*before* any AUC existed. The rule did not rescue the arm — and choosing it afterwards would have
produced a **+0.0010** "gain" that was pure artefact (§3k). Pre-register the decision rule, in the
file, in writing, every time.

---

## 0. What the data description actually tells us

Read carefully, two sentences reframe the whole competition:

> "**Only a small subset of training studies carry per-condition labels.** We also provide the
> original text of the radiology report **from which you may wish to derive the labels for the
> remaining studies.**"

and

> "Report — the free-text radiology report. **May be in any of several languages**, depending on
> the reporting institution."

### 0.1 Confirmed: no reports at test time
`train.csv` has `Report`; `test.csv` has **only** `StudyInstanceUID`. So text is a
**training-time-only** signal. The image model must stand alone at inference. ✅

### 0.2 This is a weakly-supervised competition wearing a vision competition's clothes
The organizers are telling you outright that the labeled set is small and that **you are expected
to manufacture the rest of your training labels from reports.** That means:

> **The report→label extractor is the critical path, not a phase-4 enhancement.**
> Your effective training-set size — and therefore your ceiling — is set by how well you pseudo-label.

Two teams with identical vision stacks will be separated almost entirely by label quality and
label *quantity*. This is where the competition is won.

### 0.3 The reports are multilingual, which is the real difficulty
"Any of several languages" kills the standard playbook (English regex + NegEx, or a
PubMedBERT/CheXbert-style English clinical model). You need one of:

- **A multilingual instruction-tuned LLM run offline on your own machine** over ~all training
  reports, prompted to emit structured JSON per study. This is the strongest option and it is
  entirely training-side, so the no-internet rule doesn't apply.
- Language-ID → per-language rule sets. Brittle, but a useful cross-check on the LLM.
- A multilingual encoder (XLM-R / mBERT / multilingual-e5) fine-tuned on the small labeled subset,
  used to propagate labels to the unlabeled reports. Cheap, and a good *ensemble partner* to the
  LLM — disagreement between the two flags studies worth manual review.

Do at least two of these and reconcile. Radiology reports are templated, so agreement will be
high and the disagreements are exactly the informative cases.

### 0.4 Series metadata is handed to you — free
`train_series.csv` / `test_series.csv` provide **`Fluid_Sensitive`, `Fat_Suppression`,
`Anatomical_Plane`** for every series, at both train *and* test time. This deletes what would
normally be a week of DICOM-tag reverse-engineering. Use these columns directly as the series
router and as embedding inputs to the pooling layer. (Spot-check them against
`ImageOrientationPatient` on a sample, but expect them to be correct.)

### 0.5 Scale
819,640 files, 569.76 GB. At a median of 30 slices/series that's roughly **27,000 series**;
at ~4–6 series/study, roughly **5,000–7,000 training studies** (confirm from `train.csv` row
count). **Test set is ~1,300 studies** — small, which matters a lot for §7.

---

## 1. Data logistics — solve this before anything else

**You cannot host this dataset locally.** 570 GB zip + 570 GB extracted ≈ 1.14 TB; this machine
has 56.6 GB free on C: and 293 GB on D:. The browser "Download All" will fail. Plan:

| Need | Where |
|---|---|
| `train.csv` (reports + labels), `train_series.csv` | **Local.** Tiny. `kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv` |
| Report parsing / LLM extraction / label modeling | **Local.** Text only — no images needed. This is the critical path and it runs on your own GPU. |
| Sample of ~300–800 studies for pipeline development | **D: drive.** Script the Kaggle API over a stratified subset of study folders (~30–80 GB). |
| Full-scale image training | **Kaggle notebooks** (data pre-mounted at `/kaggle/input`, no download, free T4/P100 quota) **or rented cloud GPU with fast NVMe.** |
| Preprocessed cache (resampled uint8 volumes) | Build **once** as a Kaggle Dataset; it will be a small fraction of 570 GB and makes every training run cheap. |

The cached-preprocessing step is what makes the rest of the competition tractable — treat it as a
deliverable, not an implementation detail. Aim for fixed-size uint8 arrays at ~256–320 px,
16–32 slices per series, only the series you actually feed the model.

---

## 2. Critical path: reports → labels (weeks 1–3)

> **CLOSED 2026-08-10 — this was the critical path and it is now a download.** §2.1 below calls
> for "a multilingual LLM offline over every training report". Four such tables are published
> as free Kaggle Datasets, the best scores **0.893** on gold-58, and the rule extractor built
> instead scores **0.777** and loses on 12/12 labels (`IMPROVEMENTS.md` §2f). Everything in §2
> is kept for provenance and because §2.1's *schema* — structured attributes, not twelve bits,
> with per-finding confidence — was right and is what the public readers emit. The build was
> the error, not the design. Do not spend further time on §2.1–§2.4.

### 2.1 Extract structured attributes, not just the 12 bits
Run a multilingual LLM offline over every training report; emit JSON per study:

- The **12 target labels**, each with a confidence/certainty level.
- **Severity/grade**: meniscus grade 1/2/3, mild/moderate/severe OA, effusion small/moderate/large,
  MCL grade I–III, cartilage grading.
- **Morphology**: partial vs complete tear, horizontal/radial/bucket-handle, root tear, ramp lesion.
- **Sub-location**: anterior horn / body / posterior horn; ACL proximal/mid/distal.
- **Laterality of the knee** if the report states it (cross-check against DICOM — see §3.2).
- **Certainty**: definite / probable / possible / "cannot exclude" / negated.
- **Extra findings** outside the 12: chondral defect, plica, loose body, bursitis, tendinopathy,
  hardware, prior ACL reconstruction. These become auxiliary heads.

### 2.2 Build your own validation set — the provided one is unusable
**Measured:** only **58** studies carry gold labels, the rarest positive class among them is
MCL at **n=9**, and the set is clearly *enriched* (positive rates 16–60%) rather than sampled.
Worse, per-language gold coverage is: English 28, Spanish 8, Turkish 6, Croatian 4, Dutch 4,
Greek 3, Russian 3, German 2, **French 0**.

You cannot validate a 12-label multilingual extractor on that. So:

> **Hand-label a stratified 250–400 reports spanning all 9 languages.** Machine-translate to
> read them — you are labelling, not practising fluency. This is the single highest-leverage
> manual task in the competition and most teams will skip it.

Then: use all 58 gold studies for measurement (never fitting), plus your own set, and report
per-label **and per-language** agreement. A single weak language silently poisons ~10% of the
corpus and you would never see it in an aggregate number.

### 2.3 Turn extraction into training signal
- **Soft targets from certainty**: definite→0.98, probable→0.85, possible→0.65,
  "cannot exclude"→0.55, negated→0.02. The metric is a *ranking* metric, so hedged findings
  genuinely belong between clean and definite.
- **Confidence-weighted loss**: down-weight studies where the extractor was unsure or where the
  LLM and the encoder model disagreed.
- **Ordinal severity heads** (CORAL / cumulative logit) — using expected grade as the ranking
  score often beats the plain binary head, because it orders the positives correctly.
- **Auxiliary heads** for the extra findings, loss weight ~0.2–0.5. Denser gradient per study,
  and it forces the encoder to represent anatomy the 12 labels ignore.

### 2.4 Curriculum
Pretrain on the large pseudo-labeled set → fine-tune on the small gold-labeled set. Keep the gold
subset in a held-out fold for honest evaluation of the *whole* pipeline.

---

## 3. Imaging pipeline

### 3.1 Preprocessing
1. Group by series (folders already do this); sort slices by `ImagePositionPatient` projected onto
   the slice normal — **not** `InstanceNumber`.
2. ~~**Transfer-syntax spread is a real cost**: uncompressed, JPEG Lossless, JPEG 2000, Implicit
   VR. JPEG 2000 decode is slow.~~ **Measured 2026-08-07 and false.** The corpus is **200/200
   Explicit VR Little Endian** at **3.1 ms/slice** — one syntax, uncompressed, cheap. No
   `pylibjpeg`, no GDCM, no `dicomsdl`. See `FINDINGS.md` §6.
3. MRI has no HU standard → per-volume robust percentile normalization (clip 0.5/99.5 → [0,1]).
4. Resample in-plane to fixed mm/px; centre-crop/pad to a fixed FOV. The knee is protocol-centred,
   so a detector is overkill.
5. Fixed slice count per series (16–32).
6. Cache as uint8 `.npy`.

### 3.2 Laterality — still the sharpest trap
Four labels are side-specific (Medial/Lateral Meniscus, Medial/Lateral OA). "Medial" flips between
left and right knees, so a model fed raw mixed-handedness studies sees medial findings on both
sides of the image.

**Canonicalize every study to one handedness.** **Answered 2026-08-07** (`FINDINGS.md` §6.2):
`(0020,0060) Laterality` survives but is **empty on half the corpus** — 2,203 of 4,407 studies
carry a usable value. The geometry fallback works, but *not* at the obvious boundary: `x < 0`
agrees with the tag only 89.3%, while **`x < -62` agrees 97.7%** (cross-validated 97.32% ± 0.72%),
because the scanned knee sits at isocentre rather than the patient being centred. Tag first,
geometry second, source recorded per series. A pixel-based L/R classifier is no longer needed.

> **TTA trap:** horizontal-flip TTA is *invalid* here unless you also swap the Medial↔Lateral
> output pairs. Either swap them or don't use hflip.

### 3.3 Architecture
**2.5D CNN + slice transformer + series attention** — the family that has won recent RSNA
volumetric competitions.

```
per series:
  slices → groups of 3–5 adjacent as channels
        → 2D backbone (ConvNeXt-tiny / EfficientNetV2-s / MaxViT-tiny), shared weights
        → per-slice features [S, D]
        → 2-layer transformer over slice axis (+ positional encoding)
        → attention-pool → series embedding [D]

study:
  series embeddings [K, D]
        + series-type embedding from (Anatomical_Plane, Fluid_Sensitive, Fat_Suppression)  ← given!
        → transformer / gated attention pool over series
        → head → 12 logits (+ auxiliary heads)
```

Findings are plane-specific, so feed multiple planes: MCL is coronal; PF OA and Baker's are axial;
ACL and menisci are sagittal; contusion needs **fat-suppressed fluid-sensitive** series. A
sagittal-only model caps out.

**Measured series structure** (see `FINDINGS.md` §3):
- `Fluid_Sensitive` and `Fat_Suppression` are **perfectly redundant** (10,361 at 0/0, 14,010 at
  1/1, zero off-diagonal). Collapse to one flag → **6 series types**, not 12.
- Median 5 series/study (range 3–14).
- Coverage: Axial FS **100%**, Sagittal nonFS 96.8%, Coronal FS 96.4%, Sagittal FS 94.2%,
  Coronal nonFS 77.3%, **Axial nonFS 19.4%**.
- **87.2% of studies are missing at least one of the six types.** Axial FS is the only series
  you can always count on.

Augmentation: affine (±10°, ±10% scale/translate), intensity/gamma jitter, slice-axis jitter,
coarse dropout, bias-field, Rician noise, and **series dropout — mandatory, not defensive**,
given that missing series is the norm (§5). No hflip (§3.2).

### 3.4 External data — likely decisive
Only **4,407** training studies, nearly all pseudo-labelled. That is small for a 12-label 3D
task, and the rules explicitly permit "freely & publicly available external data, including
pre-trained models." Candidates, in rough order of value:

> **PROMOTED 2026-08-09.** This section was never carried into any phased route, despite its own
> heading. It is now **Phase 2** in §9. The reason it moved up: IMPROVEMENTS §2d shows the six
> failing labels are millimetre-scale structural findings, and **MRNet supervises three of those
> six directly** (ACL, meniscal tear) with real expert image reads rather than pseudo-labels.
> OAI covers the three OA labels, which are already our strongest — so MRNet is the higher-value
> of the two despite being the smaller corpus.

- **OAI (Osteoarthritis Initiative)** — thousands of knee MRIs with expert OA gradings.
  Directly supervises Medial OA / Lateral OA / PF OA, three of the twelve.
- **MRNet (Stanford)** — 1,370 knee MRIs labelled for ACL tear, meniscal tear, abnormality.
- **fastMRI+** — lesion annotations layered over the fastMRI knee corpus.
- **KneeMRI (Rijeka)** — ACL injury grades; plausibly the same source as the Croatian reports.

Pretrain on these → fine-tune on the pseudo-labelled competition corpus. Check each licence
against the competition's external-data rules and post the datasets to the forum thread as
required.

### 3.5 Upgrades in priority order
1. Aux heads + soft labels from §2 — cheapest real gain.
2. Per-plane specialist models + a light stacker on OOF predictions.
3. A 3D branch (X3D-M, R(2+1)D-18, Video Swin-T) for ensemble diversity.
4. Cross-modal CLIP-style image↔report alignment pretraining, then fine-tune the image tower alone.
   Multilingual text encoder required. Stretch goal — only after the baseline is solid.
5. ~~Label-correlation stacker on the 12 OOF logits.~~ **⛔ CLOSED 2026-08-13, §3r.** Measured two
   ways: a fitted cross-label stacker gains +0.0033 on report labels and loses **−0.0132 on gold**
   (opposite signs — §3l-2's mechanism); unfitted PC1 shrinkage degrades monotonically on both.
   **PC1 is 52% of the variance and it is SIGNAL** — the shared axis is real comorbidity.

---

## 4. Validation

- ~~**MultilabelStratifiedGroupKFold**, 5 folds, grouped by patient.~~ **There is no patient to
  group by** — `PatientID` is present in every DICOM and is *unique per study* (4,407 IDs for
  4,407 studies), so it is de-identified per study. Plain multilabel-stratified 5-fold, ungrouped;
  `fusion/folds.py`. Stratify on the rarest labels.
- Keep gold-labeled and pseudo-labeled studies **identifiable** in every fold; always report
  metrics on the **gold** subset — pseudo-labels inherit the extractor's biases and will flatter you.
- Studies are from a "diverse international mix of imaging sites" → **site shift is the #1 CV↔LB
  divergence risk.** Monitor per-manufacturer/per-language performance if that's recoverable.
- **Prevalence differs across train / public LB / private LB** (stated explicitly). AUC is
  prevalence-insensitive *within* a label, so this is survivable — but it means public LB will be
  noisy and will not track private. **Trust CV.**
- **The public leaderboard is 30% of test; the final standing is the other 70%.** Confirmed
  2026-08-08. Two consequences, and the second is the one that costs people prizes:
  - On the ~1,300-study estimate above, the public LB is scored on **~390 studies**, and a label
    at Fracture-like prevalence contributes on the order of a hundred positives to it. A per-label
    AUC off that base has a CI wide enough to swallow most of the improvements we plan to make.
    **A public-LB move smaller than the gap between two of our CV folds is not evidence.**
  - The 30/70 split is *disjoint*, so every point of public-LB tuning is fitted to studies that
    contribute nothing to the final rank. Combined with the stated prevalence shift, public→private
    shake-up is the default expectation, not the tail risk. This is why §4's final-two-submission
    rule is (a) best CV, (b) smallest CV–LB gap — **neither of them is "best public LB".**
- Bootstrap CIs per label. With ~1,300 test studies and a rare label like Fracture, the private
  AUC for that label has a wide CI — some of the final ranking is luck. Don't chase fold noise.
- Final two submissions: (a) best CV ensemble, (b) most robust / smallest CV–LB gap.

---

## 5. Inference notebook

- Decode test DICOMs on the fly; **multiprocess** it (CPU-bound, will otherwise starve the GPU).
  Overlap decode with GPU inference via a prefetch queue.
- AMP fp16, `channels_last`, optional `torch.compile`.
- **Handle missing series.** Some test studies will lack a plane. Train with **series dropout** so
  the attention pool is robust, and explicitly unit-test single-series and degenerate inputs. This
  is the classic submission-time crash.
- Exact header:
  `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`
  Note `Baker's` contains an apostrophe and several columns contain spaces — quote handling matters.
- Guard against NaN, clip to [0,1], and wrap per-study inference in try/except emitting 0.5 on
  failure so one bad DICOM can't zero the submission.

---


---

## 8. Risks

| Risk | Mitigation |
|---|---|
| **Pseudo-label quality caps the whole solution** | Two independent extraction methods; validate per-language; confidence-weight the loss |
| **A weak language silently poisons a subset** | Per-language agreement metrics, not just overall |
| **Laterality tag stripped from the 86-tag allowlist** | Check first; fall back to geometry or a pixel-based L/R classifier |
| **Site/scanner/protocol shift** (explicitly international) | Site-aware CV, heavy intensity aug, per-site monitoring |
| **Prevalence shift train→public→private** (stated) | AUC is prevalence-insensitive within a label; don't calibrate to train prevalence; trust CV |
| **Rare labels + only ~1,300 test studies → wide CIs** | Bootstrap CIs; accept that some ranking is luck; don't overfit folds |
| ~~**JPEG 2000 decode dominates runtime**~~ **retired 2026-08-07** | Measured: 200/200 Explicit VR Little Endian, 3.1 ms/slice. The risk does not exist |
| ~~**GPU time is the cache-build constraint**~~ **corrected 2026-08-08** | It is not GPU time, it is **per-file open latency** on the mount (§6.3.1). ~700k opens at ~19 ms is hours before the GPU matters. Shard via `SHARD`/`N_SHARDS`; the build resumes, so a killed session loses one study; do the 224 pass first |
| **Kaggle assigns a P100 on roughly 4 of 5 draws, and `accelerator` in kernel-metadata.json does not override it** | Its PyTorch dropped Pascal, so a P100 session cannot run this at all — and `torch.cuda.is_available()` is True on one. `pick_device()` fails closed in the first seconds of `main()`, so a bad draw costs seconds and re-rolling is a viable strategy rather than a way to burn the weekly quota |
| **A silently-serial worker pool costs a whole session** | Three cache attempts died this way (21 h, then 9 h on the serial curve). The pool is spawn-context, workers are pinned to one thread, and the PROBE at 25/100/400 series warns inside minutes when the rate is near single-worker. `--self-test` now constructs the real pool and asserts it fans out |
| **Missing series in test studies** | Series-dropout augmentation + explicit degenerate-input tests |
| **570 GB won't fit locally** | Kaggle notebooks / cloud for pixels. ~~Local work is text-only~~ — **no longer true**: cached frozen DINOv2 embeddings are ~2.4 GB, so the fusion head trains locally (`README.md` Training) |
| ~~**Public weak labels commoditize the extractor**~~ **retired 2026-08-07** | A/B run and passed — 0.777/0.749 vs 0.672/0.672 on gold, 0.864/0.862 vs 0.757 on hand labels. Ours wins on both references and both metrics |
| ~~**Extractor gains are no longer measurable**~~ **superseded 2026-08-10** | The true statement is stronger: the extractor is **0.116 behind a free alternative** and is closed as a label source (`IMPROVEMENTS.md` §2f). "Out of instrument" framed a dead component as a measurement problem, and that framing cost five days |
| **Everyone shares one backbone** | DINOv2 is universal, so it cannot differentiate. Push the edge into the fusion layer (§3.3), the labels (§2), and external data (§3.4) — the three places our work is already strongest |

---


---

## 9f. THE WORKSTREAMS `2026-08-13 pm — the live execution plan, kept here so it survives the session`

Written after §3p retired the label-ceiling thesis. **The binding constraint is model capacity**,
so every stream below is aimed at the model→teacher gap, not at labels.

### A — harvest the free arms `1 submission`

| arm | state |
|---|---|
| `ft_b` | **DONE, §3o.** OOF 0.8522, blend **+0.0284**. |
| **the two-arm blend** | **RUNNING** — kernel `raahimnawaz/rsna-knee-f6-two-arm-blend`, no site prior, so the LB read is a clean test of §3l-2's offset. **Predicts 0.915–0.926.** |
| `tonylica` | **⛔ DROPPED, §3q.** 0.7880 and *more* correlated than `ft_b` (0.704 vs 0.632). Weaker and more redundant. |
| RadImageNet | **expected to fail the same bar** — frozen encoder (§2e's weak configuration), `best_val 0.8167`, i.e. tonylica's profile. Cheap to check with `fold_recover.py` once the encoder is wired; do not assume it pays. |
| DINOv3 | **the only remaining family likely to earn a slot.** Genuinely fine-tuned. **⛔ It does NOT condition on site** — that claim is retracted in §9e and confirmed dead in §9h: `n_sites: 109` is a config field with no parameter behind it, and `enc.tok.weight` (7, 384) conditions on **slot type**. **ARCHITECTURE HALF IS BUILT (§9h)** — `fusion/dinov3_model.py`, 5/5 folds strict on timm 1.0.28. Still **no fingerprint to check the transcription against**; what is left is `dinov3_pixels.py`, and §9h carries four corrections to the spec that a builder must read first. |

**The rule this stream now runs under (§3q):** free to run is not a reason to include. Every arm is
scored on `score_gold.py` and blended only if it beats the 2-arm baseline **and** clears the prior
— comparable strength, lower correlation. The public notebooks vote 24 members and never checked.

### B — licence, one forum post `open`

Both non-DINOv2 arms rest on custom licences. **Assessment: permissible.** DINOv3's licence permits
commercial use and grants derivative works; RadImageNet is research-use with commercial licensing
separate; rules **§2.5.a** explicitly carves *"pretrained models with an incompatible license"* out
of the winner's open-sourcing duty, and the winner licence is **CC-BY-NC 4.0** anyway. We consume
both as public Kaggle Datasets, which **§3.6.b** deems OSI-licensed.

**Unresolved:** §2.6.a's *"equally accessible to all Participants"* where upstream weights sit
behind a click-through. **This is the same question `REFERENCE.md` §1.3 has open for MRNet — asked
twice by others, never answered.** Ask it naming DINOv3 and RadImageNet. Also a dependency risk:
the submission would rest on a third party's Dataset that could be deleted.

### C — train an arm, and make it a CNN `parallel, laptop`

**Decided 2026-08-13: the trained arm should be a fine-tuned CNN, not another ViT.**

*Why a CNN specifically, rather than "more capacity":*

1. **Diversity by construction.** Every arm in the blend is a ViT (DINOv2 ×2 families, DINOv3), and
   the one CNN in the public field — RadImageNet R50 — is **frozen**, which §2e measured as the
   configuration that cannot adapt. A *fine-tuned* CNN is the one family nobody in this competition
   is running.
2. **The literature says so for exactly our failing labels.** Independent work on meniscal-tear MRI
   reports **ResNet50 consistently outperforming other architectures**, and **ViTs
   *under*performing ResNet**, attributed to the difficulty of training transformers on limited
   medical data. Our corpus is **4,407 studies** — squarely in that regime.
3. **It is an inductive-bias argument, not an information one.** A ViT-B/14 at 336 px represents
   the image as 14 px tokens (≈5–7 mm at our pitch) against findings of ~1–2 mm. The patch
   embedding is not lossy — 14×14×3 = 588 values into 768 dims — so the detail is not *destroyed*.
   What differs is that a CNN has locality and translation equivariance built in, while a ViT must
   learn them from 4,407 studies. **That is a sample-efficiency claim, and it is testable.**
4. **It aims at the right labels.** §3p: all remaining headroom is in Lateral Meniscus, Synovitis,
   Medial Meniscus, ACL and PF OA — the focal, millimetre-scale findings.

*How:* `fusion/ft_b_pixels.py` + `ft_b_model.py` are a transcribed, working recipe — swap the
backbone. Spend the budget on **unfreezing and schedule, not resolution** (`sadamtorres` measured
fine-tuning at +0.09 LB against resolution's +0.017; §2d separately measured **data ≈ 2×
resolution**, and we sit at 81.5% of the corpus). Judge it on its **blend delta**, never its solo
score — §2y and §3q both closed arms that were fine alone.

#### C — the executable spec, so this can be picked up cold

**Do not re-derive the reasoning above; it is settled. Start here.**

| decision | value | why |
|---|---|---|
| backbone | **RadImageNet ResNet-50, FINE-TUNED** — already on disk at `data/external/radimagenet_r50_encoder/ResNet50.pt` | **See the survey below. This is the pick.** Fallbacks if it disappoints: `convnext_small.fb_in22k_ft_in1k` or `tf_efficientnet_b3.ns_jft_in1k` (timm, permissive). Pick one and **do not A/B them on gold** (§3b). |
| input | **reuse `ft_b_pixels.py` unchanged** — K=32 slices, per-series, full-frame after background trim, 336 px, ImageNet stats, canonicalised to a RIGHT knee | Already transcribed and validated (§3n/§3o). CNNs take any resolution, so no `img_size` surgery. |
| head | **reuse `FTBHead`** from `fusion/ft_b_model.py`, with `d_in = backbone.num_features` (not ×2 — a CNN has no CLS token, so pool the feature map) | The head is the arm's proven half; changing both halves at once makes a failure undiagnosable. |
| targets | `data/targets.csv` (= `steven_v2`) | §3p: labels are not the ceiling. Do not spend on F4 first — **§3x challenged this on 08-17 and §3p's per-label column held.** |
| folds | `data/folds_site.csv` (site-grouped) | §2j: +0.024 of site leakage otherwise. |
| schedule | fine-tune **all** blocks; ~1.6 h/fold under `caffeinate -i`; `--resume` exists (§2u) | §2e: a frozen encoder cannot adapt, and RadImageNet is the field's cautionary example. |

**The gate, to be pre-registered before the first run:** score fold-resolved on
`fusion/score_gold.py`, then measure the **blend delta against the current best blend**, not the
solo score. §3q's bar: it must clear **both** the measurement *and* the prior (comparable strength
to ~0.85, correlation below `ft_b`'s 0.632). If it lands at tonylica's ~0.79, drop it — that arm
was free and still did not pay.

**Watch for:** MPS memory. §2p's "the port is memory-bound" was never established (§2v), but the
17.2 GB box is real and a ConvNeXt-S at 336 with K=32 slices per series is a large batch. Reduce
slices per step before reducing resolution.

### ⚠️ Topological methods — deprioritised, NOT closed `superseded in part by §9g`

> **⛔ READ §9g's "Topological CNN" BEFORE ACTING ON THIS SECTION. It revises the verdict below.**
> This survey closed TDA outright. §9g, written later the same day, found the closure **too
> dismissive on its first bullet**: there *is* classification evidence on **clinical** images
> (topological features boosting both CNNs and ViTs in breast-cancer screening; TopOC on
> ovarian/breast), not only on derived quantitative maps and microscopy.
>
> **The revised verdict is "not closed, but ranked below Workstream C"** — and it is ranked there
> on *different* grounds than this section gives: the clinical evidence is 2-D single-image, §3p's
> headroom is 1 mm focal while topology speaks to larger-scale connectivity, §3r measured that the
> structure TDA would most naturally capture (PC1, the shared abnormality axis, 52% of variance)
> is **already signal the model has**, and it pays at submission time out of a decode-dominated
> budget. Bullets 2 and 3 below stand; bullet 1 is the one that was overstated.
>
> Keeping a bold ⛔ 130 lines ahead of its own correction is how [[record-findings-durably]] fails.

Persistent homology / TDA is real in medical imaging, but the evidence does not transfer here:

* **It runs on derived quantitative maps, not clinical volumes.** The knee-OA results use
  compositional MRI (T2 mapping), morphological grading and biomechanics variables. This
  competition ships **clinical, multi-vendor, multi-protocol** MRI with no quantitative maps. The
  headline accuracies (≈98.7%) are **bone microstructure under non-linear microscopy** — a
  different problem at a different scale.
* **Its strengths land where we have no headroom.** TDA is naturally good at connected
  fluid-filled structures — effusion, Baker's cyst. §3p measured those at **0.981 and 0.980**,
  already *beating the teacher*. It has nothing obvious to say about a 1 mm posterior-horn tear,
  which is where all five open gaps are.
* **It would be a weak arm, and weak arms have now failed twice here** — §2y's port at −0.111 and
  §3q's tonylica at −0.0089. Diversity alone does not pay; §3o's gain needed strength *and*
  diversity.

**Reopen only if** a quantitative map becomes available, or if a cheap experiment shows a
topological feature separating the *focal* labels. Not on the critical path.

**§9g names that cheap experiment**, so this is no longer a hypothetical exit: a topological
feature block screened on `data/features_*` (or a small pixel sample), judged on
`fusion/score_gold.py`, pre-registered, before any training commitment. Same shape as §9g's
second-order-pooling screen, and it shares that screen's gate. Both are **blocked on the same
instrument** — see §9g.

#### C — survey of reusable CNNs, and the pick `2026-08-13 pm`

[[check-whats-free-first]] applied before writing a training loop.

| candidate | verdict |
|---|---|
| **RadImageNet ResNet-50** — `data/external/radimagenet_r50_encoder/ResNet50.pt`, **already downloaded** | **✅ THE PICK.** Verified: `backbone.*` keys load **STRICT** into a torchvision `resnet50` trunk, 23.5 M params, 2048-dim features, forward at 336 gives (2048, 11, 11). |
| MRNet repos (`MisaOgura`, `dazcona`, `Elzawawy`, `rowantahseen`) | Code, not weights — AlexNet/ResNet/NASNet trained on the Stanford MRNet set. Re-training means acquiring MRNet, which is **`REFERENCE.md` §1.3's unresolved accessibility question**, asked twice and never answered. Not worth gating a workstream on. |
| Kaggle `knee` models | Hobby-grade (X-ray osteoporosis, OA X-ray nets). Nothing trained on knee **MRI** at this scale. |
| ImageNet ConvNeXt / EfficientNet | Available and permissive, but **natural-image** pretraining — the thing `sadamtorres` measured as costing +0.09 LB to fix by fine-tuning. Fallback only. |

**Why RadImageNet fine-tuned is the strongest CNN move available, and it is three arguments
converging:**

1. **It is a CNN** — locality and translation equivariance built in, which is the sample-efficiency
   argument above, aimed at the five focal millimetre-scale labels §3p isolated.
2. **It is radiology-pretrained, not ImageNet-pretrained.** `sadamtorres` measured **domain
   adaptation at +0.09 LB against resolution's +0.017**; this backbone starts already adapted.
3. **The entire field uses it FROZEN** — `mattiaangeli`'s heads sit on a frozen R50, and §2e
   measured that a frozen encoder cannot adapt at any resolution. **Fine-tuning it is a
   differentiator that costs one training run**, not a download nobody else has.

**It also lets F6's RadImageNet arm be dropped without loss.** §9f-A expected that arm to fail
tonylica's bar because it is frozen. The same weights, fine-tuned, become Workstream C instead —
the useful half of that family, without the weak arm.

**Licence note:** research-use with commercial licensing separate, same category as DINOv3, same
§2.5.a carve-out (Workstream B). And its accessibility is *de facto* established here — the public
notebooks already attach it.

### 9g. Two method questions, answered `2026-08-13 pm`

#### Can we use tensorised images at submission time? **No — and this is structural**

The test studies are **not visible until the scored run starts**. Nothing derived from test pixels
can be precomputed and attached, so `data/tiles336`, `data/features_*` and every other cache is a
**training-time artefact only**. What *can* ride along is anything computed from train data:
weights, fixed transforms, priors, normalisation constants.

**Consequence, and it is the one that shapes every plan here:** tensorisation happens *inside* the
9 h run, on CPU, and **decode is what dominates** (§3e — the 74 s is forward passes; a real run
decodes the whole hidden test). Any method that needs an expensive per-study transform pays for it
at submission, every time. That is the budget a topological or radiomic feature would come out of.

#### Second-order (covariance) pooling — the strongest untested "statistics" idea

**Every arm we run pools FIRST-order.** pilkwang is `cls_mean`; `ft_b` is attention-pool over
slices then mean within plane. Neither represents *interactions* between feature channels.

The established result: global average pooling captures only first-order statistics and misses the
high-order feature interactions that **fine-grained** classification needs; bilinear / covariance
pooling with matrix square-root normalisation reports **+2–3%** on fine-grained benchmarks, and
second-order pooling has been applied to medical classification (SOP + SENet on ResNet-50).

**Why it fits here specifically:** §3p localised every remaining point to five **focal,
millimetre-scale** labels. "Fine-grained" is exactly that regime, and it is the same reasoning that
motivates the CNN in Workstream C — texture and local interaction rather than global shape.

**It is cheap to screen and needs no GPU, no quota and no pixel path.** `data/features_*` are
frozen DINOv2 embeddings (94×1536 per study) already on disk. Fit a head on covariance-pooled
features against one on mean-pooled features, same folds, same targets. **This is the F3-shaped
screen and it should be run before any training arm commits to a pooling choice.** Judge on
`score_gold.py`; pre-register the comparison.

**Caveat that must be priced in:** covariance of 1536 channels is 1536², so it needs low-rank or
compact bilinear pooling to be tractable. Budget that before claiming it is cheap.

#### Topological CNN — the earlier survey was too dismissive, and here is the correction

**This supersedes §9f's "Topological methods" closure in part; that section now carries a pointer
here.** §9f closed TDA on the grounds that its evidence is on derived quantitative maps and
microscopy. **That was incomplete.** There is classification evidence on *clinical* images: topological
features integrated with pretrained models in **breast-cancer screening** report consistent boosts
to both CNNs and ViTs, and TopOC applies topological deep learning to ovarian/breast diagnosis.
Separately, persistent-homology **loss functions** are differentiable and established — but note
those are for **segmentation**, where topology *is* the output structure, which is not our task.

**What still argues against it here, and why it stays behind the CNN:**

1. **The evidence is 2-D single-image** (mammogram, histopathology tile). Ours is multi-series 3-D
   knee MRI with twelve labels and a slot structure.
2. **§3p's headroom is in 1 mm focal findings.** Topology speaks to connectivity and holes at a
   larger scale — and §3r just measured that the structure it would most naturally capture (the
   shared abnormality axis, PC1, 52% of variance) is **already signal the model has**.
3. **It pays at submission time**, out of the decode-dominated budget above.

**Revised verdict: not closed, but ranked below Workstream C.** The honest cheap test is the same
one as second-order pooling — a topological feature block screened on `data/features_*` or on a
small pixel sample, judged on `score_gold.py`, before any training commitment.

### 9h. DINOv3 — the complete architecture spec, read off the WEIGHTS `2026-08-13 late`

> **✅ THE ARCHITECTURE HALF IS BUILT — `fusion/dinov3_model.py`, 2026-08-13 late.**
> `python fusion/dinov3_model.py --check` → **all 5 folds load strict, 23.5 M params each**, on
> **timm 1.0.28** despite the dataset shipping a `timm-1.0.22` wheel (162/162 encoder tensors
> matched, nothing unexpected). A forward pass runs. **The pixel path is NOT built** — see the
> four corrections below, two of which change what a `dinov3_pixels.py` has to do.
>
> **The kernel source is now saved**, which it was not when this section was written:
> `data/external/kernel_sources/mattiaangeli_bend-the-knee-to-dinov3-ensembled.py` (extracted
> from the `.ipynb` beside it). `data/` is gitignored, so re-pull with
> `kaggle kernels pull mattiaangeli/bend-the-knee-to-dinov3-ensembled -p <dir> -m`. §9h below was
> derived from source that was read and not kept — the same [[record-findings-durably]] gap that
> cost a day on `kaggle_01c`.

**⛔ FOUR CORRECTIONS FROM THE ACTUAL TRANSCRIPTION `2026-08-13 late`. Read these before building
the pixel path — two of them are silent-error traps.**

1. **`stem: 'native'` means THE 16 SLICES ARE INPUT CHANNELS, not 16 forward passes.**
   `patch_embed.proj.weight` is **(384, 16, 16, 16)** — the encoder is built
   `in_chans=cfg['n_slice']`. The spec below said `native` only *removes* `DepthCompress`; it also
   *replaces* what "a slice" means. One series = one forward pass of a (16, 336, 336) tensor.
2. **⛔ THE SLOT SCHEME IS NOT PILKWANG'S, AND `data/slots_pilkwang.csv` DOES NOT TRANSFER.**
   This arm's slots are **`[(Sagittal,1), (Sagittal,0), (Coronal,1), (Coronal,0), (Axial,1),
   (Axial,0)]`** — (plane, fat-suppression) pairs — against pilkwang's recovered
   `[SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1]`. **Different
   membership AND different order**, and `enc.tok.weight` is indexed by that order, so reusing our
   table would condition every series on the wrong token and still run. **The good news: it is
   free.** `build_study` selects on the competition's own `Anatomical_Plane` and `Fat_Suppression`
   columns and takes `sub.iloc[0]` — one series per slot, first match. No `annotate()`, no header
   parquet, no slot recovery. (`Fat_Suppression` being byte-identical to `Fluid_Sensitive` over all
   24,371 series does not matter here — the arm was *trained* against this column as it stands.)
3. **The pixel constants differ from pilkwang's and from `ft_b`'s.** `CROP_MM 130.0` (same),
   **`SLICE_BAND (0.12, 0.88)`** against pilkwang's `(0.2, 0.8)`, `SIZE 336`, `N_SLICE 16`,
   `INTENSITY 'slice'` (per-slice 1/99 percentile — note `ft_b` is per-*series* 0.5/99.5, and
   mixing them is silent). `norm: 'none'`, so pixels are just `uint8 / 255` with **no ImageNet
   normalisation** — again unlike `ft_b`. Laterality: flip in-plane iff `plane != 'Sagittal'` and
   `ImagePositionPatient[0] < 0`; **sagittal is never flipped.**
4. **Slice order is `int(InstanceNumber)` ascending — nothing geometric.** §2n measured
   InstanceNumber at **56.9%** concordance with the true direction, but that is irrelevant to
   *reproducing* this arm: it was trained on InstanceNumber order, so InstanceNumber order is
   correct here and K16 must **not** be applied. This is the one place where our better instrument
   is the wrong one to use.

**And one fear that turns out to be unfounded.** §9h below calls the RoPE branch "the single
highest-risk part — get it wrong and the model loads strict, runs, and is quietly wrong." **On
timm 1.0.28 it is not quiet.** `ViTSlotToken.__init__` bumps `vit.num_prefix_tokens` *and* every
block's `attn.num_prefix_tokens` by one; ablating the bump raises `RuntimeError: The size of
tensor a (442) must match the size of tensor b (441)` immediately, because `EvaAttention.forward`
applies rope to `q[:, :, npt:, :]` against a 441-entry table. Measured by ablation, not assumed.
The branch is still copied verbatim — but it fails loudly, so it is not the thing to fear.

**The remaining subtlety, which IS silent:** `Net.forward` keeps `torch.cat([f[:, :1],
f[:, orig:]], 1)` = **CLS ++ slot_token ++ patches**. So `CodexResidualPool` pools CLS in `base`
and attends over **the slot token together with the patches** in the delta. Dropping the slot
token from the KV changes nothing about shapes and quietly changes the model.

#### ⚠️ THE SLICE-ORDER PROBLEM IS LOCAL-ONLY, AND IT IS MEASURED `2026-08-13 late`

**`fusion/dinov3_pixels.py --probe`, n=47 gold studies × 5 folds, thresholds written before the
numbers were seen.**

| | |
|---|--:|
| reversal Δ (mean \|Δsigmoid\|) | **0.0692** |
| between-fold Δ, identical input — *the reference scale* | 0.1024 |
| **ratio** | **0.675** |
| corr(forward, reversed) | 0.9200 |

Stable across n (0.678 at n=4 → 0.675 at n=47). **The model is NOT direction-robust: reversing
the 16-slice stack costs about two-thirds of what swapping in a different fold costs.** Worst
label is ACL (0.0909), which is a sagittal structure, so this is mechanistically sensible rather
than noise. Expected — `stem:'native'` puts the slices in as **channels**, and channels are not
exchangeable.

**Why it arises at all:** this arm orders slices by `int(InstanceNumber)`, and InstanceNumber is
**not on disk in any form**. The NIfTIs carry no patient frame (§3n) and
`dicom_headers_zhukovoleksiy.parquet` is **one row per SERIES** — 24,371 rows, nothing per slice.
§2n priced the residual ambiguity at a **direction bit, not a scramble** (`inst` and `loc` at
|rho| = 1.000), correct **56.9%** of the time.

> **⛔ AND HERE IS THE POINT THAT CHANGES THE VERDICT: THIS IS AN ARTEFACT OF OUR LOCAL CORPUS,
> NOT OF THE SUBMISSION.** On Kaggle the arm reads the **DICOMs**, where `InstanceNumber` is
> simply present — their `ordered_files` does `int(ds.InstanceNumber)` and is exactly
> reproducible. **The ordering problem exists only in local scoring.**
>
> So the local gold read is a **lower bound** on what this arm contributes on the board, biased in
> the safe direction — the same shape of handicap §9e accepted for the fold problem ("biased
> toward *the new arm adds nothing*"). **If it clears §3q's bar locally despite being fed ~43% of
> series backwards, it will do better than that in a submission.** A miss, by contrast, is
> ambiguous and must not be read as "the arm is weak".

**PRE-REGISTERED, BEFORE ANY GOLD AUC FOR THIS ARM EXISTS (§3b):** local scoring of the DINOv3 arm
uses **direction TTA — render each series both ways and mean the sigmoids.** Reason: guessing is
fully wrong on ~43% of series, whereas TTA is half-right on all of them, and at corr 0.92 the two
renders are similar enough to average without one dominating. **The submission path does NOT use
TTA** — it sorts on InstanceNumber like the original, and its cost stays single. This is written
down now precisely so it cannot be chosen after seeing which scores better.

**Coverage is measured and is NOT a blocker `2026-08-13 late`.** The slot scheme reproduces
directly from `data/train_series.csv` — no header parquet, no recovery step:

| slot | 0 SAG fs=1 | 1 SAG fs=0 | 2 COR fs=1 | 3 COR fs=0 | 4 AX fs=1 | 5 AX fs=0 |
|---|--:|--:|--:|--:|--:|--:|
| all 4,407 studies | 94.2% | 96.8% | 96.4% | 77.3% | 100.0% | 19.4% |
| 3,599 with local pixels | 94.0% | 96.9% | 96.4% | 76.7% | 99.9% | 19.7% |

**Mean 4.84 / 6 filled, and no study has zero** — better than pilkwang's 20,130 slots over 4,407
(4.57). Slot 5 (axial, non-fluid-sensitive) is rare at ~19%, which is a property of the corpus,
not of our copy. **All 47 of the gold studies `ft_b` was scored on have local pixels**, at a mean
fill of **4.91 / 6** — so when `dinov3_pixels.py` exists, the gold read is a *paired* comparison
against pilkwang's 0.8516 and `ft_b`'s 0.8522 on exactly the same 47 studies, which is what §9h's
gate (c) asks for.

**One caveat on what the slots MEAN.** `Fluid_Sensitive` and `Fat_Suppression` are byte-identical
over all 24,371 series (re-verified), so this arm's six slots are really **(plane × fluid-
sensitive)** — a *coarser* partition than pilkwang's six, which separate T1 from non-fat-sat fluid
by parsing `SeriesDescription`/`SequenceName`. Two different models of the same anatomy, and that
difference is a reason to expect genuine diversity rather than a reason to distrust either.

Everything below is established from `m_f0.pt` and the kernel source, and is confirmed by the
transcription except where the four corrections above amend it.

```
cfg: backbone vit_small_patch16_dinov3.lvd1689m · img 336 · n_slice 16 · stem 'native'
     pool 'xcodex' · cond 'token' · meta 'none' · n_meta 0 · norm 'none' · pe_init 'tiled'
     n_sites 109  <- DEAD FIELD, no parameter behind it
state dict: enc.* 163 tensors (enc.vit.* + enc.tok.weight) + readout.* 17 tensors
```

**`stem: 'native'` and `n_meta: 0` mean `DepthCompress`, `SlotDepthMixer` and `meta_mlp` are all
unused** — that removes three classes from the transcription. `cond != 'post'` removes `slot_emb`.

| module | shape | what it is |
|---|---|---|
| `enc.tok.weight` | (7, 384) | `nn.Embedding(N_SLOT_TYPES+1=7, d=384, padding_idx=MASK_IDX)` — **slot-type** conditioning |
| `readout.pres_emb.weight` | (7, 64) | `nn.Embedding(7, pe=64, padding_idx=0)` |
| `readout.pool.q / dw / db / gate` | (12,384) (12,384) (12,) (12,) | `_GatedDelta` — one query per label |
| `readout.pool.attn.in_proj_weight` | (1152, 384) | `nn.MultiheadAttention(384, n_heads, batch_first=True)`; 1152 = 3×384 |
| `readout.pool.base.0 / .2` | (832,) / (12,832) | `LayerNorm(2d+pe)` → `Linear(832, 12)`; **832 = 2×384 + 64** confirms `d=384, pe=64` |

**So the readout is exactly `Readout('xcodex', d=384, n_labels=12, pe=64)` with
`CodexResidualPool`**, whose `base` takes `_seg_mean_max(tok[:, 0], sidx, B)` — i.e. the **CLS**
token segment-pooled, plus the presence embedding — and adds `gate * delta`, where `delta` is
label-query cross-attention over the **patch** tokens via `_pad_kv`.

**The conditioning mechanism**, and it is the fiddly part:

```python
tok = self.tok(cat).unsqueeze(1)
x = torch.cat([x[:, :npt], tok, x[:, npt:]], dim=1)   # npt = num_prefix_tokens
```

The slot token is inserted **after** the prefix tokens (CLS + registers) and **before** the
patches. **DINOv3 uses RoPE**, and the kernel has an explicit `if rope is not None: if
getattr(v, 'rope_mixed', False): ...` branch to keep positions right once an extra token is
inserted. **That branch is the single highest-risk part of the transcription** — get it wrong and
the model loads strict, runs, and is quietly wrong. Copy it exactly; do not reimplement from
understanding.

**The gate, and it must be pre-registered before the run:** the arm ships **no fingerprint and no
OOF**, so the only checks available are (a) strict load of all 5 folds, (b) fold-resolved OOF via
`fusion/fold_recover.py` with a χ²-flat partition, and (c) landing in a plausible band against
pilkwang's **0.8516** and `ft_b`'s **0.8522** on the same 47 gold studies. **§3q's bar applies:
comparable strength AND correlation below `ft_b`'s 0.632, or it does not earn a slot.**

#### C — the NVIDIA / MONAI category, surveyed and EMPTY `2026-08-13 late`

[[check-whats-free-first]] applied to a category §9f-C's survey never checked. It looked at
RadImageNet, MRNet repos, Kaggle knee models and ImageNet CNNs, and **never looked at NVIDIA's
medical-imaging stack at all.** Closed now, with a negative result, so nobody re-opens it.

| NVIDIA asset | verdict |
|---|---|
| **MONAI Model Zoo** | **Nothing usable.** Enumerated the published bundles: spleen/pancreas/renal/lung-nodule CT, BraTS brain MRI, prostate MRI anatomy, endoscopic, pathology (×4), MedNIST. **Zero musculoskeletal, zero knee**, and only **three classification** bundles in the whole zoo (breast density = mammography, endoscopic in-body, pathology nuclei) — the rest are segmentation. There is no MSK encoder to fine-tune. |
| **DALI / nvJPEG2000 GPU decode** | **⛔ DEAD, and this repo already had the measurement.** It targets the submission's dominant cost, so it looked like the one real lever — but `FINDINGS.md` §6 measured the corpus at **200/200 Explicit VR Little Endian, uncompressed**. There is no compression to accelerate. Worse, the same section's correction shows the bottleneck is **file-open I/O latency over ~700k opens**, not pixel decode, and no GPU touches that. |
| **TensorRT / mixed precision** | Targets the **forward pass**, which is not the bottleneck — §9g/§3e establish a submission is decode/IO-dominated. Same reasoning that killed the C++ port at §6.1 (93 s per 0.001 AUC). |
| **`OrthoDiffusion`** (arXiv 2602.20752, 2026) | **Exactly on target and unavailable.** A diffusion foundation model for MSK MRI, three orientation-specific 3D models trained on **15,948 unlabeled knee MRI scans**, doing 11-structure segmentation and **8 knee abnormality detections**, transferring to ankle and shoulder. **No public weights or code released** as of this read. **Watch item** — if weights appear, this is the strongest candidate encoder that would exist for this task. |

**What the category IS worth, and it is not nothing.** NVIDIA/MONAI's central thesis is
*domain-pretrained encoder + fine-tune on the target task*, which is **exactly Workstream C's
thesis** (RadImageNet R50, fine-tuned, §9f-C). So the largest medical-imaging platform in the
industry independently backs the direction already chosen. **It offers validation, not a
shortcut** — and the shortcut it would have offered, a pretrained MSK encoder, does not exist in
public weights today.
