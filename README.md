# RSNA Knee Abnormality Detection

Twelve-label knee-MRI classification, macro-AUROC. Final submission **2026-10-22**.

---

## START HERE — state as of 2026-08-17

> # ⏭️ PICK UP HERE
>
> ## 🔴 SESSION HANDOFF — 2026-08-18, three things are IN FLIGHT. Read this first.
>
> ### A. ✅ SCORED **0.914** — the arm earned its slot, at HALF the predicted gain
>
> **`raahimnawaz/rsna-knee-f6-three-arm-blend-radimagenet`, version 2.** Three families:
> pilkwang + `ft_b` + the public RadImageNet arm (§4c, +0.0146 gold / +0.0160 at 4.4σ large-n).
> Built by **`notebooks/build_f6_rad_kernel.py`** — never hand-edit the notebook, rebuild and push.
>
> ```
> .venv/bin/python -m kaggle kernels status raahimnawaz/rsna-knee-f6-three-arm-blend-radimagenet
> .venv/bin/python -m kaggle kernels output raahimnawaz/rsna-knee-f6-three-arm-blend-radimagenet -p <dir>
> ```
>
> **⛔ BEFORE SUBMITTING, GREP THE LOG FOR `rad:`. It must say `blended 3 families`.** If it says
> `2 families`, a guard refused the arm and the file is just the banked 0.908 path — submitting it
> gains nothing and burns a submission.
>
> **✅ THAT CHECK HAS RUN — 2026-08-18 13:09, version 2, `KernelWorkerStatus.COMPLETE` in 132 s.**
> The log reads `blended 3 families at equal weight: pilkwang + ft_b(3/3) + rad(3/3)`, and every
> upstream guard passed with it: **20/20 pilkwang members fingerprint-match within 4.6e-06**, all
> **5 `ft_b` folds strict-OK**, `rad` **6/6 SHA-256 verified, 8/8 slots filled, pixel contract
> restored**, `submission.csv` **(3, 13), nulls 0**. No warnings in the log beyond nbconvert
> boilerplate. **The slot-filter fix works and the arm shipped. This file is submittable.**
>
> **✅ AND IT IS SUBMITTED — `55608011`, 2026-08-18 20:13 UTC, 4 submissions left that day.**
> ⚠️ **This is a CODE competition: submit the KERNEL, never the file.** The in-notebook
> `submission.csv` is 3 rows off the dummy test set; Kaggle re-runs the kernel on the hidden 1,322
> studies (~2 h of the 30 h weekly quota, §3e). The command that worked:
> `kaggle competitions submit rsna-knee-abnormality-detection -k <kernel> -v 2 -f submission.csv -m ...`
> **✅ RESULT: 0.914** (banked 0.908, predicted 0.921). **The arm is real — +0.006 — and the
> prediction was optimistic by the same amount.** It landed short and the cause is where this
> file said to look: **§4b's debias/offset chain, not the arm.** See `IMPROVEMENTS.md` §4c-4.
> ⛔ **Standing correction: price a future arm's LB gain at ~HALF its gold gain.** Gold +0.0135
> and large-n +0.0102 at 8.8σ both delivered +0.006 shipped.
>
> ⚠️ Only soft note: `rad: prob mean 0.3985 std 0.2671` against a local reference of `0.345/0.261`.
> **On a 3-study dummy set that is noise** — the std matches to 0.006 — but if a full run ever shows
> the same +0.05 offset on 1,322 studies, that is a calibration drift worth reading.
>
> **v1 ALREADY FAILED THAT WAY, and the guard is why we know.** Log read:
> `rad: slot order [] != ['SAG_FLUID_FS', ...]; refusing the arm`. The notebook's `SLOTS` entries
> are **4-tuples `(name, plane, fluid, fat_sat)`, not strings**, so the filter matched nothing.
> The model half was perfect — all 6 SHA-256 verified, 3,174,924 params — and the arm still
> correctly refused rather than blending a wrongly-conditioned tensor. **Fixed in v2 (`s[0]`).
> Keep the guard; it earned its place on its first run.**
>
> ⚠️ **The in-notebook run sees a ~3-study dummy test set**, so a 4-line `submission.csv` is normal
> and tells you nothing about coverage. The log lines are the signal.
>
> ### B. ⏳ ORTHODIFFUSION — STAGE 0 HALF DONE (`PLAN.md` §C-3, §C-3.2b)
>
> **Recipe is READ, from their code, and none of it was guessable** — input `[1, 16, 256, 256]`,
> centre-crop depth to 16, resize 256² bilinear, **per-volume min-max → [−1,+1]** (a new trap-table
> row; everything else here uses a 1/99 percentile), loaded by bare `nib.load` — **which is exactly
> how `data/nifti/nifti_train` already stores 19,859 series.** Features are intermediate denoising
> activations: linear probe `timestep 100 / mid_2`; 3-pose fusion **`timestep 200,150,50`,
> `blockname mid_0,mid_0,mid_2`** — different per orientation.
>
> **⛔ BLOCKER BEFORE STAGE 1: which `pose_id` is which plane is UNRESOLVED.** It comes from a
> per-file CSV, not a constant, and the checkpoints are named `sagittal/coronal/axial`. Feeding a
> plane its neighbour's timestep runs perfectly and scores wrongly (§9h). **Resolve first.**
>
> **What is left in Stage 0:** pull the 1.66 GB from `hf://models/lanstat0123/orthodiffusion`, load
> one, emit a feature for one study. Transcribe from `linear_fusion.py` (the 3-plane version),
> `linear_classifier.py`, `pooling.py`. Repo cloned notes in §C-3.2b. **MIT, verified in `LICENSE`.**
>
> ### C. ✉️ THE AUTHORS WERE EMAILED 2026-08-18
>
> Dingyu Wang (`wang_dingyu@pku.edu.cn`), cc Dong Jiang — the two corresponding authors on **both**
> OrthoDiffusion and OrthoFoundation. Asks whether **OrthoFoundation's weights** will be released
> (still none) and asks them to **confirm MIT permits competition use**. **A reply may be waiting.**
>
> ### D. STILL OPEN, IN PRIORITY ORDER
>
> 1. **✅ §3z-4's confound — FIXED 2026-08-18.** The decomposition is pre-registered in
>    `IMPROVEMENTS.md` §3z-4: three masked reads (FULL / −DEPTH / −BOX) off the one Stage 1
>    checkpoint, **no retraining** — `SlotHead.forward` already takes a slot mask and training runs
>    `slot_dropout=0.2`, so a masked read is in-distribution in kind. **Writing it turned up two
>    things the section's own table got wrong:** (a) **`sag_pf` is listed as a "resolution" slot but
>    shares the *sagittal-series* availability of the two depth slots** — `has_sag_med ==
>    has_sag_lat == has_sag_pf` on all 3,599 studies — so an unrestricted read silently compares a
>    4-slot arm against a **3-slot** arm on 216 of them; hence **all reads are paired on the 3,254
>    studies with all six slots present**. (b) **Handedness stays confounded with depth** —
>    `needs_direction=True` on exactly `sag_med`/`sag_lat`, and the bit is **50.4% reversed**, so it
>    is doing real work. **A Stage 1 depth gain still may not be cited for `SAGITTAL_LR=1`**, and
>    there is **no non-flipped anatomical cache** to test it against — only `tiles336_lr1` has one.
> 2. **§3z — `fusion/band_ab.py` is BUILT and UNRUN.** MPS is free. 3 cache builds; do not run it
>    beside a training job.
> 3. **🆕 `aagatti/nnunet_knee` IS NOW FUNDED — `PLAN.md` §C-4, pre-registered 08-18.** MIT, verified
>    on the Hub. **Funded as an OFFLINE CALIBRATOR only** — run once over training studies, fit
>    §3y's `centre_mm`/`box_mm` from real anatomy, ship *calibrated fixed boxes*. It never touches
>    the scored path, so §10's ruling that an nnU-Net 3D cascade is unaffordable at inference does
>    not bind it. ⚠️ `fold_1` only, not the 5-fold ensemble. Gate 1 is free (they ship their own
>    `test_prediction.nii.gz`); **Gate 2 is the real risk and must be read per plane × sequence**,
>    since OAI-ZIB is DESS sagittal and this corpus is clinical multi-protocol.
>    **🆕 It may also dissolve §3z-4's handedness blocker** — medial/lateral meniscus are
>    distinguishable in a mask, giving a second independent instrument on that axis with **no cache
>    rebuild**, where §2n left the measured K16 bit with no cross-check at all.
> 4. **⚠️ The RadImageNet trunk is `CC-BY-NC-SA-4.0`** (§4c-3). Shipping A accepts that knowingly.
>    `REFERENCE.md` §1.3 is unanswered. **§C-3's OrthoDiffusion is MIT and is the clean route.**
>
> **0. ⛔ FIRST: WE ARE BELOW THE FREE PUBLIC CEILING AGAIN. SHIP BEFORE YOU BUILD (`IMPROVEMENTS.md`
> §4a, surveyed live 2026-08-18).** Rank **547 of 1,904** at 0.908 — **the last submission was
> 2026-08-13 and 1,162 teams have submitted since.** Six *public notebooks* sit at **0.917–0.922**,
> each confirmed against its author's own LB row, **including `pilkwang` at 0.919** — the author of
> the 0.891 fork we are still running. The board is compressed where we sit: **rank 100 = 0.920,
> rank 500 = 0.910, +0.012 is worth ~444 ranks.**
>
> **The field gets there by rank-mean-blending DINOv3 + RadImageNet + tonylica — §3t, §3w-2 and
> §3q, the three arms we each tested and rejected.** The obvious explanation is checked and wrong:
> we already rank-mean (`blend_test.py:59`). **What survives is that §3t's CI was [−0.0182,
> +0.0133] while the entire gap is +0.012 — it measured a NULL and the record wrote it down as a
> refutation.** Every "does not earn its slot" verdict resting on gold-47/58 is suspect for the
> same reason. Strength findings (DINOv3 0.8025, tonylica 0.788) stand; the inference from them
> does not.
>
> **✅ THAT TEST HAS RUN — §4b, same day. The instrument is FINE; the SOURCING was not.**
> Scored `tonylica`'s shipped `v52_e11_oof.csv` through §9e's pre-registered rule on gold-47:
>
> | | macro | Δ vs banked | draws |
> |---|--:|--:|--:|
> | 2-family blend (banked) | 0.8798 | — | — |
> | 3-family **+ their RadImageNet** | **0.8932** | **+0.0135** | **98%**, CI [+0.0004, +0.0262] |
> | 3-family + DINOv3 (§3t, re-run) | 0.8775 | −0.0022 | 38% — **reproduced to the digit** |
>
> Large-n says the same at **8.8σ** (+0.0102 ± 0.0012, n=4,349). **Debiased and offset, this
> predicts LB 0.921 — against a public frontier of 0.917–0.922.** The instrument lands on the board.
>
> **⛔ SO §4a-3's "the instrument cannot resolve it" IS RETRACTED, and so is the false-negative
> story.** Every rejection was **correct** — §2y 0.7323, §3w-2 0.6924, §3q 0.788, §3t 0.8025 were
> all genuinely weak. The winning arm is at **parity (0.8514)** and its Spearman is **0.713**, *less*
> diverse than DINOv3's 0.644. **Screening on strength was right the whole time.**
>
> **The real failure: §3w-2 spent Workstream C training a RadImageNet to 0.6924 while a 0.8514 one
> was a free download.** §3w-2 is **narrowed, not retracted** — our *training* is refuted at 4.8σ,
> the *architecture* is not.
>
> ### 🆕 ORTHODIFFUSION'S WEIGHTS ARE PUBLIC AND MIT — `PLAN.md` §C-3, found 2026-08-18
>
> `hf://models/lanstat0123/orthodiffusion` — **axial / coronal / sagittal, 553 MB each, `license:mit`**,
> plus MIT training code at `github.com/lt-0123/OrthoDiffusion`. Three orientation-specific 3D models,
> self-supervised on **15,948 unlabeled knee MRI scans**.
>
> **It is the only domain-matched encoder this project has ever had access to** — everything else is
> DINOv2/v3 on natural images or RadImageNet on general radiology — the only **3D** one, and **MIT**,
> so unlike the RadImageNet route it carries **no non-commercial exposure** and §1.3 does not gate it.
>
> **⛔ AND WE MISSED IT TWICE.** §C called the category empty on 08-13; the HF repo was last updated
> **29 May 2026**, eleven weeks earlier. **Not an expiry — the claim was false when written.** The
> model card is 603 bytes tagged only `en` / `arxiv:...` / `license:mit`: **no `knee`, no `mri`, no
> `musculoskeletal`**, so a registry keyword search *cannot* find it. **Standing rule: for any paper
> that matters, walk paper → GitHub → weights by hand. A registry search has false negatives.**
>
> **Gate is pre-registered in §C-3** (§4b's precedent: strength is binding — **stop below 0.83**).
> ⚠️ **A diffusion model is not an encoder out of the box** — features are intermediate UNet
> activations at a chosen timestep/layer; **read their code first, do not invent the recipe** (§3s).
> **⛔ It does not jump the queue: it is many steps from a number, the RadImageNet arm is four.**
>
> ### ✅ THE RadImageNet ARM PASSED ITS GATE — 4 of 5 steps done (`IMPROVEMENTS.md` §4c)
>
> **Strength 0.8486 gold-47 (bar 0.83) · blend +0.0146 gold · +0.0160 ± 0.0036 = 4.4σ, 100% of
> draws on large-n.** **§4b's unshippable e11 arm scored +0.0135 / +0.0102 — this one is BETTER.**
> **⛔ STILL NOT SUBMITTED. The remaining blocker is the LICENCE, not the score — see below.**
>
> | # | step | state |
> |---|---|---|
> | 1 | **`fusion/rad_model.py`** — reproduce the arm | ✅ **DONE.** Encoder + all 5 head SHA-256s verify against `rad_heads_manifest.json`, every fold loads **strict**, parameter count **3,174,924** matches the manifest to the digit, forward OK incl. the missing-planes case |
> | 2 | **`fusion/rad_pixels.py`** | ✅ **DONE.** 224 px · **full-frame, no crop** · **fat-suppressed only** · 3 planes × **8 slices** · band **(0.12, 0.88)** · per-series 1/99 · **grayscale→RGB at `x/127.5−1`** (NOT ImageNet stats). Add a column to `ARCHITECTURES.md`'s trap table |
> | 3 | local OOF (`fusion/rad_arm.py`) | ✅ **DONE.** gold-47 in 10 s, n=600 in 66 s on MPS |
> | 4 | **blend delta, §9e rule** | ✅ **PASS.** +0.0146 gold (95%), **+0.0160 at 4.4σ large-n** |
> | 5 | kernel → push → run → submit | ⏳ **BLOCKED ON LICENCE, not on the number.** Their config is `_RAD_ALPHA = 0.50`, `_RAD_EXCLUDE = ("Baker's", "Fracture")`; §3b forbids tuning either on 47 studies — reproduce theirs or ship the flat §9e rank-mean actually measured. **Do not invent a third option at submission time** |
>
> **✅ THE VARIANT WARNING RESOLVED THE GOOD WAY.** §4b's +0.0135 was measured on `e11` (130 mm
> crop) and these are `folds_v1` (full-frame) — a different arm, so it was re-measured rather than
> assumed. It transfers **and is larger**. folds_v1's own `best_val` ≈ 0.807 looked like DINOv3
> territory; on our instrument it reads **0.8486**, because `best_val` is a different reference
> (§2o, three numbers near 0.89). **The screen is what settled it, not the vendor's number.**
>
> **⛔ THE LICENCE IS THE BLOCKER AND THE PUBLIC HEADS DO NOT SOLVE IT.** Using
> `mattiaangeli/...foldsv1-heads` removes the *private-bundle* problem, not the *non-commercial* one:
> the 12.7 MB heads are useless without the **official RadImageNet ResNet-50 trunk, which is
> `CC-BY-NC-SA-4.0`** and is pinned by SHA-256 in the manifest. **`REFERENCE.md` §1.3 — asked twice,
> never answered — is now load-bearing for prize eligibility.** The whole 0.917–0.922 public frontier
> carries the same exposure, which is context, not a licence. **This is a call to make deliberately.**
>
> **⛔ Note what it does NOT block: §C-3's OrthoDiffusion is MIT.** If NC resolves badly, that is the
> route that survives.
>
> ⚠️ **Licence, unchanged:** ship from **`mattiaangeli/rsna-knee-radimagenet-foldsv1-heads`** (58 MB,
> public) — *not* from tonylica's bundle, whose own README says it must stay private and includes
> **CC-BY-NC-SA-4.0** assets. RadImageNet is NC either way, so `REFERENCE.md` §1.3's unanswered
> question is now **load-bearing for a prize**, not academic. The whole 0.917–0.922 public frontier
> carries the same exposure.
>
> **Fallback if step 4 fails:** the two-arm F6 kernel is unchanged and still submittable on its own.
>
> **§3z and §3y stay valid and stay free. They are simply SECOND.**
>
> **1. ✅ THE F6 SUBMISSION LANDED: `55490186` = 0.908. BANKED, and it is the new best.**
> Predicted 0.915–0.926; **came in below the band at 0.908**, +0.009 over the old 0.899.
> **§3v has the full reconciliation and it reprices three things** — read it before planning:
> * **The two gains DO NOT ADD.** TTA pooling (+0.008) and the `ft_b` blend (+0.015–0.020
>   expected) delivered **+0.017 together**, ~40% overlap. **Variance-reduction levers are
>   sub-additive** — never price a new arm against the plain fork, price it against 0.908.
> * **The offset is +0.039, not +0.046.** §3p's coherence check matched pilkwang's *TTA-free*
>   gold to our *TTA-included* LB and absorbed the +0.008 into the constant. `score_gold.py` is
>   updated. Debias the gold read first: 0.869 + 0.039 = **0.908 exactly**.
> * **§3p's direction stands, its headline does not.** No shared label ceiling and model capacity
>   is still the binding constraint — but a perfect learner of the free labels is now **0.938 vs
>   a 0.940 prize line**, i.e. borderline, not "10th place or better".
>
> **Board moved faster than we did (§3v-5), and again since (§4a): top 0.951, 10th/prize
> **0.941**, **1,904 teams**, we are **rank 547**. Gap to prize **0.033**. **65 days left.**
>
> **2. ⛔ DINOv3 STAYS PARKED — ITS RE-OPEN CONDITION IS REFUTED, NOT UNMET (§3v-6).** §3t said
> re-open only if local numbers ran *pessimistic*. They ran **optimistic** (0.880 gold predicted
> 0.915–0.926, delivered 0.908). **Do not revisit.** Same for `ft_a` (same family as `ft_b`, the
> least diverse arm available) and tonylica (§3q) — sub-additivity makes both worse than they
> looked. **F6 is spent; it has no arm left worth adding.**
>
> **§3t's original verdict, unchanged:** built, audited, scored, parked.
> OOF **0.8025** vs pilkwang 0.8516 / `ft_b` 0.8522; 3-family blend **0.8775 vs 0.8798 banked**,
> delta **−0.0022**, positive in 38% of draws. **Spearman vs `ft_b` 0.571 — the most diverse arm
> this project has measured**, and it still fails on strength. Third time: the port 0.639/0.7323,
> tonylica 0.704/0.788, this 0.571/0.8025. **Diversity has never been the binding constraint.**
> **Parked, not dropped** — its direction handicap is local-only (§9h). Re-open only if item 1
> shows local numbers running pessimistic. **Do not re-derive any of this.**
>
> **📐 `IMPROVEMENTS.md` §3s — what is actually inside it.** Both headline features are nearly
> inert: the `xcodex` cross-attention is gated to ~0.001 (deleting it costs **+0.0003**) and slot
> conditioning enters at **2.1%** of a patch token. Its slot usage is **anatomically correct**.
>
> **3. ⛔ WORKSTREAM C IS CLOSED — REFUTED AT 4.8σ, 2026-08-17 (`IMPROVEMENTS.md` §3w-2).**
> The gate was pre-registered before the first epoch and the arm lost:
>
> | | macro report-OOF, fold 0, n=681 |
> |---|--:|
> | `runs_cnn` RadImageNet R50 | **0.6924 ± 0.0089** |
> | `runs_port` dinov2-small | **0.7323 ± 0.0086** |
> | **paired delta** | **+0.0399 ± 0.0084 → 4.8σ, P=1.000, 11/12 labels** |
> | bar was | ≥ 0.739 |
>
> **The stop rule fires: do NOT build the 35–78 GB cache, do NOT run Stage 2.** `runs_port`
> reproduced at 0.7323 exactly, vindicating §3w's pre-training anchor correction.
>
> **⚠️ AND THE TRAINING LOSS LOOKED GREAT THE WHOLE WAY** — monotone 0.4592 → **0.3686**, 36.7% of
> the way prior→floor, past the ViT's recorded epoch-1 0.4523 by epoch 3. **It measured fitting,
> not ranking, and was worth nothing as a signal.** 23.5M unfrozen params at 1e-4 on 2,871 studies
> fit the soft targets hard and generalised worse. Gating on `score_oof.py` is the only reason this
> was caught before a cache build.
>
> **"Exactly one variable" was too strong:** the encoder swap bundles architecture **and**
> pretraining (dinov2-small self-supervised on LVD-142M vs RadImageNet R50 supervised on
> radiology). What is refuted is the **RadImageNet R50 arm**, cleanly. **✅ It settles §3y's
> backbone: build multi-scale on the ViT port.**
>
> **STAGE 1 BAR: ≥ 0.739** against `runs_port` **0.7323** (`score_oof.py`, NOT summary.json's
> 0.7298 — §2o). **Below 0.7323 → stop**, the inductive-bias argument is refuted and the big pixel
> cache is not worth funding. **In [0.7323, 0.739) → the question becomes the pixel path, not the
> backbone; do not run Stage 2 on this cache.** ⛔ **Stage 1 is NECESSARY, NOT SUFFICIENT** — the
> control is 0.7323 against pilkwang's 0.8434, so beating it proves the CNN bias helps, *not* that
> the arm earns a slot. §3w records the expectation that it probably does **not**.
>
> **Two traps already paid for, both in §3w — do not re-derive.** ① `SlotDataset` zeroes absent
> slots, which is inert under a ViT's LayerNorm and **silently poisons every BN layer** under a
> CNN. ② The obvious fix (gather only present slots) makes the tensor size vary and **MPS
> recompiles per shape: 1.57 → 44.20 s/step**. What ships is *forward all slots with BN frozen* —
> 1.70 h/fold, faster than the ViT control's 3.7 h.
>
> **§3v sets its real bar: the arm must beat +0.009**, which is what the best free diverse family
> on the board delivered on top of what we already had.
>
> **⛔ F1 IS DEAD — §3u. Do not queue the site prior.** The "+0.0023, free, unshipped" line was
> an estimator with no deployable counterpart: built in the form a submission runs, it is
> **−0.0057, positive in 0/2,000 draws**. The gain splits by *prior source*, not by
> fit/score matching. **§3f's harmonising-away result is untouched.**
> **Standing rule it produced: re-measure any gain in the exact configuration the submission will
> run before shipping it — an A/B measures an estimator, not an idea.**
>
> **Cheap screens waiting, no GPU or quota (§9g):** second-order/covariance pooling on
> `data/features_*`; a topological feature block on the same.
>
> **📐 `ARCHITECTURES.md` is new — how all five external arms actually work**, in one place:
> the load recipe per arm, and a **cross-arm trap table** of the ten conventions that differ and
> are silent when crossed (resolution, slice count, band, crop, window, normalisation, laterality,
> slice ordering, slot scheme). Read it before touching any pixel path.
>
> **3b. ✅ §3y STAGE 0 IS DONE — `IMPROVEMENTS.md` §3y-3, landed 2026-08-18, 100.3 min.**
> **`runs_port_lr1` 0.7358 ± 0.0086** vs **`runs_port` 0.7323** (reproduced exactly), **paired
> +0.0035 ± 0.0025, 1.4σ**, flip better in 91.6% of draws.
> **▶️ STAGE 1's CONTROL IS 0.7358. ITS BAR IS 0.7428.**
>
> **⛔ DO NOT CITE THIS AS EVIDENCE FOR `SAGITTAL_LR=1`.** The four labels that depend on the
> medial/lateral axis moved **≤0.008 and in both directions** (Lateral Meniscus **−0.003**);
> the macro rides on **Fracture +0.027** and **MCL +0.017**, which do not depend on it. That is
> expected — at **depth 0.5** reversal yields a different **TILE**, not a corrected **AXIS**
> (§3y-2's `GROUP=3` point). **Handedness cannot bind until depth 0.25/0.75, i.e. Stage 1.**
> The flip still changed the pixels for 28.4% of studies, which is why the control was
> mandatory — both halves of §3y-2 stand.
>
> **⛔ FIX §3z-4's CONFOUND BEFORE STAGE 1 IS READ — this is the last cheap moment.**
> Caches are BUILT:
> **`data/tiles336_lr1`** (`SAGITTAL_LR=1`, ~12 GB) holds protocol (17,403 tiles, 80.6% fill,
> 15.7 min) *and* anatomical (**20,684 tiles, 95.8% fill**, 9.8 min). **K16 covered it completely
> — 8,048 series carry a bit, 0 tiles skipped.** `data/tiles336` was **deliberately preserved**
> (§3y said rebuild in place; that would have made `runs_port` and `runs_cnn` unreproducible).
>
> **⛔ §3y-2: "protocol tiles are unaffected at depth 0.5" is MEASURED FALSE** — corpus-wide,
> **1,021 of 3,599 studies (28.4%)** get different protocol pixels under the flip, and ~⅔ of those
> are a genuinely different tile, not a channel permutation. `GROUP=3` has no fixed point under
> reversal. Corrected in six places. **This is why Stage 0 is mandatory, not precautionary.**
>
> ```
> # DONE 2026-08-18 — outputs in fusion/runs_port_lr1/ (fold0.pt, oof_all.csv, summary.json, stage0.log)
> caffeinate -i .venv/bin/python fusion/train_port.py --cache data/tiles336_lr1 \
>     --tag protocol --run-folds 0 --out fusion/runs_port_lr1 --verbose
> .venv/bin/python fusion/score_oof.py fusion/runs_port_lr1 fusion/runs_port
> ```
>
> ⚠️ For the NEXT run: it pages in cold for ~100 steps (1.4 → 30 img/s); `data/.metadata_never_index`
> stops Spotlight indexing the caches. Do not read the early `img/s` — it is a cumulative average.
>
> **⛔ AND §3y IS CONFOUNDED AS DESIGNED (§3z-4) — fix before Stage 1 is READ, not before it is
> run.** Its six anatomical slots are two treatments, not one: `sag_med`/`sag_lat` have **no box**
> and buy **depth coverage**; the other four have **no depth change** and buy **resolution**. §3y
> attributes a gain to resolution in advance (0.25 vs 0.48 mm/px), and the two readings imply
> opposite follow-ups. Pre-register the 2-vs-4 split as an ablation, or read the per-diagnosis
> attention mass `SlotHead` already computes. **Stage 0 is unaffected** — protocol slots only.
>
> **3c. ▶️ NEXT, AND IT IS FREE: §3z — THE SLICE BAND.** Pre-registered 2026-08-18 before any
> cache exists; harness is built and smoke-tested (`fusion/band_ab.py`). **Every arm that rivals
> pilkwang looks at more of the volume than pilkwang does** — `ft_b` 32 slices/full stack, DINOv3
> 16 at 0.12–0.88, against **our 12 at 0.20–0.80**. On sagittal the slice axis **is** medial–
> lateral, so **the twenty members we ship have never seen the outer 20% of any sagittal stack** —
> where **Lateral Meniscus (0.720, our worst label, gap +0.146)** lives. §3y's own `sag_lat` sits at
> depth 0.75, just inside a band the members were never trained past.
>
> **Free, and the K19 trap cannot recur here:** §3g proved `fingerprint()` takes `img_size` and not
> `SLICE_BAND`, and `SlotHead` has `slot_emb` (per-**slot**) with **no per-slice embedding**, so
> there is no learned index to misalign. The only real risk is domain shift outside (0.2, 0.8),
> which is the empirical question. **It improves the 0.8434 arm, so the vehicle problem does not
> apply to it** — that is most of why it ranks first.
>
> ```
> .venv/bin/python fusion/band_ab.py --n 600 --out data/_band_ab_n600.npz   # AFTER Stage 0 exits
> .venv/bin/python fusion/band_ab.py --gold --out data/_band_ab_gold.npz    # sign only, §3b
> ```
>
> ⚠️ **Do not run it while Stage 0 holds MPS** — ~4.9–6.5 GB of cache on a 17.2 GB box that already
> swaps at 336. **Four arms, because coverage and density are separated on purpose (§3z-3):**
> A = control rebuilt in-run · B = wider band, **coverage only** · C = more slices, **density
> only** · D = A+B pooled per target, the shippable arm. **`pool_arm_d` is fixed in code and §3b
> forbids retuning it on this run.** All four outcomes are committed to in §3z-3, including
> "both ≤ 0 → the axis is closed".
>
> **⛔ AUPRC IS THE WRONG CURRENCY (§3z-1). Do not optimise for it.** The metric is macro-AUROC;
> the host states prevalence differs train→public→private; **AUROC is prevalence-insensitive within
> a label and AUPRC is not**, so a local AUPRC gain has no stable relation to the private LB. Legit
> as a *diagnostic* of where in a ranking a label fails; never as a gate or objective.
>
> **📐 0.965 = ORACLE-PARITY, and it beats the current leader (§3z-2).** It equals closing every gap
> to a teacher that reads the report at test time, which §3p calls unattainable — against a board
> top of **0.951**. **Plan against the 0.940 prize line: gap 0.032, 65 days.**
>
> **4. 🎯 WHERE THE SCORE ACTUALLY IS — `IMPROVEMENTS.md` §3x, new 08-17. A DECOMPOSITION, NOT A
> RESULT.** §3p's seven positive gaps sum to **+0.047 macro**, so **gold 0.916–0.927 → LB
> 0.955–0.966**: *"reach 0.965"* and *"close every gap to the report teacher"* are the same
> instruction. Treat 0.965 as the optimistic end — the **+0.039 offset was calibrated over gold
> 0.85–0.90 and has never been checked above 0.90**, and AUROC compresses up there.
>
> **⛔ A severity-label / F4 revival was argued on 08-17 and §3p's own per-label column refutes
> it.** The model already **beats** the mention labels on every volumetric-threshold label
> (effusion **+0.141**, medial OA **+0.075**, Baker's +0.037) and **ACL's mention teacher scores
> 0.995 against gold** — nothing to repair. Better labels help least where the model is
> bottlenecked. **F4 stays deprioritised.**
>
> **The real split is SIZE.** Model wins on the 5 large diffuse findings; loses on the 5 small
> localised ones (meniscal surface contact, synovial thickening, ACL fibres, PF cartilage) — the
> findings a 336 downsample destroys. Agreeing evidence: **§2d's 224→518 = +0.013** (the only
> positive resolution reading here, dropped when §2e reframed it) and §3f localising the lateral
> deficit to the **posterior horn**. **§3k killed crop-INSTEAD-OF-context — its own mechanism
> ("the periphery carries ranking signal") predicts that crop-PLUS-context as extra slots works,
> and that has never been run.** §2l's canonical 132/132 axes place the boxes without a detector.
> **§3y PRE-REGISTERS THAT TEST — read it before touching the cache.** `pipeline/slot_cache.py`
> already defines six `ANATOMICAL` slots (written 08-11, never built) whose own comment states the
> mechanism: **84 mm boxes at 336 px = 0.25 mm/px against 0.48 for the full field**. Two traps it
> names in advance: ① the anatomical slots need **`SAGITTAL_LR=1`** and `tiles336` was built with
> **`0`**, so `assert_caches_compatible()` refuses and **`runs_port` 0.7323 is NOT a valid
> control** — a mandatory **Stage 0** retrains the 6-slot port on the rebuilt cache first
> (§2o's error class, caught in advance for once). ② The six slots cover only **+0.027 of the
> +0.047** — there is no ACL or synovitis slot.
> ⛔ **The vehicle problem is the real obstacle**, and §3y says so up front: the port is 0.7323
> against pilkwang's 0.8434 and §2y closed it at **15.4σ**, so even a big Stage 1 pass does not
> earn a blend slot. Carrying a gain into strength means fine-tuning pilkwang's own CC0 weights
> with the slot embedding extended 6 → 12. **Which backbone Stage 1 uses is decided by §3w's
> fold 0, not by preference.**
>
> **What NOT to re-derive:** F2-as-replacement (§3k+§3m, closed both instruments) · tonylica (§3q,
> dropped) · label-correlation stacking (§3r, closed two ways) · F4 (§3p, and §3x re-confirmed it
> against a fresh challenge).

## The state as of 2026-08-17

> **Banked 0.908** (submission `55490186` = pilkwang 20-member + per-target TTA pooling + `ft_b`).
> Board (§4a, 08-18): top **0.951**, 10th/prize **0.941**, **1,904 teams**, we are **rank 547**.
> **65 days left.** The 0.899 that appears in older notes is the superseded previous best; 0.891 is the
> original unmodified fork.
>
> **F6 is DONE and SPENT (§3v).** It delivered +0.009 and has no arm left worth adding.
> **Next is Workstream C, the trained CNN** — see item 3 above.
>
> **⛔ AND THE LABELS WERE NEVER THE CEILING — §3p, still the most important measurement here,
> with §3v's correction applied.** The best public label table, used as a predictor, scores
> **0.898 on gold → LB 0.938** (§3p said 0.944 on the old +0.046 offset), against a prize line of
> **0.940**. pilkwang reproduces **its own training labels at only 0.849** over 4,407 studies.
> **A model that perfectly learned the free, already-downloaded labels would land right at the
> prize boundary** — comfortably ahead of anything we or the public field has built, but no longer
> a proven top-ten score on its own. Nobody is label-limited; the whole field sits ~0.04 under a
> ceiling anyone can download.
> **F4 is deprioritised. Model capacity is the entire game.** All remaining headroom is in five
> focal labels — Lateral Meniscus (+0.146), Synovitis (+0.139), Medial Meniscus, ACL, PF OA — and
> the blend already *beats* its teacher on the other five.
> **§3x adds what those five have in common: they are the SMALL, LOCALISED findings**, and the
> five the blend wins are the large diffuse ones. That reads as a resolution/localisation
> bottleneck, and it makes the +0.047 those gaps sum to the whole remaining game.

### The two things that changed on 08-13 pm, and they reframe the project

**1. The "free public ceiling is reached" claim lasted one day (§3l-1).** It was surveyed 08-12
and was true then. The field now runs a **three-family** vote — 20-member DINOv2 + 5-fold
**DINOv3 ViT-S/16** + 5-fold **RadImageNet ResNet-50** + tonylica's four — all from **published,
downloadable weights**. We run pilkwang alone.

| arm | what it is | claim | size |
|---|---|--:|--:|
| `sadamtorres/rsna-ft-b` | 5-fold DINOv2 **ViT-B/14 @336**, backbone+head per ckpt | **0.883 solo** | 1.61 GB |
| `sadamtorres/rsna-ft-a` | same recipe @224 — the efficiency arm | 0.866 solo | 1.61 GB |
| `mattiaangeli/knee-mri-fold-weights` | **DINOv3 ViT-S/16**, 5-fold, fine-tuned on this comp | — | 439 MB |
| `mattiaangeli/rsna-knee-radimagenet-foldsv1-heads` | **RadImageNet ResNet-50** heads, 5-fold | — | 58 MB |

**This is the lever §2y proved we could not build.** The port was closed for being *weak* (0.7323
vs 0.8434), never for lacking diversity — its rank correlation was **0.639**, which is real. The
missing piece was an arm that is diverse *and* strong, and it is now a download. Scores above are
the authors' claims, read from kernel **source** on 08-13; treat as unverified until scored here.

**2. `score_oof.py` measures the teacher, not the target (§3l-2).** It scores a **report-derived**
source; the board scores expert **image** reads. A model that gets better at *seeing* departs from
the report exactly where the report was wrong, so **a real vision gain is partly booked as
disagreement.** Independently measured amplification: **3–5×**.

`fusion/score_gold.py` is the fix. pilkwang on the 58 image-read studies reads **0.8400** (95% CI
[0.799, 0.875]), implying **gold → LB = +0.051** against its known 0.891. That is the fourth
independent pair; three competitive systems agree at **+0.046, spread ±0.005**, inside the noise
the 58 studies impose.

### THE ROUTING RULE — which instrument answers which question

Getting this wrong is the error class §2s named, and it has now cost **five** measurements.

| instrument | n | precision | use it for |
|---|--:|--:|---|
| `fusion/score_oof.py` | 2,612 | ±0.005 | did this run break; epoch choice; **inference-side A/Bs on frozen weights** — §3m shows it agrees with gold to 0.0006 there |
| **`fusion/score_gold.py`** | 58 | ±0.038 | **is this direction worth pursuing** — sign only, `+0.046` → LB. **Required whenever weights change** |
| the leaderboard | — | ~2 h, ~15/week | anything under ~0.02 |

**§3b is unrepealed: never SELECT on the 58.** A weight chosen there claims +0.0137 and delivers
−0.0034, negative in 92% of draws; selection needs n≈400. Gold *evaluates* a fixed decision, it
never *chooses* one.

### What the gold instrument says about the labels (§3l-3)

Four of twelve labels put the report reading **outside** the gold 95% CI; chance gives ~0.6, so
the instruments differ **systematically**. Direction is the severity thesis (`REFERENCE.md` §2.1).

| label | report-OOF (§3f) | **gold-58** | |
|---|--:|--:|---|
| Lateral Meniscus | 0.767 | **0.642** | worst on both, and **§3m shows the crop still cannot move it** (−0.0055 on gold) |
| Lateral OA | 0.829 | **0.708** | |
| Synovitis | 0.886 | **0.742** | **looked 3rd best, is 3rd worst** |
| Medial OA / Effusion / Baker's | 0.872 / 0.855 / 0.887 | 0.950 / 0.943 / 0.955 | report *understates* the diffuse findings |

Per-label gold numbers are **indicative only** — 9–35 positives, half-widths ±0.10 to ±0.24. The
claim they carry is the *count* of CI violations, not any single value.

### CLOSED ROUTES — do not re-derive these

Each was measured, not argued. The section named is where the evidence lives.

| route | verdict | where |
|---|---|---|
| our port as an ensemble member | **no weight helps**, −0.111 at 15.4σ | §2y |
| crops as extra TTA windows (F2-cheap) | **DEAD ON BOTH INSTRUMENTS.** report −0.0031; **gold-47 −0.0038**, pre-registered | §3k, §3m |
| blending the fork's shipped `merge_gain` arm | +0.0007, 0/12 labels | §3h-2 |
| K16 from DICOM header rules | 56.9–60.8% vs ~50% chance; resolved by *measurement* | §2n, §2m |
| our rule extractor as a label source | **last of six**, 0.777 vs a free 0.893, 0/12 labels | §2f |
| per-label fusion of public label readers | near-duplicates (\|r\| 0.87–0.95); loses to best single | §2i |
| rank-mean `pilkwang` + `prvsiyan` | prvsiyan already *contains* pilkwang | §3c |
| re-fitting blend weights on site-grouped folds | **−0.0000** | §2z |
| harmonising scanner/site away (ComBat-style) | **−0.013 to −0.032** — case mix is real | §3f |
| selecting anything on gold-58 | claims +0.0137, delivers −0.0034 | §3b |
| post-hoc calibration / thresholds / priors | AUC is invariant to per-label monotone transforms | §3a |
| forking prvsiyan for its 0.906 | needs a **private** dataset; degrades to 0.899 | §3c |
| CoPAS, MRI foundation models, Gold Loss Correction | surveyed and rejected, with reasons | §3a |
| a C++ port for the efficiency track | 93 s per 0.001 AUC; decode is already native | `PLAN.md` §6.1 |

### Standing cautions

- **A submission costs ~2 h, not 74 s (§3e).** The 74 s is the members' forward passes; a real run
  also decodes the whole hidden test from DICOM. **~15 runs/week against a 30 h quota.** Batch.
- **A leaderboard number older than a few days is not evidence (§2x).** Re-read before any
  submission decision. **In a live competition a ceiling claim has a shelf life of days** — §3l-1
  is the fourth belief about the outside world to expire here, and it lasted 24 hours.
- **Always launch long runs under `caffeinate -i` (§2v).** ~2 of fold 0's 3.6 h were *asleep*.
  A fold is ~1.6 h awake. `caffeinate` cannot stop Thermal Emergency Sleep.
- **Read a competitor's *code*, never its description.** Three claims about `pilkwang` came from
  its description and two were wrong.
- **Baseline is macro 0.7229 ± 0.0048**, site-grouped report-OOF over 2,612 studies. Every figure
  before 2026-08-10 was inflated by ~0.024 of site leakage (§2j).
- **Live hazard: `data/tiles336` is `SAGITTAL_LR=0`; anatomical slabs need `SAGITTAL_LR=1`.**
  `pipeline/slot_cache.assert_caches_compatible()` raises on any run consuming both.
  `data/tiles336lr` is the rebuilt one — built, never consumed.

## Read these in order

| doc | what it holds |
|---|---|
| **`PLAN.md`** | strategy, the F-series, efficiency-track maths. §9a is the current plan; §9d is the disk inventory |
| **`IMPROVEMENTS.md`** | **the measurement record.** Every number and every closed route. §3l is the most recent |
| **`REFERENCE.md`** | **external ground truth** — host rulings, official label criteria, forum claims checked against ours, literature. Every claim carries a source and a read-date |
| **`FINDINGS.md`** | measured facts about the data — label coverage, languages, series structure |
| `COMPETITION_RULES.txt` | the competition rules, verbatim |
| `labeling/README.md` | hand-labelling workflow |

## The facts that shape everything

1. **58 of 4,407 training studies carry labels (1.3%).** The rest must be derived from the
   free-text radiology reports, and **the test set carries no report at all** — so the pipeline
   has exactly one shape: reports → targets, train a pure imaging model, discard the text.
2. **Reports span 9 languages, 61% non-English.** English-only clinical NLP is useless here.
3. **Gold thresholds SEVERITY; reports report MENTIONS.** 8 of 12 official criteria carry an
   explicit severity cut (ACL >50% fibres, OA ~1 cm at >50% thickness, effusion and Baker's
   moderate-or-large, MCL acute), and borderline cases were graded *negative*. This is the
   mechanism behind §2b's 84.7% agreement, it is a *ranking* problem, and §3l-3 is it showing up
   in the data. `REFERENCE.md` §2.1.
4. **0.9 is table stakes.** The board hit 0.932 within 48 h of opening on forks of public DINOv2
   notebooks; weak labels for all 12 findings are public. The backbone cannot differentiate us.
5. **Local and leaderboard are different scales, and the conversion is now MEASURED, not
   interpolated.** `gold-58 + 0.046 ≈ LB`, across four independent systems (§3l-2). The older
   two-foreign-anchor interpolation — "our 0.719 gold-37 is worth ~LB 0.75" — is superseded.
6. **The moat is retracted (§2f).** Our rule extractor is **last of six** label sources. Four
   LLM-read tables are free Kaggle Datasets. `data/targets.csv` is `steven_v2`.
   `extractor/bench_public_labels.py` is the live gate; `compare_methods.py` is the stale one.

## Layout

```
data/                  competition CSVs + derived label tables      (gitignored)
extractor/             report -> 12-label extraction, method A and method B
  rule_extractor.py      the rule extractor (clause-scoped, 9 languages). stdlib only
  run_extract.py         run over corpus + evaluate on gold. NB: works at import time
  llm_extract.py         method B via the Claude Batch API. --dry-run needs no key
  compare_methods.py     every label source scored head-to-head. the ORIGINAL decision gate,
                         stale: its baseline is nekkon's week-one binary CSV
  bench_public_labels.py the same gate re-pointed at the four public LLM readers. THIS is the
                         live comparison, and we lose it 0/12. IMPROVEMENTS.md 2f
  metrics.py             auc / bal_acc, shared. separate module so importing it is safe
  diagnose.py            bootstrap CIs + targeted failure diagnostics
  verify_claim.py        stress-tests the "labels aren't report-derived" claim
  calibrate_states.py    fits the soft-target ladder to P(gold=1|state). IMPROVEMENTS.md 1.3a.
                         The fit LOST (0.743 -> 0.699); kept as a measurement, not a setting
  to_fork_table.py       our labels -> the 25-column table pilkwang's notebook reads.
                         The __conf column is NEW and a guess -- see its docstring
labeling/              hand-labelling workflow
  glossary.json          multilingual terms for 12 findings + negation/uncertainty/severity
  compartment_patch.py   adds _side_* / _compartment_struct. IMPROVEMENTS.md 2.2
  build_labeler.py       generates the offline labelling UI
  model_labels.csv       labels produced so far (86/303, all 30 blind gold done)
  item_id_map.csv        item_id -> StudyInstanceUID. the durable fingerprint, no report text
  rekey_labels.py        binds labels to StudyInstanceUID; --check catches sample drift
  eval_model_labels.py   vs gold and vs rule extractor
  error_direction.py     splits disagreement into calibration vs true divergence
pipeline/              shared by BOTH machines -- the one definition of preprocessing
  preprocess.py          DICOM *and NIfTI* -> model input. Fingerprinted; see "Preprocessing parity"
  build_cache_local.py   the FROZEN-EMBEDDING cache, M5 from NIfTI. 1,062 studies/h @224.
                         Provenance only -- a cached embedding cannot fine-tune (IMPROVEMENTS 2e)
  bench_port.py          times the training port for real: 28.5 img/s, 2.6 h/fold. IMPROVEMENTS 2h
  bench_cache_build.py   times the 336 slot-pixel cache: ~16 min for the corpus. IMPROVEMENTS 2h
  validate_nifti.py      5 checks that the NIfTI repackaging matches the DICOMs. All pass
fusion/                the differentiator (PLAN.md 3.3). Trains on the M5, MPS
  model.py               slice transformer -> attention pool -> series attention -> 12 logits
  dataset.py             cached features -> padded batches. `python fusion/dataset.py` self-tests
  folds.py               5-fold, grouped by patient-proxy. NB: there is no patient column
  train.py               training loop + pooled-OOF gold eval. --synthetic needs no cache.
                         Also writes oof_all.csv -- every study, which is the real instrument
  instrument_test.py     proves oof_all is 6.7x tighter than gold-37, and finds the one thing
                         it CANNOT arbitrate: label sources. IMPROVEMENTS.md 2g
notebooks/             Kaggle-side only; the 570 GB of pixels never come local
  kaggle_01_dicom_audit.py    did (0020,0060) Laterality survive? + decode cost per syntax
  kaggle_01b_patients_laterality.py  header-only pass over all 4,407: PatientID + laterality
  kaggle_01c_series_geometry.py  per-series geometry + thumbnails so the NIfTI conversion can
                              be validated locally. CPU-only, no GPU lottery, ~5 min
  kaggle_02_dinov2_cache.py   frozen DINOv2 -> ~2.4 GB of per-slice features, shardable.
                              `--self-test` runs the whole scheduler locally, no DICOMs needed
  kaggle_03_submit.py         test DICOM -> submission.csv. No internet; weights from a Dataset
eda_01_labels.py       label coverage, prevalence, co-occurrence
eda_02_langs.py        language identification + series structure
eda_03_langid.py       lingua-based language detection (supersedes the heuristic)
eda_04_metadata_baseline.py   series metadata alone scores 0.471 on gold. "do not retest" was
                       written from n=58 and is retracted -- a public probe gets 0.5954 from the
                       same four columns over 4,407 (IMPROVEMENTS 2i-b). Still not a shortcut
```

## Setup

```bash
python -m pip install -r requirements.txt
# auth: new-style KGAT_ token at ~/.kaggle/access_token (legacy kaggle.json no longer works)

# all four CSVs are needed -- eda_01 reads test.csv/test_series.csv too
for f in train.csv train_series.csv test.csv test_series.csv; do
  python -m kaggle competitions download -c rsna-knee-abnormality-detection -f "$f" -p data
done

python eda_01_labels.py            # writes data/lang_guess.csv   <- eda_03 needs this
python eda_03_langid.py            # writes data/lang_detected.csv
python extractor/run_extract.py    # writes data/pseudo_labels.csv  (extractor RETIRED as a
                                   # target source -- IMPROVEMENTS 2f. Still needed for
                                   # extract_states.csv, which 2k flags as unexploited)
```

### Regenerating the derived data `data/` is gitignored, so this is the only record

Competition data is licensed and does not belong in git, which means **none of the tables the
current plan depends on survive a fresh clone.** Rebuild them in this order — total ~2 minutes,
no GPU:

```bash
# 1. targets: the public LLM labels, and the benchmark that chose them (IMPROVEMENTS 2f, 2i-c)
python extractor/bench_public_labels.py --download     # -> data/public_llm_labels/
python -c "
import pandas as pd
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
   'Effusion','Synovitis',\"Baker's\",'Contusion','Fracture']
src='data/public_llm_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv'
tr=pd.read_csv('data/train.csv')
pd.read_csv(src).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID') \
  .reindex(tr.StudyInstanceUID)[L].reset_index().to_csv('data/targets.csv', index=False)
"

# 2. scanner fingerprints -- NO Kaggle run needed, the probe author published the headers
mkdir -p data/external
kaggle kernels output zhukovoleksiy/rsna-metadata-probe -p data/external
mv data/external/headers.parquet data/external/dicom_headers_zhukovoleksiy.parquet
python pipeline/site_fingerprint.py    # expect 265 fingerprints, top 20 = 45.5%

# 3. both fold sets. Build BOTH: the gap between them is the site-leakage number (2j)
python fusion/folds.py --out data/folds.csv
python fusion/folds.py --group-by site --out data/folds_site.csv
```

`pipeline/site_fingerprint.py` prints a warning if the fingerprint count leaves 200–340. **Take
it seriously** — §2j records a version of that file which silently put 24% of the corpus in one
group while reporting a healthy count.

**Run `eda_01` before `eda_03`.** `eda_03_langid.py` reads `data/lang_guess.csv` to diff the
lingua result against the old heuristic, and `eda_01_labels.py` is the only thing that writes
it. Skipping straight to `eda_03` fails with `FileNotFoundError`.

**On a fresh clone, re-key the hand labels before trusting any evaluation** —
`labeling/model_labels.csv` joins on `item_id`, which is assigned *positionally* after a
shuffle (`sample_for_labeling.py:70-71`) and is not bound to `StudyInstanceUID`. If `lingua`
resolves to a different version, the language strata shift, the shuffle lands differently, and
those 86 labels silently describe the wrong studies. Hence the exact pin in `requirements.txt`.

```bash
python labeling/sample_for_labeling.py     # regenerates labeling_sample.csv
python labeling/rekey_labels.py --check    # fails loudly if the mapping drifted
```

Whole text pipeline runs in **6 s** on CPU (measured on an M5, 2026-08-07).

Then compare every label source, which needs no GPU either:

```bash
python extractor/compare_methods.py --fetch   # pulls the public weak labels into data/
python extractor/compare_methods.py           # the table above
```

## Kaggle-side work

**Read the competition's code, never its description.** Every finding on 2026-08-10 came from
pulling the notebook itself; three claims in "what is wrong" §6 had been written from the public
description and two of them were wrong — including "it ensembles two backbones", and the belief
that 0.891 was a *training* score when `find_weights()` makes it inference from 20 published
members in 74 s.

```bash
python -m kaggle kernels pull raahimnawaz/rsna-knee-lb-baseline-fork -p /tmp/fork -m
```

Re-pull it rather than trusting any summary in these documents, including this one.


The images are 570 GB and stay on Kaggle. The scripts run there in this order — the audits come
first, because if laterality were unrecoverable then four of the twelve labels would be
unreliable and that changes what the feature cache is worth building against. **Both audits are
done** (`FINDINGS.md` §6); the cache is not.

```
notebooks/kaggle_01_dicom_audit.py           -> /kaggle/working/dicom_audit.json        [done]
notebooks/kaggle_01b_patients_laterality.py  -> /kaggle/working/laterality_check.json   [done]
notebooks/kaggle_02_dinov2_cache.py          -> /kaggle/working/features  (publish as a Dataset)
```

**A bad Kaggle GPU draw is the single biggest way to lose a session here.** Kaggle assigns a
Tesla P100 on roughly four of five draws and `accelerator` in `kernel-metadata.json` does not
reliably override it; its PyTorch dropped Pascal, so a P100 cannot run this at all — while
`torch.cuda.is_available()` returns True on one. `pick_device()` checks compute capability
against the installed wheel's arch list in the first seconds of `main()` and refuses, so a bad
draw costs seconds and re-running is a viable strategy rather than a way to burn the weekly
quota. Run `python pipeline/preprocess.py` for the fingerprint; the guard prints the GPU it drew.

**Watch the PROBE line.** `kaggle_02` prints measured throughput at 25 / 100 / 400 series with
the implied hours for the shard, and warns when the rate is near single-worker. Three earlier
attempts burned 21 h, then two ~1 h sessions, then 9 h before anyone could tell the pool had
stopped parallelising. If that warning fires, kill the session.

The cache is the reason local work is **no longer text-only**. Frozen DINOv2 embeddings are
~2.4 GB for the whole corpus, so the §3.3 fusion head — slice transformer, attention pool,
series-type embedding, series attention — trains on a laptop in minutes per experiment. The
backbone is the same one everyone else has; the fusion layer, the labels, and external data are
where the edge has to come from.

`kaggle_02` shards via `SHARD` / `N_SHARDS` and skips finished studies on restart, because
decode is the bottleneck and this will not complete in one session.

## Training (the M5 is enough)

Two machines, one pipeline. Pixels never leave Kaggle; the laptop only ever sees vectors.

```
Kaggle   kaggle_01 audit  ->  kaggle_02 cache  ->  publish Dataset   [~2.4 GB]
M5       download  ->  fusion/folds.py  ->  fusion/train.py          [MPS, minutes/experiment]
Kaggle   upload fold*.pt  ->  kaggle_03_submit.py  ->  submission.csv
```

```bash
python fusion/dataset.py                              # self-test: shapes, masks, degenerate studies
python notebooks/kaggle_02_dinov2_cache.py --self-test # scheduler, resume, spawn pool. No DICOMs
python fusion/train.py --synthetic                    # whole loop on random features. No cache
python fusion/folds.py                                # writes data/folds.csv
python fusion/train.py --features data/features
python fusion/train.py --features data/features --limit 500   # fast iteration; keeps all gold
```

### Building the cache locally (the route that actually worked)

```bash
# 1. pixels: one NIfTI per series, ~178 GB zipped / ~386 GB extracted. Delete each zip as you go
python -m kaggle datasets download -d davidadekanmi/rsna-knee-nifti-part1 -p data/nifti --unzip

# 2. prove the repackaging matches the DICOMs BEFORE building anything (5 checks)
python pipeline/validate_nifti.py --geometry data/series_geometry.csv --thumbs data/series_thumbs.npz

# 3. build. IMG_SIZE feeds the fingerprint, so 224 and 518 caches cannot be confused
IMG_SIZE=224 caffeinate -dimsu python pipeline/build_cache_local.py --validated --out data/features_224
caffeinate -dimsu python pipeline/build_cache_local.py --validated --out data/features
```

Measured on a 16 GB M5: **1,062 studies/h at 224** (~4.2 h for the corpus), ~26 h at 518. The
`--validated` flag is a deliberate speed bump — without it the builder warns on every run that
checks 4/4b have not been re-earned for the current corpus revision.

**The NIfTI mirror is incomplete.** Parts 1–12 and 16–18 exist; 13–15 return 403. That is
**81.7% of studies** and **47 of 58 gold**. Re-check periodically — parts have appeared twice.

`kaggle_02 --self-test` is the one that guards a Kaggle session rather than a laptop run. It
drives the whole scheduling loop on synthetic series — prefetch window, per-study completion
accounting, the resume path, the throughput probe — and it constructs the **real spawn pool** and
asserts it fans out across workers, pins each to one thread, and does not re-search the mount on
re-import. Every one of those has cost a session at least once.

**`--synthetic` is not a toy.** It emits the exact shapes, dtypes, masks and edge cases the real
cache produces — single-series studies, unknown series types, minimum slice counts — so every
consumer is testable before a DICOM has been decoded. On random features it scores macro **0.518**
on the 58 gold studies. That is the point: chance is the correct answer, and anything meaningfully
above it would mean a leak.

**The M5 laptop is sufficient — no Studio, no 980 Ti.** The fusion head is 3.7M parameters over
cached vectors, so the binding constraint is RAM for the cache, not the accelerator. Measured on a
16 GB M5: **2.18 GB peak** with the full cache, the model and optimizer steps all live. The 980 Ti's
6 GB would fit too, but Maxwell has no tensor cores and PyTorch has deprecated sm_52, and it cannot
help with the one genuinely heavy step — the frozen backbone — because that runs on Kaggle either
way. Everything here is fp32 on purpose: 3.7M parameters gain nothing from mixed precision.

**Do not containerise the training.** Docker Desktop on macOS has no Metal passthrough — containers
run in a Linux VM and there is no Apple-silicon `--gpus`. A dockerised run trains on CPU. torch MPS
needs native macOS. Reproducibility comes from the pins in `requirements.txt` plus the preprocessing
fingerprint below.

### Preprocessing parity — the failure with no symptom

`pipeline/preprocess.py` is the single definition of DICOM → model input, imported by the cache
builder *and* the submission notebook. If those two ever disagree, the model is fed a distribution
it never trained on and simply scores badly — no traceback, no warning, and it would surface only
as an unexplained CV/LB gap. So every constant that changes a feature value is hashed into
`PREPROCESS_VERSION`, written into the cache manifest, and asserted by `kaggle_03` before it reads
a single study.

The manifest has to *travel with the weights* for that assert to mean anything: `kaggle_02` writes
`_shard*.json` beside the feature cache, `fusion/train.py` copies it to `manifest.json` beside
`fold*.pt`, and `kaggle_03` refuses to run at all if it cannot find one. A submission that cannot
prove which preprocessing built its heads is the exact failure this section is about, so it exits
rather than warning.

```bash
python pipeline/preprocess.py    # prints the manifest, including the fingerprint
```

Change `IMG_SIZE`, `TARGET_MM`, `FOV_MM`, `SLICES_PER_SERIES`, the model, or the canonical side, and
the fingerprint changes and the existing cache is invalid. That is the intended behaviour.

### What the folds can and cannot do

`fusion/folds.py` groups on a hash of the report text, because **`train.csv` has no patient
column** — the only same-patient signal available is that 150 studies share a report with another
(bilateral knees or follow-ups). It is a floor, not a solution: two studies on one patient with
genuinely different reports still leak and we cannot detect it.

Gold is scored **pooled out-of-fold, never per fold**. 58 gold studies over 5 folds is 8–16 each,
and MCL lands at zero positives in two of them.


## Guards that exist because they were paid for

- **K17 — a `--synthetic` smoke run once overwrote the real result directory and the guard passed
  it.** `--out` now resolves to `fusion/runs_synthetic` under `--synthetic`, and synthetic runs
  write a self-marking manifest, so a submission of pure noise cannot inherit a real fingerprint.
- **Per-epoch checkpointing + `--resume` (§2u)** are required infrastructure on a machine that
  sleeps mid-run. The `--resume` config guard includes the labels path and `n_train`, so it
  refuses to resume across a label change or a grown corpus. **A resumed run recovers but does not
  reproduce** — the dataset RNG restarts — so don't A/B a resumed fold at the third decimal.
- **K16 is resolved by measurement** — `data/slice_direction_resolved.csv`, 8,048 sagittal series,
  50.4% reversed, cross-validated 21/21 against an independent instrument. It gates `sag_med` /
  `sag_lat` only. ⛔ **The old rider here — "protocol tiles sit at depth 0.5 where reversal maps
  the middle slice to itself" — is MEASURED FALSE (§3y-2, 08-18): 28.4% of studies get different
  protocol pixels under the flip, and ~⅔ of those are a genuinely different tile, not a channel
  permutation. `GROUP=3` has no fixed point under reversal.**
  **It is not a header rule and it is our repair for a problem we created by converting to NIfTI
  (§3i-5)** — not an edge over the fork, whose own `order_slices` already sorts correctly.
- **`fusion/pilkwang_model.py --check`** fingerprints all 20 members in ~2 min and is **the first
  thing to run if anything looks wrong**, because it separates a model problem from a pixel one.
- **Pre-register the decision rule, in the file, in writing, every time.** `crop_ab.py` fixed its
  pooling rule before any AUC existed. The rule did not rescue the arm — and choosing it afterwards
  would have produced a +0.0010 "gain" that was pure artefact (§3k).

## The failure mode this project keeps repeating

**The instrument entangled with the thing it measures.** Five instances now — §2d, §2i (void at
6.0σ), §2o (a live bug), §2s, and §3l. Every one was caught *after* the run; §3l was caught after
a fortnight of them.

**Before any A/B: state the reference and show it is neutral to both arms.** It costs minutes.

The second repeat is cheaper to fix and just as costly: **a claim scoped to one source carried as
if it were scoped to the world.** The fork's six slots were "not reconstructable" (true of the
competition metadata, false of what was on disk, §3h-1); the pixels "must live on Kaggle" (an
inference from 570 GB, never a measurement); the public ceiling "is reached" (true for one day,
§3l-1). See `IMPROVEMENTS.md` and the memory note *check-whats-free-first*.
