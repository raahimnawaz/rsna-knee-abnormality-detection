# PREMORTEM — it is 2026-10-22, C6 failed. What killed it?

Written **2026-09-07**, before Stage 0 ran, against `PLAN-C6.md` and `RUNBOOK.md` as they stand.
Ranked by probability × impact. **Three of these are flaws in the plan itself, not risks to it** —
they are wrong now, and would have been discovered late and expensively.

---

## 1. ⛔ Stage C never ran. It is 5–10× over budget and cannot fit in a Kaggle session

**This is the one that kills it, and it is arithmetic, not judgement.**

`build_of_screen_kernel.py` records the measured cost of OrthoFoundation-L: **16.95 ms/tile,
101.71 ms/study**. Divide: **101.71 / 16.95 = 6 tiles per study.** That benchmark was taken on the
*old* 6-slot representation.

C6 asks for **54 windows per study** — nine times the tiles.

| | measured (6 tiles) | C6 (54 windows) |
|---|--:|--:|
| forward per study | 101.7 ms | **915 ms** |
| training step ≈ 3× forward | ~0.3 s | **~2.75 s** |
| one epoch, ~3,525 train studies | ~18 min | **~2.7 h** |
| 30 epochs (`MODEL.md` §7) | ~9 h | **~80 h / fold** |
| 10 epochs | ~3 h | **~27 h / fold** |

`PLAN-C6.md` budgets Stage C at **~15 GPU-h for two arms.** The real figure is **50–160 GPU-h**.

Two hard limits break before the budget does:
- **A Kaggle session caps at 12 h.** Even 10 epochs at 27 h/fold cannot run as one kernel; it needs
  checkpoint-and-resume across three sessions per arm.
- **The weekly quota is 30 GPU-h.** One arm is a week or more. Stage C needs two.

And memory: `MODEL.md` §7 requires *"batch = studies, not windows — every window of a study must
share a forward pass or the attention is meaningless."* 54 windows through a ViT-L/16 at 336 px is
441 tokens × 54 images in a single forward, with six blocks' activations retained for backward. At
batch 8 studies that is 432 images and will OOM a 16 GB T4. Batch drops to 1–2 studies, which
makes the throughput above optimistic, not pessimistic.

⚠️ **The 80 h figure is itself an extrapolation, and this project has a standing rule against
exactly that:** *"Never price a forward pass by parameter count — bench it. Predicted 14×,
measured 6.33×."* Extrapolating 6 → 54 tiles is the same error in a different variable. **Bench
one epoch at the real window count before believing either number.**

**Fix:** cut windows for the ViT-L arms — 54 → 18–24, *matched* across C0 and C1 so the ablation
stays clean — and re-bench. Accept that Stage C runs on a reduced representation and say so, or
accept it does not run at all.

---

## 2. The calendar never closed

45 days remain. The stages are **sequential and gated**, so slack does not pool.

| stage | plan | realistic |
|---|--:|--:|
| 0 | ~0 h | 1–2 days incl. queue |
| A | 9 h | ~1 week (8 kernels, queued, plus a re-run) |
| B | 14 h | ~1 week |
| C | 15 h | **3–5 weeks at the corrected price** |
| D | 6 h | ~1 week |

That is **6–8 weeks against 45 days**, before counting submissions — which cost **~2 h each against
the same 30 h quota**, so every leaderboard check is taken out of training.

**Fix:** the plan must fit the calendar or lose a stage. Deciding *which* stage to drop is cheaper
now than discovering it in week five.

---

## 3. ⛔ Gate B's bar is built on a control the record itself invalidated

`PLAN-C6.md` sets **B0 ≥ 0.7591 = 0.7323 + 0.0268**. But §3y says, in the repo, in bold:

> `tiles336` was built with `SAGITTAL_LR=0`, so **`runs_port` 0.7323 is NOT a valid control**

The valid-cache number is **0.7358** on `tiles336_lr1` (`contract_audit.py:49-50`), so the bar is
also arithmetically wrong by +0.0035.

The deeper problem is not the constant. **C6 changes the cache, the slot layout, the slice count
and the crop simultaneously.** Neither 0.7323 nor 0.7358 is a like-for-like control for a
5-slot / 64-slice / 140 mm representation. Gate B compares across representations — **the same sin
D3 is criticised for**, committed by the gate that is supposed to enforce rigour.

A gate that cannot fail cleanly does not stop anything. When B0 lands at 0.75, the session will
not know whether the plan failed or the bar was wrong, and will argue rather than stop.

**Fix:** Stage B needs its own control **on the C6 cache** — the old 6-slot / 18-slice / `depth 0.5`
layout, rebuilt through the new pipeline, so the only variable is the pipeline. Then coverage, then
the head. Three arms, not two; more GPU time, but the only version where the number means anything.

---

## 4. ⛔ The noise floor was never measured, and the gates likely cannot resolve their effects

Everything runs **fold 0 only**, n ≈ 681. The plan never measures its own noise floor.

A competitor measured his: **0.0020 macro, on 3 folds and 2,652 studies, same config, different
seed — measured, not assumed.** At one fold and a quarter of the studies, ours is roughly **2×
that, ~0.004–0.005.**

Now the effects the gates test:

| gate | effect | σ at a ~0.005 floor |
|---|---|--:|
| B1 − B0, per-label attention | plausibly +0.005–0.015 | **1–3σ** |
| C1 − C0, knee pretraining | unknown, plausibly +0.010 | **~2σ** |

**Both gates may return numbers too small to read.** And that is precisely the failure the record
already paid for: §3t measured a null on an instrument too small to separate null from refutation,
wrote it down as a refutation, and lost a genuinely diverse arm.

**Fix:** measure the floor first — same config, two seeds, one extra run. It is the cheapest thing
in the plan and every gate depends on it. Pre-register each gate as *effect > 2 × floor*, not as a
bare number.

---

## 5. D3's 0.085 may not be recoverable by rebuilding

> **2026-09-23: worse than suspected.** The fork's weights score the same on our NIfTI files as on
> their DICOMs (§3m), so the repackaging is not the cost. The port trained on one centre window per
> series. `DECISIONS.md` D3 is amended, and `RUNBOOK.md` §2b is the test.

Stage A's whole justification is D3. The ledger records: *"Same config, folds, labels. ours 0.7597,
theirs 0.8453, 14.1σ."* The **model** side is controlled — but *theirs* is a different pixel
pipeline end to end: different decode, different slice selection, different crop, different
resolution. Attributing the entire 0.085 to **the repackaging specifically** is an inference, not a
measurement.

If only a third of it is the repackaging, Stage A recovers ~0.03, Gate B fails, and the plan stops
at its first real test having spent a week of quota.

**Fix:** Stage A is worth doing regardless — it also buys the crop and coverage — but the gate
should not be priced as though all 0.085 returns.

---

## 6. D2 was diagnosed, never tested — and there is public evidence against it

The coverage argument is an inference from per-label AUROC (Lateral Meniscus 0.6422 and MCL 0.6433
against midline ACL 0.8689), not a coverage experiment. **No one has ever run this corpus with more
slices and measured what happened.**

Against it: a competitor measured **3 slices beating 9** by ~4× his noise floor. He states the
result is confounded — with one anchor the code takes the window centre, with three it spreads to
the window edges — so it is not decisive. But it points the wrong way for a 64-slice design, and it
is the only direct evidence anyone has published.

**Fix:** Stage 0.2 tells us whether the slices even exist. Stage B's arms should include a
slice-count contrast, or D2 remains an assumption wearing a measurement's clothes.

---

## 7. The plan has no way to beat the teacher, so "best model" caps at ~0.8985

D4: the labels used as a predictor read gold **0.8985**; the model reproduces its own training
labels at **0.8486**. Stages A–D close the *model→teacher* gap. **None of them closes the
teacher→truth gap**, because none of them touches the labels.

So the best case is approaching 0.8985 ≈ LB 0.938 — a good model, roughly rank 125, and **below the
0.950 prize line**. The user's goal is explicitly the best model rather than a placing, which makes
this worse, not better: the plan's ceiling is its teacher's score.

The one mechanism that exceeds it — **the model sees pixels the report never described**, which is
most of what synovitis is — is described in `DATA.md` §5 and **is not scheduled as a stage.**

**Fix:** add a label stage. It is CPU/API, runs on a different machine, and parallelises with
everything else, so it costs no GPU quota. Emit the `mentioned` mask (impossible to add later
without re-billing all 4,407 reports) and run U-SelfTrained imputation after Stage B.

---

## 8. It was abandoned before Stage D, and nothing had shipped

Every payoff is at the end. Stage 0 produces tables; A produces a cache; B and C produce numbers
that only matter as inputs to D. **There is no point before week five where the project is better
off than it is today** in any visible way.

This project was deleted once on 2026-09-07 and restarted the same day. A four-stage sequential
plan with stop-gates, over 45 days, on a volatile schedule, is fragile to exactly that.

**Fix:** make every stage ship something — a submission, a written finding, a public post. Stage 0
alone answers a question nobody in the competition has published: what the corpus's real anisotropy
and protocol composition are.

---

## 9. C-PRE turned out to be unresolvable, and Stage C was uninterpretable anyway

The fix requires reading `image_mean`/`image_std` from OrthoFoundation's checkpoint. **It ships no
LICENSE at all** — the only grant is the authors' email of 2026-08-18 — so it may well ship no
preprocessing config either. `RUNBOOK.md` §4 says to "measure them from the pretraining corpus"
if absent. **We do not have the pretraining corpus.**

If the constants cannot be resolved, every Stage C number carries a confound whichever way it goes,
and a guess is a confound too.

**Fix:** resolve this **before** Stage A, not before Stage C. It is a download and a file
inspection, costs no GPU, and it determines whether Stage C is worth planning for at all. If the
config is absent, email the authors — they answered once already.

---

## 10. The blend did not survive the new cache `LOWER CONFIDENCE — CHECK, DO NOT ASSUME`

Stage D blends C6's winner with `ft_b`, pilkwang and public RadImageNet. Those arms' OOFs came from
their own fold partitions, some **recovered** rather than known — and §3t's recovery failed its
pre-registered χ² check at 9.91 against a 9.49 critical value.

Rank-mean over families is sound as long as each arm's prediction is genuinely out-of-fold *for
that arm* and all arms are evaluated on a common study set. That is probably fine. But **the 0.8932
was measured under one specific alignment**, and C6 uses `folds_site.csv`. This needs checking
before it is relied on for the plan's largest single number.

---

## What I would change, in order

1. **Resolve C-PRE now** (§9) — free, and it gates whether Stage C exists.
2. **Bench one epoch at the real window count** (§1) — their own standing rule; the 80 h figure is
   an extrapolation and extrapolation is what the rule forbids.
3. **Measure the noise floor** (§4) — one extra run; every gate is unreadable without it.
4. **Rebuild Gate B's control on the C6 cache** (§3) — otherwise the first real gate cannot fail cleanly.
5. **Add the label stage** (§7) — CPU/API, parallel, no quota, and it is the only route past 0.8985.
6. **Then decide which stage to drop** (§2), with real prices instead of the ones in `PLAN-C6.md`.

**The single most likely epitaph:** *the plan was priced on a benchmark taken at 6 tiles per study
and then written to use 54.*
