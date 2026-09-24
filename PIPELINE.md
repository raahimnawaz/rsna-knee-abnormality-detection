# The build plan — 2026-09-07 → 2026-10-22

**45 days. Reassessed 2026-09-06 against a live board pull, and the target changed.**

## The number that breaks the old plan

`DECISIONS.md` and `README.md` both said *"prize line 0.940 ≈ rank 100."* The board was pulled
live on 2026-09-06 (3,243 teams, `kaggle competitions leaderboard -d`):

| | old read `08-12` | **live `09-06`** |
|---|--:|--:|
| top | 0.946 | **0.9540** |
| **10th — the last prize** | 0.934 | **0.9500** |
| rank 100 | — | 0.9410 |
| free public kernel | 0.936 | **0.9360**, 506 teams on it exactly |
| us | 0.891 → 0.914 | 0.914, **rank ~1389** |

**The prize line moved +0.016 in 25 days and is 0.950, not 0.940.** Eleven teams are already
there. 877 are at or above the free 0.936.

**Now price this plan's own best case.** `MODEL.md` says that if slice coverage and the per-label
head both land, one model reaches **~0.90 gold-58**. At §3v's +0.039 offset that is **LB ~0.939 —
rank ~125.** Not a prize. And to actually reach 0.950 the model needs gold ~0.911, which is
**above the best public label table** (steven_v4, 0.8927): it would have to beat its own teacher
by ~0.018, while `§3p` measures us 0.05 *below* it. That is a 0.07 swing, in 45 days, on one
laptop.

**So the accuracy track is not reachable on the evidence, and the plan should stop pretending
otherwise.** What follows is built on that.

---

## The two tracks, and why they share a pipeline

$77,000 total: ten accuracy prizes ($9,000–$5,000) and **three efficiency prizes
($7,000 / $6,000 / $5,000)**. `PLAN.md` §6.1 verified the formula from the competition page:

    Efficiency = (maxAUC − AUC) / (maxAUC − 0.500) + runtime_seconds / 32400      [minimize]

At the live `maxAUC` of 0.954, **0.001 AUC ≈ 71 seconds**. Worked with live numbers:

| entry | AUC | runtime | efficiency |
|---|--:|--:|--:|
| **0.941 + tuned I/O** | 0.941 | 15 min | **0.0564** |
| **free 0.936 + tuned I/O** | 0.936 | 12 min | **0.0619** |
| 0.936, hard-trimmed | 0.930 | 6 min | 0.0640 |
| 10th place, tuned | 0.950 | 30 min | 0.0644 |
| free 0.936, as published | 0.936 | 120 min | 0.2619 |
| top team, heavy ensemble | 0.954 | 5 h | 0.5556 |

**A model we can download today, run lean, beats a prize-winning accuracy model run heavy.** Three
prizes instead of ten — and the field stacked 877-deep at 0.936 is stacked there on *accuracy*,
mostly running an unmodified public notebook.

The two tracks share everything up to the final submission: a strong single model is the
efficiency entry, and the same model inside a public blend is the accuracy entry. Rules §2.2.a
allows **two final submissions**, so both can be selected.

✅ **VERIFIED 2026-09-24 on the live Evaluation page:** the formula stands (Benchmark = the all-0.5
sample submission = 0.5); GPU notebooks are allowed; only the two SELECTED finals are eligible, and they
must beat the benchmark on the private leaderboard. The note below is kept as written.

⚠️ **Verify before committing to this.** The formula above is an 08-10 read of the competition
page, and this project has retracted five beliefs about the outside world. **S0 includes
re-reading it**, plus confirming how the efficiency track selects among the two finals. A stale
formula here is the same class of error as the 0.940 prize line.

---

## Standing rules — these are what stop the circling

1. **Every stage ends in an artifact** — a Dataset, a checkpoint, a submission. Not a finding.
2. **One gate per stage, written before the run. Fail → execute the written fallback.**
3. **Ship every week.** A submission every 7 days minimum.
4. **Re-pull the board weekly.** It moved +0.016 in 25 days while the plan quoted a dead number.
5. **Check what is already public before building it.** On 09-06 this plan proposed building a
   severity extractor that was already sitting in `data/external/kernel_sources/` and scored
   *worse* than the labels we had. One `kaggle datasets list` also found four label tables the
   survey had missed. **That check costs minutes and it has now paid twice.**

---

## S0 — bank 0.936, and read the efficiency page `Sep 7 · 1 day · 1 submission` ⭐

The highest-value action available, and it does double duty: rank 1389 → ~372, **and it is the
entry ticket for the efficiency track**, which needs a strong model before it needs a fast one.

Fork the public 0.936 notebook, run unmodified, submit the **kernel** not the file. Grep the log
to confirm every arm loaded. **Record the kernel's wall-clock runtime** — it is the S1 baseline
and nobody has ever written it down.

Same day, on the competition website: **re-read the efficiency formula and prize terms**, and
**post the MRNet/OAI question** — `REFERENCE.md` §1.3 has had it drafted and unsent for 27 days.
It costs one post, it is the only route to image-derived supervision, and the answer takes days.

**GATE** LB ≥ 0.930 and a recorded runtime. **FALLBACK** keep 0.914 and continue.

---

## S1 — profile the runtime `Sep 8–10 · local · free`

**Nothing gets optimised before it is measured.** §6.3 already retracted one runtime belief:
"3.1 ms/slice ⇒ GPU-bound" timed decode on files it had already opened and missed that the real
cost is per-file **open** latency. §3k left the fork's members runnable locally with verified
fidelity, so slice-count, window-count and resolution trade-offs price **offline** instead of at
2 h a submission.

**Artifact:** a per-stage second count for the banked kernel — decode, feature extraction, head,
I/O — and an offline harness that converts a proposed change into Δefficiency.

**GATE** the profile accounts for ≥90% of measured wall time. **FALLBACK** if it does not, the
harness is wrong and S2 cannot be trusted; fix it before spending a submission.

---

## S2 — the efficiency build `Sep 10–25 · local + ~6 h Kaggle` ⭐ PRIMARY

Levers ranked in §6.3 and priced in S1's units: slice count, backbone size, batching, decode path.
**Do not sacrifice meaningful accuracy for speed** — §6.2's table shows 0.845 → 0.800 costs 0.129
units against a whole runtime budget of ~1.0.

**GATE** efficiency ≤ **0.065** on our own measurement, i.e. beating "10th place, tuned" above.
**FALLBACK** if the banked kernel cannot get under 30 min without losing >0.005 AUC, this base
cannot win the track; fall back to S4's model and re-price.

---

## S3 — labels `any time · 1 day · $0` — demoted, deliberately

⛔ **The week-long severity build is cancelled.** `DATA.md` §0 has the evidence: the frontier
already ships that extractor and it scores **0.8617 against our incumbent's 0.8873**; source
weighting converges to the best single member because the sources are one read repeated
(steven_v4 ↔ v2, **r = 0.995**); and `§3p`/D4 says labels are not the binding constraint anyway.
**No API spend.**

What is left is a day of free work:

    - pull the four label tables the 08-10 survey missed   (bench_external.py --download)
    - LEAK-CHECK them: two have gold baked in and score a fake 1.0000
    - take the frontier's __conf / __npos / __nneg as the `mentioned` mask, free
    - hand per-cell confidence weighting to S4 as a pre-registered experiment

**GATE** paired ≥ +0.010 vs `data/targets.csv` on gold-58, 95% CI excluding 0, ≤3 candidates
scored. **FALLBACK** keep `targets.csv`. **Do not spend a second day here.**

---

## S4 — the model `Sep 12 – Oct 5 · ~3 h Kaggle per experiment`

`MODEL.md` is the spec and D2 is real: 18 slices all at `depth 0.5` against the frontier's 64
across 6–94%. This is the only work that raises **both** tracks — a better single model is a
better efficiency entry and a better accuracy entry.

| # | change | gate |
|---|---|---|
| 1+2 | 64-slice coverage **and** per-label attention, shipped together | Lat. Meniscus and MCL **+0.05 each**; macro **+0.02** from the head |
| 3 | 140 mm crop | measured at +0.027 (§W) — fold in, do not re-test |
| 4 | per-cell confidence weighting from S3 | paired ≥ +0.01 gold-58, or drop |
| 5 | OrthoFoundation-L, fine-tuned not frozen | parity with the DINOv2 arm, or park |

**Honest expectation: ~0.88–0.90 gold-58 single model ≈ LB 0.92–0.94 ≈ rank 125–400.** That is a
good model and a strong efficiency entry. **It is not a top-10 accuracy finish, and this plan no
longer claims it is.**

---

## S5 — pixels `Sep 12–20 · CHECK BEFORE BUILDING`

D3 stands: our NIfTI path costs 0.085 and is abandoned. ⛔ **Amended 2026-09-23** (`DECISIONS.md`): the
0.085 is real but the NIfTI files are not its cause. The fork's weights score the same on them, and the
port saw one centre window per series. `RUNBOOK.md` §2b measures it. **The 458 GB local NIfTI corpus was
deleted on 2026-09-07**, so "rebuild tiles" is no longer 8 GPU-hours — it is a ~178 GB
re-download first. That raises the bar on the public-dataset check below, it does not lower it. But **before spending 8 h of Kaggle GPU
rebuilding tiles**, check the two public preprocessed pixel datasets found on 09-06 —
`barun2104/rsna-knee-mri-processed-3d-volumes` (21,238 downloads) and
`alenic/rsna-knee-abnormality-detection-jpeg-224x224`. Neither carries a description, so read the
files, not the title. If either is geometrically sound, S5 costs a download instead of 8 GPU-hours.

**GATE** DINOv2-small fold 0 ≥ 0.83 on the 691-study window, whichever source is used.
**FALLBACK** under 0.80, stop building pixels and run the efficiency track on the public base.

---

## S6 — buffer and two finals `Oct 13–22`

No new arms after Oct 12. **Select two finals: one lean (efficiency), one blended (accuracy).**
Kernel hardening, the ~2 h re-run margin, writeup. **Reserve 3 days for the submission failing.**

## Budget

| resource | available | planned |
|---|--:|--:|
| Kaggle GPU | ~195 h (30/wk) | ~25 h |
| submissions | ~97 (15/wk) | ~12 |
| M5 / MPS | unlimited | S1, S2, S3, scoring |
| API spend | — | **$0** (was ~$180) |

**Resource-rich, time-poor, and now aimed at a prize the arithmetic actually supports.**
