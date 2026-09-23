# RUNBOOK — read this first, then `PLAN-C6.md`

**Purpose: any session can pick this up cold and know the next command to type.** `PLAN-C6.md`
says *why*; this says *what to run, what to expect, and what to do with the answer.*

> ⛔ **Read [`PREMORTEM.md`](PREMORTEM.md) before Stage A.** Four defects in the plan are already
> known: Stage C is priced off a 6-tile benchmark and written for 54 windows (**5–10× over, and
> over the 12 h session cap**); Gate B's bar is built on `runs_port` 0.7323, which §3y states is
> **not a valid control**; the noise floor was never measured, so both paired gates may be
> unreadable at one fold; and no stage closes the teacher→truth gap, which caps the whole plan at
> ~0.8985. Stage 0 is unaffected — run it. Do not fund Stage A/B/C on the prices in `PLAN-C6.md`.

> ⛔ **D3 was amended on 2026-09-23** (`DECISIONS.md`). The NIfTI files are not what cost 0.085:
> the fork's own weights score the same on them as on their DICOMs (§3m). What differed is the
> input: our port trained on **one 3-slice window per series, at the centre**, while the fork trains
> on **12 slices across 20–80%**. **Before funding Stage A, run the sampling test (§2b).** It is the
> one run that says whether our own training reaches public strength when fed public-style input.

Last updated **2026-09-07**; D3 amended and §2b added **2026-09-23**. If today is much later than that, re-run §0.3 before trusting any
claim here about the outside world — five beliefs about it have already expired in this project,
one after 24 hours.

---

## 0. State of the world

### 0.1 What exists right now

| file | status | verified how |
|---|---|---|
| `PLAN-C6.md` | current plan, Stages 0–D | — |
| `notebooks/build_stage0_kernel.py` | **ready to push** | smoke-tested against a synthetic fixture |
| `notebooks/stage0_fixture.py` | reproducible smoke test | r recovered to 0.01 on all 3 planes |
| `pipeline/slots64.py` | **provisional — Stage 0 may overturn it** | `python pipeline/slots64.py` self-test |
| `fusion/perlabel_head.py` | ready, unused until Stage B | `python fusion/perlabel_head.py` self-test |
| `fusion/train_port.py` | has `OrthoNet`, never run | ⚠️ carries the C-PRE bug, see §4 |
| `notebooks/build_sampling_test_kernel.py` | **ready to push — §2b** | whole path smoke-tested off Kaggle on a synthetic DICOM fixture |
| `fusion/contract_audit.py` | fixed 09-23: `slices`/`band` read off the path `train_port` trains on | — |

### 0.2 What is NOT done

- **Nothing has touched a GPU.** No arm has been trained on a C6 cache. There is no C6 number.
- The local corpus was deleted 2026-09-07. **This repo is the only copy of the project.**
- `data/` does not exist. Anything reading `data/*.csv` will fail until a stage recreates it.

### 0.3 Facts with an expiry date

Re-measure before citing. Last pulled **2026-09-07**:

- Leaderboard: **3,289 teams**, top **0.954**, 10th (prize line) **0.950**, ours **0.914** @ rank 1446.
- Public notebook tier sits **0.917–0.922**; leaders at 0.942+ publish nothing.
- Public-source survey: last full survey **2026-08-10** and it was already 27 days stale by 09-06,
  having missed four label tables. **Re-run `kaggle datasets list -s "rsna knee"`.**

---

## 1. Prerequisites — one blocker, and it is not code

### 1.1 Kaggle credentials `THE ONLY THING BLOCKING STAGE 0`

`~/.kaggle/` currently holds an `access_token` (38 bytes) but **no `kaggle.json`**, which is what
the API needs.

```bash
# kaggle.com -> your avatar -> Settings -> API -> "Create New Token" -> downloads kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 1.2 A Python env with the `kaggle` package

The project venv was deleted with the corpus. Any 3.12 env works; Stage 0 needs `kaggle` only to
*push*, and nothing else locally.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -q kaggle
.venv/bin/python -m kaggle competitions list -s "rsna knee"   # proves auth works
```

### 1.3 What Stage 0 does NOT need

Stage 0 declares `dataset_sources: []` and `enable_internet: false` **deliberately** — it imports
nothing from `raahimnawaz/rsna-knee-code`, so the stale code Dataset is **not** a blocker for it.
That refresh becomes a blocker at Stage A, not before. See §5.1.

---

## 2. STAGE 0 — run this first `CPU only · no GPU quota · minutes`

### 2.1 Smoke it locally (optional, ~10 s, catches arithmetic errors not crashes)

```bash
.venv/bin/pip install -q pydicom pandas numpy
.venv/bin/python notebooks/stage0_fixture.py          # writes /tmp/s0fix/input
python3 - <<'EX' > /tmp/s0fix/work.py
import re, pathlib
print(re.search(r"WORK_SRC = r'''(.*?)'''", pathlib.Path("notebooks/build_stage0_kernel.py").read_text(), re.S).group(1))
EX
KAGGLE_INPUT=/tmp/s0fix/input KAGGLE_WORKING=/tmp/s0fix/out .venv/bin/python -u /tmp/s0fix/work.py
```

**Expected:** `r_median` exactly **10.00** Sagittal, **10.00** Coronal, **8.00** Axial;
`sample_n = 10` of 11 series (the 2-slice series is correctly excluded);
`pct_with_2plus_sagittal = 50.0`, `pct_with_2plus_coronal = 25.0`.
Anything else means the reader changed and the arithmetic is no longer trustworthy.

### 2.2 Smoke it on Kaggle — 20 studies, ALWAYS do this first

```bash
.venv/bin/python notebooks/build_stage0_kernel.py --out /tmp/s0 --limit 20
.venv/bin/python -m kaggle kernels push -p /tmp/s0
.venv/bin/python -m kaggle kernels status raahimnawaz/rsna-knee-c6-stage0
```

Every cache build in this project's history died hours in on something a 20-study run would have
caught. The smoke costs about a minute.

**Check in the log before going further:**
1. `root:` names the real competition mount, not a stray directory.
2. `train_series.csv (N, M) cols=[...]` — `Anatomical_Plane` and `Fluid_Sensitive` are present.
   If they are not, **stop**: §0.3's composition table cannot be built and no plane may be
   inferred by a rule here (`slot_cache.py:175`).
3. `n_errors` is 0 or near it.

### 2.3 Run it in full

```bash
.venv/bin/python notebooks/build_stage0_kernel.py --out /tmp/s0          # no --limit
.venv/bin/python -m kaggle kernels push -p /tmp/s0
# when Complete:
.venv/bin/python -m kaggle kernels output raahimnawaz/rsna-knee-c6-stage0 -p data/stage0/
```

Outputs: `stage0_series.csv` (one row per series), `stage0_spacing.csv` (the 400-series true-spacing
sample), `stage0_summary.json` (the aggregates).

### 2.4 What to do with the answer — decision rules, fixed in advance

**A · `r_median_overall` — the 3D question.** The threshold was set in `PLAN-C6.md` §2 *before*
this number existed. Apply it as written; do not renegotiate it against the number you got.

| result | consequence |
|---|---|
| **r ≥ 4** | isotropic 3D is **contraindicated**. Triad and MedicalNet-class encoders are excluded — they require isotropic 1 mm resampling, which would interpolate data never acquired. A 3D arm, if ever built, uses nnU-Net-style anisotropic kernels `(1,3,3)` with in-plane-only pooling. Deferred to Stage E. |
| **r < 4** | an isotropic 3D arm is admissible; §2's exclusion is void and Triad returns to the table **on anatomy grounds only** — it is still brain/breast/prostate with no MSK evaluation. |

Also compare `true_spacing_mm_median` against `tag_thickness_mm_median`. If they differ, there is
an inter-slice **gap**, and any downstream code using `SliceThickness` as the sampling interval is
wrong. Record the delta in `DATA.md`.

**B · `stack_depth_by_plane` — does `slots64.py` survive?**

| result | consequence |
|---|---|
| sagittal median **≥ 30** | 18 slices over 6–94% is a real sub-sample. Keep the spec. |
| sagittal median **20–30** | 18 slices is most of the stack; `pick_depths` repeat-padding becomes load-bearing. Keep, but check the p10. |
| sagittal median **< 20** | ⚠️ the spec asks for more slices than exist. **Reduce the slot counts** to the p10, or a large share of studies gets padded repeats that teach the encoder nothing. |

**C · `slots64_assumption` — the claim I made without evidence.**
`slots64.py` assumes two usable sagittals and two coronals per study.

| result | consequence |
|---|---|
| `pct_with_2plus_coronal` **≥ 60** | keep `cor_2`. |
| **< 60** | drop `cor_2` and redistribute its 8 slices, or most studies carry a permanently masked slot — 8 slices of zeros per study, paid at every forward pass. |

Same test for `sag_2` against `pct_with_2plus_sagittal`.

**D · Write it down.** Add the three tables to `DATA.md` under a new `## 7. C6 Stage 0` with the
date. **No AUC is produced or quoted at this stage** (§3b, `never-select-on-a-tiny-set`).

### 2.5 Failure modes seen before in this project

| symptom | cause | fix |
|---|---|---|
| kernel "Complete" but no output files | worker died; parent still exited 0 | check the log tail for `exited <rc>` |
| `no competition root under /kaggle/input` | competition not attached | `competition_sources` in `kernel-metadata.json` |
| plausible table, wrong numbers | reader trusted filename order | the fixture catches this — filenames are shuffled UIDs |
| queue wait then instant import error | stale code Dataset | not applicable to Stage 0 (`dataset_sources: []`) |

---

## 2b. THE SAMPLING TEST — why our own models lose `one T4 kernel · ~4–6 GPU-h · smoke on CPU first`

**The question.** Everything trained here scores 0.69–0.77 and everything downloaded 0.85. D3 put
the last 0.0856 of that on the NIfTI files, and the record contradicts it (banner above). This run
changes **one variable**: `train_port.py`, fold 0, every setting as in `runs_port_repro`, but fed
the fork's own pixels (12 slices over 20–80%, built by the fork's own notebook cells from the
competition DICOMs). It is scored **paired** against `runs_port_repro` on the same 691 studies. The
builder's docstring has the full design and the known residue.

```bash
# 1. smoke -- CPU, no GPU quota, 64 + 16 studies, one epoch
.venv/bin/python notebooks/build_sampling_test_kernel.py --out /tmp/st --smoke
.venv/bin/python -m kaggle kernels push -p /tmp/st
.venv/bin/python -m kaggle kernels status raahimnawaz/rsna-knee-sampling-test
```

**Check in the smoke log before going further:**
1. `scratch candidate …: N GB free`. The full run's cache is **~29 GB** and it refuses below **~37 GB**
   free (a 10% margin plus 5 GB). If no candidate has that, the full run refuses at its first cell. Nothing in this repo has measured a
   Kaggle machine's disk, so this line is new information.
2. `partition check: 691/691 control studies are in the rebuilt fold 0`. The smoke runs the real
   check. Anything less means an upstream label table or header parquet changed since 09-06.
   **Stop there**: the paired read would be contaminated.
3. `pixel contract = the members' own: {… 'CACHE_SLICES': 12 …}` and `SMOKE: the whole path ran`.

```bash
# 2. the real run
.venv/bin/python notebooks/build_sampling_test_kernel.py --out /tmp/st
.venv/bin/python -m kaggle kernels push -p /tmp/st
# when Complete:
.venv/bin/python -m kaggle kernels output raahimnawaz/rsna-knee-sampling-test -p fusion/runs_forksample_kaggle/
```

**What to do with the answer. The rule was fixed in the builder's docstring before the run.** The
verdict is in `sampling_test_verdict.json`:

| paired Δ vs `runs_port_repro` | consequence |
|---|---|
| **≥ +0.043** (half the gap) | **The input is the main cause.** Our training reaches public strength on public-style input. C6's coverage direction is right for a reason D3 never gave. Fund Stage A, and build its Gate B control from this run rather than `runs_port` (PREMORTEM §3). |
| **≤ +0.015** | **The input is not the main cause.** What remains is training and evaluation: site-grouped folds, one seed vs four, last epoch vs best-on-holdout, the label table. **Do not fund Stage A on coverage grounds.** |
| between | Both contribute. Next, split the fork's reader from its sampling. |

⚠️ **Do not compare the absolute number with pilkwang's 0.8453.** That figure is a 4-seed mean, and
per the fork's notebook its holdout is not site-grouped and its best epoch is picked on that same
holdout. The bar is our own run.

---

## 3. Standing rules — violating any of these has already cost this project

1. **Never select on gold-58.** n=58, per-label SE 0.04–0.09. Selecting on it claims +0.0137 and
   delivers **−0.0034**. Use `score_oof.py` large-n (n≈681, site-grouped).
2. **The training loss is not evidence.** Workstream C's loss fell monotonically through 36.7% of
   its range and the arm lost by **4.8σ**.
3. **Pre-register every gate before the run.** §3t measured a null, the record wrote it down as a
   refutation, and it cost a genuinely diverse arm. §3y and §3z were pre-registered and survived.
4. **A null is a result.** Record it as a null, not as a refutation.
5. **Read a competitor's code, never its description.** Three claims about `pilkwang` came from its
   description; two were wrong.
6. **Never price a forward pass by parameter count — bench it.** Predicted 14×, measured 6.33×.
7. **All-member gold reads are inflated by +0.1474.** Only fold-resolved numbers compare.
8. **Two scales exist.** gold-47/58 ≈ 0.85–0.89 and LB ≈ 0.91–0.95, offset **+0.039**. Never
   compare across them. Price a future arm's LB gain at ~half its gold gain.
9. **Write geometry into the manifest.** §R happened because `box_mm` was recorded nowhere.
10. **Never benchmark a downloaded label table without the leak check.** `rayanbabur_soft` and
    `yunus_v5_merged` reproduce all 58 gold rows and score a fake **1.0000**.

---

## 4. ⛔ C-PRE — fix before Stage C, or Stage C answers the wrong question

`fusion/train_port.py:274-275` registers ImageNet constants inside `OrthoNet`:

```python
self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
```

and `fusion/orthofoundation_model.py` resolves **no** normalisation at all. Correct for DINOv2; an
**assumption** for OrthoFoundation-L, continued-pretrained on 1.25 M greyscale knee MRIs.

A competitor hit this exact bug swapping in RAD-DINO (`image_mean` 0.5307, `image_std` 0.2583,
native 518 px). Their arm lost from epoch 1, the gap widened every epoch, and **training loss was
lower** — fitting worse, not overfitting. It reads as "medical pretraining does not transfer."
<https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154>

**Required:** resolve `image_mean`/`image_std` from OrthoFoundation's own checkpoint or config. If
it ships none, measure them from the pretraining corpus or fall back to greyscale stats — and
**record which**, because a guessed constant is a confound either way. Assert the C0 control
resolves to byte-identical ImageNet constants so the comparison stays clean.

---

## 5. Stages A–D — the shape, and what each needs first

Full rationale in `PLAN-C6.md`. Gates are binding.

### 5.1 Stage A — the cache `8 kernels · ~9 GPU-h`

**Blocked on:** Stage 0's answer (slot spec may change), **and** refreshing
`raahimnawaz/rsna-knee-code` with `pipeline/slots64.py` + `fusion/perlabel_head.py`:

```bash
.venv/bin/python -m kaggle datasets version -p <code-dir> -m "C6: slots64 + perlabel_head"
```

Before writing a builder from scratch, **evaluate forking the published CPU pixel cache**
(`stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache`) — it persists decoded pixels as sharded
memory-mapped `.npy`. Check its crop and slice-position conventions against `slots64.py` first;
`check-whats-free-first` and §4b both say measure the free thing before building.

**GATE A:** manifest records `fov_mm 140`, `depth_range`, resolved spacing · slot fill ≥ Stage 0's
prediction per slot · render 20 studies at the 6% and 94% positions and confirm real anatomy, not
wrap or air. **No AUC.**

### 5.2 Stage B — do the data fixes pay? `2 kernels · ~14 GPU-h`

Encoder held at dinov2-small so cache and head are the only variables. **B0** = C6 cache + shared
pool; **B1** = C6 cache + per-label head.

**GATE B:** `B0 ≥ 0.7591` (the `runs_port` **0.7323 ± 0.0086** anchor + the crop's own **+0.0268**)
· `B1 − B0 > 0` paired · **Lateral Meniscus and MCL must both rise** vs 0.6422 / 0.6433.
**STOP if B0 < 0.7591** — re-open D2 before spending T4 hours on a 303 M encoder.

### 5.3 Stage C — does knee pretraining pay? `2 kernels · ~15 GPU-h`

**Do §4 first.** **C0** = stock `vit_large_patch16_dinov3`; **C1** = OrthoFoundation-L. Same
architecture, params, RoPE, head, cache, folds, schedule — `orthofoundation_model.py:1` maps Ortho
onto that exact timm model, so the only difference is the knee corpus. This is the ablation §3w-2
could not give, where domain and architecture moved together.

**GATE C:** `C1 − C0 > 0`, paired, fold 0, large-n. A null is a real result.

⚠️ **OrthoFoundation has no LICENSE.** The only grant is the authors' email of **2026-08-18**.
Preserve it — it is the entire claim. Competition §2.5.a carves out incompatibly-licensed
pretrained models, so this is not a shipping blocker.

### 5.4 Stage D — the blend `~6 GPU-h`

Where the model actually is: best single **0.8522**, 3-family blend **0.8932**. Equal-weight
rank-mean over families, **nothing fitted** — §2z measured refitting weights at **−0.0000**.
Members: C6's winner, `ft_b` (0.8522), public RadImageNet folds_v1 (0.8486), pilkwang (0.8516).
Screen on strength first; every arm ever rejected here was genuinely weak (0.6924–0.8025) and
every rejection was correct.

---

## 6. Closed — do not re-derive

- **Weighting label sources by accuracy** — reverse-log `−log(1−acc)` **−0.0005**, `acc^8`
  **−0.0005**, equal rank-mean **−0.0041**. Sharpening converges the blend *to* the best member.
  Cause: the sources are one read repeated (`steven_v4`↔`v2` **r=0.995**, LLM-to-LLM mean 0.819).
  **Re-opens only if a source appears at r < 0.7** — measure correlation before proposing it again.
- **Refitting blend weights** on site-grouped folds — −0.0000.
- **Post-hoc calibration / thresholds / priors** — AUROC is invariant to monotone transforms.
- **Crops as extra TTA windows** — dead on both instruments, pre-registered.
- **A C++ port for the efficiency track** — 93 s per 0.001 AUC; decode is already native.
- The full ledger is `DECISIONS.md`. **Consult it; do not extend it.**

---

## 7. The one-paragraph version

The labels are not the constraint — used as a predictor they read gold **0.8985** (≈ LB 0.938)
while the model reproduces its own training labels at **0.8486**. We are further from the labels we
have than those labels are from truth. The gap is the image path. Its largest measured piece is
**0.085** between our port and the fork on the same instrument, larger than every architecture
delta combined. **But D3's reading of it, "the NIfTI files", was amended on 09-23**: the fork's
weights score the same on those files, and what our port lacked was coverage, one centre window
per series. §2b measures whether that is the 0.085. Stage A is worth its price only if §2b says the
input is the cause, and Stage 0 still measures before anything is designed on it. Beating the
teacher afterwards is possible for one specific reason: **the model sees pixels the report never
described**, which is most of what synovitis is.
