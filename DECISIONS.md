# Decisions ledger

One line per settled question. **Check the tag before trusting a row.**

| tag | meaning |
|---|---|
| `CLOSED` | measured on a valid instrument. Do not re-derive. |
| `CONFOUNDED-160` | scored from a model trained on `slot_cache` tiles at 0.476 mm/px. **The verdict does not stand.** See §R below. |
| `OPEN` | live |

Evidence for every row is in `docs/archive/IMPROVEMENTS.md` and `docs/archive/PLAN.md` at the
section named. Those files are provenance — **not maintained, not read in full.**

---

## §R. THE RESOLUTION CONFOUND — read this before anything else `2026-09-06`

Every model this project trained on the `slot_cache` path was trained at the **wrong resolution**,
and it was one unset parameter.

`slot_cache.py`'s protocol slots carried `box_mm=None`, so `crop_box` returned the full 457 px
resample grid — **160 mm across 336 px = 0.476 mm/px**. The fork's `manifest.json` is unanimous
across all 20 members that average **0.8398**: `crop_mm=130.0` → **0.387 mm/px**. 23% finer, on
the six millimetre-scale labels this project measured itself losing.

`slot_cache.py:27` had already written down the cost: *"At 160 mm across 336 px a tear line is
about one pixel."*

**What was measured, with the instrument named on each row** (they are not directly comparable —
§9e's routing rule still binds):

| pixel path | crop | mm/px | arm | macro | instrument |
|---|--:|--:|---|--:|---|
| `slot_cache` tiles | 160 mm | 0.476 | `runs_cnn` RadImageNet R50 | 0.6924 | fold-0 OOF, n=691 |
| `slot_cache` tiles | 160 mm | 0.476 | `runs_port` dinov2-small | 0.7323 | fold-0 OOF, n=691 |
| `slot_cache` tiles | 160 mm | 0.476 | `runs_port_lr1` | 0.7358 | fold-0 OOF, n=691 |
| `slot_cache` tiles | 160 mm | 0.476 | `runs_of` OrthoFoundation-L | 0.7710 | large-n, n=4,349 |
| `dinov3_pixels.py` own path | **130 mm** | **0.387** | DINOv3 one model | **0.8025** | gold-47 |
| their DICOM path (shipped OOF) | 130 mm | 0.387 | pilkwang | 0.8469 | large-n, n=4,349 |
| their DICOM path (shipped OOF) | 130 mm | 0.387 | `ft_b` / RadImageNet | 0.8522 / 0.8514 | gold-47 / gold-47 |

**The one local arm that cropped at 130 mm — `dinov3_model.py:58`, `CROP_MM = 130.0` — is the one
local arm that escaped the band.** That is suggestive, not decisive: 0.8025 is a gold-47 number and
the band figures are not, so it is a different instrument and cannot be read as a clean rung.

**What this does and does not license.** It does **not** show any of these arms would reach 0.84;
no run at 130 mm has finished. It shows that *"five encoders, one band, therefore the encoder is
not the lever"* compared five models that shared a defect none of them was tested without. The
conclusion may still be true. It has not been measured.

⚠️ **On re-opening the arms below.** The stop rules were honoured and that discipline stands
(§3u, §3z-3) — a result is not renegotiated because it disappointed. This is the other case: the
**instrument** is now known to be miscalibrated, which is grounds to re-measure independently of
which way the number went. Do not let this become a habit of reopening losses.

---

## CONFOUNDED-160 — verdicts that rest on a 0.476 mm/px model

| # | claim as written | why it does not stand |
|---|---|---|
| §2y | our port earns no ensemble slot, −0.111 at 15.4σ | the port was 160 mm. Already partly reopened by §2y-2 |
| §2w step 4 | closed | reopened by §2y-2, 2026-08-19 |
| §3w-2 | Workstream C refuted at 4.8σ; *our training* is refuted | **both arms were 160 mm**, so the CNN-vs-ViT *paired* delta (+0.0399 ± 0.0084) survives as a relative result; the absolute verdict does not |
| §3y-3 | Stage 0 control 0.7358, Stage 1's bar is 0.7428 | the bar was set from a 160 mm control, so it is the wrong bar |
| §C-5 | OrthoFoundation-L fails its gate at 0.7710 | frozen features from a 160 mm cache. Best-matched encoder available; **re-measure before the arm stays parked** |
| §9e / `5d8f6fb` | five encoders, one band, so the encoder is not the lever | all five shared the defect |
| §3i-3 | our pixels are ≈34% of scrambled, hence the band | separately refuted on its own terms by §3i-6 (0.0165 of perturbation costs 0.0013) |

## CLOSED — valid instrument, do not re-derive

| route | verdict | where |
|---|---|---|
| crops as extra TTA windows (F2-cheap) | dead on both instruments; report −0.0031, gold-47 −0.0038, pre-registered | §3k, §3m |
| blending the fork's shipped `merge_gain` arm | +0.0007, 0/12 labels | §3h-2 |
| K16 from DICOM header rules | 56.9–60.8% vs ~50% chance | §2n, §2m |
| our rule extractor as a label source | last of six, 0.777 vs a free 0.893, 0/12 labels | §2f |
| per-label fusion of public label readers | near-duplicates (\|r\| 0.87–0.95); loses to best single | §2i |
| rank-mean `pilkwang` + `prvsiyan` | prvsiyan already contains pilkwang | §3c |
| re-fitting blend weights on site-grouped folds | −0.0000 | §2z |
| harmonising scanner/site away (ComBat-style) | −0.013 to −0.032 — case mix is real | §3f |
| selecting anything on gold-58 | claims +0.0137, delivers −0.0034 | §3b |
| post-hoc calibration / thresholds / priors | AUC is invariant to per-label monotone transforms | §3a |
| forking prvsiyan for its 0.906 | needs a private dataset; degrades to 0.899 | §3c |
| a C++ port for the efficiency track | 93 s per 0.001 AUC; decode is already native | PLAN §6.1 |
| slice **reversal** as a decode fault | null: +0.0007 at 1.5σ, 13/20 members | §3i-7 |
| the residual's *magnitude* explains the band | refuted: 0.0165 of perturbation costs 0.0013, band gap is 0.076 | §3i-6 |
| tensorised images at submission time | structurally impossible, not merely costly | PLAN §9g |

## Standing cautions — these were paid for

- **A submission costs ~2 h, not 74 s.** ~15 runs/week against a 30 h quota. Batch them. §3e
- **A leaderboard number older than a few days is not evidence.** Five beliefs about the outside
  world have expired here; one lasted 24 hours. Re-survey before any submission decision. §2x, §3l-1
- **Launch long runs under `caffeinate -i`.** ~2 of fold 0's 3.6 h were once spent asleep. §2v
- **Read a competitor's code, never its description.** Three claims about `pilkwang` came from its
  description; two were wrong.
- **Baseline is macro 0.7229 ± 0.0048**, site-grouped report-OOF over 2,612 studies. Anything from
  before 2026-08-10 is inflated by ~0.024 of site leakage. §2j
- **Price a future arm's LB gain at ~half its gold gain.** Gold +0.0135 and large-n +0.0102 at 8.8σ
  both delivered +0.006 shipped. §4c-4
- **Never price a forward pass by parameter count — `bench` it.** 303M/22M predicted 14×; measured
  6.33×, and the absolute cost was +1.9 min on a 9 h ceiling. §C-5
- **`data/tiles336` is `SAGITTAL_LR=0`; anatomical slabs need `SAGITTAL_LR=1`.**
  `assert_caches_compatible()` raises on any run consuming both.
- **RadImageNet's trunk is `CC-BY-NC-SA-4.0`** and is currently shipped knowingly. §4c-3
- **OrthoFoundation has no LICENSE file.** The only grant is the authors' email of 2026-08-18
  (Dingyu Wang / Dong Jiang) confirming competition use. **Preserve that email.** §C-5

## The failure mode this project repeats

A real measurement gets made, then used to answer a question it does not address.

- the training-loss curve, against a ranking gate — `[[loss-curve-is-not-evidence]]`
- the pixel residual's magnitude, against the band gap — §3i-6
- the memorisation probe, whose full dynamic range is 0.0102, against a 0.076 effect — §3i-7
- `crop_mm` read from `pilkwang_pixels.py`, a module `train_port.py` does not import — §2y-2
- and §R: seven runs sharing one defect, read as evidence about encoders

**Before believing a number, name the question it can and cannot answer.**
