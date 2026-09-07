# The build plan — 2026-09-07 → 2026-10-22

**45 days. One deliverable per stage. Every stage ships something runnable.**

## The decision this plan rests on

**The local NIfTI training path is abandoned for competition purposes.** §W measured it: same
config, same folds, same labels — our pixels 0.7597, their pixels 0.8453, 14.1σ. Three weeks and
five closed divergences did not move it. We stop paying for it.

**Everything moves to Kaggle, against DICOMs, at the field's proven geometry (140 mm / 336 px).**
The M5 keeps what it is genuinely good at: head training on cached features, blending, scoring.

## Standing rules — these are what stop the circling

1. **Every stage ends in an artifact** — a published Dataset, a trained checkpoint, a submission.
   Not a finding.
2. **One gate per stage, written before the run.** Fail → take the written fallback. **Do not
   investigate a gate failure.** The fallback is already chosen; execute it.
3. **Ship every week.** A submission every 7 days minimum, even if nothing improved. Never go
   19 days blind again.
4. **New findings go in `DECISIONS.md` as one line.** No new `§` essays. If a finding does not
   change a stage in this file, it does not get written down at all.

---

## Stage 0 — bank the floor `Sep 7, 1 day, 1 submission`

We sit at 0.914 / rank ~1374 while 0.936 is free. Fix that first.

- Fork the public 0.936 notebook. Run it unmodified. Submit the **kernel**, not the file.
- Verify in the log that every arm loaded before submitting — a refused arm ships the old path.

**GATE:** LB ≥ 0.930. **FALLBACK if under:** keep the 0.914 kernel as the banked submission and
carry on to Stage 1 regardless. This stage is insurance, never the strategy.

**Artifact:** a banked submission at ~0.936.

---

## Stage 1 — our own DICOM tile cache `Sep 8–13, ~8 h Kaggle GPU`

The one thing that has never existed here: a pixel path we control that is not crippled.

- New kernel `notebooks/kaggle_04_dicom_tiles.py`. Read DICOMs directly, **never NIfTI**.
- Geometry copied from the frontier, which is public and proven: `CROP_MM = 140.0`, `IMG = 336`,
  crop in **millimetres via pixel spacing**, then resize. Six protocol slots as in `slot_cache`.
- Shard 8 ways (`build_cache` skips finished studies; one kernel per shard, Kaggle caps at 2
  concurrent GPU sessions).
- Publish as a private Kaggle Dataset `rsna-knee-tiles140`. **This is the asset. Everything
  downstream reuses it and it never gets rebuilt.**

**GATE — the whole plan turns on this one.** Train DINOv2-small on it, fold 0, 30 epochs, the exact
`train_port.py` recipe that scored 0.7597 on NIfTI. On the same 691-study window, same labels:

    >= 0.83  -> the pixel path is fixed. Proceed to Stage 2.
    0.80-0.83 -> partially. Proceed anyway; note the residue and do NOT chase it.
    <  0.80  -> our DICOM decode is broken too.

**FALLBACK if under 0.80:** stop building arms. Go directly to Stage 4 and compete purely as a
blender of public weights. That path is already worth ~0.94 and needs no pixels of our own.

**Artifact:** `rsna-knee-tiles140` Dataset + a fold-0 checkpoint and its score.

---

## Stage 2 — OrthoFoundation-L, the arm nobody else has `Sep 14–20, ~10 h Kaggle GPU`

Every public arm surveyed in §S is DINOv2 / DINOv3 / RadImageNet / CoAtNet. **OrthoFoundation-L is
DINOv3-L continued-pretrained on 1.25 M knee images and appears in no public blend.** Its 0.7710
was measured on the 160 mm cache and is void (§R, §W).

- `--backbone orthofoundation` against `rsna-knee-tiles140`. Fine-tune, not frozen — frozen is what
  scored 0.7710.
- 5 folds. ~7.3 h on a T4 per §C-5's measured bench.
- Cost at inference is **+1.9 min on a 9 h ceiling** — already benchmarked, not a barrier.

**GATE:** parity with the Stage 1 DINOv2 arm on the large-n instrument, paired, same folds.
Below parity → **park it permanently and do not re-open.** This is its second and last funding.

**Artifact:** 5 fold checkpoints + OOF, published as a Dataset.

---

## Stage 3 — compartment heads `Sep 21–30, mostly M5`

§U: Lateral Meniscus has **+0.222** of headroom and Lateral OA **+0.123**, and the lateral head
predicts *medial* truth better than its own. Eight of twelve labels are already near their label
ceiling; these two are not. Macro weights all twelve equally.

- Run §V Stage A first — three forward passes per study, `FULL` / `MASK-LEFT` / `MASK-RIGHT`, plus
  the random-location control. **Timebox: 1 day.** It is label-free and needs no direction bit.
- If the heads are not compartment-specific, add compartment-conditioned pooling: separate
  medial/lateral crops from the same tile, one head each.

**GATE:** **+0.02 on each of Lateral Meniscus and Lateral OA**, large-n, paired, per-label — *not*
macro, which would dilute it to invisibility. Bar set at +0.02 not +0.12 because the report
instrument is attenuated ~3× on exactly these labels (§V).

**FALLBACK:** drop it, keep Stages 1–2, proceed to Stage 4. **Timebox 10 days; do not extend.**

**Artifact:** either a compartment arm or a one-line `DECISIONS.md` entry closing it.

---

## Stage 4 — the blend `Oct 1–12`

Rank-mean over **families**, equal weight, nothing fitted — §9e's pre-registered rule. §2z measured
that re-fitting weights on site-grouped folds is worth −0.0000, so do not fit.

Families, most already on disk:

| family | source | status |
|---|---|---|
| pilkwang (DINOv2, 20 members) | public | held |
| RadImageNet R50 | public | held, NC licence — §5a of the rules carves it out |
| CoAtNet ×4 | `dreaddevelopment/raptor-*` | **public, unused, architecturally novel to us** |
| DINOv2 @ 140 mm | Stage 1 | ours |
| OrthoFoundation-L | Stage 2 | ours, and unique |
| compartment arm | Stage 3 | ours, if it clears |

**GATE per arm:** it enters only if it is at parity or better on the large-n instrument. §4b:
every arm below ~0.83 lost. Screen on **strength** first — `[[diversity-is-not-the-constraint]]`.

**Submit weekly from Oct 1.** Price each arm's LB gain at **~half** its gold gain (§4c-4).

---

## Stage 5 — buffer `Oct 13–22`

No new arms after Oct 12. Final submission selection, kernel hardening, the ~2 h re-run margin,
and the writeup. **Reserve 3 days for the submission itself failing.**

---

## Budget

| resource | available | planned |
|---|--:|--:|
| Kaggle GPU | ~195 h (30/wk) | ~25 h |
| submissions | ~97 (15/wk) | ~15 |
| M5 MPS | unlimited | Stage 3 + all blending |

**We are resource-rich and time-poor.** Every stage that can run on Kaggle should be running while
the M5 does something else. Never leave both idle — that is what today cost.
