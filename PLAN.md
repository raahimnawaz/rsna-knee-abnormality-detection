# PLAN — RSNA Knee, goal 0.97 macro AUROC

Finals **2026-10-22 16:59 PDT**. Branch `labels-small-backbone` (the only work branch; `main` is the 10-03
portfolio snapshot). **This page is the current state**; [`README.md`](README.md) explains the code and how to run
it; [`docs/LOG.md`](docs/LOG.md) holds every past run and the rule written before it. Last updated 2026-10-18 evening.

## Where we are

| submission | public LB | status |
|---|--:|---|
| free public base (heliosli's 0.9576 notebook, per-finding Raptor weights) | 0.958 | the base we build on |
| **B**: the 0.957 recipe + R6 + R7 in one decode + our five teacher folds (`57244451`) | 0.957 | **the main final**, unless the candidate below reads the same tick or higher |
| main candidate: B rebuilt in final mode on our private copies, pinned (`57378289`) | ⏳ | its rule: ≥ 0.957 → it becomes the main final |
| **D**: R6 + R7 + our five consensus folds, no Raptor (`57244659`) | 0.954 | **the OAI-free final**; stands as scored |
| KDC fold 0, Lateral OA probe (`57374913`) | 0.530 | **passed** (AUROC ~0.860; bar 0.529) |
| KDC fold 0, PF OA probe (`57378480`) | ⏳ | bar: shown ≥ 0.530 |

Left this week: ~11 GPU-h (estimate), five submissions a day (reset 17:00 PDT). Our rank at 0.957 was ~825 on 10-16;
0.958 held ~430 teams.

## Where the score is lost

0.97 is the mean of twelve AUROCs. Eight findings sit near 0.966; **four hold the macro down**, and 0.97 needs them
near 0.95:

| finding | our five folds, single-finding probe | why |
|---|--:|---|
| Synovitis | 0.884 | labels: radiologists mark it in 27 of the 58 gold knees; reports mention it in 14–22 % |
| Lateral OA | 0.896 | vision: the reports have it, the model doesn't see it |
| PF OA | 0.896 | vision |
| Lateral Meniscus | 0.932 | both |

**The shortcut (measured 10-18):** every model trained on report labels reads one shared "OA-ness" into all three
OA compartments (Lateral ~ Medial OA ρ 0.82 in our models; radiologists 0.32, OAI's image grades 0.06), so they all
make the same mistakes on the four. A member can only move the base if its errors are its own: at a base of 0.896 a
member at ρ 0.5 must reach ~0.84 alone to add one board tick (+0.012) on its finding; at ρ 0.8 it must match the base
alone. **KDC is the first member of ours to do it:** labels decoupled from Medial OA cut its ρ to ~0.5, and its
Lateral OA probe reads ~0.860. Nothing measured so far reaches 0.97: one tick on one finding is +0.001 on the macro.

## Open decisions (the owner's)

1. **KDC in the main final.** By its rule it joins on Lateral OA (and on PF OA if `57378480` reads ≥ 0.530). The
   weight: the two-score formula at ρ 0.49 against our five folds gives a ~0.37–0.41 share and +0.016–0.020 on the
   finding, but against the 0.958 base the ρ is unmeasured (at 0.7 the gain is +0.004–0.007). Building it in is a
   new final: build, CPU mount test, GPU commit run (~20 min), one submission.
2. **KDC folds 1–2** (~4.5 GPU-h each, two at once, ~9 of the ~11 h left): a five-fold model reads ~+0.003–0.007 above
   one fold (estimate), and two more folds would get part of that.
3. **Which two submissions are the finals** on 10-22: the main final (B, the candidate, or either with KDC) and D.

## Closed (do not re-propose; detail in the LOG)

- **Report-label routes on the four:** K's crops on report labels (folds 0–2 tie the teacher; Lateral OA probe 0.896
  = the base), W4 (same labels, not trained), a bigger backbone (ViT-L −0.011), denser slices (Stage C), pseudo-label
  rounds, LLM re-reads of the reports (tables cap ~0.90 on gold), Qwen labels, training on the 58 gold studies.
- **Ways to fold the shortcut away without new labels:** read-time erasure (≤ ~0.01), the cell screen (boosted trees
  on location-fixed cells of K's crops: 0.68–0.78), cartilage / meniscus measurements from the SKM-TEA segmenter
  (Lateral OA 0.654, PF OA 0.619).
- **OAI images (10-18):** the package is 983 GB, a two-series subset ~35 GB with no axial; it would land by Wednesday
  at best for +0.001–0.004 (estimate). The MOAKS tables stay in `~/oai_data/` for after the competition (their
  mapping code, `labeling/oai_moaks_targets.py`, is in git history at `c6b2c65`).
- **Blending tricks:** weight tuning on the public board, consistency rules between findings, distilling public models
  into ours, R7's mirror pass at inference, the public small CNN, fixed half-crops, a per-site prior.
- Synovitis from Effusion (0.70 on gold); "traumatic cartilage loss counts as OA" (a 6-knee artefact).

## How to judge a change

- **Judge weak-four work on its finding**, with a single-finding probe: +0.03 on one finding is visible there; +0.002
  on the macro is not (one macro score ±0.008).
- **The held-out report reference only says "not broken"**, and only for models trained on the same kind of labels.
  It understated KDC by ~0.15 (0.711 held-out, ~0.860 on the board) against ~0.07 for K: a model built to disagree
  with the reports must be gated on the probe.
- Write the rule before the run, in the LOG; read the result by that rule.

## After the finals: dead code still wired into live files

Left in place this week because the finals and any KDC fold embed these files, and cutting them means rewriting the
trainer and the inference dispatch: the Qwen VLM archs (`fusion/qwen_vlm.py`, the `is_qwen` branches and LoRA flags in
`train_windows.py`, the optional `transformers` requirement), the OrthoFoundation arch (`fusion/orthofoundation_model.py`,
`build_submit_kernel.py`'s default weights), Stage C (`pipeline/slots118.py`, the `stage_c` layout), and W4's flags
(`--focus`, `--loc-bias`, `w4_fixture.py`).

## Rules

- **Ask before spending GPU hours and before a batch of submissions.** A go covers what it named.
- **OAI data never enter this repo, GitHub or Kaggle** (`~/oai_data/` only, mode 700; DUC §3).
- No Claude attribution in this repo's commits. Commit and push after every step.
- Kernels are generated from committed code: commit first, build into a scratch folder, push.
- Two sessions may work this repo at once: pull before editing, commit small.
