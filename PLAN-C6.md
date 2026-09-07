# C6 — the architecture fixes, and OrthoFoundation on top of them

**Goal: the best single model this project can build.** Not a prize, not an efficiency entry.
That changes what gets built: the efficiency track rewarded a small fast encoder, and this plan
deliberately spends parameters and time instead.

Written 2026-09-07, after the local corpus was deleted. **Everything here runs on Kaggle**, where
the competition DICOMs mount directly. Nothing needs a local download, and nothing may be built
from a repackaging — see D3 below.

---

## 0. What this fixes, and why each is a measured defect

The record diagnosed four things. C5's OrthoFoundation run fixed **none** of them and is therefore
not evidence about OrthoFoundation.

| # | defect | measured cost | fixed by |
|---|---|--:|---|
| **D3** | pixels are a third-party NIfTI repackaging | **0.085** @ 14.1σ | build on Kaggle from DICOMs, Stage A |
| **§2e** | frozen encoder cannot adapt; everything lands near 0.70 | caps the whole project | `OrthoNet` already unfreezes 6 blocks |
| **D2** | 18 slices, every slot at `depth 0.5` | Lat.Men **0.6422**, MCL **0.6433** vs midline ACL **0.8689** | `slots64.py`, 64 slices over 6–94% |
| **§R** | 160 mm crop → 0.476 mm/px | **+0.0268** @ 4.1σ | `slots64.py`, 140 mm mm-crop |
| **§5** | one shared pool over all 12 labels | never measured | `perlabel_head.py` |

⛔ **D3 is the largest and it is upstream of everything.** A cache built from *any* repackaging —
ours, `alenic/…jpeg-224x224`, or `barun2104/…3d-volumes` — reintroduces it. Those datasets are a
throughput convenience and this plan does not use them. 0.085 is larger than every architecture
delta in the record combined.

⛔ **Coverage and per-label attention ship together or not at all** (`MODEL.md` §5). Adding outer
slices to a shared pool dilutes the midline findings that already work. The gates below are
written so a coverage-only run cannot pass one.

---

## Stage A — the cache `8 kernels, ~1.1 h each`

`notebooks/build_c6_cache_kernel.py --shard N --n-shards 8`

64 slices/study over 6–94% of each stack, five slots, 140 mm mm-crop, 336 px, uint8. Built from
competition DICOMs. Windows are **not** cached — consecutive 2.5D triples share two of three
slices, so caching 54 windows would store every slice three times for no extra information
(95 GB vs 32 GB). The Dataset assembles them.

Sharded eight ways because Kaggle caps kernel output at 20 GB; each shard is ~4 GB and mounts
into the training kernel via `kernel_sources`, so **nothing is ever downloaded**.

**GATE A** — this is a plumbing gate, not a strength gate. No AUC may be quoted from it (§3b,
[[never-select-on-a-tiny-set]]).
1. `manifest.json` records `spec: c6-64slice`, `fov_mm 140`, `depth_range [0.06, 0.94]`.
2. Slot fill rate ≥ the `tiles336` build's, per slot. A slot that silently stops filling would
   read as a coverage gain and be a coverage *loss*.
3. Spot-render 20 studies: the 6% and 94% slices must show real anatomy, not wrap or air.

**STOP** if fill rate drops on any slot — the assignment rule regressed, and a masked slot costs
more than a mis-preferred one.

---

## Stage B — do the architecture fixes pay? `2 kernels × ~7 h`

The encoder is held at the fork's **dinov2-small**, so the only variables are cache + head. This
is the arm the record never ran and it must run *before* OrthoFoundation, or a C6 gain and a
knee-pretraining gain are confounded in exactly the way §3w-2 confounded domain with architecture.

| arm | cache | head | encoder |
|---|---|---|---|
| **B0** control | C6 | **shared** | dinov2-small, 6 blocks |
| **B1** | C6 | **per-label** | dinov2-small, 6 blocks |

Anchor: `runs_port` = **0.7323** ± 0.0086, fold 0, n=681, report-OOF (`score_oof.py`).

**GATE B, pre-registered:**
- **B0 ≥ 0.7591.** That is the anchor plus the crop confound alone (+0.0268). B0 gets the crop and
  the coverage and the correct pixels; if it cannot clear the crop's own value, coverage is not
  paying and D2 is refuted on this data.
- **B1 − B0 > 0**, paired, same folds and seed. This is the §5 claim, measured for the first time.
- **Per-label read:** Lateral Meniscus and MCL must both improve on B1 vs the 0.6422 / 0.6433
  baseline. Macro alone can rise while the two structures D2 named stay flat, which would mean
  the gain came from somewhere else and D2's mechanism is still unproven.

**STOP** if B0 < 0.7591: the cache is not the lever, and OrthoFoundation on top of a cache that
does not pay is a second variable on an unvalidated instrument. Re-open D2 before spending T4
hours on a 303 M encoder.

⛔ **The training loss is not evidence** — Workstream C's loss fell monotonically through 36.7% of
its range and the arm still lost by 4.8σ. Only `score_oof.py` large-n decides these gates.

---

## Stage C — is knee pretraining worth anything? `2 kernels × ~7.3 h`

Only now, and only if Gate B passed. The control is chosen so the comparison is clean:

| arm | encoder | differs from C0 by |
|---|---|---|
| **C0** control | `vit_large_patch16_dinov3`, stock | — |
| **C1** | **OrthoFoundation-L** | **1,251,655 knee images. Nothing else.** |

`fusion/orthofoundation_model.py:1` maps OrthoFoundation-L onto timm's `vit_large_patch16_dinov3`,
so C1 *is* C0 continued-pretrained. Same architecture, same 303 M parameters, same RoPE, same
head, same cache, same folds, same schedule. **This is the ablation §3w-2 could not give**, because
there RadImageNet was domain-pretrained *and* the only CNN, so domain and architecture moved
together.

**GATE C, pre-registered:** **C1 − C0 > 0**, paired, fold 0, `score_oof.py` large-n.
- Positive → knee pretraining pays, and C-5's 0.7710 is confirmed as an artefact of freezing.
- **Null or negative → that is the answer**, and it is worth having. It says a 1.25 M-image
  domain corpus does not transfer to this label set, which no public result reports either way.

⚠️ **A null here is a real result, not a failed experiment.** The record's repeated error was
reading a null on a small instrument as a refutation (§4a). `score_oof.py` large-n at n≈681 is
the instrument for this, **never gold-58** — §3b: selecting on gold-58 claims +0.0137 and delivers
−0.0034.

---

## Budget

Kaggle gives 30 GPU-h/week.

| stage | kernels | GPU-h | week |
|---|--:|--:|--:|
| A cache | 8 | ~9 | 1 |
| B0 + B1 | 2 | ~14 | 1–2 |
| C0 + C1 | 2 | ~15 | 2–3 |

Fold 0 only throughout. Five folds is 5× and buys a tighter interval on a question that fold 0
already answers directionally; spend it only after a gate passes.

---

## What must be true before any of this runs

1. **`raahimnawaz/rsna-knee-code` must be refreshed** with `pipeline/slots64.py` and
   `fusion/perlabel_head.py`. The kernels import from that Dataset; a stale copy fails at import,
   after the queue wait.
2. **Kaggle API credentials.** `~/.kaggle/` currently holds only an `access_token`, no
   `kaggle.json`, so `kaggle kernels push` cannot authenticate.
3. **Smoke each kernel with `--limit 20` before the full shard.** Every cache build in this
   project's history died hours in on something a 20-study run would have caught
   (`normalise_and_resample`'s `torch.quantile` ceiling, the pool restart, mount latency).
