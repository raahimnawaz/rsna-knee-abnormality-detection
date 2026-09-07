# Archived

One-off experiments and closed-route arms. Kept for provenance; **not maintained, not imported.**
Every route these tested is recorded in `../../DECISIONS.md`.

| file | route | verdict |
|---|---|---|
| `crop_ab.py`, `crop_neutrality_test.py` | crop geometry | settled — §R/§W, 140 mm is the answer |
| `harmonise_test.py` | ComBat-style site harmonisation | CLOSED, −0.013 to −0.032 |
| `site_prior_test.py` | site priors | CLOSED |
| `fold_scheme_test.py` | fold construction | CLOSED, site-grouped folds win |
| `blend_test.py`, `selection_optimism.py` | blend weight fitting / selection bias | CLOSED, −0.0000 and §3b |
| `instrument_test.py`, `error_analysis.py` | scorer diagnostics | superseded by `score_oof` / `score_gold` |
| `pixel_fidelity.py` | NIfTI decode fidelity probe | moot — the NIfTI path is abandoned (§W) |
| `band_ab.py` | fork's members on our pixels | moot for the same reason, and never run |
| `tonylica_arm.py` | tonylica arm | CLOSED, too weak |
| `train_cnn.py` | Workstream C, RadImageNet R50 | CLOSED |
| `dinov3_*.py` | §3t DINOv3 arm | CLOSED, 0.8025, did not earn a slot |
