# Log — append-only, newest last

The detail behind [`RUNBOOK.md`](../RUNBOOK.md): a pre-registration once it has been read, the numbers, the
diagnosis, and owner Q&A that belongs to no other doc. Date and time every entry (PDT). Never edit an old entry;
correct it with a new one that names the old.

Everything before 2026-09-28 ~20:45 PDT is in [`archive/RUNBOOK-2026-09-28.md`](archive/RUNBOOK-2026-09-28.md).

---

**2026-09-28 ~20:45 — the RUNBOOK was cut to one page** (owner: "do 1"). Why: it had grown to 169 KB (1,729 lines).
Its state, pre-registrations and Q&A were interleaved, and parts contradicted the present. Its closing summary still
said "the labels are not the constraint", which D4 reversed on 09-27. The README still gave the 09-24 state. A cold
session reading top to bottom could act on a stale sentence. The full text is frozen in the archive, unchanged apart
from a header. Also decided: **no "safe" final.** The owner declined locking the public stack as a fallback ("2 is for
cowards"), so both finals aim to win (RUNBOOK, *Open* item 6).

**2026-09-28 ~20:50 — the Friday list is READY: v2 smokes read, build pinned to `b5331ff`.** Both v2 smokes (Kaggle CPU,
from `b5331ff`, pushed ~19:04) read at 20:40. Ortho (`rsna-knee-c7-train-ortho-student-r1` v2): targets
`targets_round1_cnn.csv` sha `75757be8c9b5c8d1`, not `steven_v2`; partition check 691/691 in fold 0; train 3,502 | val
691; strict load, RoPE = theirs; epoch 0 loss **0.9177** (v1: 0.9177); DDP weight checksums identical per rank; 18.8
min; no traceback. CoAtNet (`rsna-knee-c6-train-coatnet-student-r1` v2): the same targets sha; init by its full path
`/kaggle/input/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt` (sha `20d45e07`), 486 tensors strict, with no
6-minute `rglob` (15.0 min vs v1's 22.0); epoch 0 loss **0.6574** (v1: 0.6574); no traceback. Identical epoch-0 losses
across v1 and v2 show the off-by-default `--zoom` left training untouched. **Why pinned:** `train_windows.py` changed
again after `b5331ff` (`--extra-corpus`, `be32fdc`), and the Friday task builds from the working tree. So its build
commands create a detached worktree at `b5331ff` and build from there. Every embedded file is tracked, including the
two `oof_all.csv` controls. Test builds at 20:40, not pushed: both print `from b5331ff`, `sha256 75757be8c9b5c8d1`,
`mode=full` with 7 / 14 epochs, `enable_gpu` true, the right datasets and shards, 0.69 MB each.
