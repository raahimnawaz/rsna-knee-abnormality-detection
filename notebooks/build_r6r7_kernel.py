"""Build a submission kernel on R6 + R7 in one decode (bundle raahimnawaz/rsna-knee-r6r7-dual, notebooks/build_r6r7_bundle.py),
with heliosli's worker patched so R6's Baker's state stacker runs beside a second group.

The worker refuses --state-stacker with --models2 because, with two groups, score() returns both groups' fold logits
and apply_state_stacker averaged them all. Two edits make it exact: the refusal goes, and the stacker's z is the mean of
group 1's fold logits only (R6), as it was fitted. w2 is 0 on Baker's, so Baker's equals R6's own output. The worker's
sha256 in SOURCE_SHA is replaced by the patched file's, so the notebook's own source check still holds.

    python notebooks/build_r6r7_kernel.py <repo> <0.9576 notebook dir> <out dir> [build_public_blend_kernel args]
    # B: ... --read --tag r6r7-teach1cv5          (our five teacher folds, the 0.25 / 0.10 weights)
    # D: pass --weights-kernel / --weights-sha of the consensus folds after the extra args, and --drop-cells 2,3,4,5
    #    --final-from ours_sub.csv --drop-dataset nartaa/rsna-knee-publication-swa-weights-20261007 --w-ours 0.35,0.20

Found 10-15: without these edits the worker exits at start-up ("--state-stacker: one model group only") and every
study falls back (kernel rsna-knee-sub-r6r7-teach1cv5 v1, ERROR).
"""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

repo, pub, out = (Path(x) for x in sys.argv[1:4])
nb_path = next(pub.glob("*.ipynb"))
cell = "".join(json.load(open(nb_path))["cells"][6]["source"])
src_line = next(l for l in cell.splitlines() if l.startswith("SOURCES = {"))
sources = ast.literal_eval(src_line[len("SOURCES = "):])
ci = sources["compact_infer.py"]

# the two edits, on the worker's own text
E1_OLD = ("        if models2:\n            raise SystemExit(\"--state-stacker: one model group only (the json is fitted on one "
          "line's z and state read-out)\")\n")
E1_NEW = "        pass   # [ours] two groups allowed: the state stacker reads the group 1 fold logits only (below)\n"
E2_OLD = "        zf = zs.mean(0).double().cpu().numpy()                # the deployed group-1 logit"
E2_NEW = "        zf = zs[:len(models)].mean(0).double().cpu().numpy()  # [ours] group 1 (R6) only: the deployed group-1 logit"
assert ci.count(E1_OLD) == 1 and ci.count(E2_OLD) == 1
ci2 = ci.replace(E1_OLD, E1_NEW).replace(E2_OLD, E2_NEW)
old_sha = hashlib.sha256(ci.encode("utf-8")).hexdigest()
new_sha = hashlib.sha256(ci2.encode("utf-8")).hexdigest()

# the same edits as they appear inside the cell's SOURCES literal (repr text), found by repr of each fragment
def lit(s: str) -> str:
    return repr(s)[1:-1]

patches = []
for o, n in ((E1_OLD, E1_NEW), (E2_OLD, E2_NEW), (old_sha, new_sha)):
    lo, ln = (lit(o), lit(n)) if o != old_sha else (o, n)
    assert cell.count(lo) == 1, (o[:60], cell.count(lo))
    patches.append(f"{lo}=>{ln}")
print("worker sha256", old_sha[:16], "->", new_sha[:16])

shas = ("1c58b2bcfe5adc0168fffba48f38735e5888ce86a3291db3ab6cb96155d93b20,683857fad528ce5cc3518bb8e5be3314bfcfc89ff9d276a587c6c3a038aa5de6,"
        "479cf64b1b5123f4684183e4f048be8b09fba8c34e4b03223aec3e3f97c88d75,4921d0968d3616a90e3ec22948f40d60529f9f1ab3f4ed458b510da2a862269b,"
        "c910af266a41af08042d48da83f2db6e7a43deeb7b6365531c5cab30b0fd4685")
args = [str(repo / ".venv/bin/python"), str(repo / "notebooks/build_public_blend_kernel.py"), "--public-dir", str(pub),
        "--weights-kernel", "raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-cv,raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-f12-cv,"
                            "raahimnawaz/rsna-knee-c7-train-convnext-small-teach1-f34-cv",
        "--weights-sha", shas,
        "--replace-dataset", "heliosli/rsna-knee-r6-frozen-fe055406=raahimnawaz/rsna-knee-r6r7-dual",
        "--cell-patch", "FLIP_TTA = True   # [d957] package E: v4c's flip TTA on R6=>FLIP_TTA = False   # [ours] R6 + R7 in one decode (bundle 0aa1bca3) in place of R6's flip pass",
        "--cell-patch", "BUNDLE_ID = 'fe055406fad6a12f'=>BUNDLE_ID = '0aa1bca3fbde76e6'",
        "--cell-patch", "C_SLUG = 'rsna-knee-r6-frozen-fe055406'=>C_SLUG = 'rsna-knee-r6r7-dual'"]
for p in patches:
    args += ["--cell-patch", p]
args += sys.argv[4:] + ["--out", str(out)]
subprocess.run(args, check=True)
