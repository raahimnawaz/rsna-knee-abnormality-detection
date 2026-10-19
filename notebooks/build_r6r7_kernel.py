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
    # the main final (10-18): ... --main958 --wheel-pins <json> --tag <t> (final mode: no --read)

Builder-only flags, taken out before the rest goes to build_public_blend_kernel:
  --main958          the free 0.958 base's per-finding Raptor weights (heliosli's blend-mine-in-private-lb notebook,
                     diffed 10-18 against the saved 0.9576: this line and the R6 dataset slug are its only changes) and
                     our private copies of the Raptor weights and of heliosli's code dataset (the notebook finds a mount
                     by slug, so its cells stay unchanged)
  --wheel-pins F     a json {wheel file name: sha256} from the mount test; every final pins the vendored MONAI tree
                     and every wheel before anything is installed or imported (review finding 3, 10-16)
  --draft            no wheel pins: the notebook prints WHEELS_UNPINNED instead. Only for building a mount test
                     (notebooks/build_finals_mounttest.py); never push a draft to the GPU

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

# builder-only flags (docstring); everything else goes to build_public_blend_kernel
extra, main958, draft, pins = [], False, False, None
it = iter(sys.argv[4:])
for x in it:
    if x == "--main958":
        main958 = True
    elif x == "--draft":
        draft = True
    elif x == "--wheel-pins":
        pins = json.loads(Path(next(it)).read_text())
    else:
        extra.append(x)
if (pins is None) == (not draft):
    raise SystemExit("pass exactly one of --wheel-pins <json> (a submission) or --draft (a mount test's source)")

if main958:
    args += ["--cell-patch", "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}=>"
                             "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}",
             "--replace-dataset", "raahimnawaz/rsna-knee-publication-swa-weights-20261007",
             "--replace-dataset", "raahimnawaz/rsna-orthofoundation-fold0-epoch7-20261004"]
    # Raptor's two stages look their checkpoint up in a FIXED list of paths (nartaa's); our copy mounts at
    # /kaggle/input/datasets/raahimnawaz/<slug> on today's Kaggle, which no list holds (found 10-18, before any run).
    # Ours goes last in each list; the stages' own sha256 checks (7e5315da.., 3394fdd7..) still guard the file.
    for tail in ("rsna-knee-oai-original384-s42-pair", "rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224"):
        last = f"    Path('/kaggle/input/datasets/nartaa/{tail}') / _weight_name,\n]"
        ours = (f"    Path('/kaggle/input/datasets/nartaa/{tail}') / _weight_name,\n"
                "    Path('/kaggle/input/datasets/raahimnawaz/rsna-knee-publication-swa-weights-20261007') / _weight_name,"
                "   # [ours] our private copy\n]")
        args += ["--cell-patch", last + "=>" + ours]

# supply-chain pins, after the code dataset is unpacked and its wheels listed, before anything is installed or imported
VENDOR_TREE_SHA = "1dcf1dabb83ab80eec50a5c0abfc407b3ba785bf9f30ae6c0e25da124ef87b05"   # = pipeline/kcache.py's
PIN_AT = "wheels = sorted(str(p) for p in WHEELS.glob('*.whl'))"
PIN_CODE = r'''
# [ours, finals 10-18] supply-chain pins before anything from the code dataset is installed or imported: the vendored
# MONAI tree (heliosli's 10-04 version, hashed as pipeline/kcache.py's tree_sha256) and every wheel in dependencies.zip
def _ours_tree_sha(root):
    _rows = []
    for _f in sorted(Path(root).rglob('*')):
        if _f.is_file() and '__pycache__' not in _f.parts:
            _rows.append(_f.relative_to(root).as_posix() + '\t' + hashlib.sha256(_f.read_bytes()).hexdigest() + '\n')
    return hashlib.sha256(''.join(_rows).encode()).hexdigest(), len(_rows)
_ours_vt, _ours_vn = _ours_tree_sha(CODE / 'training' / 'vendor_runtime' / 'monai060')
assert _ours_vt == '__VT__', ('vendored MONAI tree differs from the pin', _ours_vt, _ours_vn)
_ours_wh = {Path(_w).name: hashlib.sha256(Path(_w).read_bytes()).hexdigest() for _w in wheels}
_OURS_WHEEL_PINS = __PINS__
if _OURS_WHEEL_PINS is None:
    print('WHEELS_UNPINNED', json.dumps(_ours_wh, sort_keys=True), flush=True)
else:
    assert _ours_wh == _OURS_WHEEL_PINS, ('wheels differ from the pins', sorted(set(_ours_wh.items()) ^ set(_OURS_WHEEL_PINS.items())))
print('SUPPLY_CHAIN_OK', json.dumps({'monai_tree': _ours_vt[:16], 'monai_files': _ours_vn, 'wheels': len(_ours_wh),
                                     'wheels_pinned': _OURS_WHEEL_PINS is not None}), flush=True)'''
args += ["--cell-patch", PIN_AT + "=>" + PIN_AT + PIN_CODE.replace("__VT__", VENDOR_TREE_SHA).replace("__PINS__", repr(pins))]

args += extra + ["--out", str(out)]
subprocess.run(args, check=True)
