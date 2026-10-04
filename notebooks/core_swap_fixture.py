"""Offline test of the core swap: a student checkpoint through the builder's cells into the public stack's own loading code.

    python notebooks/core_swap_fixture.py [--work <dir>]
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJ / "fusion"), str(PROJ / "pipeline")]
import train_windows as TW  # noqa: E402

PUB_DIR = PROJ / "data" / "public_notebooks" / "bend-the-knee-2fc5a420"
PUB_V5 = PROJ / "data" / "external" / "raptor" / "raptor_ft_coatnet_v5_full_swa.pt"
META = PROJ / "fusion" / "runs_coatnet_raptor_check" / "last_meta.json"   # a real Stage A CoAtNet run's
BUILDER = PROJ / "notebooks" / "build_submit_kernel.py"
KERNEL = "raahimnawaz/fixture-coatnet-student"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def defs(src: str, names: set[str], imports: bool = False) -> str:
    """The top-level definitions (and, optionally, imports) named, as source, from a cell."""
    tree = ast.parse(src)
    keep = []
    for n in tree.body:
        name = getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign)
                                            and isinstance(n.targets[0], ast.Name) else None)
        if name in names or (imports and isinstance(n, (ast.Import, ast.ImportFrom))):
            keep.append(ast.get_source_segment(src, n))
    return "\n\n".join(keep)


def build(out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(BUILDER), "--public-dir", str(PUB_DIR), "--out", str(out), *extra],
                          capture_output=True, text=True)


def refuses(fn, what: str) -> None:
    try:
        fn()
    except RuntimeError as e:
        print(f"  refused, as it must: {what} -- {str(e)[:110]}")
        return
    raise SystemExit(f"FIXTURE FAILED: did not refuse {what}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=None, help="scratch dir (default: a new temp dir)")
    a = ap.parse_args()
    work = a.work or Path(tempfile.mkdtemp(prefix="core_swap_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    torch.manual_seed(0)
    timm_name = TW.ARCHS["coatnet"]["timm"]

    def window_mil():
        import timm
        enc = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool="avg")
        return TW.WindowMIL(enc, 384).eval()

    # ---- 1. a student last.pt, saved the way train_windows saves one
    pub = torch.load(PUB_V5, map_location="cpu", weights_only=False)
    ren = {"clsW": "w", "clsb": "b"}                          # train_windows --init's mapping
    sd = {("encoder." + k[len("backbone."):]) if k.startswith("backbone.") else ren.get(k, k): v
          for k, v in pub["model"].items()}
    student = window_mil()
    student.load_state_dict(sd, strict=True)                  # --init's key set IS WindowMIL's
    with torch.no_grad():                                     # "training": head and one encoder tensor move
        student.w.add_(0.05 * torch.randn_like(student.w))
        student.att[3].weight.add_(0.05 * torch.randn_like(student.att[3].weight))
        next(p for n, p in student.named_parameters() if n.startswith("encoder.")).mul_(1.02)
    run = work / "input" / "fixture-coatnet-student" / "runs_coatnet"
    run.mkdir(parents=True)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in student.state_dict().items()},
               run / "last.pt")
    meta = dict(json.loads(META.read_text()), zoom=0.0)
    (run / "last_meta.json").write_text(json.dumps(meta, indent=1))
    s_sha = sha(run / "last.pt")
    assets = work / "public_assets" / "raptor-knee-maxspan"
    assets.mkdir(parents=True)
    (assets / PUB_V5.name).symlink_to(PUB_V5)
    print(f"1. student last.pt {s_sha[:16]}: {len(sd)} tensors, strict into WindowMIL, fp16, perturbed")

    # ---- 2. the build
    r = build(work / "built", "--core-kernel", KERNEL, "--core-sha", s_sha, "--tag", "fixture")
    if r.returncode:
        raise SystemExit(f"FIXTURE FAILED: the build: {r.stderr[-800:]}")
    print("2. " + r.stdout.strip().replace("\n", "\n   "))
    kmeta = json.loads((work / "built" / "kernel-metadata.json").read_text())
    nb = json.loads((work / "built" / f"{kmeta['id'].split('/')[1]}.ipynb").read_text())
    srcs = ["".join(c["source"]) for c in nb["cells"]]
    pub_nb = json.loads((PUB_DIR / json.loads((PUB_DIR / "kernel-metadata.json").read_text())["code_file"]).read_text())
    i_roots = next(i for i, s in enumerate(srcs) if re.search(r"^ASSET_ROOTS = \[", s, re.M))
    i_core = next(i for i, s in enumerate(srcs) if "Upgraded stack, part 1" in s)
    i_check = next(i for i, s in enumerate(srcs) if "Upgraded stack, part 3" in s)
    ke = srcs[i_check - 1]
    core_file = f"student_core_fixture_{s_sha[:12]}.pt"
    assert kmeta["id"] == "raahimnawaz/rsna-knee-sub-core-fixture" and KERNEL in kmeta["kernel_sources"], kmeta
    assert i_core == i_roots + 1 and i_check == len(srcs) - 2, (i_roots, i_core, i_check, len(srcs))
    assert srcs[-1].startswith("# Publish submission.csv only after"), "the publish cell is not last"
    assert len(srcs) == len(pub_nb["cells"]) + 3 and not any("Submission B, part" in s for s in srcs)
    assert ke.count(core_file) == 2 and PUB_V5.name not in ke, "the swap did not land on exactly two arms"
    changed = [i for i, (x, y) in enumerate(zip([s for k, s in enumerate(srcs) if k not in (0, i_core, i_check)],
                                                ["".join(c["source"]) for c in pub_nb["cells"]])) if x != y]
    assert changed == [len(pub_nb["cells"]) - 2], f"public cells changed beyond cell -2: {changed}"
    print(f"   core cell after the ASSET_ROOTS cell ({i_roots}); check before publish; only cell -2 changed, "
          f"its two core arms now load {core_file}")

    # ---- 3. the CORE cell, as built, against public cell 1's own asset lookup
    ns: dict = {"ASSET_ROOTS": [str(work / "public_assets")]}
    exec(defs(srcs[i_roots], {"_asset_walk", "_asset_find_asset", "_speed_original_asset_walk",
                              "_speed_original_find_asset", "_speed_asset_lock", "_speed_asset_catalogue",
                              "_speed_asset_hits"}, imports=True), ns)
    assert "_speed_original_find_asset" in ns, "cell 1 no longer wraps its lookup: re-read it"
    core_src = srcs[i_core]
    for old, new in (('_CPath("/kaggle/input")', f'_CPath({str(work / "input")!r})'),
                     ('_CPath("/tmp/student_core")', f'_CPath({str(work / "student_core")!r})')):
        assert core_src.count(old) == 1, old
        core_src = core_src.replace(old, new)
    bad_meta = dict(meta, layout="stage_b span [0.02, 0.98] img 384 canon True")
    (run / "last_meta.json").write_text(json.dumps(bad_meta))
    refuses(lambda: exec(core_src, dict(ns, ASSET_ROOTS=list(ns["ASSET_ROOTS"]))), "a Stage B checkpoint")
    (run / "last_meta.json").write_text(json.dumps(meta, indent=1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(core_src, ns)
    print("3. " + buf.getvalue().strip())
    assert ns["_C_MOVED"] > 1e-3, f"relative change {ns['_C_MOVED']}: the perturbation did not survive"

    # ---- 4. the public stack's own load path, on the swapped ARMS
    ke_src = re.search(r"_KE_SRC = r'''(.*?)\n'''", ke, re.S).group(1)
    audit = {"events": []}
    c3 = next(s for s in srcs if "def rsna_strict_load" in s)
    kns: dict = {"_asset_find_asset": ns["_asset_find_asset"],
                 "rsna_event": lambda kind, **d: audit["events"].append({"kind": kind, **d})}
    exec("import hashlib\nfrom pathlib import Path\n" + defs(c3, {"rsna_strict_load", "rsna_sha"}), kns)
    exec(defs(ke_src, {"_SLOTS64", "_SLOTS44", "ARMS", "build_backbone", "RaptorClassifier", "load_model",
                       "find_weight_file"}, imports=True), kns)
    arms = kns["ARMS"]
    assert [x["file"] for x in arms] == [core_file, "raptor_ft_coatnet_v10_full.pt", core_file,
                                         "raptor_ft_coatnet_v8_full_swa.pt"], [x["file"] for x in arms]
    assert all(arms[0][k] == arms[2][k] for k in ("file", "arch", "res", "img", "slots", "span", "k_eval"))
    wpath = kns["find_weight_file"](arms[0]["file"])
    model, res = kns["load_model"](wpath, arms[0]["arch"], arms[0]["res"], "cpu")
    ev = audit["events"][-1]
    assert Path(wpath).resolve() == (work / "student_core" / core_file).resolve() and res == 384
    assert ev["kind"] == "raptor_checkpoint" and ev["sha256"] == ns["_C_OUT_SHA"], ev
    print(f"4. the public load_model: {Path(wpath).name}, strict, res {res}; logged sha256 {ev['sha256'][:16]} = "
          f"the core cell's; maxspan-v5 and its reverse share it")
    events_ok = list(audit["events"])
    # a fresh lookup (cell 1's catalogue and memo empty) without our folder must FAIL, not fall back
    roots, hits = list(ns["ASSET_ROOTS"]), ns["_speed_asset_hits"]
    ns["ASSET_ROOTS"][:] = [r for r in roots if not r.endswith("student_core")]
    ns["_speed_asset_catalogue"], ns["_speed_asset_hits"] = None, {}
    refuses(lambda: kns["find_weight_file"](core_file), "our core file missing from ASSET_ROOTS (no fallback)")
    ns["ASSET_ROOTS"][:] = roots
    ns["_speed_asset_catalogue"], ns["_speed_asset_hits"] = None, hits

    # ---- 5. output parity, with its control
    ours = window_mil()
    ours.load_state_dict(torch.load(run / "last.pt", map_location="cpu", weights_only=True), strict=True)
    public, _ = kns["load_model"](str(PUB_V5), arms[0]["arch"], arms[0]["res"], "cpu")
    x = torch.randint(0, 256, (1, 3, 3, 384, 384), dtype=torch.uint8)
    xn = (x.float() / 255.0 - ours.mean.view(1, 1, 3, 1, 1)) / ours.std.view(1, 1, 3, 1, 1)
    with torch.no_grad():
        lo, lp, lpub = ours(x), model(xn), public(xn)
    d_ours, d_pub = float((lo - lp).abs().max()), float((lp - lpub).abs().max())
    print(f"5. logits, 3 windows x 12 findings: |public code on the swapped core - our WindowMIL| {d_ours:.2e}; "
          f"control |swapped - public weights| {d_pub:.2e}")
    if not (d_ours < 1e-4 and d_pub > 1e-3):
        raise SystemExit(f"FIXTURE FAILED: parity {d_ours:.2e} (want < 1e-4), control {d_pub:.2e} (want > 1e-3)")

    # ---- 6. the CORE_CHECK cell
    check_src = srcs[i_check]
    assert check_src.count('"/kaggle/working/core_receipt.json"') == 1
    check_src = check_src.replace('"/kaggle/working/core_receipt.json"', repr(str(work / "core_receipt.json")))
    other = {"kind": "raptor_checkpoint", "file": "/kaggle/input/raptor-knee-native384/raptor_ft_coatnet_v8_full_swa.pt",
             "sha256": "0" * 64}

    def check(events):
        cns = dict(ns, _RSNA_AUDIT={"events": events})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            exec(check_src, cns)
        return out.getvalue().strip()

    print("6. " + check(events_ok + [other]))
    refuses(lambda: check(events_ok + [dict(other, file=str(PUB_V5))]), "a load of the public v5 file")
    refuses(lambda: check([other]), "a load log without our file")
    refuses(lambda: check([dict(events_ok[-1], sha256="f" * 64)]), "our file name with another sha256")

    # ---- 7. the builder's refusals
    for what, extra in (("--ours-only", ["--ours-only"]), ("a short sha", ["--core-sha", s_sha[:16]]),
                        ("no --tag", ["--tag", ""])):
        args = ["--core-kernel", KERNEL, "--core-sha", s_sha, "--tag", "fixture"]
        for k in range(0, len(extra), 2):
            if extra[k] in args:
                args[args.index(extra[k]) + 1] = extra[k + 1]
            else:
                args.append(extra[k])
        if build(work / "no", *args).returncode == 0:
            raise SystemExit(f"FIXTURE FAILED: the builder accepted {what}")
    if build(work / "no", "--add-family").returncode == 0:
        raise SystemExit("FIXTURE FAILED: the builder accepted --add-family without --core-*")
    r = subprocess.run([sys.executable, str(BUILDER), "--out", str(work / "no"), "--core-kernel", KERNEL,
                        "--core-sha", s_sha, "--tag", "fixture"], capture_output=True, text=True)
    if r.returncode == 0:
        raise SystemExit("FIXTURE FAILED: the builder accepted --core-* without --public-dir")
    fam = build(work / "fam", "--core-kernel", KERNEL, "--core-sha", s_sha, "--tag", "fixture", "--add-family",
                "--weights-kernel", "raahimnawaz/fixture-ortho", "--weights-sha", "a" * 64, "--alone-lb", "0.93")
    if fam.returncode:
        raise SystemExit(f"FIXTURE FAILED: --add-family build: {fam.stderr[-600:]}")
    fmeta = json.loads((work / "fam" / "kernel-metadata.json").read_text())
    fsrc = ["".join(c["source"]) for c in json.loads((work / "fam" / f"{fmeta['id'].split('/')[1]}.ipynb").read_text())["cells"]]
    assert "Submission B, part 1" in fsrc[1] and "Submission B, part 3" in fsrc[-3] and "part 3 -- no submission" in fsrc[-2]
    assert fmeta["kernel_sources"][-2:] == [KERNEL, "raahimnawaz/fixture-ortho"], fmeta["kernel_sources"]
    print("7. the builder refuses --ours-only, a short sha, no --tag, --add-family alone, no --public-dir; "
          "--add-family puts ours first and its blend before the core check")
    print(f"CORE SWAP FIXTURE PASSED ({work})")


if __name__ == "__main__":
    main()
