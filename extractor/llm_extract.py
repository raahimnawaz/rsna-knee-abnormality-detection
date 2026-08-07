"""Method B: LLM extractor over the 9-language report corpus, via the Batch API.

Emits the SAME 5-state vocabulary as the rule extractor (pos/hedged/weak/neg/absent) and
reuses its SCORE constants, so llm_states.csv drops straight into eval_model_labels.py and
diff cleanly against extract_states.csv. Two methods that share a schema but not a designer
is the whole point -- see IMPROVEMENTS.md 2b.

The system prompt encodes the measured error directions from IMPROVEMENTS.md 2b-ii, where
37 of 57 gold disagreements were one-directional threshold mismatches rather than genuine
report/image divergence. Telling the model where gold's thresholds actually sit is the
cheapest available accuracy.

Four modes, in the order you use them:

  --dry-run           build the requests, write the JSONL, print a cost estimate. No API call,
                      no key needed. Inspect data/llm_requests.jsonl before spending anything.
  --submit            create the batch; writes the batch id to data/llm_batch_id.txt
  --status            poll the batch
  --collect           fetch results -> llm_states.csv / llm_pseudo_labels.csv / llm_evidence.csv

Scope flags: --sample restricts to the 303 hand-labelled studies (a few cents, and the only
subset you can actually score); --limit N takes the first N. Default is the full 4,407.
"""
import argparse, json, sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_extractor import SCORE  # single source of truth for the soft targets (2.9)

PROJ = Path(__file__).resolve().parents[1]
D, LAB = PROJ / "data", PROJ / "labeling"
BATCH_ID_FILE = D / "llm_batch_id.txt"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
STATES = ["pos", "hedged", "weak", "neg", "absent"]

MODEL = "claude-opus-5"
EFFORT = "medium"
MAX_TOKENS = 16000

# Rough, for the dry-run estimate only. Non-Latin scripts tokenize worse, and Greek+Bulgarian
# are 12.3% of the corpus. Replace with messages.count_tokens once a key exists.
CHARS_PER_TOKEN = 2.8
EST_OUTPUT_TOKENS = 700          # 12 findings x (state + short evidence), plus thinking
PRICE = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0)}   # $/Mtok in, out


SYSTEM = """\
You extract 12 structured knee-MRI findings from free-text radiology reports.

The corpus is 9 languages -- English, Spanish, Turkish, Croatian, Greek, German, Bulgarian,
Dutch, French -- and 61% of it is not English. Read the report in its own language. Do not
translate first; translation loses the hedging and negation cues that decide the state.

For each of the 12 findings, return exactly one state:

  pos     the report affirms the finding
  hedged  affirmed with uncertainty -- "possible", "suspected", "cannot exclude", "suggestive of"
  weak    affirmed but explicitly minimal/early/mild in a way the report itself downplays.
          Applies mainly to the OA and meniscus findings (degenerative grading language).
  neg     the report explicitly denies it -- "no ACL tear", "menisci intact", "b.o."
  absent  the report does not mention it at all

`neg` and `absent` are different and the difference matters. `neg` means the radiologist looked
and said no. `absent` means silence. Do not collapse them.

The 12 findings:
  ACL               anterior cruciate ligament tear/rupture/injury
  MCL               medial collateral ligament injury
  Medial Meniscus   medial meniscal tear or degeneration
  Lateral Meniscus  lateral meniscal tear or degeneration
  Medial OA         medial compartment osteoarthritis / chondral loss
  Lateral OA        lateral compartment osteoarthritis / chondral loss
  PF OA             patellofemoral osteoarthritis / chondromalacia patellae
  Effusion          joint effusion / fluid in the joint
  Synovitis         synovial thickening, synovitis, plica syndrome
  Baker's           Baker's / popliteal cyst
  Contusion         bone contusion / bone marrow oedema / bone bruise
  Fracture          fracture

CALIBRATION -- these come from measured disagreements against expert image-derived labels, and
they are the most common way this task goes wrong. Follow them even when the report's own
wording pushes the other way:

  Effusion    Do NOT mark pos for an effusion the report calls small, mild, minimal, or trace.
              Those read as negative against the reference standard. Use `weak` for those.
              Reserve `pos` for a moderate or large effusion, or an unqualified one.
  Fracture    Does NOT include osteochondral impaction, insufficiency fracture, subchondral
              fracture, or stress reaction. Those are `absent` for this label unless a true
              cortical fracture is also described.
  Synovitis   Rarely named explicitly and frequently present anyway. If the report describes
              synovial thickening, an inflamed plica, or marked effusion with synovial
              enhancement, mark `hedged` rather than `absent`.

Some studies contain two concatenated reports under one ID (flagged in the text as a bilateral
note). When you see that, extract the union of findings across both, and do not let a negation
in one report scope over the other.

Quote evidence verbatim from the report in its original language. Keep it under 120 characters.
Leave it empty for `absent`. Never invent, translate, or paraphrase evidence."""


def _finding_schema():
    return {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": STATES},
            "evidence": {"type": "string"},
        },
        "required": ["state", "evidence"],
        "additionalProperties": False,
    }


def output_schema():
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {lab: {"$ref": "#/$defs/finding"} for lab in LABELS},
            "required": LABELS,
            "additionalProperties": False,
            "$defs": {"finding": _finding_schema()},
        },
    }


def load_corpus(sample: bool, limit: int | None) -> pd.DataFrame:
    tr = pd.read_csv(D / "train.csv")
    lang = pd.read_csv(D / "lang_detected.csv")
    df = tr[["StudyInstanceUID", "Report"]].merge(
        lang[["StudyInstanceUID", "lang"]], on="StudyInstanceUID", how="left")
    df = df[df.Report.notna() & df.Report.astype(str).str.strip().astype(bool)]

    if sample:
        sp = LAB / "labeling_sample.csv"
        if not sp.exists():
            sys.exit(f"missing {sp.relative_to(PROJ)}\n"
                     f"  regenerate with: python labeling/sample_for_labeling.py")
        keep = set(pd.read_csv(sp).StudyInstanceUID)
        df = df[df.StudyInstanceUID.isin(keep)]
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def build_requests(df: pd.DataFrame, model: str) -> list[dict]:
    """Batch request bodies. The system prompt is one cached block -- it is identical across
    all 4,407 calls, so it is written once and read at ~0.1x thereafter."""
    reqs = []
    for i, row in df.iterrows():
        reqs.append({
            "custom_id": f"r{i:05d}",
            "params": {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "system": [{"type": "text", "text": SYSTEM,
                            "cache_control": {"type": "ephemeral"}}],
                "output_config": {"format": output_schema(), "effort": EFFORT},
                "messages": [{"role": "user", "content":
                              f"Language: {row.lang}\n\nReport:\n{row.Report}"}],
            },
        })
    return reqs


def estimate(df: pd.DataFrame, model: str) -> None:
    n = len(df)
    report_chars = df.Report.astype(str).str.len().sum()
    sys_tok = len(SYSTEM) / CHARS_PER_TOKEN
    in_tok = report_chars / CHARS_PER_TOKEN
    cache_tok = sys_tok * max(n - 1, 0)
    out_tok = EST_OUTPUT_TOKENS * n
    pin, pout = PRICE.get(model, PRICE[MODEL])

    cost = (in_tok * pin + cache_tok * pin * 0.1 + sys_tok * pin * 1.25
            + out_tok * pout) / 1e6
    print(f"\n{'='*66}\nESTIMATE -- {model}, {n:,} reports\n{'='*66}")
    print(f"  report text     {in_tok:>12,.0f} tok")
    print(f"  system cached   {cache_tok:>12,.0f} tok read + {sys_tok:,.0f} written")
    print(f"  output (est)    {out_tok:>12,.0f} tok  @ {EST_OUTPUT_TOKENS}/report")
    print(f"\n  standard        ${cost:>8.2f}")
    print(f"  batch (-50%)    ${cost/2:>8.2f}   <- what --submit costs")
    print("\n  Estimate only: char/token ratio is approximate and output length is a guess.")
    print("  The batch response reports actual usage; trust that over this.")


def cmd_dry_run(args):
    df = load_corpus(args.sample, args.limit)
    reqs = build_requests(df, args.model)
    out = D / "llm_requests.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in reqs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(PROJ)}  ({len(reqs):,} requests)")
    print(f"  contains verbatim report text -- gitignored, do not commit")
    estimate(df, args.model)
    print(f"\nInspect a request:  head -1 {out.relative_to(PROJ)} | python -m json.tool")
    print(f"Then submit:        python extractor/llm_extract.py --submit"
          + (" --sample" if args.sample else ""))


def _client():
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic  (or: pip install -r requirements.txt)")
    return anthropic.Anthropic()


def cmd_submit(args):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    df = load_corpus(args.sample, args.limit)
    reqs = build_requests(df, args.model)
    client = _client()
    batch = client.messages.batches.create(requests=[
        Request(custom_id=r["custom_id"],
                params=MessageCreateParamsNonStreaming(**r["params"])) for r in reqs])

    BATCH_ID_FILE.write_text(batch.id)
    (D / "llm_batch_index.csv").write_text(
        pd.DataFrame({"custom_id": [r["custom_id"] for r in reqs],
                      "StudyInstanceUID": df.StudyInstanceUID,
                      "lang": df.lang}).to_csv(index=False))
    print(f"batch {batch.id}  ({len(reqs):,} requests, status {batch.processing_status})")
    print(f"  id -> {BATCH_ID_FILE.relative_to(PROJ)};  index -> data/llm_batch_index.csv")
    print(f"  poll: python extractor/llm_extract.py --status")


def _batch_id(args) -> str:
    if args.batch_id:
        return args.batch_id
    if BATCH_ID_FILE.exists():
        return BATCH_ID_FILE.read_text().strip()
    sys.exit("no batch id -- pass --batch-id, or run --submit first")


def cmd_status(args):
    b = _client().messages.batches.retrieve(_batch_id(args))
    c = b.request_counts
    print(f"{b.id}: {b.processing_status}")
    print(f"  processing {c.processing}  succeeded {c.succeeded}  errored {c.errored} "
          f"canceled {c.canceled}  expired {c.expired}")
    if b.processing_status == "ended":
        print("  collect: python extractor/llm_extract.py --collect")


def cmd_collect(args):
    idx_path = D / "llm_batch_index.csv"
    if not idx_path.exists():
        sys.exit(f"missing {idx_path.relative_to(PROJ)} -- it is written by --submit")
    idx = pd.read_csv(idx_path).set_index("custom_id")

    client = _client()
    bid = _batch_id(args)
    if client.messages.batches.retrieve(bid).processing_status != "ended":
        sys.exit(f"batch {bid} has not ended yet -- check --status")

    rows, evid, errors = [], [], []
    for res in client.messages.batches.results(bid):
        cid = res.custom_id
        if res.result.type != "succeeded":
            errors.append((cid, res.result.type))
            continue
        msg = res.result.message
        if msg.stop_reason == "refusal":
            errors.append((cid, "refusal"))
            continue
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            errors.append((cid, "no text block"))
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            errors.append((cid, "unparseable json"))
            continue

        meta = idx.loc[cid]
        row = {"StudyInstanceUID": meta.StudyInstanceUID, "lang": meta.lang}
        ev = dict(row)
        for lab in LABELS:
            f = data.get(lab) or {}
            row[lab] = f.get("state", "absent")
            ev[lab] = (f.get("evidence") or "").replace("\n", " ")
        rows.append(row)
        evid.append(ev)

    if not rows:
        sys.exit(f"no usable results ({len(errors)} failures) -- nothing written")

    states = pd.DataFrame(rows)[["StudyInstanceUID", "lang"] + LABELS]
    soft = states.copy()
    for lab in LABELS:
        soft[lab] = states[lab].map(SCORE)

    states.to_csv(D / "llm_states.csv", index=False)
    soft.to_csv(D / "llm_pseudo_labels.csv", index=False)
    pd.DataFrame(evid)[["StudyInstanceUID", "lang"] + LABELS].to_csv(
        D / "llm_evidence.csv", index=False)

    print(f"wrote llm_states.csv / llm_pseudo_labels.csv / llm_evidence.csv "
          f"({len(states):,} studies)")
    if errors:
        print(f"\n{len(errors)} failures:")
        for cid, why in errors[:10]:
            print(f"  {cid}  {why}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more")
    print("\nllm_evidence.csv holds verbatim report text -- gitignored, do not commit.")
    print("Compare against the rules: python extractor/compare_methods.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for flag, fn in (("--dry-run", cmd_dry_run), ("--submit", cmd_submit),
                     ("--status", cmd_status), ("--collect", cmd_collect)):
        g.add_argument(flag, dest="fn", action="store_const", const=fn)
    ap.add_argument("--sample", action="store_true",
                    help="only the 303 hand-labelled studies")
    ap.add_argument("--limit", type=int, help="first N reports (smoke test)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch-id", help="override the stored batch id")
    a = ap.parse_args()
    a.fn(a)
