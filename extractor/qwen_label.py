"""Qwen3.5 reads the reports on this Mac (MLX): one GRADED answer per finding, taken from the
next-token distribution over six letters -- never from generated text.

Why this shape (`DATA.md` §1, §4):
  * graded output is worth +0.086 over bits on the labels' own AUROC, so every finding gets a
    distribution, and the score is its expectation -- ties are rare and the ranking is fine-grained;
  * the official cut-points (`REFERENCE.md` §2) are in the system block verbatim, and the letters
    separate "present but BELOW the threshold" (trace effusion, low-grade sprain) from "absent" and
    from "not mentioned" -- the severity axis every public reader collapses;
  * `P(F)` is the `mentioned` mask §5 asks for, so Synovitis imputation can be revisited without a
    second pass over the corpus.
A different model family from every GPT-read table we train on: `steven_v2` <-> the public Qwen3-8B
reads correlate at rho 0.72 on non-gold rows, against 0.82 LLM-to-LLM (RUNBOOK, 09-26 bench).

Cost: the system block is cached once for the whole corpus, each report is prefilled once, and the
twelve questions run as ONE right-padded batch on a copy of that cache. No decoding at all.

    python extractor/qwen_label.py --limit 40             # smoke: s/report, letter mass, a few rows
    caffeinate -i python extractor/qwen_label.py          # all 4,407; resumable, appends as it goes
    python extractor/qwen_label.py --finalize             # probs.jsonl -> targets.csv + manifest.json

Writes data/qwen_labels/<tag>/: `probs.jsonl` (one row per study: the six letter probabilities per
finding and the letter mass), `targets.csv` (StudyInstanceUID + 12 scores in [0, 1]), `manifest.json`
(model, prompt hash, VALUE, schema version). ⚠️ gold-58 is the arbiter and it is spent in
pre-registered reads only (`RUNBOOK.md`, Qwen block): never iterate this prompt against gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
D = PROJ / "data"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LETTERS = "ABCDEF"
MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
SCHEMA = "qwen-graded-v1"

# Letter -> score. Fixed before any gold read, and the raw probabilities are kept so another
# mapping can be derived later without re-running the corpus. Order: meets > probably meets > on the
# fence > present-but-below > not mentioned > explicitly absent. Silence sits above denial because a
# radiologist who looked and wrote "no effusion" is stronger evidence than one who wrote nothing.
VALUE = {"A": 1.0, "B": 0.75, "C": 0.45, "D": 0.12, "E": 0.0, "F": 0.05}
# Synovitis is mentioned in 12.4% of reports (DATA.md §5): silence says nothing, so its F mass takes
# half the study's Effusion score instead of 0.05 -- steven_v2's documented fill, in graded form.
SYNOVITIS_FILL = 0.5

SYSTEM = """\
You are a musculoskeletal radiologist. You will read one knee MRI report and answer questions about it.

Every study in this dataset was ALSO read from the images by expert musculoskeletal radiologists using \
strict image-based thresholds; borderline or "on the fence" findings were graded NEGATIVE. For each \
question, judge from the report text whether the finding meets that expert POSITIVE criterion.

Reports are in English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch or French. Read each \
report in its own language. Watch negation carefully, including negation placed AFTER the term (Turkish \
"efüzyon izlenmedi" = no effusion) and abbreviations meaning normal ("o.B.", "b.o.", "gb"). A report may \
list the clinical question or history -- that is not a finding.

POSITIVE criteria (official):
- ACL: high-grade partial or complete tear -- discontinuity, or more than 50% of fibres disrupted. \
Below threshold: mild signal change, mucoid degeneration, thickening, sprain without discontinuity, \
low-grade partial tear, intact reconstruction graft.
- MCL: high-grade partial or complete ACUTE tear with disrupted fibres and edema. Below threshold: \
low-grade sprain, periligamentous edema only, chronic or remote changes (thickening, old injury).
- Medial Meniscus: tear of the medial meniscus -- abnormal signal that definitely reaches the articular \
surface, or a morphologic abnormality (truncated, diminutive, absent segment, displaced fragment, \
bucket-handle, flap, root tear). Below threshold: intrasubstance (grade 1-2) degeneration or signal \
not reaching the surface.
- Lateral Meniscus: the same criteria for the lateral meniscus.
- Medial OA: an area of about 1 cm or more of cartilage loss deeper than 50% of cartilage thickness in \
the MEDIAL femorotibial compartment (high-grade or full-thickness chondral loss, grade 3-4 \
chondromalacia, advanced osteoarthritis, bone-on-bone). Below threshold: mild or superficial cartilage \
loss, grade 1-2 chondromalacia, fissuring, a small focal defect, osteophytes alone.
- Lateral OA: the same criteria for the LATERAL femorotibial compartment.
- PF OA: the same criteria for the patellofemoral compartment (patella or femoral trochlea).
- Effusion: MODERATE or LARGE joint effusion distending the joint. Below threshold: trace, minimal, \
small or physiological fluid.
- Synovitis: inflammation AND thickening of the synovial lining (synovitis, synovial thickening or \
hypertrophy).
- Baker's: MODERATE or LARGE popliteal (Baker's) cyst in the typical location. Below threshold: a small \
cyst or a little fluid in the bursa.
- Contusion: bone marrow edema-like signal from an impact injury (bone bruise, contusion), without a \
discrete fracture line at that site. Below threshold: degenerative subchondral edema or cysts.
- Fracture: an ACUTE fracture -- a cortical break or fracture line, including osteochondral and \
impaction fractures. Below threshold: an old or healed fracture, a stress reaction without a line.

Answer every question with ONE letter and nothing else:
A = the report clearly describes this finding at or above the positive threshold
B = probably at or above the threshold (hedged wording, or the severity is implied but not stated)
C = the finding is present but the report gives no severity or acuity -- on the fence
D = the finding is present but clearly BELOW the threshold
E = the report explicitly says it is absent or normal
F = the report does not mention this structure or finding at all"""

QUESTION = "QUESTION: {label}. Which letter (A-F)?"


def prompt_hash() -> str:
    return hashlib.sha256((SYSTEM + QUESTION + json.dumps(VALUE) + MODEL).encode()).hexdigest()[:16]


def load_model(name: str):
    from mlx_lm import load
    model, tok = load(name)
    return model, tok


def render(tok, report: str, label: str) -> list[int]:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'REPORT:\n"""\n{report.strip()}\n"""\n\n' + QUESTION.format(label=label)}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def lcp(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs)
    for i in range(n):
        t = seqs[0][i]
        if any(s[i] != t for s in seqs[1:]):
            return i
    return n


class Reader:
    """Holds the model, the letter token ids and the cached system block."""

    def __init__(self, name: str = MODEL):
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, KVCache
        self.mx, self.ArraysCache, self.KVCache = mx, ArraysCache, KVCache
        self.model, self.tok = load_model(name)
        self.letter_ids = []
        for c in LETTERS:
            ids = self.tok.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit(f"letter {c!r} is not one token: {ids}")
            self.letter_ids.append(ids[0])
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        # The part every prompt shares: two different reports, same label -> common prefix.
        a, b = render(self.tok, "alpha", LABELS[0]), render(self.tok, "omega", LABELS[0])
        self.n_sys = lcp([a, b])
        self.sys_ids = a[: self.n_sys]
        self.sys_cache = self._run(self.sys_ids, None)[1]
        self.n_uncached = 0

    # -- cache plumbing -------------------------------------------------------------------------
    def _fresh(self):
        return self.model.language_model.make_cache()

    def _expand(self, cache, B: int):
        """Copy a batch-1 cache to batch B (new arrays: the source is never written again)."""
        mx, out = self.mx, []
        for c in cache:
            if isinstance(c, self.KVCache):
                n = self.KVCache()
                k, v = c.state
                n.keys, n.values, n.offset = mx.repeat(k, B, 0), mx.repeat(v, B, 0), c.offset
            else:
                n = self.ArraysCache(size=len(c.cache))
                n.cache = [None if s is None else mx.repeat(s, B, 0) for s in c.cache]
            out.append(n)
        return out

    def _run(self, ids: list[int], cache):
        """Prefill `ids` (batch 1) on `cache` (fresh if None); returns (last hidden, cache)."""
        mx = self.mx
        cache = cache if cache is not None else self._fresh()
        h = self.model.language_model.model(mx.array(ids)[None], cache)
        mx.eval(h, *[x for c in cache for x in c.state if x is not None])
        return h[:, -1], cache

    def _logits_at(self, h):
        lm = self.model.language_model
        return lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)

    # -- one report ------------------------------------------------------------------------------
    def read(self, report: str) -> dict:
        mx = self.mx
        prompts = [render(self.tok, report, lab) for lab in LABELS]
        n_rep = lcp(prompts)
        if prompts[0][: self.n_sys] == self.sys_ids:
            _, cache = self._run(prompts[0][self.n_sys: n_rep], self._expand(self.sys_cache, 1))
        else:   # the report's first characters merged across the boundary: no shared block this time
            self.n_uncached += 1
            _, cache = self._run(prompts[0][:n_rep], None)
        tails = [p[n_rep:] for p in prompts]
        L = max(len(t) for t in tails)
        batch = mx.array([t + [self.pad] * (L - len(t)) for t in tails])
        last = mx.array([len(t) - 1 for t in tails])
        # Right padding is safe: every layer is causal (attention and the delta-rule recurrence),
        # and each row is read at its own last real token.
        h = self.model.language_model.model(batch, self._expand(cache, len(LABELS)))
        h = h[mx.arange(len(LABELS)), last]
        logits = self._logits_at(h).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sel = mx.exp(lp[:, mx.array(self.letter_ids)])
        mx.eval(sel)
        P = np.array(sel)
        mass = P.sum(1)
        P = P / mass[:, None]
        return {"probs": P.round(5).tolist(), "mass": mass.round(4).tolist(), "n_tokens": len(prompts[0])}


def score_rows(rows: list[dict]) -> pd.DataFrame:
    """probs.jsonl rows -> one score per finding (the VALUE expectation, Synovitis filled)."""
    v = np.array([VALUE[c] for c in LETTERS])
    out = {"StudyInstanceUID": [r["StudyInstanceUID"] for r in rows]}
    P = np.array([r["probs"] for r in rows])          # [n, 12, 6]
    S = P @ v
    eff, syn = LABELS.index("Effusion"), LABELS.index("Synovitis")
    fF = LETTERS.index("F")
    S[:, syn] += P[:, syn, fF] * (SYNOVITIS_FILL * S[:, eff] - VALUE["F"])
    for j, lab in enumerate(LABELS):
        out[lab] = S[:, j].round(5)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="qwen35_9b_v1")
    ap.add_argument("--limit", type=int, default=0, help="first N studies (smoke)")
    ap.add_argument("--ids", type=Path, help="csv with StudyInstanceUID: read only these")
    ap.add_argument("--finalize", action="store_true", help="probs.jsonl -> targets.csv + manifest.json")
    a = ap.parse_args()

    out = D / "qwen_labels" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "probs.jsonl"
    done = {}
    if jl.exists():
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            done[r["StudyInstanceUID"]] = r
    if done and next(iter(done.values())).get("prompt") != prompt_hash():
        sys.exit(f"{jl} was written with a different prompt/VALUE/model -- use a new --tag")

    tr = pd.read_csv(D / "train.csv", usecols=["StudyInstanceUID", "Report"])
    if a.ids:
        keep = set(pd.read_csv(a.ids).StudyInstanceUID.astype(str))
        tr = tr[tr.StudyInstanceUID.isin(keep)]
    if a.limit:
        tr = tr.head(a.limit)

    if not a.finalize:
        todo = tr[~tr.StudyInstanceUID.isin(done)]
        print(f"{len(tr)} studies, {len(done)} already read, {len(todo)} to go -> {jl}", flush=True)
        if len(todo):
            rd = Reader(a.model)
            print(f"model {a.model} | system block {rd.n_sys} tokens, cached once", flush=True)
            t0, low_mass = time.time(), 0
            with open(jl, "a") as f:
                for i, (uid, rep) in enumerate(zip(todo.StudyInstanceUID, todo.Report.fillna(""))):
                    r = rd.read(rep)
                    r.update(StudyInstanceUID=uid, prompt=prompt_hash())
                    low_mass += int(min(r["mass"]) < 0.5)
                    f.write(json.dumps(r) + "\n")
                    done[uid] = r
                    if (i + 1) % 25 == 0 or i + 1 == len(todo):
                        f.flush()
                        dt = (time.time() - t0) / (i + 1)
                        print(f"  {i + 1}/{len(todo)}  {dt:.2f} s/report  eta {dt * (len(todo) - i - 1) / 60:.0f} min"
                              f"  | reads with letter mass < 0.5: {low_mass}", flush=True)

    rows = [done[u] for u in tr.StudyInstanceUID if u in done]
    t = score_rows(rows)
    t.to_csv(out / "targets.csv", index=False)
    mass = np.array([r["mass"] for r in rows])
    man = {"schema": SCHEMA, "model": a.model, "prompt_hash": prompt_hash(), "value": VALUE,
           "synovitis_fill": SYNOVITIS_FILL, "letters": LETTERS, "n": len(rows),
           "letter_mass_min": float(mass.min()) if len(rows) else None,
           "letter_mass_median": float(np.median(mass)) if len(rows) else None,
           "written": time.strftime("%Y-%m-%d %H:%M")}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"wrote {out / 'targets.csv'} ({len(t)} studies); letter mass min {man['letter_mass_min']}, "
          f"median {man['letter_mass_median']}")


if __name__ == "__main__":
    main()
