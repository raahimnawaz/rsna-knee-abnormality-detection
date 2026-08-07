"""Rule-based multilingual report -> 12-label extractor.

Deliberately transparent: every prediction carries the clause that produced it, so
disagreements with the LLM (method B) can be adjudicated by reading, not guessing.

Core model
----------
A finding is POSITIVE when, inside one clause, an ANATOMY term co-occurs with the
PATHOLOGY term that finding requires, and no negation cue is present in that clause.
Anatomy alone is never enough -- 'medial meniscus' appears in every normal report too.
Some findings (effusion, synovitis, Baker's, contusion, fracture) are their own pathology
and need no second term.

Normalisation is byte-for-byte identical to the labeler's JS (NFKD + accent strip +
final-sigma fold), so Greek's MICRO SIGN problem is handled here too. See FINDINGS.md 2.2.
"""
import json, re, unicodedata
from pathlib import Path

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# label -> (anatomy glossary key, list of acceptable pathology keys or None)
SPEC = {
    "ACL":              ("ACL",                   ["_tear"]),
    "MCL":              ("MCL",                   ["_tear"]),
    "Medial Meniscus":  ("Medial Meniscus",       ["_tear"]),
    "Lateral Meniscus": ("Lateral Meniscus",      ["_tear"]),
    "Medial OA":        ("_compartment_medial",   ["_OA_generic"]),
    "Lateral OA":       ("_compartment_lateral",  ["_OA_generic"]),
    "PF OA":            ("PF OA",                 ["_OA_generic"]),
    "Effusion":         ("Effusion",              None),
    "Synovitis":        ("Synovitis",             None),
    "Baker's":          ("Baker's",               None),
    "Contusion":        ("Contusion",             None),
    "Fracture":         ("Fracture",              None),
}

# state -> soft target. Ranking metric, so hedged findings sit between clean and definite.
SCORE = {"pos": 0.95, "hedged": 0.65, "weak": 0.45, "neg": 0.03, "absent": 0.08}
RANK = {"pos": 4, "hedged": 3, "weak": 2, "neg": 1, "absent": 0}

CLAUSE_RE = re.compile(r"[.;\n\r]+|\s[-–—•]\s")


def norm(s: str) -> str:
    """Lowercase + NFKD + strip combining marks + final sigma fold.

    Output length ALWAYS equals input length so match offsets stay usable.
    NFKD is what folds MICRO SIGN (U+00B5) onto GREEK MU (U+03BC).
    """
    out = []
    for ch in s:
        c = unicodedata.normalize("NFKD", ch.lower())
        c = "".join(x for x in c if not unicodedata.combining(x))
        c = c[0] if c else ch.lower()
        out.append("σ" if c == "ς" else c)
    return "".join(out)


def _is_word_start(text: str, pos: int) -> bool:
    return pos == 0 or not text[pos - 1].isalpha()


def find_term(hay: str, needle: str, word_start: bool) -> bool:
    """Substring search; word_start=True for CUES only.

    Finding stems must stay substring-matched ('artroz' inside 'gonartroza'), but cues
    must not ('normal' inside 'abnormal' would invert the finding).
    """
    p = hay.find(needle)
    while p != -1:
        if not word_start or _is_word_start(hay, p):
            return True
        p = hay.find(needle, p + 1)
    return False


class RuleExtractor:
    def __init__(self, glossary_path: Path):
        g = json.loads(Path(glossary_path).read_text(encoding="utf-8"))
        self.F = {k: {l: [norm(t) for t in ts] for l, ts in v.items()}
                  for k, v in g["findings"].items()}
        self.NEG = {l: [norm(t) for t in ts] for l, ts in g["cues"]["negation"].items()}
        self.UNC = {l: [norm(t) for t in ts] for l, ts in g["cues"]["uncertainty"].items()}
        self.MIN = {l: [norm(t) for t in ts]
                    for l, ts in g["cues"].get("minimal", {}).items()}

    def _has(self, clause: str, key: str, lang: str) -> bool:
        """A term containing '~' is a conjunction: every part must appear in the clause.

        Needed because some languages localise a finding as a generic word plus a
        qualifier that sits elsewhere in the sentence -- Bulgarian writes Baker's cyst as
        'поплитеална киста' and never 'Бейкер', while bare 'киста' matches ganglion and
        meniscal cysts too. 'киста~поплитеал' expresses that without a bespoke rule.
        """
        for t in self.F.get(key, {}).get(lang, []):
            if "~" in t:
                if all(find_term(clause, p, False) for p in t.split("~") if p):
                    return True
            elif find_term(clause, t, False):
                return True
        return False

    def _cue(self, clause: str, table: dict, lang: str) -> bool:
        return any(find_term(clause, t, True) for t in table.get(lang, []))

    def extract(self, text: str, lang: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {"states": {L: "absent" for L in LABELS},
                    "scores": {L: SCORE["absent"] for L in LABELS}, "evidence": {}}

        clauses = [c for c in CLAUSE_RE.split(text) if c and c.strip()]
        states = {L: "absent" for L in LABELS}
        evid = {}

        def bump(label, st, raw):
            if RANK[st] > RANK[states[label]]:
                states[label] = st
                evid[label] = raw.strip()[:220]

        for raw in clauses:
            c = norm(raw)
            neg = self._cue(c, self.NEG, lang)
            unc = self._cue(c, self.UNC, lang)
            # severity: 32% of detected effusions are qualified 'trace/minimal/small'.
            # The metric is a ranking metric, so a trace effusion must not tie with a
            # tense one. Downgrade rather than drop -- it is still a real finding.
            mild = self._cue(c, self.MIN, lang)
            positive = "hedged" if (unc or mild) else "pos"

            for L, (anat, paths) in SPEC.items():
                if not self._has(c, anat, lang):
                    continue
                if paths and not any(self._has(c, p, lang) for p in paths):
                    continue
                bump(L, "neg" if neg else positive, raw)

            # --- ambiguity resolution -------------------------------------------------
            # tricompartmental OA names no compartment but implies all three
            if self._has(c, "_tricompartmental", lang) and not neg:
                for L in ("Medial OA", "Lateral OA", "PF OA"):
                    bump(L, "hedged" if unc else "pos", raw)

            # unqualified OA ('gonarthrosis', 'degenerative change') with no compartment
            # named: real evidence of OA, but we cannot say which. Weak on all three.
            if (self._has(c, "_OA_generic", lang) and not neg
                    and not self._has(c, "_compartment_medial", lang)
                    and not self._has(c, "_compartment_lateral", lang)
                    and not self._has(c, "PF OA", lang)):
                for L in ("Medial OA", "Lateral OA", "PF OA"):
                    bump(L, "weak", raw)

            # unqualified meniscal tear with no side named -> weak on both
            if (self._has(c, "_meniscus_generic", lang) and self._has(c, "_tear", lang)
                    and not self._has(c, "Medial Meniscus", lang)
                    and not self._has(c, "Lateral Meniscus", lang)):
                for L in ("Medial Meniscus", "Lateral Meniscus"):
                    bump(L, "neg" if neg else "weak", raw)

        return {"states": states,
                "scores": {L: SCORE[states[L]] for L in LABELS},
                "evidence": evid}
