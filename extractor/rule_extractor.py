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


def iter_spans(hay: str, needle: str, word_start: bool = False):
    """Yield (start, end) for every match of a stem. The one search implementation.

    A stem may opt into word-start matching by carrying a leading '^', regardless of the
    flag. Needed where a stem is a suffix of an unrelated word: English '^chondral' must not
    fire inside 'subchondral' (the bone *under* the cartilage -- subchondral fracture,
    contusion and bone island are not OA) or 'osteochondral'. Same class of bug as R5
    ('normal' inside 'abnormal').
    """
    if needle.startswith("^"):
        needle, word_start = needle[1:], True
    p = hay.find(needle)
    while p != -1:
        if not word_start or _is_word_start(hay, p):
            yield p, p + len(needle)
        p = hay.find(needle, p + 1)


def find_term(hay: str, needle: str, word_start: bool) -> bool:
    """Substring search; word_start=True for CUES only.

    Finding stems must stay substring-matched ('artroz' inside 'gonartroza'), but cues
    must not ('normal' inside 'abnormal' would invert the finding).
    """
    return next(iter_spans(hay, needle, word_start), None) is not None


class RuleExtractor:
    # A compartment is a side adjective plus a structure belonging to exactly one
    # compartment, so these two keys are satisfied by the pair as well as by an explicit
    # phrase. See labeling/compartment_patch.py and IMPROVEMENTS.md 2.2.
    DERIVED = {"_compartment_medial": "_side_medial",
               "_compartment_lateral": "_side_lateral"}

    # Max characters between the side adjective and the structure it must modify. Measured,
    # not guessed, and chosen before looking at gold: over the 299 studies the derived rule
    # newly resolves, the gap is 3 at the median and 22 at p90, then a long tail out to 188
    # that is ~80% false positives -- 'Patellofemoral compartment cartilage: ... medial
    # patellar facet' (the medial PATELLAR facet is in the patellofemoral compartment, not
    # the medial tibiofemoral one), impaction fractures, meniscal degeneration. 25 sits past
    # the elbow and keeps 90.6%.
    SIDE_WINDOW = 25

    def __init__(self, glossary_path: Path):
        g = json.loads(Path(glossary_path).read_text(encoding="utf-8"))
        self.F = {k: {l: [norm(t) for t in ts] for l, ts in v.items()}
                  for k, v in g["findings"].items()}
        self.NEG = {l: [norm(t) for t in ts] for l, ts in g["cues"]["negation"].items()}
        self.UNC = {l: [norm(t) for t in ts] for l, ts in g["cues"]["uncertainty"].items()}
        self.MIN = {l: [norm(t) for t in ts]
                    for l, ts in g["cues"].get("minimal", {}).items()}

    def _near(self, clause: str, side_key: str, struct_key: str, lang: str) -> bool:
        """True when a side adjective sits within SIDE_WINDOW chars of a structure.

        Bare co-occurrence in a clause is too loose for compartment attribution: the side
        has to actually modify the structure, not merely share a sentence with it.

        Both keys must hold plain stems. A '~' conjunction has no single span, so it would
        silently never match here -- the failure mode this repo keeps rediscovering.
        """
        sides = [s for t in self.F.get(side_key, {}).get(lang, [])
                 for s in iter_spans(clause, t)]
        if not sides:
            return False
        structs = [s for t in self.F.get(struct_key, {}).get(lang, [])
                   for s in iter_spans(clause, t)]
        # Gap between two spans, 0 when they touch or overlap, order-independent.
        return any(max(0, max(s0, t0) - min(s1, t1)) <= self.SIDE_WINDOW
                   for s0, s1 in sides for t0, t1 in structs)

    def _has(self, clause: str, key: str, lang: str) -> bool:
        """A term containing '~' is a conjunction: every part must appear in the clause.

        Needed because some languages localise a finding as a generic word plus a
        qualifier that sits elsewhere in the sentence -- Bulgarian writes Baker's cyst as
        'поплитеална киста' and never 'Бейкер', while bare 'киста' matches ganglion and
        meniscal cysts too. 'киста~поплитеал' expresses that without a bespoke rule.

        The two compartment keys are additionally DERIVED: reports name the compartment far
        more often through its anatomy ('medialen Femurcondyle', 'Lateral eklem aralığında')
        than through the formal phrase, which in 6 of 9 languages never appears at all. See
        labeling/compartment_patch.py and IMPROVEMENTS.md 2.2.
        """
        if key in self.DERIVED and self._near(clause, self.DERIVED[key],
                                              "_compartment_struct", lang):
            return True
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
