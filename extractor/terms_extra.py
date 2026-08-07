"""Glossary additions + two correctness fixes found while designing the extractor.

FIX 1 - Bulgarian 'цялост' (integrity) was listed as a NEGATION cue. It appears in both
        'със запазена цялост' (preserved -> negative) and 'нарушена цялост' (disrupted ->
        POSITIVE for a tear). As a bare cue it inverts every tear it touches. Removed;
        the qualified forms carry the meaning.

FIX 2 - English 'normal' as a negation cue matches inside 'ABnormal'. Cue matching now
        requires a word-start boundary (handled in rule_extractor.norm_find and in the
        labeler JS). Finding STEMS deliberately keep substring matching, because
        'artroz' must still fire inside Croatian 'gonartroza'.

ADDITIONS - unqualified meniscus terms (reports often say just 'menisci are normal') and
        tricompartmental phrases, both needed to resolve labels the side-specific terms miss.
"""
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

P = PROJ / "labeling" / "glossary.json"
d = json.loads(P.read_text(encoding="utf-8"))

# FIX 1
bg = d["cues"]["negation"]["bulgarian"]
if "цялост" in bg:
    bg.remove("цялост")
    d["cues"]["negation"]["bulgarian"] = bg + ["запазена цялост", "запазен интегритет"]

# ADDITIONS
d["findings"]["_meniscus_generic"] = {
    "english":   ["meniscus", "menisci", "meniscal"],
    "spanish":   ["menisco", "meniscos", "meniscal"],
    "turkish":   ["menisküs", "menisk"],
    "croatian":  ["menisk", "meniska", "menisci"],
    "greek":     ["μηνίσκ"],
    "german":    ["meniskus", "menisken"],
    "bulgarian": ["менискус", "мениск"],
    "dutch":     ["meniscus", "menisci"],
    "french":    ["ménisque", "menisque", "ménisques"],
}
d["findings"]["_tricompartmental"] = {
    "english":   ["tricompartmental", "all three compartments", "three compartments",
                  "global osteoarthr", "generalised osteoarthr", "generalized osteoarthr"],
    "spanish":   ["tricompartimental", "tres compartimentos", "los tres compartimentos"],
    "turkish":   ["üç kompartman", "trikompartman"],
    "croatian":  ["tri odjeljka", "trokompartmentaln", "sva tri odjeljka"],
    "greek":     ["και των τριών", "τριών διαμερισμάτων"],
    "german":    ["pangonarthrose", "alle drei kompartimente", "trikompartimentell"],
    "bulgarian": ["и трите", "трите отдела", "трикомпартментал"],
    "dutch":     ["tricompartimenteel", "alle drie de compartimenten"],
    "french":    ["tricompartimental", "trois compartiments", "les trois compartiments"],
}

P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("glossary patched:")
print("  bulgarian negation cues ->", d["cues"]["negation"]["bulgarian"])
print("  added _meniscus_generic, _tricompartmental for",
      len(d["findings"]["_meniscus_generic"]), "languages")
