"""Compartment attribution: a compartment is a SIDE plus a compartment-bearing STRUCTURE.

IMPROVEMENTS.md 2.2 -- 1,415 studies land in the `weak` state for Medial OA and 1,582 for
Lateral OA, i.e. OA is named but no compartment is, so all three OA labels get a flat 0.45
that never crosses threshold and contributes nothing to ranking. Both labels sit near 0.71
because of it.

2c diagnosed this as a SCOPE problem and proposed propagating the compartment from an
enclosing section header. Measured against the corpus, that is wrong: in 6 of 9 languages
the formal phrase ('medial compartment') never appears anywhere in the report at all, so
there is nothing to propagate. It is the FINDINGS.md 3 pattern yet again -- the vocabulary
is wrong, not the logic. The compartment is usually named in the SAME clause, through
anatomy the glossary did not know:

  german     'Knorpelirregularitaeten an der medialen Femurcondyle'   condyle, not compartment
  turkish    'Medial tibiofemoral eklem duzeyinde'                    word order flipped
  croatian   'degenerativne promjene FT zgloba'                       abbreviated
  dutch      'Mediaal femorotibiaal gewrichtscompartiment'            different inflection
  turkish    'Lateral eklem araliginda kikirdakta %50den fazla kayip' joint space

So instead of one phrase per language, express the concept: laterality adjective
(`_side_*`) co-occurring with a structure that belongs to exactly one compartment
(`_compartment_struct`). rule_extractor._has() derives the compartment from the pair.

MENISCI ARE DELIBERATELY NOT COMPARTMENT-BEARING. 'Medial meniscus' does sit in the medial
compartment, but a degenerative meniscal tear is not OA, and `_OA_generic` contains
'degenerativ' -- including it would fire Medial OA on every degenerative medial meniscal
tear in the corpus. Cartilage, condyle, plateau, joint space and the femorotibial joint
itself are the compartment-defining structures; the meniscus is not.

Recovers 368 of the 1,389 unresolved weak Medial OA studies (26 -> 394). Spanish barely
moves (+2/213) and that is a true negative: Spanish reports name both compartments together
('cartilagos de los compartimentos femorotibiales') rather than one side.
"""
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
P = PROJ / "labeling" / "glossary.json"
d = json.loads(P.read_text(encoding="utf-8"))
F = d["findings"]

# Laterality adjectives. Stems where the language inflects predictably ('medial' covers
# medialen/mediale/medialer), explicit forms where it does not (Dutch mediaal vs mediale).
# Spanish/French carry the interno/externo convention alongside medial/lateral.
SIDE_MEDIAL = {
    "english":   ["medial"],
    "german":    ["medial"],
    "dutch":     ["mediaal", "mediale"],
    "turkish":   ["medial"],
    "croatian":  ["medijaln"],
    "spanish":   ["medial", "interno", "interna"],
    "french":    ["médial", "interne"],
    "greek":     ["έσω"],
    "bulgarian": ["медиал"],
}
SIDE_LATERAL = {
    "english":   ["lateral"],
    "german":    ["lateral"],
    "dutch":     ["lateraal", "laterale"],
    "turkish":   ["lateral"],
    "croatian":  ["lateraln"],
    "spanish":   ["lateral", "externo", "externa"],
    "french":    ["latéral", "externe"],
    "greek":     ["έξω"],
    "bulgarian": ["латерал"],
}

# Structures that belong to exactly one compartment. No menisci -- see the docstring.
# Croatian 'ft zglob' is kept but bare 'ft ' is NOT: terms are substring-matched, so 'ft '
# would fire inside 'soft ', which appears in the English fragments Croatian reports carry.
COMPARTMENT_STRUCT = {
    "english":   ["compartment", "femorotibial", "tibiofemoral", "femoral condyle",
                  "condyle", "tibial plateau", "plateau", "joint space", "joint line"],
    "german":    ["kompartiment", "femorotibial", "tibiofemoral", "femurcondyl",
                  "femurkondyl", "kondyl", "condyl", "tibiaplateau", "plateau",
                  "gelenkspalt"],
    "dutch":     ["compartiment", "femorotibia", "tibiofemora", "femurcondyl", "condyl",
                  "tibiaplateau", "plateau", "gewrichtsspleet"],
    "turkish":   ["kompartman", "femorotibial", "tibiofemoral", "femoral kondil", "kondil",
                  "plato", "eklem aralığ", "eklem araliğ"],
    "croatian":  ["kompartm", "femorotibijaln", "tibiofemoraln", "ft zglob", "kondil",
                  "plato", "zglobn prostor"],
    "spanish":   ["compartimento", "femorotibial", "tibiofemoral", "cóndilo femoral",
                  "cóndilo", "platillo", "meseta tibial", "espacio articular", "interlínea"],
    "french":    ["compartiment", "fémoro-tibial", "femorotibial", "condyle fémoral",
                  "condyle", "plateau tibial", "plateau", "interligne"],
    "greek":     ["διαμέρισμα", "διαμερίσματ", "μηροκνημιαί", "κόνδυλ", "πλατώ", "μεσάρθρι"],
    "bulgarian": ["компартм", "феморотибиал", "тибиофеморал", "кондил", "плато",
                  "ставна цепка", "ставно пространство"],
}

for key, table in [("_side_medial", SIDE_MEDIAL),
                   ("_side_lateral", SIDE_LATERAL),
                   ("_compartment_struct", COMPARTMENT_STRUCT)]:
    F[key] = {lang: sorted(set(F.get(key, {}).get(lang, []) + terms))
              for lang, terms in table.items()}

# Found by auditing what the rule above newly fires on. English '_OA_generic' carries the
# stem 'chondral', which substring-matches inside 'subchondral' (846 hits) and
# 'osteochondral' (170) -- so subchondral marrow oedema, subchondral insufficiency fracture,
# a subchondral bone island and an osteochondral fracture all read as cartilage pathology.
# 25 of the studies the compartment rule newly resolved rested on nothing else. R5 again
# ('normal' inside 'abnormal'), so use the same remedy: require word-start via '^'.
#
# Checked in all nine languages; English is the only one that needs it. Dutch 'artrose'
# inside 'gonartrose', German 'arthrose' inside 'gonarthrose'/'retropatellararthrose' and
# Bulgarian 'артроза' inside 'гонартроза' are all knee OA -- that is the substring matching
# working as designed. Greek 'χόνδρ' fires inside 'οστεοχονδρινο'/'ενχόνδρωμα' 20 times of
# 64, which is real but too small to fix without measuring what word-start would cost on
# Greek's heavily prefixed compounds. Recorded in IMPROVEMENTS.md 2.11 instead.
F["_OA_generic"]["english"] = ["^chondral" if t == "chondral" else t
                               for t in F["_OA_generic"]["english"]]

P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("added _side_medial / _side_lateral / _compartment_struct for 9 languages")
print("english _OA_generic: 'chondral' -> '^chondral' (word-start; excludes subchondral)")
