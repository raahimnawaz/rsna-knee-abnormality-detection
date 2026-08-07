"""Round-1 glossary fixes. Every change is justified by corpus evidence from diagnose.py,
NOT by gold AUC -- with n=58 the per-label CIs are 0.17-0.35 wide and cannot resolve these.
"""
import json
from pathlib import Path

P = Path(r"C:\Users\Raahim\rsna-knee-mri\labeling\glossary.json")
d = json.loads(P.read_text(encoding="utf-8"))
F = d["findings"]

# FIX A -- Greek medial/lateral meniscus were 100% identical, because plain 'μηνίσκ' sat in
# BOTH lists (correct for highlighting, fatal for extraction). Remove it; the unqualified
# form is already in _meniscus_generic, which routes to the 'weak on both' path.
for side in ("Medial Meniscus", "Lateral Meniscus"):
    F[side]["greek"] = [t for t in F[side]["greek"] if t not in ("μηνίσκ",)]

# FIX B -- Croatian compartments never fired: 'odjeljak' appears 0/406 times. The corpus
# uses 'femorotibijalnom' (117/406) and 'kompartm' (105/406). Bare 'medijaln' is useless
# on its own (402/406), so require co-occurrence.
F["_compartment_medial"]["croatian"] = [
    "medijaln~femorotibijaln", "medijaln~kompartm", "medijalnom femorotibijalnom"]
F["_compartment_lateral"]["croatian"] = [
    "lateraln~femorotibijaln", "lateraln~kompartm", "lateralnom femorotibijalnom"]

# FIX C -- Bulgarian Baker's fired on 62.7% of reports because bare 'киста' matches
# ganglion / meniscal / subchondral cysts. 'Бейкер' appears 0/220; 'поплитеал' 90/220.
F["Baker's"]["bulgarian"] = ["киста~поплитеал", "поплитеална киста", "бейкър", "бейкер"]

# FIX D -- severity cues. New cue class; a positive finding in a clause qualified as
# minimal is downgraded to 'hedged' (0.65) instead of 'pos' (0.95).
d["cues"]["minimal"] = {
    "english":   ["trace", "minimal", "small", "mild", "scant", "tiny", "slight", "subtle"],
    "spanish":   ["leve", "mínim", "minim", "pequeñ", "escas", "discret", "ligero"],
    "turkish":   ["minimal", "hafif", "küçük", "az miktar", "silik"],
    "croatian":  ["manji", "minimaln", "blag", "diskret", "oskudn", "neznatn"],
    "greek":     ["ελάχιστ", "μικρ", "ήπι", "διακριτικ"],
    "german":    ["gering", "minimal", "diskret", "leicht", "klein", "mäßig"],
    "bulgarian": ["малък", "малка", "минимал", "лек", "оскъдн", "неголям"],
    "dutch":     ["gering", "minimaal", "klein", "licht", "weinig", "subtiel"],
    "french":    ["minime", "discret", "faible", "petit", "léger", "modér"],
}

P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("round-1 fixes applied:")
print("  A greek meniscus  ->", F["Medial Meniscus"]["greek"], "/", F["Lateral Meniscus"]["greek"])
print("  B croatian medial ->", F["_compartment_medial"]["croatian"])
print("  C bulgarian baker ->", F["Baker's"]["bulgarian"])
print("  D minimal cues    ->", len(d["cues"]["minimal"]), "languages")
