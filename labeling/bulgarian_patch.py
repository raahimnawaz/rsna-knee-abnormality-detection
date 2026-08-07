"""Replace the (wrong-language) Russian glossary with Bulgarian, and strengthen Greek.

The Cyrillic block in this corpus is Bulgarian, not Russian -- confirmed by lingua and by
inspection ('МР находка', 'ставен излив', 'кръстна връзка', 'б.о.'). Terms below are taken
from the actual reports where possible.
"""
import json
from pathlib import Path

P = Path(r"C:\Users\Raahim\rsna-knee-mri\labeling\glossary.json")
d = json.loads(P.read_text(encoding="utf-8"))

BG = {
    "ACL":              ["предна кръстна", "предната кръстна", "предн кръстн", "пкв"],
    "MCL":              ["колатерал", "медиален колатерал", "вътрешн странич"],
    "Medial Meniscus":  ["медиалния менискус", "медиален менискус", "медиалниот менискус",
                         "вътрешния менискус", "медиалния мениск"],
    "Lateral Meniscus": ["латералния менискус", "латерален менискус", "външния менискус",
                         "латералния мениск"],
    "_OA_generic":      ["артроза", "гонартроза", "дегенератив", "хондромалац", "остеофит",
                         "хрущял", "изтънен", "изтъняване", "стеснен"],
    "_compartment_medial":  ["медиалната страна", "медиалния кондил", "медиално ставно",
                             "медиална гонартроза"],
    "_compartment_lateral": ["латералната страна", "латералния кондил", "латерално ставно",
                             "латерална гонартроза"],
    "PF OA":            ["пател", "ретинакул", "феморопател", "пателофемор"],
    "Effusion":         ["ставен излив", "излив", "течност в", "хидартроза"],
    "Synovitis":        ["синовит", "синовиал", "синовиум"],
    "Baker's":          ["бейкър", "бейкер", "поплитеална киста", "подколянна киста"],
    "Contusion":        ["костномозъчен едем", "костен едем", "контузия", "оток на костния",
                         "субхондрален едем"],
    "Fracture":         ["фрактура", "счупване", "авулзи"],
    "_tear":            ["руптура", "скъсване", "разкъсване", "увреда", "лезия",
                         "нарушена цялост"],
}
BG_NEG = ["няма", "без ", "не се", "нормално", "нормален", "нормална", "запазена",
          "запазен", "б.о.", "непроменен", "интактн", "цялост"]
BG_UNC = ["вероятно", "възможно", "суспектн", "не се изключва", "съмнени", "вероятн"]

# Greek was the weakest Latin-script-adjacent glossary (10/16 keys firing); add stems.
GR_EXTRA = {
    "MCL":              ["έσω πλάγιου", "πλάγιος σύνδεσμος", "πλαγίου συνδέσμου"],
    "_compartment_medial":  ["έσω διαμερίσματ", "έσω μηροκνημιαία"],
    "_compartment_lateral": ["έξω διαμερίσματ", "έξω μηροκνημιαία"],
    "Fracture":         ["κάταγμ", "καταγματ"],
    "Contusion":        ["οίδημα του μυελού", "οστικού οιδήματ", "μώλωπ"],
    "Synovitis":        ["υμενίτιδα", "υμενικ", "αρθρικού υμέν"],
}

for k, terms in BG.items():
    d["findings"][k].pop("russian", None)
    d["findings"][k]["bulgarian"] = terms
for k, terms in GR_EXTRA.items():
    d["findings"][k]["greek"] = sorted(set(d["findings"][k].get("greek", []) + terms))

d["cues"]["negation"].pop("russian", None)
d["cues"]["uncertainty"].pop("russian", None)
d["cues"]["negation"]["bulgarian"] = BG_NEG
d["cues"]["uncertainty"]["bulgarian"] = BG_UNC

d["languages"] = ["english","spanish","turkish","croatian","greek","german",
                  "bulgarian","dutch","french"]
d["_comment"] += (" CORRECTION 2026-08-06: the Cyrillic block is BULGARIAN, not Russian; "
                  "the 'latin' bucket was terse ENGLISH. Language IDs now come from lingua "
                  "(data/lang_detected.csv), not the stopword heuristic.")

P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
missing = {k: [l for l in d["languages"] if not v.get(l)]
           for k, v in d["findings"].items()}
print("glossary updated. languages:", d["languages"])
print("\nfinding-keys with no terms for a language (should be empty):")
for k, langs in missing.items():
    if langs:
        print(f"  {k}: {langs}")
