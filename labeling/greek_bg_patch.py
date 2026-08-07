"""Greek and Bulgarian glossary terms taken from the actual report text.

Greek reports are terse and use forms my first pass guessed wrong:
  'Χωρίς ενδαρθρική συλλογή υγρού'      no intra-articular fluid collection
  'Φυσιολογικά απεικονίζονται οι μηνίσκοι'  menisci appear normal  (often unqualified,
                                            so plain 'μηνίσκ' must fire for BOTH menisci)
  'ολικής ρήξης'                        complete tear
  'έσω μεσάρθριο'                       medial compartment
  'επιχείλια οστεόφυτα'                 marginal osteophytes
Bulgarian: Baker's is written just 'киста' (+ location), never 'Бейкер' in this corpus.
"""
import json
from pathlib import Path

P = Path(r"C:\Users\Raahim\rsna-knee-mri\labeling\glossary.json")
d = json.loads(P.read_text(encoding="utf-8"))
F = d["findings"]

def add(key, lang, terms):
    F[key][lang] = sorted(set(F[key].get(lang, []) + terms))

# 'μηνίσκ' unqualified is extremely common -> fire it for both menisci so the human reads
# the sentence and decides which side. Highlighting is an attention aid, not a classifier.
add("Medial Meniscus",  "greek", ["μηνίσκ", "έσω μηνίσκ", "έσω κερατ"])
add("Lateral Meniscus", "greek", ["μηνίσκ", "έξω μηνίσκ", "έξω κερατ"])
add("ACL",   "greek", ["χιαστ", "πρόσθιου χιαστού", "πρόσθιο χιαστό"])
add("MCL",   "greek", ["πλάγι σύνδεσμ", "πλάγιοι σύνδεσμοι", "πλαγίων συνδέσμων", "έσω πλάγι"])
add("Effusion", "greek", ["συλλογή υγρού", "ενδαρθρική συλλογή", "αρθρική συλλογή",
                          "ενδαρθρικ", "υγρού"])
add("_OA_generic", "greek", ["αρθρικός χόνδρος", "αρθρικού χόνδρου", "χόνδρου",
                             "επιχείλια", "οστεόφυτα", "εκφύλισ", "απώλεια"])
add("_compartment_medial",  "greek", ["έσω μεσάρθριο", "έσω μεσαρθρίου"])
add("_compartment_lateral", "greek", ["έξω μεσάρθριο", "έξω μεσαρθρίου"])
add("Contusion", "greek", ["οστικός μυελός", "οστικού μυελού", "μυελού των οστών",
                           "οίδημα", "μυελ"])
add("Synovitis", "greek", ["αρθρικού υμένα", "υμεν", "υμέν"])
add("_tear", "greek", ["ρήξ", "ολικής ρήξης", "μερικής ρήξης", "ασαφοποίηση"])
add("PF OA", "greek", ["επιγονατιδ", "μηροεπιγονατιδ"])

add("Baker's", "bulgarian", ["киста", "кистата", "поплитеалн", "подколянн"])
add("Synovitis", "bulgarian", ["синови", "синовиум", "синовиал"])

P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("patched Greek + Bulgarian terms")
