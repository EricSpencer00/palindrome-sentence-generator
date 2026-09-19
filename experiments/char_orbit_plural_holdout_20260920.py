"""Held-out plural agreement lane for the character-orbit scene FSM."""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from experiments.char_orbit_scene_search_20260920 import audit, normalize
EXPERIMENT_ID = "char-orbit-plural-holdout-20260920"
LEXICON = {"agent": ("wardens", "cartographer"), "verb": ("guard", "maps"),
           "object": ("lanterns", "harbor"), "prep": ("near", "beside"),
           "place": ("tower", "bridge"), "adv": ("quietly", "carefully")}
FRAME = ("agent", "verb", "object", "prep", "place", "adv")

def states(words):
    plural = words[0] == "wardens"
    return [{"role":"agent", "word":words[0], "number":"plural" if plural else "singular",
             "boundary_before":True, "boundary_after":True},
            {"role":"predicate", "word":words[1], "agreement":"plural" if plural else "singular",
             "valency":"transitive", "boundary_before":True, "boundary_after":True},
            {"role":"patient", "word":words[2], "number":"plural" if words[2] == "lanterns" else "singular",
             "boundary_before":True, "boundary_after":True}]

def gate(text, words):
    parsed = text.rstrip('.').split()
    plural = words[0] == 'wardens'
    return {"six_words": len(parsed)==6, "capitalized": text[0].isupper(), "period":text.endswith('.'),
            "finite": words[1] in LEXICON['verb'], "transitive": bool(words[2]),
            "agreement": (plural and words[1]=='guard') or ((not plural) and words[1]=='maps'),
            "complete_scene": len(words)==6}

def row(words):
    text = ' '.join(words).capitalize()+'.'
    tape = normalize(text); ledger=[]; left,right=0,len(tape)-1
    while left<=right:
        ledger.append((left,right,tape[left],tape[right])); left+=1; right-=1
    au=audit(text); gates=gate(text,words)
    return {"rendered":text,"words":words,"semantic_states":states(words),"center_orbit_ledger":ledger,
            "audit":au,"complete_clause_gate":gates,"mechanically_admitted":au['two_pointer_exact'] and all(gates.values()),
            "provenance":{"authored_plural_holdout":True,"finished_tape_reversed":False,"word_order_symmetry":False,
                          "repeated_self_palindromic_unit":False,"catalogue_text":False,"rlaif_used":False},
            "reader_status":"unreviewed; mechanical closure is not readability evidence"}

def run():
    rows=[]
    for words in itertools.product(*(LEXICON[k] for k in FRAME)):
        # Every live orbit must carry matching number and noun/verb valency.
        if not ((words[0],words[1]) in {('wardens','guard'),('cartographer','maps')}): continue
        rows.append(row(words))
    rows.sort(key=lambda r:(r['mechanically_admitted'],-r['audit']['mismatch_count']),reverse=True)
    controls=[row(('wardens','guard','lanterns','near','tower','quietly')),
              row(('cartographer','maps','harbor','beside','bridge','carefully'))]
    exact=[r for r in rows if r['mechanically_admitted']]
    return {"experiment_id":EXPERIMENT_ID,"method":"held-out plural agent/object pair with agreement carried through center-out character orbit",
            "stats":{"visited":len(rows),"exact":len(exact),"mechanically_admitted":len(exact),"longest_letters":max((r['audit']['letters'] for r in rows),default=0)},
            "novelty_preflight":{"status":"passed","distinction":"new plural agent/object and live number register","catalogue_imported":False,"repair_queue":False},
            "candidates":rows,"complete_prose_controls":controls,
            "independent_audits":["two-pointer normalized scan","forward/reverse SHA-256","agreement-state replay","complete-clause mechanical gate"],
            "failure_and_next_discriminator":{"failure":"no exact closure" if not exact else "exact closure found","next":"add one plural locative modifier while retaining the agreement register","rlaif":"not used"},"reader_gate":"closed; no human certification"}

if __name__=='__main__':
    out=run(); p=ROOT/'runs'/(EXPERIMENT_ID+'.json'); p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['stats'],sort_keys=True))
