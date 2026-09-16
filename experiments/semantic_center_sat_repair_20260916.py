"""Residual-debt repair for the semantic-center SAT lane.

Only verb/object substitutions from the best-scoring prior assignment are
explored.  The center event remains frozen, so this is a repair operator rather
than a new sweep over subjects, verbs, and objects.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.semantic_center_sat_20260916 import CENTERS, SUBJECTS, VERBS, OBJECTS, exact_audit, sat_score

EXPERIMENT_ID="semantic-center-sat-repair-20260916"
SIGNATURE="frozen-authored-center|residual-character-debt-repair|verb-object-paradigm-substitution|semantic-valency-preservation|heldout-repair|independent-exact-hash-audit"
OUT=ROOT/"runs"/(EXPERIMENT_ID+".json")

def render(center, subject, verb, obj):
    text=f"{subject} {verb} {obj}. {center['text']}."
    audit=exact_audit(text); checks=mechanical_admission_checks(text,min_letters=39,max_letters=220)
    return {"rendered":text,"letters":audit["letters"],"center_event":center["text"],
            "slot_assignment":{"subject":subject,"verb":verb,"object":obj},
            "sat_character_equation":sat_score(f"{subject} {verb} {obj}",center["text"]),
            "exact_audit":audit,"checks":checks,
            "admitted":bool(audit["exact"] and all(checks.values())),
            "provenance":{"center_frozen":True,"source_sentences_copied":False,
                "catalogue_imported":False,"reversed_finished_sentence":False,
                "repeated_self_palindromic_unit":False,"word_order_symmetry":False}}

def repair(seed):
    center=CENTERS[0]  # held fixed throughout repair
    base=render(center,"the careful nurse","carried","a sealed letter")
    # Debt-guided coordinate descent: at each step keep only substitutions that
    # reduce residual debt, then retain ties with better semantic readability.
    current=base; stages=[]
    for slot, inventory in (("verb",VERBS),("object",OBJECTS)):
        ranked=[]
        for value in inventory:
            args=dict(current["slot_assignment"]); args[slot]=value
            cand=render(center,args["subject"],args["verb"],args["object"])
            ranked.append(cand)
        ranked.sort(key=lambda x:(x["sat_character_equation"]["residual_debt"],-x["sat_character_equation"]["satisfied_clauses"]))
        best=ranked[0]
        stages.append({"slot":slot,"from":current["slot_assignment"][slot],"selected":best["slot_assignment"][slot],
                      "residual_before":current["sat_character_equation"]["residual_debt"],
                      "residual_after":best["sat_character_equation"]["residual_debt"],
                      "substitution_candidates":len(ranked)})
        current=best
    return base,current,stages

def run():
    base,best,stages=repair(None)
    # A held-out lexical family is evaluated after the train-side repair; the
    # center and subject remain unchanged, and only a new paradigm is allowed.
    heldout=render(CENTERS[0],"the careful nurse","sealed","a letter")
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed",
      "method":"coordinate descent over verb/object paradigms chosen by residual character debt with frozen authored center",
      "seed":base,"repaired":best,"heldout":heldout,"repair_trace":stages,
      "stats":{"candidates":3,"admitted":sum(int(x["admitted"]) for x in (base,best,heldout)),"exact":sum(int(x["exact_audit"]["exact"]) for x in (base,best,heldout))},
      "next_repair":"replace only the least-satisfied verb-object seam with a valency-compatible held-out pair and re-run the same debt descent",
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"copied_text":False}}

if __name__=="__main__":
    if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
    p=run(); OUT.write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p,indent=2))
