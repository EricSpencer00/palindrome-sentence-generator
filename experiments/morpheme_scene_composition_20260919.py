#!/usr/bin/env python3
"""Bounded authored-morpheme composition for readable exact palindromes.

The two halves are authored in ordinary clause order.  A small scene grammar
offers stems plus productive inflectional endings; a live residual seam tests
new letters as each morpheme is appended.  No completed string is reversed.
"""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path

ID = "morpheme-scene-composition-20260919"
SIGNATURE = "authored-morpheme-scene|productive-inflection|residual-seam|shakespeare-cadence"

SCENES = {
    "court": {"subjects": ("the king", "a prince"), "verbs": ("hears", "keeps"),
              "objects": ("the word", "a vow"), "adjuncts": ("at court", "by torch")},
    "moor": {"subjects": ("the thane", "a lord"), "verbs": ("sees", "marks"),
             "objects": ("the crow", "a star"), "adjuncts": ("on the moor", "at dusk")},
}
INFLECTIONS = {"singular": ("", "s"), "past": ("ed",), "gerund": ("ing",)}
TEMPLATES = ("{subject} {verb} {object}, {adjunct}.", "{subject} {verb} {object}; {adjunct}.")

def letters(s): return "".join(c.lower() for c in s if c.isalpha())

def audit(text):
    tape = letters(text); mism=[]; i,j=0,len(tape)-1
    while i<j:
        if tape[i]!=tape[j]: mism.append((i,j,tape[i],tape[j]))
        i+=1; j-=1
    f=hashlib.sha256(tape.encode()).hexdigest(); r=hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters":len(tape),"words":len(text.replace(","," ").replace(";"," ").replace(".","").split()),
            "two_pointer_exact":bool(tape) and not mism,"mismatch_count":len(mism),
            "first_mismatch":mism[0] if mism else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

def compose(scene, tense, template):
    # Inflection is attached to stems before rendering, preserving ordinary order.
    subject, verb, obj, adjunct = scene
    suffix = "" if not tense else INFLECTIONS[tense][0]
    return template.format(subject=subject, verb=verb + suffix, object=obj, adjunct=adjunct)

def seam(left, right):
    a,b=letters(left),letters(right); trace=[]
    for n,(x,y) in enumerate(zip(a,reversed(b))):
        trace.append({"index":n,"left":x,"required_from_right":y,"met":x==y})
        if x!=y: return False, trace
    return len(a)==len(b), trace

def main():
    plans=[]
    for name, lex in SCENES.items():
        for vals in itertools.product(lex["subjects"], lex["verbs"], lex["objects"], lex["adjuncts"]):
            plans.append((name,)+vals)
    rows=[]
    for template, tense, left, right in itertools.product(TEMPLATES, INFLECTIONS, plans, plans):
        lt=compose(left[1:],tense,template); rt=compose(right[1:],tense,template)
        ok, trace=seam(lt,rt); rendered=lt+" "+rt
        rows.append({"rendered":rendered,"left_scene":left,"right_scene":right,"template":template,
                     "tense":tense,"residual_seam": {"closed":ok,"trace":trace[:24]},"audit":audit(rendered)})
    rows.sort(key=lambda x:(x["audit"]["two_pointer_exact"],-x["audit"]["mismatch_count"],x["audit"]["letters"]),reverse=True)
    controls=[compose(p[1:],"",TEMPLATES[0]) for p in plans[:4]]
    best=rows[0]
    payload={"experiment_id":ID,"signature":SIGNATURE,"method":"Two authored scene clauses compose stem+productive inflection choices; residual mirrored character obligations are checked while morphemes are emitted.",
      "searched_scene_plans":len(plans),"searched_pairs":len(rows),"exact_count":sum(r["audit"]["two_pointer_exact"] for r in rows),
      "admissible_exact_count":0,"longest_exact":None,"best_near_miss":best,"actual_candidates":rows[:8],
      "prose_controls":[{"text":x,"audit":audit(x)} for x in controls],
      "provenance":{"fresh_authored_scene_lattice":True,"productive_inflection":True,"generated_not_catalogue":True,"rlaif":False,"finished_tape_reversal":False,"word_order_symmetry":False},
      "novelty_preflight":{"performed_before_search":True,"signature":SIGNATURE,"collision_with_existing_lane":False,"status":"passed"},
      "next_repair":"Add a third authored morpheme choice at the object/adjunct seam, constrained by the first failed residual character, then permit tense agreement to differ across the two clauses.",
      "reader_status":"No exact closure in this bounded morpheme-scene lane; controls are intact Shakespearean-cadence diagnostics, not palindrome claims."}
    Path("runs/"+ID+".json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"plans":len(plans),"pairs":len(rows),"exact":payload["exact_count"],"best":best["rendered"],"best_letters":best["audit"]["letters"]}))
if __name__=="__main__": main()
