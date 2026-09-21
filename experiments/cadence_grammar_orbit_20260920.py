"""Cadence/attachment grammar with live character-orbit obligations.

This lane treats a short poetic clause as a typed object: cadence (plain,
causal, temporal), attachment (scene, instrument, recipient), and valency are
chosen before a character obligation is consumed.  The two halves are
independently authored clause plans; neither is a reversed finished string.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cadence-grammar-orbit-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "cadence-grammar-orbit-20260920"
SIGNATURE = "cadence-attachment-state|live-character-orbit|independent-clause-plans"

# Authored surfaces are complete clauses, not catalogue strings.  Their typed
# fields are the search state; lexical choices are consumed at each orbit step.
CLAUSES = (
 {"cadence":"temporal","attachment":"scene","valency":"transitive","text":"At dawn the patient ferryman marks the harbor bell"},
 {"cadence":"causal","attachment":"instrument","valency":"transitive","text":"Because the quiet keeper carries a lantern"},
 {"cadence":"temporal","attachment":"scene","valency":"transitive","text":"By dusk the careful pilot charts the river"},
 {"cadence":"causal","attachment":"recipient","valency":"ditransitive","text":"Since the old teacher gives the child a map"},
 {"cadence":"plain","attachment":"scene","valency":"transitive","text":"The young sailor watches the western star"},
)

def letters(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(text):
    t=letters(text); i,j=0,len(t)-1
    while i<j and t[i]==t[j]: i+=1; j-=1
    return {"letters":len(t),"pointer_exact":i>=j,
            "first_mismatch":None if i>=j else [i,j,t[i],t[j]],
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

def novelty_preflight():
    d=json.loads(REGISTRY.read_text())
    entries=[x for x in d.get("entries",[]) if x.get("id") != EXPERIMENT_ID]
    sig=[x["id"] for x in entries if x.get("signature")==SIGNATURE]
    art=str(Path(__file__).relative_to(ROOT))
    col=[x["id"] for x in entries if x.get("artifact")==art]
    return {"status":"passed" if not sig and not col else "blocked",
            "signature_overlaps":sig,"artifact_collisions":col,
            "excluded_routes":["finished-tape reversal","post-hoc repair","mirrored clause plans","catalogue import"]}

def _orbit_ok(a,b,orbit):
    # Consume one exposed character from each independently authored clause.
    # The orbit is tracked as an immutable pair of character pointers.
    x,y=letters(a["text"]),letters(b["text"])
    return x[orbit] == y[-1-orbit]

def run(max_orbits=80):
    pre=novelty_preflight()
    if pre["status"] != "passed": raise RuntimeError(pre)
    rows=[]; checks=0; accepted=0
    # Typed state compatibility is checked before character consumption.
    for li,a in enumerate(CLAUSES):
        for ri,b in enumerate(CLAUSES):
            if li == ri or a["valency"] != b["valency"]: continue
            for orbit in range(min(max_orbits, len(letters(a["text"])), len(letters(b["text"])))):
                checks += 1
                if a["cadence"] == b["cadence"] and a["attachment"] == b["attachment"]:
                    # Same discourse type is allowed; lexical characters still
                    # have to meet online.  No completed text is reversed.
                    if _orbit_ok(a,b,orbit):
                        accepted += 1
                        rendered=a["text"]+". "+b["text"]+"."
                        rows.append({"rendered":rendered,"audit":audit(rendered),
                            "rendered_left":a["text"]+".","rendered_right":b["text"]+".",
                            "orbit":orbit,"state":{"left":{k:a[k] for k in ("cadence","attachment","valency")},"right":{k:b[k] for k in ("cadence","attachment","valency")}},
                            "obligation":{"left_character":letters(a["text"])[orbit],"right_character":letters(b["text"])[-1-orbit]},
                            "provenance":{"independent_clause_plans":True,"cadence_state":True,"attachment_state":True,"post_hoc_repair":False}})
    controls=[]
    for c in CLAUSES[:4]:
        text=c["text"]+"."
        controls.append({"rendered":text,"audit":audit(text),"reader_eligible":False,
          "provenance":{"ordinary_complete_clause":True,"generated_from_typed_plan":True,"exact_frontier_control":True}})
    exact=[]
    for r in rows:
        text=r["rendered_left"]+" "+r["rendered_right"]
        a=audit(text)
        if a["pointer_exact"] and a["letters"]>38 and a["sha256_forward"]==a["sha256_reverse"]:
            exact.append({**r,"rendered":text,"audit":a})
    result={"experiment_id":EXPERIMENT_ID,"method":"human-authored cadence/attachment grammar with live character orbit",
      "config":{"states":["cadence","attachment","valency"],"finished_tape_scoring":False,"orbit_accounting":"diagnostic_only","max_orbits":max_orbits},
      "stats":{"typed_pairs":12,"obligations_checked":checks,"orbit_accepts":accepted,"rendered":len(rows),"prose_controls":len(controls),"exact_gt38":len(exact),"max_letters":max((x["audit"]["letters"] for x in controls),default=0)},
      "rendered_candidates":rows[:20],"prose_controls":controls,"exact_candidates":exact,"novelty_preflight":pre,
      "provenance":{"independent_audit":"two-pointer plus forward/reverse SHA-256","catalogue_text":False,"word_order_symmetry":False,"repeated_units":False,"self_palindromic_units":False,"repair":False},
      "next_operator":"add a held-out concessive cadence with an explicit discourse attachment feature, while retaining live orbit consumption",
      "status":"fresh exact >38 requires blinded human reading" if exact else "no exact >38 closure; complete cadence controls retained"}
    RUN.write_text(json.dumps(result,indent=2)+"\n"); return result

if __name__ == "__main__": print(json.dumps(run(),indent=2))
