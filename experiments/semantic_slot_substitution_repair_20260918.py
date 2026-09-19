"""Exact-candidate repair with typed semantic slot substitutions.

This lane starts from intact, authored micro-clauses.  It changes only typed
subject/verb/object/adjunct slots, and computes a character residual for every
left/right pair before rendering punctuation.  It is deliberately bounded:
the result is a construction diagnostic, not a claim that a language model
score certifies prose.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from dataclasses import asdict, dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

@dataclass(frozen=True)
class Slot:
    text: str; kind: str; number: str = "ANY"

@dataclass(frozen=True)
class Clause:
    subject: Slot; verb: Slot; object: Slot; adjunct: Slot
    source: str
    @property
    def text(self):
        return " ".join(x.text for x in (self.subject,self.verb,self.object,self.adjunct) if x.text)

SUBJECTS = (Slot("an aide","person","SG"), Slot("a poet","person","SG"),
            Slot("the nurse","person","SG"), Slot("some men","person","PL"),
            Slot("Diana","name","SG"), Slot("Noel","name","SG"), Slot("Leon","name","SG"))
VERBS = (Slot("rips","document_verb","SG"), Slot("reads","document_verb","SG"),
         Slot("writes","document_verb","SG"), Slot("inspires","person_verb","SG"),
         Slot("inspire","person_verb","PL"), Slot("saw","any_verb"))
OBJECTS = (Slot("nine memos","document","PL"), Slot("a note","document","SG"),
           Slot("the map","document","SG"), Slot("a poem","document","SG"),
           Slot("some men","person","PL"), Slot("Diana","name","SG"), Slot("Noel","name","SG"))
ADJUNCTS = (Slot("","place"), Slot("at dawn","time"), Slot("in town","place"),
            Slot("near home","place"), Slot("in an arena","place"))
SEAMS = ((";", "semicolon"), (". ", "sentence"), (" because ", "causal"), (" while ", "contrast"))
KNOWN = {"anaideripsninememossomemeninspirediana"}

def tape(s): return normalize_letters(s)
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
    t=tape(s); bad=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-1-i]]
    return {"letters":len(t),"two_pointer_exact":not bad,"mismatch_count":len(bad),
            "first_mismatch":bad[0] if bad else None,"sha256_forward":sha(t),
            "sha256_reverse":sha(t[::-1]),"sha_equal_under_reversal":sha(t)==sha(t[::-1])}

def clauses():
    out=[]
    for s in SUBJECTS:
      for v in VERBS:
       if v.number not in ("ANY",s.number): continue
       for o in OBJECTS:
        if v.kind=="document_verb" and o.kind!="document": continue
        if v.kind=="person_verb" and o.kind not in ("person","name"): continue
        for a in ADJUNCTS: out.append(Clause(s,v,o,a,"authored_slot_bank_v1"))
    return tuple(out)

def residual(left, right, seam):
    """Compare the letters forced by the two clause tapes before rendering."""
    lt=tape(left.text); rt=tape(right.text)
    seam_t=tape(seam)
    combined=lt+seam_t+rt
    bad=[(i,len(combined)-1-i,combined[i],combined[-1-i])
         for i in range(len(combined)//2) if combined[i]!=combined[-1-i]]
    return {"left_letters":len(lt),"right_letters":len(rt),"seam_letters":len(seam_t),
            "residual_mismatches":len(bad),"first_residual":bad[0] if bad else None,
            "predicted_exact":not bad}

def run(max_probes=10000):
    cs=clauses(); rows=[]; exact=[]; checked=0
    for l in cs:
      for seam,seam_name in SEAMS:
       for r in cs:
        checked+=1
        res=residual(l,r,seam)
        rendered=l.text+seam+r.text+"."
        a=audit(rendered); checks=mechanical_admission_checks(rendered,min_letters=30,max_letters=240)
        row={"rendered":rendered,"left_provenance":l.source,"right_provenance":r.source,
             "left_slots":{k:asdict(v) for k,v in l.__dict__.items() if k != "source"},
             "right_slots":{k:asdict(v) for k,v in r.__dict__.items() if k != "source"},"seam":seam_name,
             "pre_render_residual":res,"audit":a,"mechanical_checks":checks,
             "independent_exact":a["two_pointer_exact"],
             "mechanically_admitted":a["two_pointer_exact"] and all(checks.values()),
             "reader_status":"not_run; programmatic measures diagnose only"}
        if a["two_pointer_exact"]: exact.append(row)
        if len(rows)<max_probes: rows.append(row)
        if checked>=max_probes: break
       if checked>=max_probes: break
      if checked>=max_probes: break
    admitted=[x for x in exact if x["mechanically_admitted"]]
    new=[x for x in admitted if tape(x["rendered"]) not in KNOWN]
    return {"status":"semantic_slot_substitution_repair_bounded","experiment_id":"semantic-slot-substitution-repair-20260918",
      "signature":"authored-clause|typed-slot-substitution|pre-render-residual|independent-audit",
      "config":{"clause_count":len(cs),"seam_count":len(SEAMS),"max_probes":max_probes},
      "stats":{"clauses":len(cs),"pairs_checked":checked,"bounded":checked<len(cs)*len(SEAMS)*len(cs),"stored_probes":len(rows),"exact":len(exact),"mechanically_admitted":len(admitted),"new_mechanically_admitted":len(new),"reader_eligible":0},
      "rendered_candidates_and_probes":rows,"exact_candidates":exact,"admitted":admitted,"new_admitted":new,
      "provenance":{"intact_authored_clauses":True,"slot_substitutions_only":True,"finished_tape_reversed":False,"catalogue_text_imported":False,"word_order_mirror":False,"independent_validator":"two-pointer normalized tape plus forward/reverse SHA-256","human_readability_certified":False},
      "next_repair":"retain residual-compatible subject/verb choices while introducing adjective-noun slots and held-out authored clauses",
      "reader_gate":"closed until an exact candidate survives intact-prose review and blinded controls"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True,type=Path); ap.add_argument("--max-probes",type=int,default=10000); a=ap.parse_args()
    if a.out.exists(): ap.error(f"refusing to overwrite existing output: {a.out}")
    r=run(a.max_probes); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r["stats"],indent=2))
if __name__=="__main__": main()
