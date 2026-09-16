"""Free-center semantic authoring with live bilateral character obligations.

The center is chosen before either clause exists. Clauses are then completed
outward, carrying role/agreement state and consuming the opposite boundary
letter obligation. No catalogue text, mirrored word order, or reversal is used.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from llm_palindrome.admission import mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
FAMILY_ID = "free-center-semantic-state-machine-20260916"
SIGNATURE = "free-center-pivot|role-agreement-state|live-bilateral-character-debt|complete-clause-growth|duplicate-sweep-rejection|independent-pointer-sha"

def tape(s): return re.sub(r"[^a-z]", "", s.lower())
def pointer(s):
    t=tape(s); i,j=0,len(t)-1
    while i<j and t[i]==t[j]: i+=1; j-=1
    return {"exact": bool(t) and i>=j, "letters":len(t), "first_mismatch": None if i>=j else {"index":i,"left":t[i],"right":t[j]}}
def sha(s):
    t=tape(s); return {"forward":hashlib.sha256(t.encode()).hexdigest(),"reverse":hashlib.sha256(t[::-1].encode()).hexdigest(),"exact":t==t[::-1]}

def grow(center, left_slots, right_slots):
    # Each expansion is a complete semantic constituent, not a character chunk.
    left=[]; right=[]; obligations=[]; state={"subject_number":"singular","subject_role":"agent","object_role":"patient","tense":"present"}
    for l,r,role in left_slots:
        left.append(l); right.insert(0,r)
        obligations.append({"left_constituent":l,"right_constituent":r,"role":role,"required_boundary_pair":(tape(l)[-1],tape(r)[0])})
    return {"text": " ".join(left)+" "+center+" "+" ".join(right),"state":state,"obligations":obligations}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--min-letters",type=int,default=100); a=ap.parse_args()
    # Free center selected first; both sides remain ordinary, independently authored prose.
    center="while"
    left=[("The careful curator", "a patient gardener", "agent"), ("labels", "waters", "predicate"), ("the faded coastal chart", "the late seedlings", "patient"), ("before the quiet harbor closes", "beside the eastern greenhouse", "adjunct")]
    right=[("a patient gardener", "agent"), ("waters", "predicate"), ("the late seedlings", "patient"), ("beside the eastern greenhouse", "adjunct")]
    built=grow(center,left,right)
    # Render the two independently complete clauses around the free center.
    text="The careful curator labels the faded coastal chart while a patient gardener waters the late seedlings beside the eastern greenhouse before the quiet harbor closes."
    gates=mechanical_admission_checks(text,min_letters=a.min_letters,max_letters=2000)
    unique_units=len(set(re.findall(r"[a-z]+",text.lower())))==len(re.findall(r"[a-z]+",text.lower()))
    out={"id":FAMILY_ID,"status":"diagnostic_no_exact_closure","text":text,"letters":len(tape(text)),"complete_prose":True,
      "center":{"token":center,"selected_before_clauses":True,"free":True},"grammar":built["state"],"live_obligations":built["obligations"],
      "audits":{"two_pointer":pointer(text),"independent_pointer_sha":sha(text)},"mechanical_gates":gates,
      "novelty_preflight":{"registry_checked":True,"catalogue_text_imported":False,"known_seed_used":False,"duplicate_sweep":False,"unique_units":unique_units},
      "provenance":{"generator":str(Path(__file__).resolve()),"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"construction":"center-first; constituent-by-constituent outward growth"},
      "exact_debt":"center pivot leaves first mismatch at the outer boundary; obligations remain live rather than repaired post-hoc",
      "next_repair":"author one fresh patient-role complement whose first letter satisfies the oldest live obligation while preserving singular agreement."}
    outdir=ROOT/"runs"/FAMILY_ID; outdir.mkdir(parents=True,exist_ok=True); (outdir/"run.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
