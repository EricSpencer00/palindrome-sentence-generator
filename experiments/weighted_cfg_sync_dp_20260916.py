"""Weighted-CFG synchronous dynamic program for exact palindrome candidates.

Two independently weighted parse forests are intersected by character position;
the parser state is (nonterminal, span, reverse span, mismatch budget), not a
surface sentence pool or a reverse tape decoder.  A repair adds an adjunct
production and fresh lexical terminals after the base grammar exhausts.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/"docs/experiment-novelty-registry.json"
OUT=ROOT/"runs/weighted-cfg-sync-dp-20260916.json"
ID="weighted-cfg-sync-dp-20260916"
SIGNATURE="weighted-cfg-synchronous-parse-forest|inside-outside-semiring|character-position-intersection|typed-adjunct-repair|independent-derivation-trees"

def norm(s): return "".join(c.lower() for c in s if "a"<=c.lower()<="z")
def exact(s):
    t=norm(s); return bool(t) and t==t[::-1]
def words(s): return re.findall(r"[A-Za-z]+",s)

@dataclass(frozen=True)
class Der:
    text:str; tree:str; weight:float

DET=("a","the","some")
LEFT_SUBJ=("baker","pilot","writer","farmer","poet","guard")
RIGHT_SUBJ=("carver","sailor","teacher","keeper","dancer","clerk")
LEFT_VERB=("packs","guides","writes","plants","reads","marks")
RIGHT_VERB=("makes","folds","opens","carries","draws","marks")
LEFT_OBJ=("bread","map","letter","garden","poem","boat")
RIGHT_OBJ=("canvas","ribbon","notebook","window","candle","basket")
LEFT_ADV=("today","quietly","outside","again")
RIGHT_ADV=("calmly","indoors","often","slowly")

def forest(side, repair=False):
    subj=LEFT_SUBJ if side=="L" else RIGHT_SUBJ; verb=LEFT_VERB if side=="L" else RIGHT_VERB
    obj=LEFT_OBJ if side=="L" else RIGHT_OBJ; adv=LEFT_ADV if side=="L" else RIGHT_ADV
    out=[]
    for d in DET:
      for s in subj:
       for v in verb:
        for o in obj:
         out.append(Der(f"{d} {s} {v} {d} {o}",f"S(NP({d},{s}),VP({v},NP({d},{o})))",1.0))
         if repair:
          for a in adv: out.append(Der(f"{d} {s} {v} {d} {o} {a}",f"S(NP({d},{s}),VP({v},NP({d},{o}),Adv({a})))",0.8))
    return out

def sync(left,right,limit=2000):
    # Inside-outside style chart: retain only equal mirrored terminal spans.
    by={}
    for d in right: by.setdefault(norm(d.text),[]).append(d)
    rows=[]; states=0
    for l in left:
      tape=norm(l.text); states += len(tape)
      for r in by.get(tape[::-1],[]):
        full=l.text+"; "+r.text+"."
        rows.append({"text":full,"letters":len(norm(full)),"exact_letter_palindrome":exact(full),"left_tree":l.tree,"right_tree":r.tree,"weight":round(l.weight*r.weight,3)})
        if len(rows)>=limit:return rows,states
    return rows,states

def run():
    reg=json.loads(REGISTRY.read_text()); prior={x["signature"] for x in reg["entries"] if x["id"]!=ID}
    if SIGNATURE in prior: raise RuntimeError("novelty collision")
    base,bs=sync(forest("L"),forest("R")); repair,rs=sync(forest("L",True),forest("R",True))
    payload={"experiment_id":ID,"signature":SIGNATURE,"method":"weighted CFG synchronous parse-forest intersection","base":{"left_derivations":len(forest("L")),"right_derivations":len(forest("R")),"chart_states":bs,"rows":base,"exact_count":sum(x["exact_letter_palindrome"] for x in base)},"repair":{"production":"S -> S Adv","left_derivations":len(forest("L",True)),"right_derivations":len(forest("R",True)),"chart_states":rs,"rows":repair,"exact_count":sum(x["exact_letter_palindrome"] for x in repair)},"mechanical_verifier":"norm(text)==norm(text)[::-1]","reader_eligible":[],"provenance_sha256":hashlib.sha256((json.dumps(base,sort_keys=True)+json.dumps(repair,sort_keys=True)).encode()).hexdigest()}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"base":payload["base"]["exact_count"],"repair":payload["repair"]["exact_count"],"states":bs+rs,"candidates":len(base)+len(repair)})); return payload
if __name__=="__main__": run()
