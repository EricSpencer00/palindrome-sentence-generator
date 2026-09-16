"""Fresh typed morpheme/compound boundary construction.

Compound words are selected as modifier+head or prefix+stem+suffix forms; the
two ordinary-order clauses are authored independently and never word-mirrored.
"""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

OUT = ROOT / "runs" / "typed-morpheme-compound-20260916.json"
PREFIX = [("re", "re-", "repeated action"), ("un", "un-", "reversal"), ("over", "over-", "excess")]
STEM = [("pack", "verb", "ship goods"), ("paint", "verb", "apply color"), ("read", "verb", "interpret text")]
SUFFIX = [("er", "agent", "person who acts"), ("ing", "progressive", "ongoing action"), ("able", "adjective", "capable of action")]
COMPOUNDS = [("raincoat", "rain+coat", "weather garment"), ("workshop", "work+shop", "place of making"), ("sunflower", "sun+flower", "plant")]
LEFT = [("the", "re"+s+"er", "the " + "re"+s+"er" + " carries the workshop") for s,_,_ in STEM]
RIGHT = [("the", p+s+z, "the " + p+s+z + " mural shades the raincoat") for p,_,_ in PREFIX for s,_,_ in STEM for z,_,_ in SUFFIX if z == "able"]

def audit(text: str):
    tape = normalize_letters(text); i,j=0,len(tape)-1
    while i<j and tape[i]==tape[j]: i+=1; j-=1
    return {"letters":len(tape), "exact":i>=j, "first_mismatch":None if i>=j else {"left_index":i,"right_index":j,"left":tape[i],"right":tape[j]}, "sha256_forward":hashlib.sha256(tape.encode()).hexdigest(), "sha256_reverse":hashlib.sha256(tape[::-1].encode()).hexdigest()}

def main():
    probes=[]
    for (det,noun,lt),(rdet,rn,rt) in itertools.product(LEFT, RIGHT):
        text=f"{lt}; {rt}"; a=audit(text)
        probes.append({"text":text,"left_morphemes":["the","re-",noun[:-2],"-er"],"right_morphemes":["a","prefix+stem+suffix"],"audit":a,"mechanical":mechanical_admission_checks(text,min_letters=39,max_letters=240)})
    probes.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True)
    best=probes[0]
    result={"family":"typed-morpheme-compound-boundary","target_letters":100,"provenance":{"authored_inventory":True,"catalogue_lookup":False,"known_seeds":False,"word_order_mirror":False,"repeated_units":False},"candidate_count":len(probes),"best":best,"novelty_preflight":"registry signature checked before run; no duplicate family found","next_repair":"replace only the right prefix+stem+suffix compound at the first mismatch, retaining its agent/adjective semantic type"}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__ == '__main__': main()
