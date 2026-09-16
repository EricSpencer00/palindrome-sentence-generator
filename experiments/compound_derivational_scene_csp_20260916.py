"""Compound-segmented scene frames with live mirrored-character obligations.

This lane is intentionally different from inflection/FST probes: each content
choice is a lexical compound with an explicit semantic decomposition (modifier,
head, scene role), optionally followed by a derivational suffix.  Two ordinary
clauses choose different scene frames while their rendered characters are
checked in the same state.  No right clause is produced by reversing words.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ID="compound-derivational-scene-csp-20260916"; SIG="compound-segmented-scene-frame|modifier-head-semantic-decomposition|derivational-suffix-choice|ordinary-order-independent-clauses|live-mirrored-character-obligations|heldout-compound-affix-repair|independent-exact-two-pointer-hash-mechanical-audits"
OUT=ROOT/"runs/compound-derivational-scene-csp-20260916.json"; MIN=39
@dataclass(frozen=True)
class Lex:
    surface:str; modifier:str; head:str; role:str; suffix:str=""; sense:str=""
@dataclass(frozen=True)
class Clause:
    frame:str; words:tuple[Lex,...]
    @property
    def text(self): return " ".join(x.surface for x in self.words)

def L(surface, modifier, head, role, suffix="", sense=""): return Lex(surface,modifier,head,role,suffix,sense)
DET=tuple(L(x,"","","determiner") for x in ("the","a","our"))
SUBJ=(L("gardener","garden","er","agent", "er", "person tending plants"),L("sailor","sail","or","agent","or","person at sea"),L("teacher","teach","er","agent","er","person guiding learners"))
COMP=(L("raincoat","rain","coat","object",sense="weather clothing"),L("sunflower","sun","flower","object",sense="garden plant"),L("lighthouse","light","house","object",sense="coastal beacon"),L("workshop","work","shop","place",sense="making room"),L("notebook","note","book","object",sense="writing book"),L("seashell","sea","shell","object",sense="shore token"))
VERB=(L("carries","carry","s","verb", "s", "moves an object"),L("opens","open","s","verb","s","makes accessible"),L("mends","mend","s","verb","s","repairs"))
ADV=(L("carefully","care","ful","manner","ful","with attention"),L("quietly","quiet","ly","manner","ly","without noise"),L("kindly","kind","ly","manner","ly","with goodwill"))
# Distinct semantic frames; neither is the reverse of the other.
FRAMES=(("harbor",("DET","SUBJ","VERB","DET","COMP","ADV")),("garden",("DET","COMP","VERB","ADV","DET","SUBJ")))
POOL={"DET":DET,"SUBJ":SUBJ,"COMP":COMP,"VERB":VERB,"ADV":ADV}
def compile_frame(name, slots, limit=80):
 out=[]
 for xs in product(*(POOL[s] for s in slots)):
  if len({x.head for x in xs if x.role in {"agent","object","place"}})<2: continue
  out.append(Clause(name,xs))
  if len(out)>=limit: break
 return out
def ptr(t):
 mm=[]; i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: mm.append((i,j,t[i],t[j]))
  i+=1;j-=1
 return {"exact":bool(t) and not mm,"pairs_checked":len(t)//2,"mismatches":mm[:8]}
def audit(c1,c2):
 text=c1.text+"; "+c2.text; tape=normalize_letters(text); p=ptr(tape)
 h1=hashlib.sha256(tape.encode()).hexdigest(); h2=hashlib.sha256(tape[::-1].encode()).hexdigest()
 mech=mechanical_admission_checks(text,min_letters=MIN,max_letters=220)
 return {"text":text,"letters":len(tape),"normalized":tape,"two_pointer":p,"hash_audit":{"exact":tape==tape[::-1],"forward":h1,"reverse":h2,"digest_equal":h1==h2},"mechanical":mech,"exact":p["exact"] and tape==tape[::-1] and all(mech.values()),"frames":[c1.frame,c2.frame],"compound_paths":[[asdict(x) for x in c1.words if x.modifier],[asdict(x) for x in c2.words if x.modifier]],"ordinary_word_order":True}
def repair(row,c2):
 m=row["two_pointer"]["mismatches"][0] if row["two_pointer"]["mismatches"] else None
 target=c2.words[1] if len(c2.words)>1 else COMP[0]; alt=next(x for x in COMP if x.surface!=target.surface)
 return {"first_mismatch":m,"held_out_slot":"right.compound","original":target.surface,"replacement":alt.surface,"operator":"replace compound while preserving modifier/head semantic decomposition","replayed":False}
def main():
 reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text()); entries=reg["entries"]
 pre={"status":"novel_exact_signature","entries_read":len(entries),"signature_collisions":[e["id"] for e in entries if e.get("signature")==SIG],"artifact_collisions":[e["id"] for e in entries if e.get("artifact")==f"experiments/{Path(__file__).name}"],"overlap_reviewed":["morphology-semantic-template-csp-20260916","inflectional-fst-clitic-tape-20260916","orthographic-compound-boundary-probe-20260915"],"distinction":"compound modifier/head segmentation and scene-role semantics are state variables; not alternate inflections"}
 if pre["signature_collisions"] or pre["artifact_collisions"]: raise SystemExit(pre)
 left=compile_frame(*FRAMES[0]); right=compile_frame(*FRAMES[1]); rows=[]
 for a,b in zip(left,right):
  r=audit(a,b)
  if r["letters"]>=MIN: r["repair"]=repair(r,b); rows.append(r)
 exact=[r for r in rows if r["exact"]]
 payload={"experiment":ID,"signature":SIG,"novelty_preflight":pre,"probe_count":len(rows),"complete_prose_probes_ge_39":sum(r["letters"]>=MIN for r in rows),"exact_count":len(exact),"best_actual_prose":max(rows,key=lambda r:r["letters"])["text"] if rows else None,"probes":rows[:48],"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"material":"hand-authored compound lexemes and semantic scene frames; no catalogue lookup"},"reader_status":"not_run; intact clauses are a construction diagnostic, not human evidence"}
 OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({k:payload[k] for k in ("experiment","probe_count","exact_count","best_actual_prose")}))
if __name__=="__main__": main()
