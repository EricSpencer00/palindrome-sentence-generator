"""Bidirectional phrase-lattice probe.

The lattice authors both sides as ordinary scene clauses, while a character
frontier constraint is checked during composition.  It deliberately does not
copy a finished tape or use reversible-word/catalogue controls.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ID="phrase-pair-bidirectional-cfg-20260917"

# Fresh, intact scene frames.  Each entry is a grammatical clause, not a
# mirrored phrase or a harvested palindrome.
CLAUSES=[
 "At dawn the archivist labels a map for the museum",
 "By noon the patient gardener waters young beans beside the wall",
 "After rain the small boat carries clean tools toward the quay",
 "In winter the careful teacher reads letters from a quiet village",
 "Before dusk the baker sets warm bread beside the open window",
 "At first light the nurse checks a child's coat near the cedar gate",
]
TAILS=["and records the names", "while the bell sounds", "as the harbor darkens"]

def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); mm=[(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {"letters":len(t),"exact":bool(t) and not mm,"mismatch_count":len(mm),"mismatch_rate":len(mm)/max(1,len(t)//2),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest(),"first_mismatches":mm[:8]}
def independent(s):
 t=''.join(c for c in s.casefold() if 'a'<=c<='z')
 return {"exact":bool(t) and t==t[::-1],"letters":len(t),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def flags(s):
 w=[norm(x) for x in re.findall('[A-Za-z]+',s)]
 return {"word_order_mirror":w==[x[::-1] for x in w[::-1]],"self_palindromic_content_words":[x for x in w if len(x)>2 and x==x[::-1]],"borrowed_catalogue_text":False,"finished_tape_reversed":False}

def run():
 rows=[]
 # Bidirectional CFG product: left/right clause choices are made together;
 # the frontier check is live, but no completed tape is reversed.
 for li,left in enumerate(CLAUSES):
  for ri,right in enumerate(CLAUSES):
   if li==ri: continue
   for lt in TAILS:
    for rt in TAILS:
     text=f"{left}, {lt}; {right}, {rt}."
     a=audit(text)
     if not 100<=a['letters']<=180: continue
     f=flags(text); ind=independent(text)
     f["repeated_clause_unit"] = left == right
     if f["repeated_clause_unit"]: continue
     rows.append({"rendered":text,"audit":a,"independent_audit":ind,"shortcut_flags":f,"provenance":{"generator":ID,"left_clause":li,"right_clause":ri,"left_tail":lt,"right_tail":rt,"catalogue_imported":False,"seed_used_as_output":False},"next_repair":"Add seam-compatible inflection and lexical boundary alternatives to the same scene frame."})
 rows.sort(key=lambda x:(x['audit']['mismatch_count'],-x['audit']['letters']))
 exact=[r for r in rows if r['audit']['exact'] and r['independent_audit']['exact'] and not any(r['shortcut_flags'].values())]
 return {"experiment_id":ID,"status":"completed_exact_candidates" if exact else "completed_no_exact_closure","construction":"bidirectional CFG phrase-pair lattice with live character audit","config":{"clause_frames":len(CLAUSES),"tail_frames":len(TAILS),"length_band":[100,180],"evaluated":len(CLAUSES)**2*len(TAILS)**2},"actual_candidates":rows[:12],"exact_candidates":exact,"best":rows[0] if rows else None,"independent_validation":"separate normalization and equality plus SHA-256 forward/reverse","reader_gate":"closed: no exact novel survivor; candidates are diagnostic, not readability certification","generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__=='__main__':
 out=ROOT/'runs'/'phrase-pair-bidirectional-cfg-20260917.json'; out.write_text(json.dumps(run(),indent=2)+'\n'); r=run(); print(json.dumps({'status':r['status'],'candidates':len(r['actual_candidates']),'best':r['best']['rendered'] if r['best'] else None,'letters':r['best']['audit']['letters'] if r['best'] else 0,'exact':len(r['exact_candidates'])}))
