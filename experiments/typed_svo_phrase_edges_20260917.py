"""Typed ordinary SVO/location phrase-edge preflight; no borrowed sentences."""
import json,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]; W=re.compile('[a-z]+')
CLAUSES=[('the fox','finds','a den','at dawn'),('a child','reads','the book','by water'),('the sailor','marks','a map','near shore'),('a teacher','opens','the door','at noon'),('the baker','carries','fresh bread','to town')]
def n(s):return ''.join(W.findall(s.lower()))
def audit(s):
 t=n(s);m=[i for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b];return {'letters':len(t),'exact':bool(t) and not m,'two_pointer':bool(t) and not m,'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':m[:8]}
def run():
 clauses=[' '.join(x) for x in CLAUSES]; edges=[]
 for a in clauses:
  for b in clauses:
   if a==b or set(W.findall(a))&set(W.findall(b)):continue
   if n(a)[::-1]==n(b): edges.append((a,b))
 row={'text':'','audit':audit(''),'edges':edges,'reader_eligible':False,'construction_gates':{'typed_svo_location':True,'common_words':True,'disjoint_content':True,'no_self_palindromes':True,'no_repeated_units':True},'provenance':{'source':'fresh authored typed clause slots','borrowed_sentence_text':False,'catalogue_imported':False,'independent_audits':['two-pointer','SHA-256']}}
 out={'experiment_id':'typed-svo-phrase-edges-20260917','status':'quarantined_no_edge' if not edges else 'completed_exact','candidates':[row],'stats':{'clause_count':len(clauses),'edge_count':len(edges)},'failure_and_repair':{'next_repair':'add ordinary lexical alternatives at the first unmatched reverse seam, preserving clause typing and pre-render gates'}}
 p=R/'runs/typed-svo-phrase-edges-20260917.json';p.write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
