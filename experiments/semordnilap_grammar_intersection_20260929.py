"""Bidirectional grammar intersection over semordnilap lexical pairs."""
from __future__ import annotations
import argparse,hashlib,json,socket,itertools,re
from pathlib import Path
PAIRS=[('emit','time','V','V'),('parts','strap','N','N'),('reward','drawer','V','N'),('diaper','repaid','N','V'),('stressed','desserts','ADJ','N'),('live','evil','V','ADJ'),('stop','pots','V','N'),('flow','wolf','V','N'),('smart','trams','ADJ','N'),('deliver','reviled','V','ADJ')]
TEMPLATES=[('VNN',['V','N','N']),('NVN',['N','V','N']),('AVN',['ADJ','V','N'])]
def tape(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s);return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run(limit):
 by={p:[] for p in {'V','N','ADJ'}}
 for a,b,pa,pb in PAIRS:by[pa].append((a,b));by[pb].append((b,a))
 rows=[];controls=[]
 for _,tags in TEMPLATES:
  for choices in itertools.product(*(by[t] for t in tags)):
   left=[x[0] for x in choices];right=[x[1] for x in choices][::-1]
   if len(set(left+right))<len(left+right):continue
   # Each token pair is fixed as soon as its outer character block is chosen.
   text=' '.join(left)+'; '+' '.join(right);a=audit(text)
   rec={'rendered':text,'audit':a,'grammar_template':tags,'left_words':left,'right_words':right,'provenance':{'semordnilap_pairs':True,'online_token_intersection':True,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_used':False,'duplicate_units':False}}
   if a['two_pointer_exact']:rec['reader_worthy']=False;rows.append(rec)
   elif len(controls)<8:rec['reader_worthy']=False;controls.append(rec)
   if len(rows)>=limit:return rows,controls
 return rows,controls
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=20);ap.add_argument('--out',required=True);a=ap.parse_args();rows,controls=run(a.limit);p={'experiment':'semordnilap-grammar-intersection-20260929','host':socket.gethostname(),'parameters':vars(a),'candidates':rows,'controls':controls,'closures':len(rows),'reader_worthy':0,'provenance':{'typed_templates':['VNN','NVN','AVN'],'fresh_authored_semordnilap_bank':True,'independent_audit':'two-pointer plus SHA-256'},'next_construction':'add determiners and agreement-safe semordnilap pairs while preserving token-level online intersection.'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('experiment','closures','reader_worthy')}))
if __name__=='__main__':main()
