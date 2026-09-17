import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/audit-matrix-lanes-1-2-5-20260916.json'
FILES=['runs/live-gpt2-character-decoder-preflight-20260916.json','runs/exact-tape-grammatical-resegmentation-20260916-luna.json','runs/earley-finite-state-grammar-intersection-20260916.json']
def main():
 out=[]
 for p in FILES:
  d=json.loads((R/p).read_text());rows=d.get('rendered_candidates') or d.get('rendered_probes') or ([d['candidate']] if d.get('candidate') else []); ss=[x['rendered'] for x in rows if x.get('rendered')]; audits=[]
  for s in ss:
   t=re.sub('[^a-z]','',s.lower());i=0
   while i<len(t)//2 and t[i]==t[-1-i]:i+=1
   f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();audits.append({'two_pointer_exact':i==len(t)//2,'first_mismatch':None if i==len(t)//2 else i,'sha256_forward':f,'sha256_reverse':r,'sha256_equal':f==r})
  out.append({'path':p,'rendered_count':len(ss),'independent_replay':audits,'provenance_present':bool(d.get('provenance')),'novelty_preflight_present':bool(d.get('novelty_preflight')),'next_repair_present':bool(d.get('repair') or d.get('next_repair_operator')),'gap':'no rendered candidate; probes are not reader evidence' if not d.get('rendered_candidates') and not d.get('candidate') else 'source reverse SHA supplied by this audit'})
 O.write_text(json.dumps({'experiment':'audit-matrix-lanes-1-2-5-20260916','search_executed':False,'lanes':out,'repair':'Independent replay only; no duplicate sweep.'},indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
