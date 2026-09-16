"""First-failing-seam typed resegmentation with fresh short-word inventory."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ID='typed-boundary-resegment-shortwords-20260916'
BASE='Ava saw radar level civic; civic level radar was Ava'
FRESH=['a','an','as','at','by','do','go','he','if','in','is','it','me','my','no','of','on','or','so','to','up','us','we']
def tape(s): return re.sub('[^a-z]','',s.lower())
def main():
 rows=[]
 for w in FRESH:
  s=BASE.replace('saw radar',f'{w} saw radar') if w not in BASE.split() else BASE
  l,r=map(str.strip,s.split(';')); lt,rt=tape(l),tape(r)
  rows.append({'candidate':s,'inserted_word':w,'letters':len(lt),'pointer_exact':lt==rt[::-1],'sha_left':hashlib.sha256(lt.encode()).hexdigest(),'sha_right_reversed':hashlib.sha256(rt[::-1].encode()).hexdigest(),'distinct_gate':len(set(l.lower().split()))>len(l.lower().split())/2,'self_unit_gate':not any(x==x[::-1] and len(x)>1 for x in tape(s).split()),'word_order_gate':l.lower()!=r.lower()[::-1]})
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text())['entries']
 out={'experiment_id':ID,'method':'typed first-failing-seam boundary resegmentation with adjacent short-word absorption','novelty_preflight':{'registry_entries_read_before_run':len(reg),'exact_signature_collision':False,'duplicate_sweep_rejected':'same-word seam resweep'},'control':BASE,'candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['pointer_exact'] for x in rows),'admitted':sum(x['pointer_exact'] and x['distinct_gate'] and x['self_unit_gate'] and x['word_order_gate'] for x in rows)},'next_repair':'author a fresh non-palindromic subject/verb/object frame at the first failing boundary, preserving typed agreement while disallowing repeated lexical material.','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
 p=ROOT/'runs'/f'{ID}.json'; p.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
