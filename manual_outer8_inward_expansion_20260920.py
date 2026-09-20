"""Bounded manual outer-equation construction with inward expansion."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/manual-outer8-inward-expansion-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
PAIRS=(('the patient cartographer maps the quiet coast at dawn','the harbor pilot watches the distant light'),('a careful teacher opens a bright window in winter','our kind neighbor carries warm bread to the station'))
def run():
 rows=[]
 for left,right in PAIRS:
  a,b=n(left),n(right)[::-1]; trace=[]; fail=None
  for i,(x,y) in enumerate(zip(a,b)):
   trace.append({'offset':i,'left':x,'right_reversed':y,'matched':x==y})
   if x!=y: fail=i; break
  text=left+'; '+right+'.'
  rows.append({'rendered':text,'outer_equation_width':8,'outer_seed_matched':fail is None or fail>=8,'first_failed_inward_offset':fail,'trace':trace,'audit':audit(text),'provenance':{'manually_authored_natural_clauses':True,'equations_chosen_before_rendering':True,'fresh_roles':True,'catalogue_text_reused':False,'mirror_pair_import':False,'post_hoc_repair':False,'finished_tape_reversal':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 best=max(rows,key=lambda r:r['audit']['letters'])
 return {'experiment_id':'manual-outer8-inward-expansion-20260920','method':'manual outer-8 equation seed followed by inward character expansion','stats':{'pairs':2,'outer8_seeds':sum(r['outer_seed_matched'] for r in rows),'exact_gt38':len(exact),'best_letters':best['audit']['letters']},'controls':rows,'best_control':best,'exact_candidates':exact,'status':'precise zero frontier: outer equations fail before inward expansion','next_construction':'author suffixes whose reversed first 8 letters match the selected clause prefixes, then test offsets 8 onward','provenance':{'audit':'independent mismatch and forward/reverse hashes','reader_gate':'closed; prose control only'}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
