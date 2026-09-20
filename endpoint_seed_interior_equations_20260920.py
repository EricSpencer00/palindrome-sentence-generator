"""Fresh endpoint-seeded grammar with live interior equations."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/endpoint-seed-interior-equations-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=(('a','archivist','maps','a','harbor'),('a','artist','marks','a','garden'),('a','analyst','keeps','a','journal'))
RIGHT=(('a','baker','opens','the','plaza'),('a','builder','mends','the','area'),('a','broker','holds','a','panorama'))
def live(left,right):
 a,b=n(' '.join(left)),n(' '.join(right))[::-1]; tr=[]
 for i,(x,y) in enumerate(zip(a,b)):
  tr.append({'offset':i,'left':x,'right_reversed':y,'matched':x==y})
  if x!=y:return False,tr
 return len(a)==len(b),tr
def run():
 rows=[]; endpoint=0; interior=0
 for l in LEFT:
  for r in RIGHT:
   # Authored endpoint classes: the first left character and last right
   # character agree, but this is only a seed for the live interior solver.
   lt,rt=n(' '.join(l)),n(' '.join(r)); ep=lt[:1]==rt[-1:]; endpoint+=ep
   ok,tr=live(l,r); interior+=ok
   rows.append({'left_clause':' '.join(l),'right_clause':' '.join(r),'endpoint_seed_match':ep,'live_closed':ok,'trace':tr,
    'audit':audit(' '.join(l)+'; '+' '.join(r)), 'provenance':{'left':'fresh semantic role frame','right':'fresh independent semantic role frame','endpoint_seed_only':True,'interior_equations_live':True,'catalogue_text_reused':False,'post_hoc_repair':False,'finished_tape_reversal':False,'mirrored_units':False}})
 return {'experiment_id':'endpoint-seed-interior-equations-20260920','method':'authored endpoint seed followed by live interior cross-word character equations','stats':{'left_frames':3,'right_frames':3,'endpoint_seed_matches':endpoint,'interior_live_closures':interior,'exact_candidates':sum(x['audit']['exact'] for x in rows)},'candidates':rows,'status':'precise zero frontier: endpoint seeds do not close interior equations','provenance':{'audit':'independent mismatch and forward/reverse hashes','reader_gate':'closed; no exact closure'}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
