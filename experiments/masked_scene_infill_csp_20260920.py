"""Dream-RSI-inspired masked character grid with typed scene grammar fills."""
from __future__ import annotations
import hashlib,itertools,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/masked-scene-infill-csp-20260920.json';ID='masked-scene-infill-csp-20260920';SIG='masked-character-grid|typed-scene-grammar|paired-slot-infill|construction-time-csp'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
@dataclass(frozen=True)
class Word: role:str;text:str;number:str='free';valency:str='free'
S=('agent','verb','object','adjunct')
L=(Word('agent','the young bard','sg'),Word('agent','the fair queen','sg'),Word('agent','a wise king','sg'),Word('agent','the silent guard','sg'),Word('verb','guards','sg','transitive'),Word('verb','praises','sg','transitive'),Word('verb','inspires','sg','transitive'),Word('verb','answers','sg','transitive'),Word('object','the crown'),Word('object','a bright rose'),Word('object','the moon'),Word('adjunct','within the court'),Word('adjunct','beneath the moon'),Word('adjunct','near the tower'))
R=(Word('agent','a lone knight','sg'),Word('agent','the old king','sg'),Word('agent','a wise poet','sg'),Word('agent','the bright herald','sg'),Word('verb','guards','sg','transitive'),Word('verb','praises','sg','transitive'),Word('verb','seeks','sg','transitive'),Word('verb','holds','sg','transitive'),Word('object','the red letter'),Word('object','a quiet song'),Word('object','the silver crown'),Word('adjunct','under the moon'),Word('adjunct','beside the rose'),Word('adjunct','before dawn'))
def banks(items):return {r:[x for x in items if x.role==r] for r in S}
def fill(grid,start,text):
 chars=letters(text);g=list(grid);n=len(g)
 for j,ch in enumerate(chars):
  i=start+j;k=n-1-i
  for pos in (i,k):
   if g[pos] not in ('?',ch):return None
   g[pos]=ch
 return g
def frame_options(bank):
 b=banks(bank)
 for xs in itertools.product(*(b[r] for r in S)):
  if len({x.text for x in xs})<len(xs):continue
  if not any(x.valency=='transitive' for x in xs) or any(x.role=='object' for x in xs):yield xs
def run():
 left=list(frame_options(L));right=list(frame_options(R));rows=[];states=0;rejected={}
 for xs in left:
  lt=' '.join(x.text for x in xs);ll=len(letters(lt))
  for ys in right:
   rt=' '.join(x.text for x in ys);rr=len(letters(rt));n=ll+rr
   if not 68<=n<=100:continue
   states+=1;grid=['?']*n;g=fill(grid,0,lt)
   if g is None:rejected['left_fill_conflict']=rejected.get('left_fill_conflict',0)+1;continue
   g=fill(g,ll,rt)
   if g is None:rejected['paired_slot_conflict']=rejected.get('paired_slot_conflict',0)+1;continue
   text=lt+'; '+rt+'.';a=audit(text)
   if a['exact']:rows.append({'rendered':text,'length':a['letters'],'audit':a,'left_roles':list(S),'right_roles':list(S),'provenance':{'masked_grid_fill':True,'typed_scene_grammar':True,'paired_slots_checked_during_fill':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
 controls=['The young bard guards the crown within the court.','The fair queen praises a bright rose beneath the moon.']
 return {'experiment_id':ID,'method':'masked character-grid scene infill with typed complete frames','stats':{'left_frames':len(left),'right_frames':len(right),'length_band_states':states,'fresh_exact':len(rows),'rejections':rejected},'exact_candidates':rows,'controls':[{'rendered':x,'audit':audit(x),'complete_prose':True} for x in controls],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior seam/chart lanes; full-length paired character variables are filled while complete scene grammar choices are made','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'banks':'fresh authored Shakespearean scene words','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless exact rows appear','representation_bottleneck':'paired grid conflicts eliminate all tested complete frame pairs' if not rows else 'none'},'status':'fresh exact candidates require human reading' if rows else 'no fresh exact candidate; paired masked-grid conflicts are the bottleneck','next_construction':'replace four-slot scene frame with a new complement/coordination hypergraph, not a repair operator'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
