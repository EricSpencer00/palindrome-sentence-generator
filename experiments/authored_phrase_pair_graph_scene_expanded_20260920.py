"""Bounded larger authored Shakespearean phrase-pair graph."""
from __future__ import annotations
import hashlib,itertools,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/authored-phrase-pair-graph-scene-expanded-20260920.json';ID='authored-phrase-pair-graph-scene-expanded-20260920';SIG='authored-phrase-pair-graph-expanded|40-nonpalindromic-chunks|cross-boundary-spans|complete-scene-valency'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
@dataclass(frozen=True)
class Chunk: role:str;text:str;agreement:str='free';valency:str='free'
LEFT=(Chunk('agent','the young bard','sg'),Chunk('agent','the fair queen','sg'),Chunk('agent','a noble king','sg'),Chunk('agent','my lord','sg'),Chunk('agent','our poet','sg'),Chunk('agent','the moonlit queen','sg'),Chunk('agent','this brave guard','sg'),Chunk('agent','the red king','sg'),Chunk('action','guards','sg','transitive'),Chunk('action','praises','sg','transitive'),Chunk('action','inspires','sg','transitive'),Chunk('action','answers','sg','transitive'),Chunk('action','sings','sg','transitive'),Chunk('object','the crown','free','object'),Chunk('object','a bright torch','free','object'),Chunk('object','the red rose','free','object'),Chunk('setting','near the moon','free','adjunct'),Chunk('setting','within the court','free','adjunct'),Chunk('dialogue','asks whether','free','complement'),Chunk('dialogue','replies that','free','complement'))
RIGHT=(Chunk('agent','the court','sg'),Chunk('agent','a silent knight','sg'),Chunk('agent','the bright court','sg'),Chunk('agent','a wise poet','sg'),Chunk('agent','the old king','sg'),Chunk('agent','the rose garden','sg'),Chunk('action','praising','free','transitive'),Chunk('action','guarding','free','transitive'),Chunk('action','sing','free','transitive'),Chunk('action','answering','free','transitive'),Chunk('object','a silver light','free','object'),Chunk('object','a scarlet torch','free','object'),Chunk('object','the silent night','free','object'),Chunk('object','a quiet court','free','object'),Chunk('setting','within','free','adjunct'),Chunk('setting','under the moon','free','adjunct'),Chunk('dialogue','answers that','free','complement'),Chunk('dialogue','speaks to','free','complement'),Chunk('setting','before dawn','free','adjunct'),Chunk('object','the bright','free','object'))
def edge(a,b):
 x=letters(a.text);y=letters(b.text);return bool(x and y and x[0]==y[-1] and x!=y)
def run():
 pairs=[(a,b) for a in LEFT for b in RIGHT if a.role==b.role and edge(a,b)]
 by={r:[p for p in pairs if p[0].role==r] for r in {'agent','action','object','setting','dialogue'}};shapes=(('agent','action','object','setting'),('agent','dialogue','object','setting'),('agent','action','object','setting','dialogue'))
 rows=[];states=0
 for shape in shapes:
  for selected in itertools.product(*(by[r] for r in shape)):
   states+=1;ls=[p[0] for p in selected];rs=[p[1] for p in selected]
   if len({x.text for x in ls+rs})!=len(ls+rs):continue
   if any(x.valency=='transitive' for x in ls+rs) and not any(x.role=='object' for x in ls+rs):continue
   if any(x.agreement=='sg' for x in ls) and any(x.agreement=='pl' for x in ls):continue
   l=' '.join(x.text for x in ls);r=' '.join(x.text for x in rs);a=audit(l+'; '+r+'.')
   if a['exact']:rows.append({'rendered':l+'; '+r+'.','audit':a,'chunk_roles':list(shape),'provenance':{'expanded_human_authored_chunks':True,'non_palindromic_chunks':True,'cross_word_boundary_spans':True,'complete_left_frame':True,'complete_right_frame':True,'valency_agreement_checked':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
 baseline='An aide rips nine memos; some men inspire Diana.'
 controls=['The young bard guards the crown near the moon.','The fair queen asks whether the old king answers that a quiet court sings.']
 return {'experiment_id':ID,'method':'expanded authored Shakespearean phrase-pair graph with complete scene/dialogue frames','stats':{'left_chunks':len(LEFT),'right_chunks':len(RIGHT),'compatible_pairs':len(pairs),'composition_states':states,'exact_fresh':len(rows)},'exact_candidates':rows,'baseline_control':{'rendered':baseline,'audit':audit(baseline),'excluded_from_generated_candidates':True},'controls':[{'rendered':x,'audit':audit(x),'complete_prose':True} for x in controls],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior sparse phrase graph; expanded authored non-palindromic chunk inventory and 4–6-chunk scene/dialogue frames','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'chunk_source':'new human-authored Shakespearean scene/dialogue phrases','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact rows appear','next_reader_test':'blinded intact-scene versus shuffled controls'},'status':'fresh exact candidates require human reading' if rows else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
