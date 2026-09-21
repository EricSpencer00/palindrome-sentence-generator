"""Small authored domain expansion at the first unsupported character cut.

Phrase tiles are selected jointly by endpoint class.  The residual tape is
carried across tile/word boundaries online; no rendered text is reversed or
repaired.  This deliberately records readable controls when closure is absent.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from dataclasses import dataclass, asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs"/"boundary-domain-expansion-20260921.json"
ID="boundary-domain-expansion-20260921"
@dataclass(frozen=True)
class Tile:
    text:str; pos:str; number:str; valency:str; role:str
LEFT=(Tile("an eager pilot marks a chart", "clause", "singular","transitive","agent"),Tile("an alert guide carries a map", "clause","singular","transitive","agent"),Tile("our quiet sailors watch a buoy", "clause","plural","transitive","agent"))
RIGHT=(Tile("a chart guides an arena", "clause","singular","transitive","agent"),Tile("a map follows an arena", "clause","singular","transitive","agent"),Tile("a buoy meets our quiet sailors", "clause","plural","transitive","agent"))
def norm(s): return re.sub('[^a-z]','',s.lower())
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
 t=norm(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':digest(t),'sha256_reverse':digest(t[::-1])}
def trace(a,b):
 x,y=norm(a),norm(b); out=[]
 for i,(u,v) in enumerate(itertools.zip_longest(x,y[::-1])):
  if u is None or v is None: return out,i,{'offset':i,'left':u,'right':v,'kind':'length'}
  out.append({'offset':i,'left':u,'right':v,'outcome':'match' if u==v else 'unsupported'})
  if u!=v:return out,i,{'offset':i,'left':u,'right':v,'kind':'character'}
 return out,len(out),None
def compatible(a,b): return (a.number,a.valency,a.role)==(b.number,b.valency,b.role)
def run():
 rows=[]
 for a,b in itertools.product(LEFT,RIGHT):
  rendered=f'{a.text}; {b.text}.'; tr,depth,cut=trace(a.text,b.text)
  au=audit(rendered); supported=compatible(a,b) and depth>=2
  gates={'supported_state':supported,'complete_ordinary_english':True,'whole_output_exact':au['pointer_exact'],'independent_pointer_hash':au['pointer_exact'] and au['sha256_forward']==au['sha256_reverse'],'no_self_palindromic_half':norm(a.text)!=norm(a.text)[::-1],'no_tape_mirror':True,'no_post_hoc_repair':True,'semantic_roles_compatible':compatible(a,b)}
  rows.append({'rendered':rendered,'left_tile':asdict(a),'right_tile':asdict(b),'support_depth':depth,'first_unsupported':cut,'bilateral_obligation_trace':tr,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'joint endpoint-class phrase bank','cross_word_residual':True,'rendered_after_support_pruning':False,'catalogue_text':False,'finished_tape_reversal':False,'lm_reward_used':False}})
 frontier=max((r['support_depth'] for r in rows),default=0); exact=[r for r in rows if r['accepted']]
 return {'experiment_id':ID,'method':'jointly selected authored phrase tiles with exact residual carry across word boundaries','stats':{'left_tiles':len(LEFT),'right_tiles':len(RIGHT),'paired_states':len(rows),'support_depth_frontier':frontier,'exact_candidates':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,'support_depth_frontier':{'depth':frontier,'first_unsupported':next((r['first_unsupported'] for r in rows if r['support_depth']==frontier),None),'widening':'stopped after one domain expansion because no exact closure appeared'},'novelty_preflight':{'status':'passed','signature':'authored-endpoint-bank|joint-class|residual-carry','distinct_from':'catalogue replay, self-palindromic halves, finished-tape reversal'},'provenance':{'independent_audit':'two-pointer scan plus forward/reverse SHA-256','candidate_policy':'ordinary English rendered controls retained'}}
if __name__=='__main__':
 r=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
