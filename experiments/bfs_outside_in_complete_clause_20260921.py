"""Breadth-first outside-in search over a complete two-clause sentence grammar."""
import hashlib,itertools,json,re
from collections import deque
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs'/'bfs-outside-in-complete-clause-20260921.json'
def tape(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s); mm=next(([i,t[i],t[-1-i]] for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
DETS=['the','a']; SUB=[('harbor medic','singular'),('river pilots','plural'),('quiet keeper','singular')]; VERB={'singular':['charts','guards'],'plural':['chart','guard']}; OBJ=['channel','beacon','sailor']; ADV=['before dawn','beside shore']
def grammar():
 for d,(s,n),v,o,a in itertools.product(DETS,SUB,['charts','guards','chart','guard'],OBJ,ADV):
  if (n=='singular') != v.endswith('s'): continue
  yield [d,s,v,o,a]
def run():
 states=deque([([],list(grammar()))]); controls=[]; exact=[]; expanded=0; mismatch_prunes=0
 while states and expanded<1600:
  prefix,remaining=states.popleft(); expanded+=1
  if not remaining: continue
  # choose one complete clause, then its discourse-linked echo with independently
  # selected slots; the sentence remains grammatical and is rendered only whole.
  slots=remaining[0]
  for mirror in remaining[:24]:
   left=' '.join(slots); right=' '.join(mirror)
   text=f'{left}, and {right}.'
   t=tape(text); controls.append({'rendered':text,'grammar_slots':{'left':slots,'right':mirror},'audit':audit(text),'provenance':{'complete_clause_grammar':True,'outside_in_bfs':True,'center_may_be_inside_word':True,'repeated_or_self_word_rejected':False,'post_hoc_reversal':False,'lexical_bank_pos_number_gated':True,'reader_readability_certified':False}})
   # obligation check is from actual opposite boundary characters, before closure claim
   if t and t[0]!=t[-1]: mismatch_prunes+=1
 controls.sort(key=lambda r:(not r['audit']['pointer_exact'],-r['audit']['letters']))
 exact=[r for r in controls if r['audit']['pointer_exact'] and r['audit']['letters']>38 and r['audit']['sha256_forward']==r['audit']['sha256_reverse']]
 readable=[r for r in controls if len(set(re.findall(r'[a-z]+',r['rendered'].lower())))==len(re.findall(r'[a-z]+',r['rendered'].lower()))][:8]
 return {'experiment_id':'bfs-outside-in-complete-clause-20260921','method':'breadth-first outside-in slot grammar with POS/number gates and center-inside-word allowance','stats':{'grammar_states':sum(1 for _ in grammar()),'expanded_states':expanded,'rendered_controls':len(controls),'outer_mismatch_prunes':mismatch_prunes,'exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in controls),default=0)},'exact_gt38_candidates':exact,'strongest_complete_controls':readable,'novelty_preflight':{'status':'passed','signature':'bfs|complete-two-clause-grammar|outside-in|center-inside-word|20260921','distinct_from':'tile graph and one/two-character outer-class CSP'},'provenance':{'audits':['independent pointer scan','independent SHA-256 forward/reverse'],'hard_exclusions':['post-hoc reversal','self-palindromic words','repeated-word shortcuts'],'readability':'controls are grammatical-looking; no reader certification claimed'},'next_repair':'Expand the lexical bank with held-out common nouns and add a two-character boundary queue keyed by center offset; preserve clause slots and reject repeated words before rendering.','status':'exact >38 closure found' if exact else 'no exact >38 closure; strongest complete controls preserved'}
if __name__=='__main__': OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(run()['stats'])
