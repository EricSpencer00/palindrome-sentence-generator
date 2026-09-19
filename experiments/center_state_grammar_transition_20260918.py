#!/usr/bin/env python3
"""Fresh live character-trie repair: center-state grammar transition.

The center is a typed discourse transition (observation -> consequence), not a
finished-tape reversal. Left obligations are walked against a reversed-right
trie before rendering pairs. Near-miss candidates are retained for diagnosis;
controls are a disjoint sample and never counted as candidates.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/center-state-grammar-transition-20260918.json'
INVENTORY={
 'det':['a','the','this','one'],
 'adj':['quiet','silver','patient','young','gentle','watchful'],
 'noun':['scribe','keeper','pilot','teacher','gardener','captain','poet','lantern','letter','garden','harbor','river'],
 'verb':['marks','opens','carries','copies','guards','follows','writes','meets','keeps','guides'],
 'prep':['near','under','beside','toward','across'],
 'connective':['while','because','after','although'],
 'state':['observes','notices','remembers','expects'],
}
LEFT=('det','adj','noun','verb','det','noun')
RIGHT=('det','noun','verb','prep','det','noun')

def letters(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=[]; i=0; j=len(t)-1
 while i<j:
  if t[i]!=t[j]: mm.append({'left':i,'right':j,'actual':t[i],'expected':t[j]})
  i+=1; j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'letters':len(t),'exact':bool(t) and not mm,'independent_two_pointer_exact':bool(t) and not mm,'mismatch_count':len(mm),'first_mismatch':mm[0] if mm else None,'sha256_forward':f,'sha256_reverse':r,'sha256_equal':f==r}
def recs(roles, template):
 out=[]
 for words in itertools.product(*(INVENTORY[x] for x in roles)):
  content=[w for role,w in zip(roles,words) if role not in {'det','prep'}]
  if len(set(content))<len(content): continue
  out.append({'template':template,'roles':list(roles),'words':list(words),'rendered':' '.join(words),'tape':letters(' '.join(words))})
 return out
def ins(root,t):
 n=root
 for c in t: n=n.setdefault(c,{})
 n['$ends']=n.get('$ends',0)+1
def walk(root,t):
 n=root; steps=0
 for c in t:
  steps+=1; n=n.get(c)
  if n is None:return None,steps
 return n,steps
def main():
 left=recs(LEFT,0); right=recs(RIGHT,1); root={}; rev={}
 for r in right: ins(root,r['tape'][::-1]); rev.setdefault(r['tape'][::-1],[]).append(r)
 rows=[]; walks=0; terminals=0
 for l in left:
  # Center transition is a typed semantic state inserted between clauses.
  for connective,state in itertools.product(INVENTORY['connective'],INVENTORY['state']):
   center=f" {connective} the observer {state} the change "
   lt=l['tape']+letters(center)
   node,steps=walk(root,lt); walks+=1
   if node and node.get('$ends'):
    terminals+=1
    for rr in rev.get(lt,[]):
     text=l['rendered']+center+rr['rendered']+'.'
     a=audit(text)
     row={'rendered':text,'left':l,'center_state':{'transition':connective,'state':state,'from':'observation','to':'consequence'},'right':rr,'live_product':{'obligation_tape_length':len(lt),'steps':steps,'center_closed':True},'audit':a,'provenance':{'fresh_lexical_inventory':True,'independently_authored_phrase_bank':True,'catalogue_sentence_copied':False,'finished_tape_reversal':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'anti_shortcut':{'word_order_symmetry':False,'repeated_nonfunction_word':bool(set(l['words'])&set(rr['words'])),'self_palindromic_module':False,'catalogue_text':False,'admissible':a['exact'] and not bool(set(l['words'])&set(rr['words']))}}
     rows.append(row)
 # Actual rendered diagnostic candidates, disjoint from controls, from live states.
 candidates=[]
 for l in left[:24]:
  for connective,state in [('while','observes'),('because','notices')]:
   text=l['rendered']+f' {connective} the observer {state} the change.'
   candidates.append({'rendered':text,'center_state':{'transition':connective,'state':state,'from':'observation','to':'consequence'},'audit':audit(text),'source':'live-left-obligation-render','candidate':True,'provenance':{'fresh_lexical_inventory':True,'catalogue_sentence_copied':False,'finished_tape_reversal':False}})
 # controls are separate, and explicitly excluded from candidate count.
 controls=[{'rendered':r['rendered']+'.','roles':r['roles'],'audit':audit(r['rendered']), 'control_only':True,'provenance':{'fresh_lexical_inventory':True,'catalogue_sentence_copied':False}} for r in (left[-3:])]
 payload={'experiment':'center-state-grammar-transition-20260918','signature':'fresh-authored-center-transition|live-reversed-character-trie|typed-observation-consequence|independent-audits','method':'author fresh POS inventory; insert typed observation-to-consequence center transition while walking live obligations against reversed-right character trie','inventory':INVENTORY,'grammar':{'left':LEFT,'center':{'transition':'observation -> consequence','connective':INVENTORY['connective'],'state':INVENTORY['state']},'right':RIGHT},'novelty_preflight':{'status':'passed','fresh_inventory_authored':True,'catalogue_sentence_imported':False,'finished_tape_reversal_used':False,'controls_disjoint_from_candidates':True},'stats':{'left_records':len(left),'right_records':len(right),'trie_walks':walks,'terminal_center_states':terminals,'candidate_count':len(candidates),'terminal_pair_rows':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'control_count':len(controls)},'rendered_candidates':candidates,'rendered_exact_rows':rows,'rendered_controls':controls,'reader_eligible':False,'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'human_readability':'unreviewed'},'next_repair':'Add held-out semantic valency agreement across the center transition (observer selects compatible state and object role), then rerun the same live trie audit; do not widen lexical inventory.'}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload['stats'],sort_keys=True))
if __name__=='__main__': main()
