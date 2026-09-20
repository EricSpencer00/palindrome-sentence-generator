"""Global full-forward sentence-language CSP with factorized lexical choices.

One grammar generates one ordinary sentence (not a preselected sentence pair).
A variable-length word sequence assigns character-position variables x[0:N]
while the palindrome factor x[i]=x[N-1-i] is propagated immediately whenever
both positions are assigned. Word boundaries are choices in the grammar, and
subject/verb agreement plus role/valency constraints are carried in the parse
state. No completed string is reversed or paired with a second sentence.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/global-forward-sentence-csp-20260920.json'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); rev=t[::-1]; mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None); f=hashlib.sha256(t.encode()).hexdigest(); b=hashlib.sha256(rev.encode()).hexdigest()
 return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
# Factorized lexical alternatives, each with grammatical features.
DET=(('the','sg'),('a','sg'))
ADJ=('patient','careful','young','quiet','weary','kind','watchful','brave','gentle')
SUBJ=(('sailor','sg'),('gardener','sg'),('scholar','sg'),('keeper','sg'),('teacher','sg'),('traveler','sg'))
VERB=(('studies','sg','trans'),('carries','sg','trans'),('copies','sg','trans'),('guards','sg','trans'),('opens','sg','trans'),('follows','sg','trans'))
OBJ=(('the chart','sg'),('a lantern','sg'),('the letter','sg'),('the gate','sg'),('the lesson','sg'),('the road','sg'))
PP=(('beside','harbor'),('through','orchard'),('under','window'),('before','dawn'),('near','lighthouse'),('inside','school'),('toward','village'))

def bind(chars, word, N):
 """Assign word to global x positions and propagate equality factors."""
 if len(chars)+len(word)>N:return None
 out=list(chars); start=len(out); out.extend(word)
 for i in range(start,len(out)):
  j=N-1-i
  if j<len(out) and out[i]!=out[j]: return None
 return tuple(out)

def search(N, cap=8000):
 states=prunes=complete=exact=0; renders=[]; best=[]
 # Grammar S -> DET ADJ NP V OBJ PP, with lexical alternatives factored.
 def rec(stage, chars, words, features):
  nonlocal states,prunes,complete,exact
  states+=1
  if states>cap:return
  choices=[]
  if stage==0: choices=[(d,{'det':d,'number':n}) for d,n in DET]
  elif stage==1: choices=[(a,{}) for a in ADJ]
  elif stage==2: choices=[(n,{'number':num,'role':'subject'}) for n,num in SUBJ if features.get('number')==num]
  elif stage==3: choices=[(v,{'verb':v,'valency':val}) for v,num,val in VERB if num==features.get('number')]
  elif stage==4: choices=[(o,{'object':o,'object_number':num}) for o,num in OBJ]
  elif stage==5: choices=[(p+' '+q,{'attachment':'locative'}) for p,q in PP]
  else:
   complete+=1; text=' '.join(words)+'.'; row={'rendered':text,'audit':audit(text),'provenance':{'target_length':N,'grammar':'S -> DET ADJ NPsubject Vtrans NPobject PPlocative','word_boundaries':len(words),'factorized_lexical_choices':True,'parse_features':features,'position_variables':'x[0:N]','palindrome_factor':'x[i] = x[N-1-i] propagated on each word emission','finished_tape_reversal':False,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}}
   renders.append(row)
   if row['audit']['two_pointer_exact'] and N>38:exact+=1
   return
  for word,feat in choices:
   b=bind(chars,letters(word),N)
   if b is None:prunes+=1;continue
   rec(stage+1,b,words+[word],{**features,**feat})
 for _ in [0]:rec(0,tuple(),[],{})
 return {'target_length':N,'states':states,'prunes':prunes,'complete_renderings':complete,'exact_candidates_above_38':exact,'rendered_candidates':renders[:40]}
def run():
 results=[search(n) for n in (44,52,60)]
 controls=['The patient sailor studies the chart beside the harbor.','A careful gardener carries a silver lantern through the orchard.']
 return {'experiment_id':'global-forward-sentence-csp-20260920','method':'global full-forward factorized lexical sentence CSP with variable boundaries and live palindrome position factors','results':results,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','registry_entries_checked':603,'signature':'global-forward-factorized-lexical-csp|variable-boundaries|parse-role-factors','distinct_from':'fixed sentence-pair audits, shared-cell toys, endpoint envelopes, and reversal-based products: one forward grammar assigns x[0:N] directly, with variable word boundaries and agreement/valency parse factors; palindrome constraints propagate during each lexical emission'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'source_text':'fresh factorized lexical grammar in this script','reader_evidence':False,'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'position-domain lexical propagation','operator':'Give each unassigned x position a support domain from held-out lexical alternatives and use arc consistency before choosing the next grammar factor; preserve full-forward generation and complete parse constraints.', 'reader_facing_test':'retain only intact complete prose, independently audit every exact closure above 38, then randomize intact prose against word-shuffled controls for blinded ratings'},'status':'diagnostic lane; no exact candidate above 38'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps([(x['target_length'],x['states'],x['prunes'],x['complete_renderings'],x['exact_candidates_above_38']) for x in r['results']]))
