"""Finite role-licensed lexical boundary completion, not corpus phrase retrieval.
Brown supplies frozen POS support; existing morphology supplies realizations.
Caller-owned semantic classes and local constructions are explicit assumptions.
"""
from pathlib import Path
import gzip,json,hashlib,re
from .bilateral_seams import norm
from .agreement_morphology import Agreement,Lexeme,MorphologyTransducer
from .typed_constituents import VERBS
ROOT=Path(__file__).resolve().parents[1]
RESOURCES=('tools/polaris/payload/brown.json.gz','llm_palindrome/typed_constituents.py','llm_palindrome/agreement_morphology.py','data/lexicon.txt')
# Semantic classes are frozen licensing decisions, not inferred by Brown tags.
CONCEPTS={'haste':('rush','answer','respond','speak'),'deliberate':('consider','decide','think'),'acknowledge':('nod','answer','respond'),'perceive':('see',)}

def lex(lemma,pos,form='base'):
 return {'lemma':lemma,'pos':pos,'form':form}
def fixed(word,pos):return [lex(word,pos)]
def verbs(concept):return [lex(v,'VERB') for v in CONCEPTS[concept]]

def recipes(role,args):
 c=args.get('construction');patient=args.get('patient_surface','question')
 if role=='perception_vp':
  return [('VP -> V(base) NP(named)',[verbs(args.get('predicate','perceive')),fixed(patient,'NOUN')])]
 if role=='opening' and c=='negative_advice':
  return [('S1 -> AUX(Do) NEG V(base)',[fixed('Do','VERB'),fixed('not','ADV'),verbs('haste')])]
 if role=='opening' and c=='direct_deliberation':
  return [('IMP -> V(transitive) DET N(question)',[[lex(v,'VERB') for v in ('consider','decide')],fixed('the','DET'),fixed(patient,'NOUN')]),('IMP -> V(think) ADP DET N(question)',[fixed('Think','VERB'),fixed('about','ADP'),fixed('the','DET'),fixed(patient,'NOUN')])]
 if role=='opening' and c=='purpose_advice':
  return [('IMP -> V(take) N(time)',[fixed('Take','VERB'),fixed('time','NOUN')])]
 if role=='opening' and c=='modal_deliberation':
  return [('S -> PRON MODAL V(pause/wait)',[fixed('You','PRON'),fixed('should','VERB'),[lex(v,'VERB') for v in ('pause','wait')]])]
 if role=='ending' and c in ('negative_advice','purpose_advice','modal_deliberation'):
  return [('RESPONSE -> TO V(base)',[fixed('to','PRT'),verbs('acknowledge')])]
 if role=='ending' and c=='direct_deliberation':
  return [('TEMPORAL -> before PRON V(base,second person)',[fixed('before','ADP'),fixed('you','PRON'),verbs('acknowledge')])]
 if role=='cognitive_link' and c in ('negative_advice','purpose_advice','modal_deliberation'):
  return [('PURPOSE -> TO V(cognitive) WH',[fixed('to','PRT'),[lex(v,'VERB') for v in ('consider','decide')],[lex('whether','ADP'),lex('how','ADV')]]),('PURPOSE -> TO V(think) ADP WH',[fixed('to','PRT'),fixed('think','VERB'),fixed('about','ADP'),[lex('whether','ADP'),lex('how','ADV')]])]
 if role=='pause_link' and c=='negative_advice':
  return [('IMP2 -> V(take) N(time)',[fixed('Take','VERB'),fixed('time','NOUN')])]
 if role=='location_link' and c in ('direct_deliberation','purpose_advice','modal_deliberation'):
  return [('PP -> in DET N(location)',[fixed('in','ADP'),fixed('the','DET'),fixed(args.get('location_surface','room'),'NOUN')])]
 if role=='bridge' and c=='direct_deliberation':
  return [('ADV -> with N(care)',[fixed('with','ADP'),fixed('care','NOUN')])]
 return []

class PhraseCompleter:
 def __init__(self,max_requests=128,max_phrases=16):
  if not 1<=max_requests<=128 or not 1<=max_phrases<=16:raise ValueError('completion cap')
  self.table=json.load(gzip.open(ROOT/RESOURCES[0],'rt'))['table'];self.lemmas=frozenset((ROOT/RESOURCES[3]).read_text().lower().split());self.max_requests=max_requests;self.max_phrases=max_phrases;self.calls=[]
  self.resource_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in RESOURCES}
 def realize(self,item,args):
  lemma=item['lemma'];base=lemma.lower();pos=item['pos'];form=item['form']
  if pos not in self.table.get(base,[]):return None,'unsupported_frozen_pos'
  if pos=='VERB' and form!='base':
   if base in VERBS and form in ('third','past'):word=VERBS[base][0 if form=='third' else 1]
   elif form=='third':word=MorphologyTransducer(Agreement(3,'singular','present')).realize(Lexeme(base,'verb'))
   else:return None,'unsupported_morphology'
  else:word=lemma
  if norm(word).lower() not in self.lemmas and base not in self.lemmas:return None,'unsupported_frozen_lemma'
  return word,None
 @staticmethod
 def advance(d,direction,word):
  exposed=norm(word) if direction=='left' else norm(word)[::-1];old=d['letters'];owner=d['side']
  if owner in ('none',direction):return {'side':direction,'letters':old+exposed},0,None
  k=min(len(old),len(exposed))
  for j in range(k):
   if old[j]!=exposed[j]:return None,0,{'reason':'letter_mismatch','offset':j,'owed':old[j],'exposed':exposed[j]}
  if len(old)>len(exposed):return {'side':owner,'letters':old[k:]},k,None
  if len(exposed)>len(old):return {'side':direction,'letters':exposed[k:]},k,None
  return {'side':'none','letters':''},k,None
 def complete_phrase(self,debt,direction,grammatical_role,semantic_arguments):
  if len(self.calls)>=self.max_requests:raise RuntimeError('completion request cap exceeded')
  if direction not in ('left','right') or debt.get('side') not in ('left','right','none') or not re.fullmatch('[a-z]*',debt.get('letters','')):raise ValueError('invalid debt or direction')
  args=semantic_arguments;bindings=args.get('bindings',{});failures=[];out=[];visited=0;truncated=False
  for k,v in args.get('required_bindings',{}).items():
   if bindings.get(k)!=v:
    result={'phrases':[],'failures':[{'reason':'semantic_or_agreement_conflict','feature':k,'expected':v,'actual':bindings.get(k)}],'receipt':{'status':'complete_rejected_bindings','lexical_states':0}}
    self.calls.append({'debt':debt,'direction':direction,'role':grammatical_role,'arguments':args,**result});return result
  grammars=recipes(grammatical_role,args)
  if not grammars:failures.append({'reason':'unlicensed_local_role','role':grammatical_role,'construction':args.get('construction')})
  for derivation,slots in grammars:
   indexes=list(range(len(slots))) if direction=='left' else list(reversed(range(len(slots))))
   def walk(i,state,chosen,trace,paid):
    nonlocal visited,truncated
    visited+=1
    if len(out)>=self.max_phrases:truncated=True;return
    if i==len(indexes):
     words=[chosen[j]['surface'] for j in range(len(slots))];text=' '.join(words)
     if not 2<=len(words)<=6:failures.append({'reason':'phrase_word_bound','words':len(words)});return
     out.append({'text':text,'derivation':derivation,'lexical_parse':[chosen[j] for j in range(len(slots))],'bindings':dict(bindings),'local_obligations':{'construction':args.get('construction'),'role':grammatical_role},'debt':state,'paid_letters':paid,'trace':trace,'resource_hashes':self.resource_hashes});return
    j=indexes[i]
    for item in slots[j]:
     word,error=self.realize(item,args)
     if error:failures.append({'reason':error,'lexeme':item,'derivation':derivation});continue
     new,n,error=self.advance(state,direction,word)
     if error:failures.append({'debt':state,'word':word,'derivation':derivation,**error});continue
     walk(i+1,new,{**chosen,j:{**item,'surface':word}},trace+[{'word':word,'direction':direction,'before':state,'after':new,'paid':n}],paid+n)
   walk(0,{'side':debt['side'],'letters':debt['letters']},{},[],0)
  result={'phrases':out,'failures':failures,'receipt':{'status':'output_truncated' if truncated else 'complete_finite_grammar','lexical_states':visited,'outputs':len(out),'failures':len(failures)}}
  self.calls.append({'debt':debt,'direction':direction,'role':grammatical_role,'arguments':args,**result});return result

_DEFAULT=None
def complete_phrase(debt,direction,grammatical_role,semantic_arguments):
 global _DEFAULT
 if _DEFAULT is None:_DEFAULT=PhraseCompleter()
 return _DEFAULT.complete_phrase(debt,direction,grammatical_role,semantic_arguments)
