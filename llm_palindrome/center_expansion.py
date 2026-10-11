"""Nested-center substitution with explicit finite full-text seam recognition.
The parser is a declared grammar for this seed, not general English.
"""
from pathlib import Path
import gzip,json,re
from .bilateral_seams import norm
from .lexicon import load_lexicon,is_real_word
from .typed_phrase_compiler import repetition_penalty
ROOT=Path(__file__).resolve().parents[1]
SEED='Some men, if editors suss roti, define memos.'
L='somemenifeditors';P='sus';R=L[::-1]
# Argument classes are fixed before candidate generation; not corpus-inferred.
FOOD_ACTIONS=('cut','slice','toast','warm','turn','tap','top','tip','draw','read','inspect','describe','suss')
OWNERS={'pat':('Pat',"Pat's",'person'),'pot':('pots',"pots'",'container'),'pit':('pits',"pits'",'location'),'ward':('wards',"wards'",'institution')}
CENTERS=('sus','draw a ward','live on no evil','step on no pets','reward a drawer',"Don't nod",'no evil did live on','nurses run','drawn inward','stop spots','sees','level','refer','repaper')

def seed_span(radius):
 if not 0<=radius<len(L):raise ValueError('radius must leave a fixed outer prefix')
 left=L[:-radius] if radius else L;right=left[::-1]
 original=norm(SEED)[len(left):len(norm(SEED))-len(right)]
 assert norm(SEED)==left+original+right and original==original[::-1]
 return {'radius':radius,'left':left,'original_center':original,'right':right}

def nested_center(inner,wrapper='s'):
 tape=norm(inner);assert tape and tape==tape[::-1]
 return wrapper+tape+wrapper[::-1]

def replace_center(radius,center):
 s=seed_span(radius);q=norm(center)
 if not q or q!=q[::-1]:raise ValueError('replacement center not palindromic')
 tape=s['left']+q+s['right'];assert tape==tape[::-1]
 return {**s,'replacement_center':q,'tape':tape,'letters':len(tape)}

class SeedSeamParser:
 def __init__(self):
  self.table=json.load(gzip.open(ROOT/'tools/polaris/payload/brown.json.gz','rt'))['table'];self.lexicon=load_lexicon(str(ROOT/'data/lexicon.txt'))
 def licenses_word(self,w,pos):return pos in self.table.get(w.lower(),[]) and is_real_word(w.lower(),self.lexicon)
 def parse(self,tape):
  # Walk shared syntactic interfaces against the candidate tape. No candidate
  # full sentences are inputs; each derivation licenses both exposed seams.
  head='somemenifeditors';tail='definememos';out=[]
  if tape.startswith(head) and tape.endswith(tail):
   middle=tape[len(head):-len(tail)]
   if middle=='sussroti':
    out.append({'text':SEED,'derivation':'NP(plural), if NP(plural) V(base suss) NP(food), V(base define) NP(document).','bindings':{'main_agent':'men1','condition_agent':'editors1','food':'roti1','document':'memos1'},'seed_control':True,'new_seams':False})
   for verb in FOOD_ACTIONS:
    if not self.licenses_word(verb,'VERB') or not middle.startswith(verb):continue
    object_tape=middle[len(verb):]
    for root,(plural_or_name,possessive,kind) in OWNERS.items():
     if not self.licenses_word(root,'NOUN'):continue
     if object_tape!=norm(possessive+' roti'):continue
     if kind!='person' and not is_real_word(plural_or_name,self.lexicon):continue
     text=f'Some men, if editors {verb} {possessive} roti, define memos.'
     out.append({'text':text,'derivation':'NP(plural), if NP(plural) V(base transitive) [NP(owner '+kind+') POSS NP(food)], V(base define) NP(document).','bindings':{'main_agent':'men1','condition_agent':'editors1','owner':root+'1','food':'roti1','document':'memos1'},'seed_control':False,'new_seams':True,'seam_roles':['center initial s realizes editors plural','center final s realizes possessive morphology'],'semantic_scope':'Conditional physical depiction/handling/checking of the same roti precedes memo definition. Relation is a declared documentation scenario, not entailed by wording.'})
  # Explicit independent clauses license the roti -> rot; I resegmentation.
  head='somemenifeditors';tail='rotidefinememos'
  if tape.startswith(head) and tape.endswith(tail):
   middle=tape[len(head):-len(tail)]
   options={'drawawards':('draw awards','V(base draw) NP(plural awards)','depict awards; separate rot/define clauses'), 'liveonnoevils':('live on no evils','V(base live) PP(on NP(no evils))','figurative moral subsistence; separate rot/define clauses')}
   if middle in options:
    vp,derivation,meaning=options[middle];text=f'Some men, if editors {vp}, rot; I define memos.'
    out.append({'text':text,'derivation':'NP(plural), if NP(plural) '+derivation+', V(base intransitive rot); NP(I) V(base define) NP(memos).','bindings':{'main_agent':'men1','condition_agent':'editors1','speaker':'speaker1','document':'memos1'},'seed_control':False,'new_seams':True,'seam_roles':['borrowed right s finishes awards/evils plural','roti is resegmented as rot + I'],'semantic_scope':meaning+'; deliberately separate from food-documentation plan and separately quality-scored.'})
  # Widened centered span may expose Ed as a licensed singular actor. This is
  # a shortening diagnostic, never a length-improvement claim.
  if tape=='somemenifedseesdefinememos':
   out.append({'text':'Some men, if Ed sees, define memos.','derivation':'NP(plural), if NP(named Ed singular) V(third-singular sees, intransitive visual ability), V(base define) NP(memos).','bindings':{'main_agent':'men1','condition_agent':'Ed1','document':'memos1'},'seed_control':False,'new_seams':True,'shortening_diagnostic':True})
  for row in out:
   assert norm(row['text'])==tape and tape==tape[::-1]
   row['repetition_cost']=repetition_penalty(row['text']);row['human_label']='positive' if row['seed_control'] else None;row['quality_review_pending']=True
  return out
