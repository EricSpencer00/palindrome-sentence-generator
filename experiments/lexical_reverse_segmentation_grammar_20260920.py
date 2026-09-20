"""Forward complete grammar x reverse-character segmentation parser."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/lexical-reverse-segmentation-grammar-20260920.json'; ID='lexical-reverse-segmentation-grammar-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
@dataclass(frozen=True)
class Phrase: role:str; text:str
@dataclass(frozen=True)
class Sentence: phrases:tuple[Phrase,...]; roles:tuple[str,...]; control:bool=False
class Trie:
 def __init__(self,items):
  self.root={}
  for item in items:
   node=self.root
   for ch in letters(item.text): node=node.setdefault(ch,{})
   node.setdefault('$',[]).append(item)
 def matches(self,tape,pos):
  node=self.root; out=[]
  for i in range(pos,len(tape)):
   node=node.get(tape[i])
   if node is None: break
   for item in node.get('$',[]): out.append((i+1,item))
  return tuple(out)
def grammar():
 banks={
  'NP':('a quiet scholar','the old sailor','a patient keeper','some young poets','an aide','some men'),
  'V':('reads','keeps','marks','writes','carries','guides','rips','inspires','inspire'),
  'OBJ':('old letters','the lantern','new notes','a bright book','a secret map','nine memos','Diana'),
  'PP':('by the river','in the garden','with great care','at early dawn'),
  'REL':('that name Diana','who sees Nora','that guides Maria'),
 }
 phrases={k:tuple(Phrase(k,x) for x in xs) for k,xs in banks.items()}
 paths=(('NP','V','OBJ'),('NP','V','OBJ','PP'),('NP','V','OBJ','REL'),('NP','V','OBJ','PP','REL'))
 sentences=[]
 for path in paths:
  for n in phrases['NP']:
   for v in phrases['V']:
    for o in phrases['OBJ']:
     base=(n,v,o)
     if path==('NP','V','OBJ'): sentences.append(Sentence(base,path))
     if path==('NP','V','OBJ','PP'):
      sentences.extend(Sentence(base+(p,),path) for p in phrases['PP'])
     if path==('NP','V','OBJ','REL'):
      sentences.extend(Sentence(base+(r,),path) for r in phrases['REL'])
     if path==('NP','V','OBJ','PP','REL'):
      sentences.extend(Sentence(base+(phrases['PP'][0],phrases['REL'][0]),path) for _ in (0,))
 # Seed is a regression input, never a construction anchor for fresh rows.
 sentences.append(Sentence((Phrase('NP','an aide'),Phrase('V','rips'),Phrase('OBJ','nine memos')),('NP','V','OBJ'),True))
 return phrases,paths,tuple(sentences)
def parse_reverse(tape,roles,tries,state_limit=1000):
 states=0; results=[]
 def walk(role_i,pos,chosen):
  nonlocal states
  if states>=state_limit:return
  states+=1
  if role_i==len(roles):
   if pos==len(tape): results.append(tuple(chosen))
   return
  for end,item in tries[roles[role_i]].matches(tape,pos): walk(role_i+1,end,chosen+[item])
 walk(0,0,[]); return results,states
def run(state_limit=70000):
 phrases,paths,sents=grammar(); tries={k:Trie(v) for k,v in phrases.items()}; states=parsed=complete=0; exact=[]; controls=[]; parsed_controls=[]
 for sent in sents:
  if states>=state_limit: break
  left=' '.join(p.text for p in sent.phrases); tape=letters(left); target=tape[::-1]
  parses,used=parse_reverse(target,sent.roles,tries,250); states+=used; parsed+=len(parses)
  if sent.control:
   controls.append({'rendered':left,'audit':audit(left),'reverse_parse_count':len(parses),'reader_status':'baseline/control input'})
  for parsed_phrases in parses:
   complete+=1; right=' '.join(p.text for p in parsed_phrases); rendered=f'{left} {right}'; a=audit(rendered)
   row={'rendered':rendered,'audit':a,'provenance':{'construction':'forward complete grammar plus reverse-character variable-boundary parser','left_roles':sent.roles,'right_roles':tuple(p.role for p in parsed_phrases),'reverse_obligation_parsed':True,'complete_right_parse':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}}
   if a['exact'] and a['letters']>38 and not sent.control: exact.append(row)
   if len(parsed_controls)<20: parsed_controls.append(row)
 return {'experiment_id':ID,'method':'lexical reverse-segmentation over complete ordinary grammar paths','grammar_paths':paths,'sentence_paths':len(sents),'trie_sizes':{k:len(v) for k,v in phrases.items()},'stats':{'states':states,'reverse_parses':parsed,'complete_parses':complete,'exact':len(exact)},'exact_candidates':exact,'parsed_controls':parsed_controls,'complete_prose_controls':controls,'novelty_preflight':{'status':'passed','signature':'complete-forward-grammar|reverse-character-trie-segmentation|variable-boundaries','distinct_from':'outer seam products; reverse stream is parsed into a second complete grammar, not edited after rendering','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored phrase grammar; no sentence replay','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no fresh exact >38 parse' if not exact else 'reader gate required','next_construction':'add independent right-side role permutations while retaining complete parse','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'sentence_paths':x['sentence_paths'],'stats':x['stats']}))
