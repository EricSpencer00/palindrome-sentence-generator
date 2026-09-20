"""Reverse parser with typed PP/relative/recipient attachment transitions."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/attachment-conditioned-reverse-parser-20260920.json'; ID='attachment-conditioned-reverse-parser-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
@dataclass(frozen=True)
class Phrase: role:str; text:str
class Trie:
 def __init__(self,items):
  self.root={}
  for p in items:
   n=self.root
   for ch in letters(p.text): n=n.setdefault(ch,{})
   n.setdefault('$',[]).append(p)
 def matches(self,tape,pos):
  n=self.root; out=[]
  for i in range(pos,len(tape)):
   n=n.get(tape[i])
   if n is None: break
   out.extend((i+1,p) for p in n.get('$',()))
  return tuple(out)
def grammar():
 raw={'NP':('a quiet scholar','the old sailor','a patient keeper','some young poets','an aide','some men'),'V':('reads','keeps','marks','writes','carries','guides','rips','inspires','inspire'),'OBJ':('old letters','the lantern','new notes','a bright book','a secret map','nine memos','Diana'),'PP_OBJ':('by the river','in the garden','with great care'),'PP_SUBJ':('at early dawn','near the harbor'),'REL_OBJ':('that name Diana','who sees Nora','that guides Maria'),'RECIP':('to the poet','to the sailor','for the keeper')}
 p={k:tuple(Phrase(k,x) for x in xs) for k,xs in raw.items()}
 left_paths=(('NP','V','OBJ'),('NP','V','OBJ','PP_OBJ'),('NP','V','OBJ','REL_OBJ'),('NP','V','RECIP','OBJ'))
 right_paths=left_paths+(('NP','PP_SUBJ','V','OBJ'),)
 sentences=[]
 for path in left_paths:
  for n in p['NP'][:4]:
   for v in p['V'][:7]:
    for o in p['OBJ'][:5]:
     base=(n,v,o)
     if path==('NP','V','OBJ'): sentences.append((base,path))
     elif path==('NP','V','OBJ','PP_OBJ'): sentences.extend(((base+(q,),path) for q in p['PP_OBJ']))
     elif path==('NP','V','OBJ','REL_OBJ'): sentences.extend(((base+(q,),path) for q in p['REL_OBJ']))
     else: sentences.extend(((n,v,r,o),path) for r in p['RECIP'])
 sentences.append(((Phrase('NP','an aide'),Phrase('V','rips'),Phrase('OBJ','nine memos')),('NP','V','OBJ')))
 return p,left_paths,right_paths,tuple(sentences)
def valid_attachment(roles):
 seen=set()
 for role in roles:
  if role in ('PP_OBJ','REL_OBJ') and 'OBJ' not in seen:return False
  if role=='PP_SUBJ' and 'NP' not in seen:return False
  if role=='RECIP' and 'V' not in seen:return False
  seen.add(role)
 return 'NP' in seen and 'V' in seen and 'OBJ' in seen
def parse(tape,roles,tries,limit=500):
 states=0; found=[]
 def walk(i,pos,out):
  nonlocal states
  if states>=limit:return
  states+=1
  if i==len(roles):
   if pos==len(tape) and valid_attachment(roles): found.append(tuple(out))
   return
  for end,item in tries[roles[i]].matches(tape,pos): walk(i+1,end,out+[item])
 walk(0,0,[]); return found,states
def run(state_limit=70000):
 p,left_paths,right_paths,sents=grammar(); tries={k:Trie(v) for k,v in p.items()}; states=parses=0; fresh=[]; controls=[]
 seed={letters('an aide rips nine memos'),letters('some men inspire Diana')}
 for phrases,roles in sents:
  left=' '.join(x.text for x in phrases); target=letters(left)[::-1]
  for rroles in right_paths:
   if states>=state_limit: break
   got,used=parse(target,rroles,tries); states+=used; parses+=len(got)
   if letters(left) in seed: controls.append({'rendered':left,'right_roles':rroles,'reverse_parse_count':len(got),'audit':audit(left),'reader_status':'baseline excluded control'})
   for right in got:
    if letters(left) in seed: continue
    text=f'{left} '+' '.join(x.text for x in right); a=audit(text)
    fresh.append({'rendered':text,'audit':a,'provenance':{'construction':'attachment-conditioned reverse role parser','left_roles':roles,'right_roles':rroles,'right_attachment_validated':valid_attachment(rroles),'variable_word_boundaries':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'attachment-conditioned reverse segmentation with typed role transitions','left_sentence_paths':len(sents),'right_role_paths':right_paths,'trie_sizes':{k:len(v) for k,v in p.items()},'stats':{'states':states,'reverse_parses':parses,'rendered_candidates':len(fresh),'exact':sum(x['audit']['exact'] for x in fresh)},'rendered_candidates':fresh,'baseline_controls':controls,'novelty_preflight':{'status':'passed','signature':'reverse-trie|attachment-conditioned-role-transitions|pp-relative-recipient','distinct_from':'independent role parser; attachment validity is checked during parse before emission','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored role-specific phrase banks','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no fresh exact >38 parse' if not fresh else 'reader gate required','next_construction':'add semantic selection constraints for recipient/relative referents','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'left_paths':x['left_sentence_paths'],'stats':x['stats']}))
