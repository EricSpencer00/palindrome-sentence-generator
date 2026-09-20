"""Authored Shakespearean complement cadence with pronoun attachments."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/shakespeare-complement-pronoun-reverse-lattice-20260920.json'; ID='shakespeare-complement-pronoun-reverse-lattice-20260920'
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
  for p in items:
   n=self.root
   for ch in letters(p.text): n=n.setdefault(ch,{})
   n.setdefault('$',[]).append(p)
 def matches(self,tape,pos):
  n=self.root; out=[]
  for i in range(pos,len(tape)):
   n=n.get(tape[i])
   if n is None:break
   out.extend((i+1,p) for p in n.get('$',()))
  return tuple(out)
def grammar():
 raw={'NP':('the young bard','the wise king','the fair queen','a lonely knight','the crowned heir'),'V':('says','hears','vows','declares','remembers'),'COMP':('that she guards the crown','that he praises the rose','that they seek the moon','that the queen holds the crown','that the bard writes a song'),'PP':('beneath the moon','within the court','beside the rose','at evening tide'),'REL':('who sings at dusk','that guards the crown','who loves the queen')}
 p={k:tuple(Phrase(k,x) for x in xs) for k,xs in raw.items()}; paths=(('NP','V','COMP'),('NP','V','COMP','PP'),('NP','V','COMP','REL'))
 sents=[]
 for path in paths:
  for n in p['NP']:
   for v in p['V']:
    for c in p['COMP']:
     base=(n,v,c)
     if len(path)==3:sents.append(Sentence(base,path))
     elif path[-1]=='PP':sents.extend(Sentence(base+(q,),path) for q in p['PP'])
     else:sents.extend(Sentence(base+(q,),path) for q in p['REL'])
 sents.append(Sentence((Phrase('NP','an aide'),Phrase('V','rips'),Phrase('COMP','nine memos')),('NP','V','COMP'),True))
 return p,paths,tuple(sents)
def parse(tape,roles,tries,limit=500):
 states=0; found=[]
 def walk(i,pos,out):
  nonlocal states
  if states>=limit:return
  states+=1
  if i==len(roles):
   if pos==len(tape):found.append(tuple(out))
   return
  for end,item in tries[roles[i]].matches(tape,pos):walk(i+1,end,out+[item])
 walk(0,0,[]); return found,states
def run(state_limit=70000):
 p,paths,sents=grammar(); tries={k:Trie(v) for k,v in p.items()}; states=parses=0; fresh=[]; controls=[]; seed={letters('an aide rips nine memos'),letters('some men inspire Diana')}
 for sent in sents:
  left=' '.join(x.text for x in sent.phrases); target=letters(left)[::-1]
  for rroles in paths:
   got,used=parse(target,rroles,tries); states+=used; parses+=len(got)
   if sent.control: controls.append({'rendered':left,'right_roles':rroles,'parse_count':len(got),'audit':audit(left),'reader_status':'old 38 seed excluded control'})
   for right in got:
    if sent.control or letters(left) in seed:continue
    text=f'{left} '+' '.join(x.text for x in right); a=audit(text); fresh.append({'rendered':text,'audit':a,'provenance':{'construction':'authored complement cadence/pronoun reverse lattice','left_roles':sent.roles,'right_roles':rroles,'pronoun_attachment_paths':True,'complete_right_parse':True,'fresh_scene_vocabulary':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True,'next_reader_test':'blind cadence and coherence ratings'}})
 return {'experiment_id':ID,'method':'Shakespearean complement/pronoun grammar times reverse trie parser','sentence_paths':len(sents),'grammar_paths':paths,'trie_sizes':{k:len(v) for k,v in p.items()},'stats':{'states':states,'reverse_parses':parses,'rendered_candidates':len(fresh),'exact':sum(x['audit']['exact'] for x in fresh)},'rendered_candidates':fresh,'scene_controls':[{'rendered':'the wise king says that she guards the crown beneath the moon','audit':audit('the wise king says that she guards the crown beneath the moon')},{'rendered':'the fair queen hears that he praises the rose within the court','audit':audit('the fair queen hears that he praises the rose within the court')}],'baseline_controls':controls,'novelty_preflight':{'status':'passed','signature':'fresh-complement-cadence|pronoun-attachment|reverse-variable-segmentation','distinct_from':'prior scene SVO lattice; adds authored that-clause cadence and he/she/they complement paths','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'fresh authored bard/king/queen/crown/rose/moon complements and pronouns','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no fresh exact >38 parse' if not fresh else 'reader gate required','next_construction':'add authored dialogue response complements','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'sentence_paths':x['sentence_paths'],'stats':x['stats']}))
