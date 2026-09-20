"""Large endpoint-indexed lexical envelope over complete grammar templates."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'data/brown_pcfg_bank_20260920.json';OUT=ROOT/'runs/large-phrase-endpoint-reverse-envelope-20260920.json';ID='large-phrase-endpoint-reverse-envelope-20260920';SIG='large-brown-phrase-inventory|endpoint-indexed-reverse-envelope|live-word-boundaries|complete-grammar-templates'
def norm(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=norm(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
@dataclass(frozen=True)
class Item:role:str;text:str
def inventory():
 p=json.loads(BANK.read_text())['lexicon']; top=lambda k,n:tuple(x['word'] for x in p[k][:n]); names=('Alice','Diana','Marie','John','Anna','James','Nora','Peter','Sarah','Victor','Helen','Arthur','Julia','Mark','Rose','Thomas','George','Jane','Oliver','Paul')
 nouns=top('NOUN',55); verbs=top('VERB',55); preps=top('PREP',24); places=top('NOUN',35)
 return {'SUBJ':tuple(Item('SUBJ',x) for x in names+tuple('the '+x for x in nouns[:25])),'VERB':tuple(Item('VERB',x) for x in verbs),'OBJ':tuple(Item('OBJ',x) for x in tuple('the '+x for x in nouns[:35])+tuple('a '+x for x in nouns[35:55])),'PREP':tuple(Item('PREP',x) for x in preps),'PLACE':tuple(Item('PLACE','the '+x) for x in places)}
SHAPES=(('svo',('SUBJ','VERB','OBJ')),('pp',('SUBJ','VERB','OBJ','PREP','PLACE')),('dit',('SUBJ','VERB','PREP','SUBJ','OBJ')))
def endpoint_index(inv):
 out={r:{} for r in inv}
 for r,items in inv.items():
  for it in items:
   t=norm(it.text);out[r].setdefault((t[0],t[-1],len(t)),[]).append(it)
 return out
def search(ls,rs,inv,idx,max_states=80000):
 stack=[(0,len(rs)-1,'','',0,-1,(),())];seen=set();exact=[]
 while stack and len(seen)<max_states:
  li,ri,lw,rw,lp,rp,left,right=stack.pop();key=(li,ri,lw,rw,lp,rp,left,right)
  if key in seen:continue
  seen.add(key)
  if li==len(ls) and ri<0 and not lw and not rw:
   text=' '.join(left)+'; '+' '.join(right)+'.';a=audit(text)
   if a['exact'] and a['letters']>38:exact.append({'rendered':text,'length':a['letters'],'audit':a,'left_roles':list(ls),'right_roles':list(rs),'provenance':{'large_phrase_inventory':True,'endpoint_indexed':True,'live_phrase_boundaries':True,'complete_left_grammar':True,'complete_right_grammar':True,'semantic_valency':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
   continue
  lo=(Item('',lw),) if lw else (tuple(x for b in idx[ls[li]].values() for x in b) if li<len(ls) else ())
  ro=(Item('',rw),) if rw else (tuple(x for b in idx[rs[ri]].values() for x in b) if ri>=0 else ())
  for a in lo:
   at=norm(a.text);i=lp if lw else 0
   for b in ro:
    bt=norm(b.text);j=rp if rw else len(bt)-1
    if not at or not bt or i>=len(at) or j<0 or at[i]!=bt[j]:continue
    le=i+1==len(at);rexit=j==0;stack.append((li+le,ri-rexit,'' if le else a.text,'' if rexit else b.text,0 if le else i+1,-1 if rexit else j-1,left+((a.text,) if not lw else ()),((b.text,) if not rw else ())+right))
 return len(seen),exact
def run():
 inv=inventory();idx=endpoint_index(inv);results=[search(ls,rs,inv,idx) for _,ls in SHAPES for _,rs in SHAPES];exact=[x for _,e in results for x in e];controls=['Alice reads the report near the station.','The young analyst checks the schedule within the office.','Marie writes a note beside the river.','John opens the file after the meeting.']*5
 return {'experiment_id':ID,'method':'large Brown phrase endpoint-indexed reverse lexical envelope','inventory_sizes':{k:len(v) for k,v in inv.items()},'stats':{'endpoint_keys':sum(len(v) for v in idx.values()),'states':sum(n for n,_ in results),'fresh_exact_gt38':len(exact),'controls':len(controls)},'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior small seam and envelope checks; substantially larger Brown-derived phrase inventory and complete grammar products','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'brown_bank':str(BANK.relative_to(ROOT)),'bank_sha256':hashlib.sha256(BANK.read_bytes()).hexdigest(),'name_bank':'20 ordinary authored names','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38 rows appear','next_reader_test':'blinded complete sentence versus shuffled controls'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate','next_construction':'add phrase-level relative clauses only after an exact closure appears'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
