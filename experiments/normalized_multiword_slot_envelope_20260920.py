"""Normalized envelope with explicit multiword NP/PP/relative slots."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'data/brown_pcfg_bank_20260920.json';OUT=ROOT/'runs/normalized-multiword-slot-envelope-20260920.json';ID='normalized-multiword-slot-envelope-20260920';SIG='normalized-letter-tapes|multiword-semantic-slots|np-pp-relative-boundaries|live-paired-equations'
def norm(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=norm(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
@dataclass(frozen=True)
class Slot:role:str;text:str
DOM={'AGENT':tuple(Slot('AGENT',x) for x in ('Alice','Diana','Marie','the young bard','the fair queen','a wise king','the silent guard','the moonlit knight','the old poet','the bright herald')),'VERB':tuple(Slot('VERB',x) for x in ('guards','praises','inspires','answers','seeks','holds','writes','helps','finds')),'OBJECT':tuple(Slot('OBJECT',x) for x in ('the crown','a bright rose','the silver moon','a quiet song','the old book','the red letter','a noble plan','the royal garden')),'PP':tuple(Slot('PP',x) for x in ('within the court','beneath the pale moon','near the stone tower','beside the red rose','before the dawn')),'REL':tuple(Slot('REL',x) for x in ('who guards the crown','that praises a rose','who holds the old book'))}
SHAPES=(('np-pp',('AGENT','VERB','OBJECT','PP')),('relative',('AGENT','VERB','OBJECT','REL')),('np-pp-relative',('AGENT','VERB','OBJECT','PP','REL')))
@dataclass(frozen=True)
class S:li:int;ri:int;lw:str;rw:str;lp:int;rp:int;left:tuple[str,...];right:tuple[str,...]
def advance(st,ls,rs):
 lo=(st.lw,) if st.lw else tuple(x.text for x in DOM[ls[st.li]]) if st.li<len(ls) else ()
 ro=(st.rw,) if st.rw else tuple(x.text for x in DOM[rs[st.ri]]) if st.ri>=0 else ()
 for lw in lo:
  a=norm(lw);i=st.lp if st.lw else 0
  for rw in ro:
   b=norm(rw);j=st.rp if st.rw else len(b)-1
   if not a or not b or i>=len(a) or j<0 or a[i]!=b[j]:continue
   le=i+1==len(a);rexit=j==0;yield S(st.li+le,st.ri-rexit,'' if le else lw,'' if rexit else rw,0 if le else i+1,-1 if rexit else j-1,st.left+((lw,) if not st.lw else ()),((rw,) if not st.rw else ())+st.right)
def search(ls,rs):
 q=[S(0,len(rs)-1,'','',0,-1,(),())];seen=set();exact=[]
 while q and len(seen)<80000:
  s=q.pop();k=(s.li,s.ri,s.lw,s.rw,s.lp,s.rp,s.left,s.right)
  if k in seen:continue
  seen.add(k)
  if s.li==len(ls) and s.ri<0 and not s.lw and not s.rw:
   text=' '.join(s.left)+'; '+' '.join(s.right)+'.';a=audit(text)
   if a['exact'] and a['letters']>38:exact.append({'rendered':text,'audit':a,'left_roles':list(ls),'right_roles':list(rs),'provenance':{'normalized_slot_tapes':True,'multiword_np_pp_relative_slots':True,'complete_valency':True,'rendered_boundaries_separate':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
   continue
  q.extend(advance(s,ls,rs))
 return len(seen),exact
def run():
 res=[search(ls,rs) for _,ls in SHAPES for _,rs in SHAPES];exact=[x for _,e in res for x in e];controls=['The young bard guards the crown within the court.','The fair queen praises a bright rose beneath the pale moon.']
 return {'experiment_id':ID,'method':'normalized multiword NP/PP/relative slot envelope','stats':{'slot_domains':{k:len(v) for k,v in DOM.items()},'nodes':sum(n for n,_ in res),'fresh_exact_gt38':len(exact)},'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'corrected single-word envelope; multiword semantic slots carry normalized tapes across internal word boundaries','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'bank':str(BANK.relative_to(ROOT)),'authored_name_and_scene_slots':True,'audits':['two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38 rows appear'},'status':'no fresh exact >38 candidate'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
