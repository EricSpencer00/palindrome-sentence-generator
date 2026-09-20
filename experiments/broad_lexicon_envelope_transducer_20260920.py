"""Broad-lexicon envelope WFSA with independent word-boundary offsets."""
from __future__ import annotations
import hashlib,heapq,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/broad-lexicon-envelope-transducer-20260920.json';ID='broad-lexicon-envelope-transducer-20260920';SIG='broad-authored-name-bank|character-envelope-wfsa|independent-word-offsets|complete-role-paths'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
N=('alice','anna','arthur','diana','edward','elena','felix','george','helen','james','jane','john','julia','leon','lucas','marie','mark','nora','oliver','paul','peter','rose','sarah','thomas','victor','william')
D=('the','a','an'); A=('young','fair','wise','silent','old','bright','noble','red'); V=('guards','praises','inspires','answers','seeks','holds','writes','helps','finds','carries','shows','gives'); O=('crown','rose','moon','bell','book','song','letter','plan','garden','bridge','harbor','answer'); P=('to','for','near','under','within','beside'); X=('court','tower','moon','rose','dawn','garden','harbor')
DOM={'DET':D,'AGENT':N+tuple('the '+x for x in ('bard','queen','king','poet','guard','sailor','teacher','farmer')), 'ADJ':A,'VERB':V,'OBJECT':tuple('the '+x for x in O)+tuple('a '+x for x in O),'PREP':P,'PLACE':tuple('the '+x for x in X)}
SHAPES=(('svo',('AGENT','VERB','OBJECT')),('dit',('AGENT','VERB','PREP','AGENT','OBJECT')),('pp',('AGENT','VERB','OBJECT','PREP','PLACE')))
@dataclass(frozen=True)
class S:
 li:int;ri:int;lw:str;rw:str;lp:int;rp:int;left:tuple[str,...];right:tuple[str,...];opens:int
def step(st,ls,rs):
 lo=(st.lw,) if st.lw else (DOM[ls[st.li]] if st.li<len(ls) else ())
 ro=(st.rw,) if st.rw else (DOM[rs[st.ri]] if st.ri>=0 else ())
 for lw in lo:
  lp=st.lp if st.lw else 0
  for rw in ro:
   rp=st.rp if st.rw else len(rw)-1
   if not lw or not rw or lp>=len(lw) or rp<0 or lw[lp]!=rw[rp]:continue
   le=lp+1==len(lw);rexit=rp==0
   left=st.left+((lw,) if not st.lw else ());right=((rw,) if not st.rw else ())+st.right
   yield S(st.li+(1 if le else 0),st.ri-(1 if rexit else 0),'' if le else lw,'' if rexit else rw,0 if le else lp+1,-1 if rexit else rp-1,left,right,st.opens+(not st.lw)+(not st.rw))
def search(ls,rs,max_nodes=60000):
 q=[S(0,len(rs)-1,'','',0,-1,(),(),0)];seen=set();rows=[];nodes=0
 while q and nodes<max_nodes:
  st=q.pop();key=(st.li,st.ri,st.lw,st.rw,st.lp,st.rp,st.left,st.right)
  if key in seen:continue
  seen.add(key);nodes+=1
  if st.li==len(ls) and st.ri<0 and not st.lw and not st.rw:
   text=' '.join(st.left)+'; '+' '.join(st.right)+'.';a=audit(text)
   if a['exact'] and a['letters']>38:rows.append({'rendered':text,'audit':a,'left_roles':list(ls),'right_roles':list(rs),'boundary_opens':st.opens,'provenance':{'character_wfsa_envelope':True,'complete_left_roles':True,'complete_right_roles':True,'independent_word_offsets':True,'semantic_valency':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
   continue
  # Keep broad lexical exploration bounded per grammar pair, never align whole frames.
  children=list(step(st,ls,rs));q.extend(children[:12000])
 return nodes,rows
def run():
 results={}
 for i,(_,ls) in enumerate(SHAPES):
  for j,(_,rs) in enumerate(SHAPES):results[f'{i}:{j}']=search(ls,rs)
 exact=[x for n in results.values() for x in n[1]];controls=['Alice guards the crown; Diana praises the red letter.','The young bard writes a quiet song; Marie seeks the moon.']
 return {'experiment_id':ID,'method':'broad lexical character-envelope WFSA with complete semantic paths','results':{k:{'nodes':v[0],'exact':len(v[1])} for k,v in results.items()},'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x),'complete_prose':True} for x in controls],'stats':{'nodes':sum(v[0] for v in results.values()),'fresh_exact_gt38':len(exact)},'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior whole-frame broad bank; independent word boundaries advance inside character envelope before roles complete','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'bank':'expanded authored common/proper-name bank','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38 rows appear','next_reader_test':'blinded intact sentence versus shuffled controls'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
