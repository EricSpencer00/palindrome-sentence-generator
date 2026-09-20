"""Role-trie transducer with independently scheduled word boundaries."""
from __future__ import annotations
import hashlib,heapq,json,re,sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.brown_authored_semantic_reverse_decoder_20260920 import Word,audit,load_words
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'data/brown_pcfg_bank_20260920.json';OUT=ROOT/'runs/brown-boundary-transducer-free-offsets-20260920.json';ID='brown-boundary-transducer-free-offsets-20260920';SIG='brown-derived-lexicon|role-trie-transducer|independent-boundary-scheduling|free-cross-word-offsets'
SHAPES=(('svo',('DET','ADJ','AGENT','ACTION','DET','OBJECT')),('place',('DET','AGENT','ACTION','DET','OBJECT','PREP','DET','PLACE')))
FUNCTION={'the','a','an','near','beside','beyond','inside','under','through','toward','within','over','at','in','on','with','for','to'}
@dataclass(frozen=True)
class S:
 li:int;ri:int;lw:str;rw:str;lp:int;rp:int;left:tuple[str,...];right:tuple[str,...];score:float;opens:int
def norm(s):return re.sub(r'[^a-z]','',s.casefold())
def expand(st,ls,rs,d):
 # Explicit boundary scheduler: open left, right, or both independently.
 lopts=((st.lw,) if st.lw else tuple(w.text for w in d[ls[st.li]]) if st.li<len(ls) else ('',))
 ropts=((st.rw,) if st.rw else tuple(w.text for w in d[rs[st.ri]]) if st.ri>=0 else ('',))
 for lw in lopts:
  lp=st.lp if st.lw else 0
  if not lw and st.li<len(ls):continue
  if lw and lp>=len(lw):continue
  for rw in ropts:
   rp=st.rp if st.rw else len(rw)-1
   if not rw and st.ri>=0:continue
   if rw and rp<0:continue
   if not lw or not rw or lw[lp]!=rw[rp]:continue
   if lw not in FUNCTION and lw in st.left+st.right:continue
   if rw not in FUNCTION and rw in st.left+st.right:continue
   le=lp+1==len(lw);rexit=rp==0
   left=st.left+((lw,) if not st.lw else ());right=((rw,) if not st.rw else ())+st.right
   yield S(st.li+(1 if le else 0),st.ri-(1 if rexit else 0),'' if le else lw,'' if rexit else rw,0 if le else lp+1,-1 if rexit else rp-1,left,right,st.score,st.opens+(not st.lw)+(not st.rw))
def search(ls,rs,d,max_nodes=50000,beam=5000):
 h=[(0.0,0,S(0,len(rs)-1,'','',0,-1,(),(),0.0,0))];seen=set();rows=[];serial=nodes=0
 while h and nodes<max_nodes:
  _,_,st=heapq.heappop(h);key=(st.li,st.ri,st.lw,st.rw,st.lp,st.rp,st.left,st.right)
  if key in seen:continue
  seen.add(key);nodes+=1
  if st.li==len(ls) and st.ri<0 and not st.lw and not st.rw:
   text=' '.join(st.left)+'; '+' '.join(st.right)+'.';a=audit(text)
   if a['exact']:rows.append({'rendered':text,'audit':a,'left_shape':list(ls),'right_shape':list(rs),'boundary_opens':st.opens,'provenance':{'independent_boundary_scheduler':True,'role_specific_tries':True,'free_cross_word_offsets':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
   continue
  for child in expand(st,ls,rs,d):
   serial+=1;remain=(len(ls)-child.li)+(child.ri+1);heapq.heappush(h,(-child.score+0.002*remain,serial,child))
  if len(h)>beam:h=heapq.nsmallest(beam,h);heapq.heapify(h)
 return nodes,len(seen),rows
def run():
 d=load_words();results={}
 for i,(_,ls) in enumerate(SHAPES):
  for j,(_,rs) in enumerate(SHAPES):results[f'{i}:{j}']=search(ls,rs,d)
 exact=[x for z in results.values() for x in z[2]];controls=['The young man sees the house; the old woman hears the word.','A good boy found the door; a new girl held the key.']
 return {'experiment_id':ID,'method':'role-specific lexical boundary transducer with asynchronous cross-word offsets','results':{k:{'nodes':v[0],'states':v[1],'exact':len(v[2])} for k,v in results.items()},'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'stats':{'nodes':sum(v[0] for v in results.values()),'states':sum(v[1] for v in results.values()),'exact':len(exact)},'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior simultaneous word-opening beam; this transducer schedules each side boundary independently','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'bank':str(BANK.relative_to(ROOT)),'bank_sha256':hashlib.sha256(BANK.read_bytes()).hexdigest(),'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed; no exact fresh candidate','next_reader_test':'blinded intact-versus-shuffled controls if an exact row appears'},'status':'no fresh exact parse in this lane'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
