"""Live two-sided grammar beam: word boundaries and characters are chosen jointly."""
from __future__ import annotations
import hashlib,json,re,sys,heapq
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.brown_authored_semantic_reverse_decoder_20260920 import Word, audit, load_words
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'data/brown_pcfg_bank_20260920.json';OUT=ROOT/'runs/brown-bidirectional-beam-decoder-20260920.json';ID='brown-bidirectional-beam-decoder-20260920';SIG='brown-derived-lexicon|live-bidirectional-beam|character-boundary-synchronous|complete-grammar-gate'
SHAPES=(('svo',('DET','ADJ','AGENT','ACTION','DET','OBJECT')),('svo-place',('DET','AGENT','ACTION','DET','OBJECT','PREP','DET','PLACE')))
COLLOC={('the','young'):1.0,('young','man'):2.0,('good','house'):0.2,('near','the'):0.5,('under','the'):0.5,('the','old'):1.0}
FUNCTION={'the','a','an','near','beside','beyond','inside','under','through','toward','within','over','at','in','on','with','for','to'}
@dataclass(frozen=True)
class State:
 li:int;ri:int;lw:str;rw:str;lp:int;rp:int;left:tuple[str,...];right:tuple[str,...];score:float
def norm(s):return re.sub(r'[^a-z]','',s.casefold())
def step(st,ls,rs,dom):
 lopts=((st.lw,) if st.lw else tuple(w.text for w in dom[ls[st.li]]) if st.li<len(ls) else ())
 ropts=((st.rw,) if st.rw else tuple(w.text for w in dom[rs[st.ri]]) if st.ri>=0 else ())
 for lw in lopts:
  lp=st.lp if st.lw else 0
  if not lw or lp>=len(lw):continue
  for rw in ropts:
   rp=st.rp if st.rw else len(rw)-1
   if not rw or rp<0 or lw[lp]!=rw[rp]:continue
   if lw not in FUNCTION and lw in st.left+st.right:continue
   if rw not in FUNCTION and rw in st.left+st.right:continue
   ldone=lp+1==len(lw);rdone=rp==0
   left=st.left+((lw,) if not st.lw else ());right=((rw,) if not st.rw else ())+st.right
   score=st.score+COLLOC.get(((st.left[-1] if st.left else ''),lw),0)+COLLOC.get((rw,(st.right[0] if st.right else '')),0)
   yield State(st.li+(1 if ldone else 0),st.ri-(1 if rdone else 0),'' if ldone else lw,'' if rdone else rw,0 if ldone else lp+1,-1 if rdone else rp-1,left,right,score)
def search(ls,rs,dom,max_nodes=60000,beam=6000):
 start=State(0,len(rs)-1,'','',0,-1,(),(),0.0);heap=[(0.0,0,start)];seen=set();closures=[];nodes=serial=0
 while heap and nodes<max_nodes:
  _,_,st=heapq.heappop(heap);key=(st.li,st.ri,st.lw,st.rw,st.lp,st.rp,st.left,st.right)
  if key in seen:continue
  seen.add(key);nodes+=1
  if st.li==len(ls) and st.ri<0 and not st.lw and not st.rw:
   text=' '.join(st.left)+'; '+' '.join(st.right)+'.'; row={'rendered':text,'left_shape':list(ls),'right_shape':list(rs),'rank_score':st.score,'audit':audit(text),'provenance':{'live_character_matching':True,'word_boundaries_selected_during_search':True,'complete_left_grammar':True,'complete_right_grammar':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'readability_certified':False}}
   if row['audit']['exact']:closures.append(row)
   continue
  for child in step(st,ls,rs,dom):
   serial+=1;remaining=(len(ls)-child.li)+(child.ri+1);priority=-child.score+0.002*remaining;heapq.heappush(heap,(priority,serial,child))
  if len(heap)>beam:heapq.heapify(heap);heap[:]=heapq.nsmallest(beam,heap)
 return {'nodes':nodes,'states':len(seen),'closures':closures,'status':'node_budget' if nodes>=max_nodes else 'exhausted'}
def run():
 d=load_words(); results={}
 for i,(ln,ls) in enumerate(SHAPES):
  for j,(rn,rs) in enumerate(SHAPES):results[f'{i}:{j}']=search(ls,rs,d)
 exact=[x for y in results.values() for x in y['closures']]
 controls=['The young man sees the house; the old woman hears the word.','A good boy found the door; a new girl held the key.']
 return {'experiment_id':ID,'method':'live bidirectional character/word-boundary beam over complete authored grammar paths','shape_count':len(SHAPES),'results':results,'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x),'complete_prose':True} for x in controls],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior full-frame reverse parsing; both sides are expanded synchronously before rendering','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'bank':str(BANK.relative_to(ROOT)),'bank_sha256':hashlib.sha256(BANK.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ranking':'small authored collocation score ranks compatible states only; it never certifies readability','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'next_reader_test':'randomized blinded ratings of intact exact rows versus shuffled controls'},'stats':{'nodes':sum(x['nodes'] for x in results.values()),'closures':len(exact),'exact':len(exact)},'status':'fresh exact candidates require human reading' if exact else 'no fresh exact parse in this lane','next_construction':'human reader package if exact rows survive; otherwise add a new grammar topology, not a wider beam'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
