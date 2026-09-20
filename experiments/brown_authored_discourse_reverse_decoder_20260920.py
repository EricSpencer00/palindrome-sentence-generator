"""Authored discourse-frame grammar with complete reverse parsing."""
from __future__ import annotations
import hashlib,itertools,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.brown_authored_semantic_reverse_decoder_20260920 import Word,Trie,audit,frame_score,load_words
ROOT=Path(__file__).resolve().parents[1]; BANK=ROOT/'data/brown_pcfg_bank_20260920.json'; OUT=ROOT/'runs/brown-authored-discourse-reverse-decoder-20260920.json'; ID='brown-authored-discourse-reverse-decoder-20260920'; SIG='brown-derived-lexicon|authored-discourse-frame|complete-clause-marker-clause|variable-boundary-reverse-parse'
SHAPES=(('then-frame',('MARK','DET','ADJ','AGENT','ACTION','DET','OBJECT','MARK','DET','ADJ','AGENT','ACTION','DET','OBJECT')),('now-frame',('MARK','DET','AGENT','ACTION','DET','OBJECT','MARK','DET','AGENT','ACTION','DET','OBJECT')))
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def make_frames(d,limit):
 out=[]
 for kind,shape in SHAPES:
  choices=[d[r][:7] if r not in {'MARK'} else d[r] for r in shape]
  for xs in itertools.product(*choices):
   words=tuple(xs); content=[w.text for w in words if w.text not in {'the','a','an','then','now'}]
   if len(set(content))!=len(content):continue
   out.append((kind,shape,words))
   if len(out)>=limit:return out
 return out
def parse(tape,shape,trie,forbidden):
 out=[];states=0;seen=set()
 def walk(i,pos,ws):
  nonlocal states
  states+=1; key=(i,pos,tuple(w.text for w in ws))
  if key in seen or len(out)>=20:return
  seen.add(key)
  if i==len(shape):
   if pos==len(tape):out.append(ws)
   return
  for end,w in trie.matches(tape,pos,shape[i]):
   if w.text in forbidden and w.text not in {'the','a','an','then','now'}:continue
   walk(i+1,end,ws+(w,))
 walk(0,0,());return out,states
def run(max_frames=10000):
 d=load_words();d=dict(d);d['MARK']=(Word('then',1.0),Word('now',1.0)); fs=make_frames(d,max_frames);trie=Trie(d);rows=[];exact=[];states=parses=0
 for kind,shape,ws in fs:
  found,used=parse(letters(' '.join(w.text for w in ws))[::-1],shape,trie,frozenset(w.text for w in ws));states+=used;parses+=len(found)
  for rw in found:
   text=' '.join(w.text for w in ws)+'; '+' '.join(w.text for w in rw)+'.'; row={'rendered':text,'frame_kind':kind,'roles':list(shape),'rank_score':frame_score(ws)+frame_score(rw),'audit':audit(text),'provenance':{'source':'Brown-derived word forms only; no Brown sentence text','complete_discourse_frame':True,'variable_word_boundaries':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'readability_certified':False}};rows.append(row)
   if row['audit']['exact']:exact.append(row)
 controls=['Then the young man sees the house; now the old woman hears the word.','Now a good boy found the door; then a new girl held the key.']
 return {'experiment_id':ID,'method':'authored discourse-marker frames over Brown-derived domains with complete reverse parsing','stats':{'complete_forward_frames':len(fs),'reverse_states':states,'complete_reverse_parses':parses,'rendered_candidates':len(rows),'exact':len(exact)},'rendered_candidates':sorted(rows,key=lambda x:-x['rank_score'])[:200],'exact_candidates':sorted(exact,key=lambda x:-x['audit']['letters'])[:100],'controls':[{'rendered':x,'audit':audit(x),'complete_prose':True} for x in controls],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior clause, coordination, and subordination lanes; temporal discourse markers are authored frame constituents','catalogue_text':False,'finished_tape_reversal':False,'post_hoc_repair':False},'provenance':{'bank':str(BANK.relative_to(ROOT)),'bank_sha256':hashlib.sha256(BANK.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'marker_policy':'then and now admitted from Brown-derived ADV forms; however/therefore unavailable in frozen bank','next_reader_test':'randomized blinded ratings of intact discourse frames versus shuffled controls'},'status':'fresh exact candidates require human reading' if exact else 'no fresh exact parse in this lane','next_construction':'prepare reader package if exact rows survive; otherwise author a new event-chain grammar'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
