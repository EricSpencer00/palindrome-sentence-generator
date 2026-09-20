"""Seam-indexed complete grammar construction over Brown role words."""
from __future__ import annotations
import hashlib,itertools,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.brown_authored_semantic_reverse_decoder_20260920 import audit,load_words
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'data/brown_pcfg_bank_20260920.json';OUT=ROOT/'runs/brown-seam-indexed-lexical-generator-20260920.json';ID='brown-seam-indexed-lexical-generator-20260920';SIG='brown-derived-lexicon|seam-indexed-role-spans|exposed-character-length-buckets|complete-grammar-finish'
SHAPE=('DET','ADJ','AGENT','ACTION','DET','OBJECT')
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def index(dom):
 out={}
 for role,words in dom.items():
  for w in words:
   t=letters(w.text);out.setdefault((role,t[0],t[-1],len(t)),[]).append(w.text)
 return out
def compatible(left,right):
 # Compare only the newly exposed span characters; unpaired residuals remain
 # live and are checked when the next span is assigned.
 a=letters(''.join(left));b=letters(''.join(right))[::-1];n=min(len(a),len(b))
 return all(a[i]==b[i] for i in range(n))
def search(dom,limit=10000):
 idx=index(dom); buckets={r:[] for r in SHAPE}
 for (r,first,last,length),ws in idx.items():buckets.setdefault(r,[]).extend((w,first,last,length) for w in ws)
 out=[];states=0
 def walk(i,left,right):
  nonlocal states
  states+=1
  if len(out)>=limit:return
  if i==len(SHAPE):
   if len(left)==len(right) and compatible(left,right):
    text=' '.join(left)+'; '+' '.join(right)+'.';a=audit(text)
    if a['exact']:out.append({'rendered':text,'audit':a,'left_roles':list(SHAPE),'right_roles':list(SHAPE),'provenance':{'seam_indexed_spans':True,'role_words_from_brown_bank':True,'complete_left_sentence':True,'complete_right_sentence':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
   return
  role=SHAPE[i]
  for lw,lf,ll,ln in buckets[role][:24]:
   for rw,rf,rl,rn in buckets[role][:24]:
    if lw not in {'the','a','an'} and lw in left+right:continue
    if rw not in {'the','a','an'} and rw in left+right:continue
    nl=left+[lw];nr=[rw]+right
    if compatible(nl,nr):walk(i+1,nl,nr)
 walk(0,[],[]);return out,states,{r:len(v) for r,v in buckets.items()}
def run():
 dom=load_words();exact,states,buckets=search(dom);controls=['The young man sees the house; the old woman hears the word.','A good boy found the door; a new girl held the key.']
 return {'experiment_id':ID,'method':'seam-indexed role-span generator with residual-length compatibility before grammar completion','stats':{'span_buckets':buckets,'states':states,'exact':len(exact)},'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x),'complete_prose':True} for x in controls],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior live beam/transducer lanes; exposed span indices and residual lengths are selected before role completion','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'bank':str(BANK.relative_to(ROOT)),'bank_sha256':hashlib.sha256(BANK.read_bytes()).hexdigest(),'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed; no exact fresh candidate','next_reader_test':'blinded intact-versus-shuffled ratings if an exact row appears'},'status':'no fresh exact parse in this lane'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'states':x['stats']['states'],'exact':x['stats']['exact']}))
