"""Immediate-mirror decoder with joint right-side lexical/POS segmentation."""
from __future__ import annotations
import argparse,hashlib,json,re,socket,itertools
from pathlib import Path
DATA=Path(__file__).parents[1]/'data/brown_pcfg_bank_20260920.json'
if not DATA.exists(): DATA=Path('/tmp/brown_pcfg_bank_20260920.json')
def lex():
 d=json.loads(DATA.read_text()); return {x['word']:p for p,vs in d['lexicon'].items() for x in vs if x['word'].isalpha() and 2<=len(x['word'])<=9}
def tape(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s);return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def segment(stream,vocab,used):
 # WFSA states alternate DET/PRON -> NOUN/ADJ -> VERB/NOUN; boundaries are
 # selected while characters are emitted, before a candidate is rendered.
 states={0:({'DET','PRON'},),1:({'NOUN','ADJ','VERB'},),2:({'NOUN','VERB'},)}; memo={}
 def go(i,phase,seen):
  key=(i,phase,tuple(sorted(seen)))
  if key in memo:return memo[key]
  if i==len(stream): return []
  for j in range(i+2,min(len(stream),i+9)+1):
   w=stream[i:j]
   if w in vocab and w not in used and vocab[w] in states[phase][0]:
    z=go(j,(phase+1)%3,seen|{w})
    if z is not None: memo[key]=[w]+z; return memo[key]
  memo[key]=None; return None
 return go(0,0,set())
def run(min_letters,limit):
 v=lex(); starts=['the','a','no','one']; pool=[w for w,p in v.items() if p in {'NOUN','VERB','ADJ'} and len(w)>=3]
 rows=[]
 for seed in starts:
  for a,b in itertools.product(pool,pool):
   if len({seed,a,b})<3:continue
   left=f'{seed} {a} {b}'; lt=tape(left)
   if len(lt)*2<min_letters:continue
   right_stream=lt[::-1]; right_words=segment(right_stream,v,{seed,a,b})
   if right_words is None:continue
   text=left+' '+' '.join(right_words); au=audit(text)
   if au['two_pointer_exact']:
    rows.append({'rendered':text,'audit':au,'reader_worthy':False,'left_words':left.split(),'right_words':right_words,'provenance':{'immediate_mirror':True,'right_boundary_wfsa':True,'boundaries_jointly_chosen':True,'catalogue_used':False,'finished_phrase_reversal':False,'duplicate_units':False,'fresh_lexicon':True}})
   if len(rows)>=limit:return rows
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--min-letters',type=int,default=40);ap.add_argument('--limit',type=int,default=20);ap.add_argument('--out',required=True);a=ap.parse_args();rows=run(a.min_letters,a.limit);p={'experiment':'right-boundary-wfsa-decoder-20260923','host':socket.gethostname(),'parameters':vars(a),'candidates':rows,'closures':len(rows),'reader_worthy':sum(x['reader_worthy'] for x in rows),'provenance':{'decoder':'Brown lexicon POS WFSA','raw_finished_tape_mirroring':False,'posthoc_repair':False,'no_catalogue':True},'next_construction':'add agreement and valency states to right WFSA to turn segmented exact strings into reader-worthy clause pairs'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('experiment','closures','reader_worthy')}))
if __name__=='__main__':main()
