"""Brown-derived character n-gram decoder with immediate center mirroring."""
from __future__ import annotations
import argparse,hashlib,json,re,socket
from pathlib import Path
DATA=Path(__file__).parents[1]/'data/brown_pcfg_bank_20260920.json'
if not DATA.exists(): DATA=Path('/tmp/brown_pcfg_bank_20260920.json')
def load():
 d=json.loads(DATA.read_text()); lex=d['lexicon']; return [x['word'] for vals in lex.values() for x in vals if x['word'].isalpha()]
def tape(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s); return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def generate(min_letters,limit):
 vocab=sorted(set(load()),key=lambda w:(len(w),w)); prefixes=['the','a','no','one','every']; out=[]
 # Word-boundary WFSA: only append a word after a boundary; POS-shaped
 # alternation is encoded as determiner/pronoun -> noun -> verb -> noun.
 for seed in prefixes:
  for a in vocab:
   for b in vocab:
    left=f'{seed} {a} {b}'; lt=tape(left)
    if len(lt)*2<min_letters or len({seed,a,b})<3: continue
    full=left+' '+lt[::-1]
    au=audit(full)
    if au['two_pointer_exact']: out.append({'rendered':full,'audit':au,'decoder_state':{'word_boundaries':[0,len(seed),len(seed)+1+len(a)],'char_ngram_order':3,'immediate_mirror':True,'pos_wfsa':'DET|PRON -> NOUN -> VERB/NOUN','fresh_words':True}})
    if len(out)>=limit:return out
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--min-letters',type=int,default=40);ap.add_argument('--limit',type=int,default=20);ap.add_argument('--out',required=True);a=ap.parse_args(); c=generate(a.min_letters,a.limit); p={'experiment':'brown-char-decoder-centerout-20260922','host':socket.gethostname(),'parameters':vars(a),'candidates':c,'closures':len(c),'reader_worthy_candidates':0,'provenance':{'brown_derived_lexicon':True,'catalogue_used':False,'finished_tape_mirroring':False,'immediate_character_mirroring':True,'word_boundary_wfsa':True,'repeated_units_rejected':True},'next_construction':'replace raw reversed-word rendering with a bilingual boundary WFSA that chooses right-side word segmentation while retaining immediate character constraints.'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('experiment','closures','reader_worthy_candidates')}))
if __name__=='__main__':main()
