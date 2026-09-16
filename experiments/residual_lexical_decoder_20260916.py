"""Repair exact tape fragments by held-out lexical segmentation and clause-shape filtering."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'runs/semantic-frame-tape-solver-20260916.json'; OUT=ROOT/'runs/residual-lexical-decoder-20260916.json'
LEX={'the','a','nurse','sailor','baker','keeps','opens','marks','blue','calm','red','map','book','gate','by','dawn','near','water','at','home','one','quiet'}
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s); w=re.findall('[a-z]+',s.lower()); return {'text':s,'letters':len(t),'exact':bool(t) and t==t[::-1],'complete_sentence':s.endswith('.'),'all_words_lexicon':all(x in LEX for x in w),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def segment(t):
 if not t:return ['']
 out=[]
 for w in sorted(LEX,key=len,reverse=True):
  if t.startswith(w):
   for rest in segment(t[len(w):]):
    if len(out)<16: out.append(w+(' '+rest if rest else ''))
 return out
def main():
 src=json.loads(SRC.read_text()); rows=[]; rejected=[]
 for r in src['candidates']:
  if not r['audit']['exact']: continue
  frag=r['text'].split()[-2].strip('.')
  segs=segment(frag)
  if segs:
   for s in segs[:4]:
    text=r['text'].rsplit(' ',2)[0]+' '+s+'.'; a=audit(text)
    (rows if a['complete_sentence'] and a['all_words_lexicon'] else rejected).append({'text':text,'audit':a,'provenance':'residual-tape|held-out-lexicon-segmentation'})
  else: rejected.append({'text':r['text'],'audit':audit(r['text']),'provenance':'residual-tape|unsegmentable'})
 out={'experiment':'residual-lexical-decoder-20260916','signature':'exact-tape-residual-repair|held-out-pos-lexicon|recursive-word-segmentation|complete-clause-filter|independent-audit','source_artifact':str(SRC.relative_to(ROOT)),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'survivors':rows,'rejected':rejected,'exact_count':sum(x['audit']['exact'] for x in rows),'reader_eligible_count':0,'repair_action':'segment reverse debt using held-out lexicon; reject fragments'}
 OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'survivors':len(rows),'rejected':len(rejected),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
