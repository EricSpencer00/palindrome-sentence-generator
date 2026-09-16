"""Reader-first coupled syntax/lexical authoring fallback."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/coupled-syntax-lexical-authoring-20260916.json'
frames=[('the baker','quietly repairs','a broken chair'),('a nurse','carefully carries','the blue folder'),('the pilot','steadily watches','a distant light')]
tails=['before sunrise','during the storm','beside the river']
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for f,t in itertools.product(frames,tails):
  s=f'{f[0]} {f[1]} {f[2]} {t}.'; rows.append({'text':s,'audit':a(s),'provenance':'deterministic-coupled-authoring','semantic_coherence':'single intact event','joint_state':f})
  r=s.replace('quietly','gently',1).replace('before','after',1);repair.append({'text':r,'audit':a(r),'provenance':'deterministic-coupled-authoring|joint-syntax-lexical-repair','repair':'change adverb and connective as one coupled state'})
 out={'experiment':'coupled-syntax-lexical-authoring-20260916','signature':'joint-syntax-lexical-state|reader-first-event-coherence|deterministic-authoring-fallback|coupled-repair|independent-exact-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh deterministic authored events; no model/corpus/catalogue or symmetry shortcut'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
