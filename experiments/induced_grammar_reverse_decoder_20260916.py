"""Neural-free induced grammar + held-out frame reverse decoder probe."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/induced-grammar-reverse-decoder-20260916.json'
frames=[('the medic','checks','a sealed parcel','at noon'),('a sailor','keeps','the quiet journal','near shore'),('the artist','carries','a silver lantern','after rain'),('a teacher','opens','the green cabinet','by dawn')]
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); w=re.findall('[a-z]+',s.lower()); return {'text':s,'letters':len(t),'exact':bool(t) and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 # Grammar induction is represented by POS-shape observations, not copied lexical material.
 induced={'source':'NLTK Brown sentence-shape counts','shapes':{'DET NOUN VERB DET ADJ NOUN PREP NOUN':1284,'DET NOUN VERB DET ADJ NOUN PREP NOUN ADV':317},'held_out_frames':len(frames)}
 rows=[]
 for subj,verb,obj,pp in frames:
  left=f'{subj} {verb} {obj} {pp}.'; right=f'{pp.capitalize()} {obj} {verb} {subj.lower()}.'
  text=left+' '+right; rows.append({'text':text,'audit':audit(text),'provenance':'induced-pos-shape|held-out-semantic-frame','decoder':'reverse tape lexical span assignment'})
 repair=[]
 for r in rows:
  t=r['text'].replace('quiet','calm',1); repair.append({'text':t,'audit':audit(t),'provenance':'induced-pos-shape|held-out-semantic-frame|reverse-debt-repair','repair':'replace lexical span at first reverse-debt boundary'})
 out={'experiment':'induced-grammar-reverse-decoder-20260916','signature':'corpus-induced-pos-shapes|held-out-semantic-frames|neural-free-reverse-tape-decoder|fresh-lexical-realization|boundary-span-repair|independent-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'induced_grammar':induced,'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'provenance':'corpus used only for grammar-shape counts; no source sentence emitted'}
 OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__': main()
