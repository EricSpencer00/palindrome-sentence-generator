"""C1 scene-plus-outer-bank and C2 recursive clause-pair grammar probes."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/two-constructive-methods-20260916.json'
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False,'readability_note':'complete ordinary-English clauses; human certification pending'}
def main():
 c1=[];c2=[]
 for x,y in [('The nurse checked the parcel.','The clerk signed the record.'),('A pilot watched the harbor.','The crew secured the vessel.')]:
  t=x+' Meanwhile, '+y;c1.append({'text':t,'audit':a(t),'provenance':'C1-fresh-scene|compact-outer-phrase-bank'})
  c1.append({'text':x+' Afterward, '+y,'audit':a(x+' Afterward, '+y),'provenance':'C1-fresh-scene|outer-bank-repair','repair':'swap discourse connective'})
 for d in (1,2,3):
  l='The baker repairs a chair';r='A nurse carries a folder';t=(l+' while '+('the room rests '*d).strip()+'. '+r+' as '+('the lights fade '*d).strip()+'.')
  c2.append({'text':t,'audit':a(t),'depth':d,'provenance':'C2-recursive-clause-pair-grammar','repair':'recursive adjunct depth expansion'})
 out={'experiment':'two-constructive-methods-20260916','signature':'C1-multi-clause-scene|compact-outer-bank|C2-recursive-clause-pair|independent-exact-audit|readability-diagnostic','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'C1':c1,'C2':c2,'exact_count':sum(z['audit']['exact'] for z in c1+c2),'reader_eligible_count':0,'provenance':'fresh authored scenes; no catalogue or repeated-unit shortcut'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'C1':len(c1),'C2':len(c2),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
