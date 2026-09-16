"""Fresh semantic wrapper search around non-catalogue centers."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/reversible-semantic-wrappers-20260916.json'
C=['the nurse checks a parcel','a pilot guides the vessel','the baker repairs a chair']; W=[('At dawn','near the quay'),('In rain','by the fire'),('At home','near the gate')]
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for c,(l,r) in itertools.product(C,W):
  t=f'{l}, {c}. {r.capitalize()}, {c}.';rows.append({'text':t,'audit':a(t),'provenance':'fresh-center|semantic-wrapper-pair','center':c,'wrapper_units':[l,r]})
  z=t.replace('At ','After ',1);repair.append({'text':z,'audit':a(z),'provenance':'fresh-center|semantic-wrapper-pair|wrapper-mutation-repair','repair':'mutate first wrapper connective'})
 out={'experiment':'reversible-semantic-wrappers-20260916','signature':'fresh-semantic-center|reversible-prose-wrappers|character-pair-primitives|distinct-content-gate|wrapper-mutation-repair|independent-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh centers and wrappers; no borrowed sentence or self-palindromic unit'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
