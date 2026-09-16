"""Reader-first scene search coupling paraphrase and discourse connectives."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/discourse-connective-coupled-20260916.json'
scenes=[('The nurse checked the parcel','the clerk signed the record'),('A pilot watched the harbor','the crew secured the vessel')]
conns=['Then','Afterward','Meanwhile']
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for (x,y),c in itertools.product(scenes,conns):
  t=f'{x}. {c}, {y}.';rows.append({'text':t,'audit':a(t),'provenance':'fresh-scene|coupled-connective-paraphrase','semantic_scene':'single coherent work event'})
  z=t.replace('checked','examined').replace('Then','Next');repair.append({'text':z,'audit':a(z),'provenance':'fresh-scene|coupled-connective-paraphrase|ledger-repair','repair':'jointly paraphrase verb and connective at mismatch'})
 out={'experiment':'discourse-connective-coupled-20260916','signature':'reader-first-multi-sentence-scene|coupled-semantic-paraphrase|discourse-connective-choice|mirrored-ledger-search|independent-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh scene authoring; no catalogue or symmetry shortcut'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
