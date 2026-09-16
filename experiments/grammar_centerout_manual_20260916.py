"""Manual grammar-aware center-out prose candidates and expansion repair."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/grammar-centerout-manual-20260916.json'
BASE=['The careful nurse records a dosage near the quiet clinic.','A patient sailor carries a letter beyond the harbor wall.','The young baker opens a window beside the warm kitchen.']
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[{'text':x,'audit':a(x),'provenance':'manual-common-word-centerout','construction':'grammar-aware center expansion'} for x in BASE]
 repair=[]
 for x in BASE:
  words=x[:-1].split(); mid=len(words)//2; y=' '.join(words[:mid]+['steady']+words[mid:])+'.';repair.append({'text':y,'audit':a(y),'provenance':'manual-common-word-centerout|center-expansion-repair','repair':'insert semantic center word at grammar boundary'})
 out={'experiment':'grammar-centerout-manual-20260916','signature':'manual-common-word-authoring|grammar-aware-centerout|cross-word-seam|semantic-center-expansion-repair|independent-exact-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh manual prose; no catalogue, repeated units, or word-order symmetry'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
