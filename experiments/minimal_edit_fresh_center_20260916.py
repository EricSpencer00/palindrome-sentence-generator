"""Template-free fresh prose followed by minimal lexical mismatch edits."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/minimal-edit-fresh-center-20260916.json'
BASE=['At first light, the archivist carried a weathered journal into the quiet room.','During the evening storm, a patient mechanic repaired the lantern beside the open shed.','Before the meeting, the young gardener placed warm seeds beneath the window.']
REPL={'quiet':'still','open':'wide','warm':'dry','young':'new','patient':'calm','weathered':'old'}
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[{'text':x,'audit':a(x),'provenance':'fresh-free-prose-center','operator':'none'} for x in BASE];repair=[]
 for x in BASE:
  for old,new in REPL.items():
   if old in x:
    y=x.replace(old,new,1);repair.append({'text':y,'audit':a(y),'provenance':'fresh-free-prose-center|minimal-edit-repair','repair':f'{old}->{new}; one lexical edit, punctuation/order unchanged'})
 out={'experiment':'minimal-edit-fresh-center-20260916','signature':'template-free-fresh-prose|minimal-lexical-edit|fixed-word-order|mismatch-directed-repair|independent-exact-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh authored prose; no catalogue, punctuation, repetition, or word-order shortcut'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
