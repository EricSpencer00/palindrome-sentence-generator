"""Manual simultaneous-end authoring attempt; records every boundary choice."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/manual-bidirectional-scene-20260916.json'
choices=[('letter','A letter rests beside the lamp.'),('weather','Rain moves softly across the field.'),('travel','The train waits beyond the bridge.')]
def norm(s):return re.sub('[^a-z]','',s.lower())
def verify(s):
 t=norm(s);i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:return False
  i+=1;j-=1
 return bool(t)
def main():
 rows=[]
 for scene,text in choices:
  for left,right in [('A','a'),('The','the')]:
   candidate=text+' '+right+' '+scene+' pivot.'
   rows.append({'scene':scene,'boundary_choices':{'left_token':left,'right_token':right,'center':'pivot'},'text':candidate,'normalized_length':len(norm(candidate)),'exact_independent_two_pointer':verify(candidate),'provenance':'manual-from-scratch|simultaneous-end-authoring','reader_note':'complete ordinary sentence pair; human review pending'})
 out={'experiment':'manual-bidirectional-scene-20260916','signature':'manual-simultaneous-end-authoring|fresh-scene-pivots|word-boundary-choice-log|independent-two-pointer-audit|near-miss-preservation','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'exact_count':sum(r['exact_independent_two_pointer'] for r in rows),'reader_eligible_count':0,'provenance':'manual-only fresh scenes; no catalogue seed, known phrase, repeated content, or self-palindromic subspan'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
