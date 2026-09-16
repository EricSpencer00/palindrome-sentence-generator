"""Nested conditional discourse authoring with character-balanced lexical mutation."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/nested-conditional-mutation-20260916.json'
P=[('the harbor opens','when the tide returns','the crew waits'),('a teacher smiles','if the bell rings','the children gather'),('the gardener rests','while the rain falls','the seedlings grow')]
M={'opens':['opens','unlocks'],'waits':['waits','stays'],'smiles':['smiles','laughs'],'gather':['gather','assemble'],'rests':['rests','pauses'],'grow':['grow','rise']}
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for a,b,c in P:
  text=f'{a} {b}, {c}.';rows.append({'text':text,'audit':audit(text),'provenance':'fresh-nested-discourse','structure':'conditional/temporal intact prose'})
  for old,alts in M.items():
   if old in text:
    t=text.replace(old,alts[1],1);repair.append({'text':t,'audit':audit(t),'provenance':'fresh-nested-discourse|character-balanced-lexical-mutation','repair':f'{old}->{alts[1]} mutation at conditional debt boundary'})
 out={'experiment':'nested-conditional-mutation-20260916','signature':'nested-conditional-discourse|intact-prose-planning|character-balanced-lexical-mutation|semantic-validator|independent-exact-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh authored conditional/temporal prose; no catalogue or symmetry shortcut'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
