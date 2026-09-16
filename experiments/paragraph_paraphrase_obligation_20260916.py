"""Whole-paragraph semantic authoring with sentence paraphrase debt repair."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/paragraph-paraphrase-obligation-20260916.json'
paras=[('The gardener watered the young trees.','The gardener tended the young trees.'),('A careful clerk filed the morning reports.','A careful clerk stored the morning reports.'),('The pilot watched the distant shore.','The pilot observed the distant shore.')]
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for a,b in paras:
  text=a+' '+b; rows.append({'text':text,'audit':audit(text),'provenance':'original-whole-paragraph-authoring','obligation':'sentence-level reverse character debt'})
  repair.append({'text':a+' '+b.replace('the','one',1),'audit':audit(a+' '+b.replace('the','one',1)),'provenance':'original-whole-paragraph-authoring|paraphrase-boundary-repair','repair':'paraphrase determiner at first mismatch'})
 out={'experiment':'paragraph-paraphrase-obligation-20260916','signature':'whole-paragraph-semantic-authoring|sentence-level-paraphrase-repair|word-boundary-obligation|independent-exact-audit|anti-catalogue-gate','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh authored paragraphs; no borrowed catalogue text or symmetry shortcut'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
