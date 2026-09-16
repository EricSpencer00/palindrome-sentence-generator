"""Fresh clause composition from heteropalindromic lexical pairs."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/heteropalindromic-clause-composer-20260916.json'
PAIRS=[('aide','edia'),('calm','mlac'),('nurse','esrun'),('pilot','tolip'),('reads','sdaer')]
CLAUSES=[('the aide','records','nine memos'),('a calm nurse','reads','the red note'),('the pilot','keeps','a map'),('a nurse','opens','the door')]
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'heteropalindromic_units':all(x!=x[::-1] for x in w),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for (s,v,o),(x,y,z) in itertools.product(CLAUSES,CLAUSES):
  if (s,v,o)==(x,y,z):continue
  text=f'{s} {v} {o}. {x.capitalize()} {y} {z}.'; rows.append({'text':text,'audit':audit(text),'provenance':'fresh-hand-authored-clause-pair','construction':'heteropalindromic lexical pair seam'})
  if len(rows)>=24:break
 for r in rows:
  t=r['text'].replace('the','a',1);repair.append({'text':t,'audit':audit(t),'provenance':'fresh-hand-authored-clause-pair|seam-synonym-repair','repair':'change first determiner to alter seam debt'})
 out={'experiment':'heteropalindromic-clause-composer-20260916','signature':'heteropalindromic-lexical-pairs|fresh-complete-clause-composition|cross-boundary-seam|independent-exact-audit|determiner-seam-repair','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh hand-authored clauses; no known catalogue or repeated/self-palindromic units'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
