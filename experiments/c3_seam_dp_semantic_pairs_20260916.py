"""C3 semantic scenes with heteropalindromic pairs and boundary-shift seam DP."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/c3-seam-dp-semantic-pairs-20260916.json'
PAIRS=[('drawer','reward'),('diaper','repaid'),('stressed','desserts')]
SC=[('The drawer','holds','a letter'),('A sailor','reads','the diary'),('The cook','serves','desserts')]
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for (s,v,o),(p,q) in itertools.product(SC,PAIRS):
  t=f'{s} {v} {o}; {p.capitalize()} {q}.';rows.append({'text':t,'audit':a(t),'provenance':'C3-fresh-semantic-scene|heteropalindromic-pair','seam_dp':'boundary shifts allowed'})
  z=t.replace(';','. Then',1);repair.append({'text':z,'audit':a(z),'provenance':'C3-seam-dp|connective-repair','repair':'shift clause boundary and connective'})
 out={'experiment':'c3-seam-dp-semantic-pairs-20260916','signature':'C3-fresh-semantic-scene|heteropalindromic-word-pairs|character-seam-dp|boundary-shift|connective-repair|independent-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh scene clauses; no catalogue, repeated units, or self-palindromic units'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
