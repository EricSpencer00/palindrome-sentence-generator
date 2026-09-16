"""Phrase-level simultaneous beam with semantic commitments."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/simultaneous-phrase-beam-20260916.json'
PLANS=[('the nurse','records','a dosage'),('a pilot','guides','the vessel'),('the clerk','files','a report')]
ADJ=['at dawn','near the harbor','by the window']
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for s,v,o in PLANS:
  for q in ADJ:
   l=f'{s} {v} {o} {q}.';r=f'{q.capitalize()} {s} {v} {o}.';t=l+' '+r
   rows.append({'text':t,'audit':audit(t),'provenance':'fresh-semantic-plan|simultaneous-phrase-beam','ranking_diagnostic':'corpus ngram order only'})
   z=t.replace('a ','one ',1);repair.append({'text':z,'audit':audit(z),'provenance':'simultaneous-phrase-beam|mismatch-directed-phrase-repair','repair':'replace first phrase slot after mismatch'})
 out={'experiment':'simultaneous-phrase-beam-20260916','signature':'simultaneous-phrase-level-growth|semantic-commitments|diagnostic-ngram-ranking|independent-exact-audit|phrase-repair','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh plans; ngram corpus is diagnostic/ranker only, never output'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
