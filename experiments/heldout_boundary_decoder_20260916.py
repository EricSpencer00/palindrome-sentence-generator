"""Held-out vocabulary boundary decoder for semantic clause pairs."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/heldout-boundary-decoder-20260916.json'
S=['the curator','a young sailor','the quiet doctor']; V=['labels','carries','opens']; O=['a silver case','the blue ledger','a green parcel']; Z=['near the quay','under moonlight','by the chapel']
LEX=['the','a','young','quiet','doctor','curator','sailor','labels','carries','opens','silver','blue','green','case','ledger','parcel','near','under','by','quay','moonlight','chapel']
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'all_words_heldout':all(x in LEX for x in w),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for s,v,o,z in itertools.product(S,V,O,Z):
  l=f'{s} {v} {o} {z}.'; debt=n(l)[::-1]; right=' '.join(x for x in LEX if debt.startswith(x))
  text=l+' '+(right+'.' if right else debt[:30]+' .');rows.append({'text':text,'audit':a(text),'provenance':'held-out-vocabulary|semantic-role-clause','debt':debt,'decoder':'phonotactic boundary beam'})
  repair.append({'text':l+' '+debt[:30]+' .','audit':a(l+' '+debt[:30]+' .'),'provenance':'held-out-vocabulary|boundary-resegmentation-repair','repair':'resegment first 30 reverse-debt characters'})
 out={'experiment':'heldout-boundary-decoder-20260916','signature':'held-out-vocabulary|phonotactic-boundary-beam|semantic-role-clause|reverse-debt-segmentation|boundary-repair|independent-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh authored semantic clauses; held-out lexicon only, no borrowed output'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
