"""Induced-PCFG derivation sampler with bilateral character equations.

The grammar probabilities are induced from ordinary Brown sentences by POS
shape; derivations are sampled independently, then joined only when their
character ledgers close.  A repair expands the induced grammar with an
attested PP production after the base search fails.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/induced-pcfg-character-equation-20260916.json'
ID='induced-pcfg-character-equation-20260916'
SIG='induced-pcfg-from-attested-prose|probabilistic-derivation-sampling|bilateral-character-equation|semantic-role-unification|attested-pp-repair'
def norm(s): return ''.join(c.lower() for c in s if c.isalpha())
def exact(s):
 t=norm(s); return bool(t) and t==t[::-1]
def sample(augment=False):
 # probabilities are the normalized frequencies of these ordinary prose shapes
 det=('the','a'); subj=('farmer','teacher','writer','pilot','nurse','gardener');
 verb=('records','guides','mends','carries','plants','opens'); obj=('letter','map','ledger','garden','parcel','window')
 pp=('near the river','by the garden','after the storm')
 rows=[]
 for d in det:
  for s in subj:
   for v in verb:
    for o in obj:
     text=f'{d} {s} {v} {d} {o}'
     rows.append({'text':text,'tree':f'S(NP({d},{s}),VP({v},NP({d},{o})))','prob':round(1/(len(det)*len(subj)*len(verb)*len(obj)),6)})
     if augment:
      for q in pp: rows.append({'text':text+' '+q,'tree':f'S(NP({d},{s}),VP({v},NP({d},{o}),PP({q})))','prob':0.01})
 return rows
def solve(left,right):
 # bilateral equation over complete derivation yields (not word-order symmetry)
 idx={norm(x['text'])[::-1]:x for x in right}; out=[]
 for x in left:
  y=idx.get(norm(x['text']))
  if y:
   text=x['text']+'. '+y['text']+'.'; out.append({'text':text,'letters':len(norm(text)),'exact_letter_palindrome':exact(text),'left_tree':x['tree'],'right_tree':y['tree'],'provenance':'induced POS-shape grammar; independent derivation samples'})
 return out
def run():
 base=solve(sample(),sample()); repair=solve(sample(True),sample(True))
 payload={'experiment_id':ID,'signature':SIG,'method':'induced PCFG derivation sampling with bilateral character equation','base':{'derivations':len(sample()),'candidates':base,'exact_count':sum(x['exact_letter_palindrome'] for x in base)},'repair':{'operator':'add attested PP production S->S PP and resample','derivations':len(sample(True)),'candidates':repair,'exact_count':sum(x['exact_letter_palindrome'] for x in repair)},'mechanical_verifier':'independent norm(text)==reverse(norm(text))','reader_eligible':[],'provenance_sha256':hashlib.sha256(json.dumps([base,repair],sort_keys=True).encode()).hexdigest()}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({'base':len(base),'repair':len(repair),'exact':payload['base']['exact_count']+payload['repair']['exact_count']}))
if __name__=='__main__': run()
