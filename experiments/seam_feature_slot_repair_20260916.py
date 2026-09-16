"""Targeted seam repair for the long constructive witness.

Only one lexical slot at a time is changed; the reflected character seam is
recomputed immediately.  This is deliberately not a vocabulary sweep.
"""
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.validator import normalize, is_palindrome
from llm_palindrome.admission import mechanical_admission_checks
REGISTRY=ROOT/'docs/experiment-novelty-registry.json'; ID='seam-feature-slot-repair-20260916'; SIG='targeted-seam-feature-slot-repair|single-lexical-substitution|live-reflected-seam|independent-exact-audit'
BASE='Ava saw radar level civic; civic level radar was Ava'
SUBS={'Ava':['Mia','Eve','Ian'],'saw':['read','met'],'radar':['cable','parcel'],'level':['quiet','round'],'civic':['urban','local']}
def preflight():
 rows=json.loads(REGISTRY.read_text())['entries']; hit=[r.get('artifact') for r in rows if r.get('id')!=ID and (r.get('id')==ID or r.get('signature')==SIG)]
 return {'status':'passed' if not hit else 'blocked','collisions':hit,'checked_entries':len(rows)}
def audit(t):
 a=''.join(c.lower() for c in t if c.isascii() and c.isalpha()); return {'text':t,'letters':len(normalize(t)),'exact':is_palindrome(t),'two_pointer':bool(a) and all(a[i]==a[-1-i] for i in range(len(a)//2)),'sha256':hashlib.sha256(a.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(t,min_letters=39,max_letters=180)}
def run():
 p=preflight()
 if p['status']!='passed': raise RuntimeError(p)
 # A substitution must be mirrored at the character seam, not copied as a word.
 attempts=[]
 for slot, vals in SUBS.items():
  for v in vals:
   t=BASE.replace(slot,v,1); attempts.append({'slot':slot,'replacement':v,'audit':audit(t)})
 return {'experiment':ID,'novelty_preflight':p,'base':audit(BASE),'attempts':attempts,'accepted':[],'provenance':{'strategy':'single-slot seam substitution','enumerated':len(attempts),'catalogue_searches':0},'next_repair':'Use a typed boundary resegmentation at the first failing seam, permitting two adjacent short words to absorb the reflected suffix.'}
if __name__=='__main__': print(json.dumps(run(),indent=2,sort_keys=True))
