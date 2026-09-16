"""Fresh typed SVO frame with live seam obligations (no post-hoc decoding)."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.validator import normalize,is_palindrome
from llm_palindrome.admission import mechanical_admission_checks
REG=ROOT/'docs/experiment-novelty-registry.json'; ID='fresh-typed-frame-live-seam-20260916'; SIG='fresh-typed-svo-frame|agreement-preserved|live-mirrored-character-obligation|seam-residual-ledger|independent-audit'
FRAME={'subject':'The baker','verb':'carries','object':'a letter','number':'sg','tense':'pres','roles':['agent','event','patient']}
def preflight():
 rows=json.loads(REG.read_text())['entries']; c=[r.get('artifact') for r in rows if r.get('id')!=ID and r.get('signature')==SIG]; return {'status':'passed' if not c else 'blocked','collisions':c,'checked_entries':len(rows)}
def audit(t):
 a=''.join(x.lower() for x in t if x.isascii() and x.isalpha());return {'text':t,'letters':len(normalize(t)),'exact':is_palindrome(t),'two_pointer':bool(a) and all(a[i]==a[-i-1] for i in range(len(a)//2)),'sha256':hashlib.sha256(a.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(t,min_letters=39,max_letters=180)}
def run():
 p=preflight();
 if p['status']!='passed':raise RuntimeError(p)
 left='The baker carries a letter'; right='near the quiet harbor'
 tape=normalize(left+' '+right); obligations=[(i,tape[-i-1]) for i in range(min(8,len(tape)))]
 return {'experiment':ID,'novelty_preflight':p,'frame':FRAME,'candidate':left+' '+right,'audit':audit(left+' '+right),'live_obligations':obligations,'provenance':{'lexical_frame':'baker/carries/letter/harbor','fresh_domains':True,'posthoc_segmentation':False},'next_repair':'replace only the adjunct boundary lexeme selected by the first residual obligation, preserving singular agreement'}
if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
