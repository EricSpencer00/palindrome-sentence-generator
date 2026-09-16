"""Single recorded adjunct-boundary repair; inventory is intentionally one item."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.validator import normalize,is_palindrome
from llm_palindrome.admission import mechanical_admission_checks
REG=ROOT/'docs/experiment-novelty-registry.json';ID='adjunct-boundary-targeted-repair-20260916';SIG='single-adjunct-boundary-repair|fresh-preposition|preserved-typed-svo|live-seam-residual|independent-audit'
def audit(t):
 a=''.join(c.lower() for c in t if c.isascii() and c.isalpha());return {'text':t,'letters':len(normalize(t)),'exact':is_palindrome(t),'two_pointer':bool(a) and all(a[i]==a[-1-i] for i in range(len(a)//2)),'sha256':hashlib.sha256(a.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(t,min_letters=39,max_letters=180)}
def run():
 rows=json.loads(REG.read_text())['entries']; col=[r.get('artifact') for r in rows if r.get('id')!=ID and r.get('signature')==SIG]; pre={'status':'passed' if not col else 'blocked','collisions':col,'checked_entries':len(rows)}
 if col: raise RuntimeError(pre)
 text='The baker carries a letter by the quiet harbor';return {'experiment':ID,'novelty_preflight':pre,'candidate':text,'audit':audit(text),'provenance':{'repair':'first residual adjunct boundary','replacement':'near -> by','frame_preserved':True,'agreement':'singular present','inventory_size':1},'next_repair':'carry the remaining residual character into the determiner slot; do not alter the SVO frame'}
if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
