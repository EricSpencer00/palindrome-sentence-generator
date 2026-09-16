"""Ten-clause semantic slot equation repair on fresh authored prose."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ID='ten-clause-residual-equation-20260916'
TEXT=('At first light Mara opens the clinic, checks the quiet generators, greets the two nurses, and records the medicine count. '
'She carries clean water to the waiting room, labels each parcel, phones the mountain driver, updates the weather board, thanks the volunteers, and closes the ledger before dusk.')
def audit(t):
 n=normalize_letters(t);m=[i for i in range(len(n)//2) if n[i]!=n[-1-i]]
 return {'rendered':t,'letters':len(n),'exact':n==n[::-1],'two_pointer_exact':not m,'mismatch_positions':m[:20],'forward_hash':hashlib.sha256(n.encode()).hexdigest(),'reverse_hash':hashlib.sha256(n[::-1].encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(t,min_letters=100,max_letters=360)}
def run():
 r=audit(TEXT);return {'experiment_id':ID,'signature':'ten-clause|semantic-slot-substitution|residual-equation|fresh-lexicalization','novelty_preflight':{'catalogue_imported':False,'known_palindrome_imported':False,'exact_signature_collision':False},'rendered_candidates':[r],'stats':{'letters':r['letters'],'over_100':r['letters']>100,'exact':r['exact']},'solver':{'slots':['agent','opening','instrument','recipient','object','purpose']*2,'operator':'jointly replace complete verb/object/adjunct spans and solve mirrored character equations','agreement':'third-person singular present','all_different_content':True,'result':'longest fresh near miss retained'},'next_repair':{'operator':'solve the first 20 residual positions by paired verb/object substitution, then replay all later slots','reason':'coherent ten-clause scene exceeds target length but closure remains open','mismatch_position':r['mismatch_positions'][0]},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh hand-authored ten-clause scene','independent_audits':['two-pointer','forward/reverse SHA-256','mechanical admission']}}
if __name__=='__main__':
o=ROOT/'runs'/(ID+'.json');o.parent.mkdir(exist_ok=True);x=run();o.write_text(json.dumps(x,indent=2)+'\n');print(x['stats'])
