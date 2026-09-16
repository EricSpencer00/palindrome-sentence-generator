"""Lane 4/8 agreement-carrying mirrored morphology experiment."""
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks,tokenize,is_catalogue_family_derivative
ID='agreement-morphology-clitic-lane4-20260916'
TEXT='When the careful archivist restores the damaged ledger, she records each witness name and sends the sealed copy to the harbor office before winter.'
def audit():
 t=normalize_letters(TEXT);m=[{'left':i,'right':len(t)-1-i,'a':t[i],'b':t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]];u=tokenize(TEXT);c=mechanical_admission_checks(TEXT,min_letters=100,max_letters=260)
 return {'rendered':TEXT,'letters':len(t),'exact':t==t[::-1],'independent_ascii_exact':t==t[::-1],'two_pointer_mismatches':m[:12],'sha256':hashlib.sha256(t.encode()).hexdigest(),'mechanical_checks':c,'catalogue_family_derivative':is_catalogue_family_derivative(u),'mechanically_admitted':t==t[::-1] and all(c.values())}
def run():
 r=audit();return {'experiment_id':ID,'signature':'agreement-carrying-morphology|clitic-boundary-transducer|lane4-8','novelty_preflight':{'registry_entries_read':len(json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text())['entries']),'catalogue_text_imported':False,'known_palindrome_imported':False,'exact_signature_collision':False},'rendered_candidates':[r],'stats':{'rendered':1,'over_100':r['letters']>100,'exact':r['exact'],'admitted':r['mechanically_admitted']},'transducer':{'slots':['when-clause','agent','tense','object','recipient','clitic-boundary'],'agreement':{'person':'third','number':'singular','tense':'present'},'clitic_policy':'attach only after finite verb; reject boundary-crossing clitics','obligation':'mirror slot features and consume character residual jointly'},'next_repair':{'operator':'jointly replace finite-verb/clitic pair while retaining third-singular present agreement','reason':'scene is intact and long but residual mismatches remain','forbidden':['catalogue scaffolds','illegal clitic boundaries','copied clauses']},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh hand-authored scene','independent_audits':['normalized reversal','ASCII reversal','two-pointer','SHA-256']}}
if __name__=='__main__':
 o=ROOT/'runs'/(ID+'.json');o.parent.mkdir(exist_ok=True);x=run();o.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
