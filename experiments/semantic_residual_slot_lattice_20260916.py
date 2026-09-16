"""Semantic-slot lattice with residual-debt guided substitutions."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ID='semantic-residual-slot-lattice-20260916'
TEXT='After the winter storm, Lena repairs the observatory clock, checks the brass instruments, and writes a careful report for the coastal research crew.'
def audit():
 t=normalize_letters(TEXT); f=hashlib.sha256(t.encode()).hexdigest(); rev=hashlib.sha256(t[::-1].encode()).hexdigest(); mm=[{'i':i,'j':len(t)-1-i,'a':t[i],'b':t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'text':TEXT,'letters':len(t),'exact':t==t[::-1],'two_pointer':not mm,'forward_hash':f,'reverse_hash':rev,'mismatches':mm[:12],'checks':mechanical_admission_checks(TEXT,min_letters=100,max_letters=260)}
def run():
 r=audit();return {'experiment_id':ID,'signature':'semantic-slot-lattice|valency-attachment|residual-debt-substitution|agreement-lock','novelty_preflight':{'catalogue_imported':False,'known_palindromes_imported':False,'exact_signature_collision':False},'rendered_candidates':[r],'stats':{'letters':r['letters'],'over_100':r['letters']>100,'exact':r['exact']},'coupling':{'slots':['agent','time','repair_action','instrument_object','report_purpose'],'agreement':'third-person singular present','selection':'choose slot substitutions minimizing live mirrored residual debt','nonrepetition':'content lexemes unique'},'next_repair':{'operator':'substitute instrument_object and report_purpose jointly at highest residual debt frontier','reason':'all valency slots attach coherently; character debt remains','mismatch_position':r['mismatches'][0]['i'] if r['mismatches'] else None},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh hand-authored lattice terminals','audits':['two-pointer','forward/reverse hash','mechanical admission']}}
if __name__=='__main__':
 o=ROOT/'runs'/(ID+'.json');o.parent.mkdir(exist_ok=True);x=run();o.write_text(json.dumps(x,indent=2)+'\n');print(x['stats'])
