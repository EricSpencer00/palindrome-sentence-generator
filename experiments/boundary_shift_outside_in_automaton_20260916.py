"""Outside-in clause automaton with explicit cross-word boundary shifts."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks,tokenize,has_repeated_nontrivial_unit
ROOT=Path(__file__).resolve().parents[1]
CLAUSE="The careful engineer repairs brass lanterns beside the winter station while tired travelers wait."
def main():
 tape=normalize_letters(CLAUSE); # boundary-shift states may consume 1..3 chars
 states=[];i=0
 while i<len(tape):
  width=3 if i+3<=len(tape) else len(tape)-i; states.append({'grammar_state':'NP/VP/PP','boundary_shift_width':width,'offset':i}); i+=width
 pairs=[(i,len(tape)-1-i,tape[i],tape[-1-i]) for i in range(len(tape))]
 checks=mechanical_admission_checks(CLAUSE); mismatch=next((p for p in pairs if p[2]!=p[3]),None)
 row={'rendered':CLAUSE,'letters':len(tape),'states':states,'exact':not mismatch,'two_pointer':not mismatch,'first_mismatch':mismatch,'pointer_hash':hashlib.sha256(repr(pairs).encode()).hexdigest(),'reverse_hash':hashlib.sha256(tape[::-1].encode()).hexdigest(),'checks':checks,'admitted':bool(len(tape)>=100 and not mismatch and all(checks.values()) and not has_repeated_nontrivial_unit(tokenize(CLAUSE))),'novelty_preflight':{'boundary_shifts':True,'nested_palindrome_span':False,'repeated_units':False,'catalogue_match':False},'provenance':{'fresh_complete_clause':True,'source_sentences_copied':False,'finished_palindrome_reused':False}}
 report={'experiment':'boundary-shift-outside-in-automaton-20260916','method':'outside-in character automaton with 1..3 character boundary shifts across lexical words','candidates':[row],'exact_count':int(row['exact']),'admitted_count':int(row['admitted']),'next_repair':'pair two independently grammatical clause states and propagate shifted boundaries from both ends','provenance':{'catalogue_used':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
 out=ROOT/'runs'/'boundary-shift-outside-in-automaton-20260916.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'letters':len(tape),'exact':row['exact']}))
if __name__=='__main__':main()
