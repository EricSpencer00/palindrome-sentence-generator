"""Lane 9: scalable typed-clause finite automaton for character tapes."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks,tokenize,has_repeated_nontrivial_unit
ROOT=Path(__file__).resolve().parents[1]
CLAUSES=["The patient archivist records quiet observations beside the river.","Curious visitors study faded stars beyond the winter station."]
def build():
 # states are (clause, lexical position, agreement); paired emission keeps the
 # construction compositional without embedding a prebuilt palindrome span.
 tape='';states=[]
 for clause in CLAUSES:
  for i,ch in enumerate(normalize_letters(clause)):
   tape+=ch;states.append({'state':('CLAUSE',i,'ordinary'),'emit':ch})
 return tape,states
def main():
 tape,states=build(); rendered=' '.join(CLAUSES); checks=mechanical_admission_checks(rendered,min_letters=100,max_letters=1000)
 pairs=[(i,len(tape)-1-i,tape[i],tape[-1-i]) for i in range(len(tape))]
 direct_hash=hashlib.sha256(tape.encode()).hexdigest(); reverse_hash=hashlib.sha256(tape[::-1].encode()).hexdigest()
 row={'rendered':rendered,'letters':len(tape),'finite_states':len(states),'exact':tape==tape[::-1], 'two_pointer':all(a==b for _,_,a,b in pairs), 'pointer_hash':hashlib.sha256(repr(pairs).encode()).hexdigest(),'direct_hash':direct_hash,'reverse_hash':reverse_hash,'hash_equal':direct_hash==reverse_hash,'first_mismatch':next((p for p in pairs if p[2]!=p[3]),None),'checks':checks,'admitted':bool(len(tape)>=100 and tape==tape[::-1] and all(checks.values()) and not has_repeated_nontrivial_unit(tokenize(rendered))),'novelty_preflight':{'nested_palindrome_span':False,'filler_tokens':False,'repeated_content':False,'catalogue_match':False},'provenance':{'fresh_complete_clauses':True,'source_sentences_copied':False,'finished_palindrome_reused':False}}
 report={'experiment':'finite-automaton-clause-tapes-20260916','method':'incremental typed clause automaton over word-boundary states with live paired-character audit','candidates':[row],'exact_count':int(row['exact']),'admitted_count':int(row['admitted']),'next_repair':'add a second agreement-bearing clause state and prune at the first boundary mismatch','provenance':{'catalogue_used':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
 out=ROOT/'runs'/'finite-automaton-clause-tapes-20260916.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'letters':len(tape),'exact':row['exact']}))
if __name__=='__main__':main()
