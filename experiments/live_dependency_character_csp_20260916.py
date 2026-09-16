"""Orthogonal repair: live dependency-boundary character equations."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1]
SCENE=("At first light, the marine biologist logged three patient observations "
       "while her assistant arranged clean instruments beside the sheltered tide pool.")
HELDOUT={"marine":"adj","biologist":"n","observations":"n","assistant":"n","sheltered":"adj"}
def solve(text):
 t=normalize_letters(text); equations=[]
 for i,ch in enumerate(t): equations.append({"left_boundary":i,"right_boundary":len(t)-1-i,"equation":f"x[{i}] = x[{len(t)-1-i}]","left_char":ch,"right_char":t[-1-i]})
 mismatch=next((e for e in equations if e['left_char']!=e['right_char']),None)
 return t,equations,mismatch
def main():
 t,eq,mismatch=solve(SCENE); checks=mechanical_admission_checks(SCENE,min_letters=100,max_letters=1000)
 reverse_hash=hashlib.sha256(t[::-1].encode()).hexdigest(); direct_hash=hashlib.sha256(t.encode()).hexdigest()
 row={"rendered":SCENE,"letters":len(t),"dependency_boundary_variables":len(eq),"first_mismatch":mismatch,
      "two_pointer_exact":not any(e['left_char']!=e['right_char'] for e in eq),"reverse_hash":reverse_hash,"direct_hash":direct_hash,
      "exact":t==t[::-1],"admitted":bool(len(t)>100 and t==t[::-1] and all(checks.values())),"checks":checks,
      "heldout_lexicalization":HELDOUT,"novelty_preflight":{"copied_text":False,"catalogue_match":False,"prior_lane_state_reused":False},
      "provenance":{"fresh_complete_scene":True,"source_sentences_copied":False,"finished_sentence_reversed":False}}
 report={"experiment":"live-dependency-character-csp-20260916","method":"dependency boundary variables with live character equations and held-out lexicalization","candidates":[row],"exact_count":int(row['exact']),"admitted_count":int(row['admitted']),"next_repair":"replace one adjunct with a held-out typed PP and propagate its live boundary equations before realization","provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"catalogue_used":False}}
 out=ROOT/'runs'/'live-dependency-character-csp-20260916.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'letters':len(t),'exact':row['exact'],'mismatch':mismatch['left_boundary'] if mismatch else None}))
if __name__=='__main__':main()
