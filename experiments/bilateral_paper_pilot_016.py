"""Paper-aligned phrase-overhang trace, bounded productive-cycle diagnostics."""
import json,time,hashlib
from pathlib import Path
from dataclasses import asdict
from llm_palindrome.bilateral_seams import Chunk,CenterOutGrammar,BilateralGrammar
from llm_palindrome.admission import normalize_letters as norm
OUT=Path('research/block-seams/bilateral-paper-016')
def fixture():return [Chunk('L1','Was it ','START','NP','question copula/subject','inherited question control'),Chunk('L2','a ca','NP','CAR_SUFFIX','determiner/partial car','car lexical lemma'),Chunk('L3','r ','CAR_SUFFIX','OR_PREFIX','car completion','car lexical lemma'),Chunk('C','o','OR_PREFIX','OR_SUFFIX','partial coordinator','or lexical lemma'),Chunk('R3','r a ','OR_SUFFIX','CAT','coordinator completion/determiner','or/a seam'),Chunk('R2','cat','CAT','REL','observed theme','cat lemma'),Chunk('R1',' I saw?','REL','END','relative observation','inherited question control',(('observer','I'),))]
def run():
 OUT.mkdir(exist_ok=True,parents=True);start=time.monotonic();cs=fixture();g=CenterOutGrammar(cs,['C']);s=g.initial();trace=[]
 for pid,side in [('L3','left'),('R3','right'),('L2','left'),('R2','right'),('L1','left'),('R1','right')]:
  s,error=g.extend(s,next(c for c in cs if c.id==pid),side);assert error is None;trace.append(dict(step=len(trace)+1,left=s['left'],center=g.center,right=s['right'],left_state=s['left_state'],right_state=s['right_state'],bindings=s['bindings'],**s['trace'][-1]))
 assert g.closed(s);assert norm(g.render(s))==norm(g.render(s))[::-1]
 control=g.search(max_steps=12,max_states=2000,max_outputs=20,seconds=2)
 # Formal grammar has productive a-loop. It supplies a^n for every n>=1,
 # explicitly NOT an English or repetition-free/readability certificate.
 formal=BilateralGrammar([Chunk('loop','a','START','START','formal diagnostic','not English grammar'),Chunk('end','a','START','END','formal diagnostic','not English grammar')]);cycle=formal.search(max_steps=20,max_states=2000,max_outputs=30,seconds=2)
 assert {r['letters'] for r in cycle['outputs']}==set(range(1,21))
 result=dict(plan=dict(max_control_states=2000,max_control_steps=12,max_formal_steps=20,max_formal_states=2000,seconds_per_search=2,models=0,paidcalls=0),normalization='ASCII-letter norm; unsupported alphabet/rendering rejected',paper_direction='Center-out:prependleft/appendright;closeonemptydebt. Outside-in retained separately:appendleft/prependright;palindromicdebtcanclose.',trace=trace,real_output=g.render(s),left_sequence=s['left_ids'],right_sequence=s['right_ids'],center_id='C',control_search=control,formal_cycle=cycle,infinite_scope='The formal a-loop witnesses arbitrarily long distinct a^n tapes; repeated letters, not infinite coherent/nonrepeating prose.',soft_repetition='repeated chunks retained with increasing cost; text histories not collapsed by debt signature',payoff='new heuristic paid_letters/added_letters,not paperresult; can validlyextend debt withratio0',elapsed_seconds=time.monotonic()-start)
 (OUT/'trace-and-results.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'chunks.json').write_text(json.dumps([asdict(c) for c in cs],indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['real_output','left_sequence','right_sequence','center_id','elapsed_seconds']},indent=2));print(json.dumps(trace,indent=2))
if __name__=='__main__':run()
