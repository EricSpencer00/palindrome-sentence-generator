"""Explicit supplemental long meaningful-clause probe after47-letter bound audit."""
import json,gzip,time,math,hashlib
from pathlib import Path
from dataclasses import asdict
from experiments.expanded_phrase_coverage_015 import slot
from llm_palindrome.bidirectional_lexical import Frame,GrammarDAG,exact_grammar_palindromes
OUT=Path('research/block-seams/expanded-phrase-015/long-clause-probe')
def run():
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 # Actual role grammar: subject helps patient because a caregiver(with a
 # restrictive relative) acted before an authority called that subject.
 frame=Frame('long_caregiving_cause',tuple([slot('human_agent','Mom'),slot('past_verb','helped','diapered','repaid'),slot('human_patient','Dad'),slot('causal_connector','because'),slot('determiner','the'),slot('caregiver','nurse','doctor','teacher'),slot('relative','who'),slot('relative_past_verb','helped','stopped','spotted'),slot('relative_patient','Dad'),slot('causal_past_verb','helped','stopped','spotted','called'),slot('causal_patient','Mom'),slot('time_connector','before'),slot('determiner','the'),slot('authority','doctor','teacher','nurse'),slot('past_verb','called','helped'),slot('human_patient','Mom')]),provenance='supplemental authored long-clause grammar; no mirrored sentences; temporal/causal meaning quality unverified')
 plan=dict(scope='separate supplemental probe, added after initial23frame47letterceiling audit; initial receipts preserved',max_work=2000000,max_seconds=8,max_paths=5000,frames=[asdict(frame)],finite_derivations=math.prod(len(s.words) for s in frame.slots),max_letters=sum(max(len(w) for w in s.words) for s in frame.slots),min_letters=sum(min(len(w) for w in s.words) for s in frame.slots),models=0,paidcalls=0)
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');g=GrammarDAG([frame],1)
 with gzip.open(OUT/'states.jsonl.gz','wt') as trace:paths,receipt=exact_grammar_palindromes(g,max_work=2000000,seconds=8,max_paths=5000,trace=lambda r:trace.write(json.dumps(r)+'\n'))
 rows=[g.materialize(p) for p in paths]
 summary=dict(plan=plan,complete=receipt['complete'],receipt=receipt,raw_outputs=rows,exact_outputs=len(rows),failed_exact_derivations=plan['finite_derivations']-len(rows),elapsed_seconds=time.monotonic()-start,no_readability_scores=True)
 (OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='raw_outputs'},indent=2))
if __name__=='__main__':run()
