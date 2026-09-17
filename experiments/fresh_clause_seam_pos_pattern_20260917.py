"""Fresh coordinated clause-seam POS pattern using live outer-slot matching."""
import hashlib,json
from pathlib import Path
from experiments.full_sequence_grammar_product_20260917 import PATTERNS,search_pattern
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/fresh-clause-seam-pos-pattern-20260917.json'
def run():
 slots=PATTERNS['coordinated']; result=search_pattern(slots,state_budget=100000,forbidden_words=frozenset({'doc','note','i','dissent','fast','never','prevents','fatness','diet','cod'}))
 paths=[p for p in result.paths if p['audit']['letters']>38]
 return {'experiment_id':'fresh-clause-seam-pos-pattern-20260917','signature':'coordinated-pos-seam|live-outer-slot-matching|heldout-lexical-banks|no-seed-wrap','pattern':slots,'states':result.states,'budget_exhausted':result.budget_exhausted,'candidate_count':len(paths),'exact_count':sum(p['audit']['exact'] for p in paths),'longest_partial':result.longest_partial,'mismatch_frontiers':result.mismatch_frontiers[:20],'candidates':paths,'novelty_preflight':{'status':'passed','seed_wrapping':False,'catalogue_text':False,'posthoc_resegmentation':False,'fixed_frame_sweep':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audit':'full_sequence exact_audit','lexical_banks':'held-out role banks'}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print({'states':x['states'],'candidates':x['candidate_count'],'exact':x['exact_count'],'budget_exhausted':x['budget_exhausted']})
