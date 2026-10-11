"""Finite frontier diagnostics for imported failed inventory + known controls."""
import json,time,zipfile,hashlib
from pathlib import Path
from llm_palindrome.phrase_inventory_search import fragment_frontier,search_fragment_api
from experiments.phrase_inventory_pilot_012 import fixture
OUT=Path('research/block-seams/debt-conditioned-013')
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
def run():
 t=time.monotonic();path=OUT/'imports/native-fragment-batch-20261010-evidence.zip'
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  name=next(n for n in z.namelist() if n.endswith('luna-fragment-inventory-raw.json'));native=json.loads(z.read(name));receipt=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/search-receipt.json'))))
 save('plan.json',dict(max_native_plans=8,max_fragments=48,diagnostics='one root frontier per imported7plans plus2partial known question frontiers',local_model_calls=0,paid_api_calls=0,max_control_product_operations=100000,max_control_seconds=5,max_control_paths=2000,no_adaptive_batch=True,selection='Independent coherence/readability before meaningful length; grade repetition, never ban; local frontier compatibility is mechanical, not a quality score'))
 diagnostics=[fragment_frontier(native,p['id']) for p in native['plans']];save('native-frontiers.json',diagnostics)
 rejection=[dict(plan_id=d['plan_id'],root_pair_count=len(d['paired_endpoint_alternatives']),compatible_pairs=sum(x['compatible'] for x in d['paired_endpoint_alternatives']),immediate_rejection=d['immediate_rejection'],reasons=[x['letter_debt'] for x in d['paired_endpoint_alternatives'] if not x['compatible']]) for d in diagnostics];save('endpoint-rejections.json',rejection)
 inv,plans=fixture();payload=dict(fragments=[dict(id=p.id,text=p.text,entry=p.entry,exit=p.exit,role=p.role,source=p.source,bindings=dict(p.bindings)) for p in inv],plans=plans)
 p=plans[0];first=fragment_frontier(payload,p['id'],[p['slots'][0][0]],[p['slots'][-1][0]]);second=fragment_frontier(payload,p['id'],[p['slots'][0][0],p['slots'][1][0]],[p['slots'][-1][0]])
 save('controlled-frontiers.json',[first,second]);baseline=search_fragment_api(payload);save('bounded-baseline-result.json',baseline)
 assert first['current']['letter_debt']['inner_required_suffix']=='t';assert second['current']['letter_debt']['inner_required_suffix']=='racat';assert len(baseline['outputs'])==7
 task=dict(task='One small native Luna frontier-conditioned fragment correction batch, not a generic inventory or whole-palindrome sweep.',max_alternatives=8,frontier=second,required=['id','text','entry','exit','role','source','bindings'],instructions=['Preserve an intelligible single question/scene and the observer I.','Propose useful alternative phrases for the open grammar states and mirror debt shown; no unrelated independent palindrome sentence.','The current bridge must satisfy inner_required_suffix racat. Existing exact control completes or a cat; copied completion is calibration, not progress.','If no natural new solution, report failure; do not repair with filler or padding.','For new scene inventory first propose compatible endpoints with explicit roles, then condition inner phrases on actual reverse debt.','Do not assume all compatible local options complete; final product search must verify whole tape.','Prefer meaningful coherent content to length; repetition is a graded cost.','Save raw batch and Library receipt before further iteration.'],native_failed_inventory_diagnostic=rejection,quality_labels='No independent rating or human acceptance inferred;50corpus010editorialAIreview not human feedback.')
 save('parent-conditioned-task.json',task)
 summary=dict(imported_library_id='libfile_a224300d7f808191b0d3ff0722370542',import_zip_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native_plans=len(diagnostics),native_root_pairs=sum(len(d['paired_endpoint_alternatives']) for d in diagnostics),native_immediate_rejections=sum(d['immediate_rejection'] for d in diagnostics),native_compatible_root_pairs=sum(x['compatible'] for d in diagnostics for x in d['paired_endpoint_alternatives']),native_original_search_receipt=receipt,baseline_exact_derivations=len(baseline['outputs']),baseline_unique_tapes=len({r['tape'] for r in baseline['outputs']}),new_readable_paragraphs_claimed=0,new_quality_scores=0,human_labels=0,elapsed_seconds=time.monotonic()-t,controlled_result='Known viable question frontier gives suffix t then racat; grammar slots and actor I remain explicit. Generic native plans rejected at endpoints before any product search.',no_new_generation=True)
 save('summary.json',summary);save('checkpoint.json',dict(status='completed_frontier_enhancement',summary=summary,next='Parent small conditioned phrase batch using saved debt/grammar/actor task; no duplicate generic batch.',human_feedback_on50='not inferred; editorial AIreview separate'))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
