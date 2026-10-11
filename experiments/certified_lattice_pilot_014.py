"""Native alternatives admitted only with exact finite continuation witnesses."""
import json,zipfile,time,hashlib
from pathlib import Path
from experiments.phrase_inventory_pilot_012 import fixture
from llm_palindrome.phrase_inventory_search import conditioned_lattice,letter_admit_alternative
from llm_palindrome.admission import normalize_letters
OUT=Path('research/block-seams/certified-lattice-014')
def save(n,o):(OUT/n).write_text(json.dumps(o,indent=2)+'\n')
def run():
 start=time.monotonic();inv,plans=fixture();payload=dict(fragments=[dict(id=p.id,text=p.text,entry=p.entry,exit=p.exit,role=p.role,source=p.source,bindings=dict(p.bindings)) for p in inv],plans=plans);p=plans[0];left=[p['slots'][0][0],p['slots'][1][0]];right=[p['slots'][-1][0]]
 archive=next((OUT/'imports').glob('*.zip'))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  native=json.loads(z.read(next(n for n in z.namelist() if n.endswith('luna-conditioned-alternatives-raw.json'))));receipt=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/search-receipt.json'))))
 save('plan.json',dict(seed='certified-lattice-014',native_alternatives=4,fixed_plans=5,inventory=48,max_work_per_lattice=100000,max_seconds_per_lattice=5,max_paths_per_lattice=2000,max_calls=9,local_models=0,paid_apis=0,no_adaptive_iteration=True,quality_policy='coherence/premise/progression before meaningful length;soft repetition; mechanical compatibility is not quality',stop='4admissionchecks+5existingplanlattices; report no novelty rather than spend model calls on stock lattice'))
 gates=[]
 for a in native['alternatives']:gates.append(dict(alternative=a,admission=letter_admit_alternative(payload,p['id'],left,right,a['slot_side'],a)))
 save('native-admission-failures.json',gates);save('native-original-search-receipt.json',receipt)
 lattices=[conditioned_lattice(payload,plan['id']) for plan in plans];save('certified-lattices.json',lattices)
 # Compare against complete104-path enumeration previously saved in012.
 control=json.load(open('research/block-seams/phrase-inventory-012/enumeration-control-all-proposals.json'));expected={(r['plan'],tuple(r['phrase_ids']),r['tape']) for r in control if r['exact']};actual={(r['plan'],tuple(r['phrase_ids']),r['tape']) for l in lattices for r in l['outputs']};assert actual==expected
 allrows=[r for l in lattices for r in l['outputs']];known=json.load(open('research/block-seams/phrase-inventory-012/raw-outputs.json'));known_tapes={r['tape'] for r in known};fresh=[r for r in allrows if r['tape'] not in known_tapes]
 bound=max(sum(max(len(normalize_letters(next(f['text'] for f in payload['fragments'] if f['id']==pid))) for pid in slot) for slot in pl['slots']) for pl in plans)
 save('semantic-selector-input.json',dict(certified_exact_paths=allrows,novel_paths=fresh,selector_should_run=bool(fresh),reason='No novel paths in this finite lattice; no quality gain can be inferred from selecting stock controls. Avoid another model pass.',quality_scores=[],human_labels=[]))
 summary=dict(native_library_id='libfile_99928fa9aea48191a916c49367d25505',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),native_alternatives=len(gates),admitted=sum(g['admission']['admitted'] for g in gates),letter_mismatch_rejections=sum(g['admission']['reason']=='letter_mismatch' for g in gates),no_closure_rejections=sum(g['admission']['reason']=='no_exact_closure_in_finite_inventory' for g in gates),certified_path_derivations=len(allrows),unique_certified_tapes=len({r['tape'] for r in allrows}),new_relative_to012=len(fresh),independent104pathcontrol_agrees=True,max_possible_letters_in_declared_grammar=bound,goal60letters_possible=bound>=60,actual_max_letters=max(r['letters'] for r in allrows),new_model_calls=0,new_quality_scores=0,elapsed_seconds=time.monotonic()-start,measured_reason_none=f'All native proposals fail deterministic admission; all7certifiedpaths are existing controls. Current5-plan grammar has a{bound}-letter maximum, so cannot reach60letters. This is a finite coverage failure, not universal impossibility.')
 save('summary.json',summary);save('checkpoint.json',dict(status='completed_negative_lattice',summary=summary,next='Expand reverse-compatible lexical/grammatical coverage before another model call. Semantic selector receives only certified complete paths; no new stock batch scoring.',human_feedback='No human acceptance inferred from editorial50review.'))
 print(json.dumps(summary,indent=2));print(json.dumps([(g['alternative']['id'],g['admission']['reason'],g['admission'].get('mismatch')) for g in gates],indent=2))
if __name__=='__main__':run()
