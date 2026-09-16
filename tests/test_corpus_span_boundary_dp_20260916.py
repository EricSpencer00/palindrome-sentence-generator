import json
from pathlib import Path
def test_corpus_span_run_audited_and_nonreader():
 p=json.loads((Path(__file__).parents[1]/'runs/corpus-span-boundary-dp-20260916.json').read_text())
 assert p['registry_preflight']['exact_signature_collisions']==[]
 assert p['exact_count']==0 and p['reader_eligible_count']==0
 assert p['repair_operator_trials']>0
