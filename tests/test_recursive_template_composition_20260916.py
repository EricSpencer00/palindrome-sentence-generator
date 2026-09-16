import json
from pathlib import Path
def test_recursive_template_composition():
 p=json.loads((Path(__file__).parents[1]/'runs/recursive-template-composition-20260916.json').read_text())
 assert p['registry_preflight']['exact_signature_collisions']==[]
 assert p['complete_sentence_count']==3 and p['reader_eligible_count']==0
 assert all(x['no_repeated_units'] for x in p['candidates'])
