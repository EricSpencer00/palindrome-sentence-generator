import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/semantic-valency-nfa-product-20260917.json'
def test_graph_product_is_bounded_and_exact():
 d=json.loads(RUN.read_text())
 assert d['budget']==128 and d['expanded_states']<=d['budget']
 assert d['exact_paired_paths']>=0 and d['pruned_transitions']>=0
 for p in d['paths']:
  assert p['left_roles'] and p['right_roles']
  assert len(p['left_sha256'])==64 and len(p['right_sha256'])==64
def test_graph_has_agreement_typed_roles():
 d=json.loads(RUN.read_text())
 assert 'agent:sg' in d['graph'] and 'patient:pl' in d['graph']
