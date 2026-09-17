import json
from pathlib import Path
def test_while_subject_alternation_lane():
 x=json.loads(Path('runs/cfg-while-subject-alternation-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert r['grammar_state']['subject_relation']=='alternating';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_object_frame():
 assert 'alternating object frame' in json.loads(Path('runs/cfg-while-subject-alternation-20260917.json').read_text())['next_repair']
