import json
from pathlib import Path
def test_paired_relation_lane():
 x=json.loads(Path('runs/cfg-while-paired-plural-object-relation-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  z=r['grammar_state'];assert z['object_relation']=='paired';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_relation_alternation():
 assert 'semantic relation alternation' in json.loads(Path('runs/cfg-while-paired-plural-object-relation-20260917.json').read_text())['next_repair']
