import json
from pathlib import Path
def test_paired_adverb_lane():
 x=json.loads(Path('runs/cfg-before-paired-adverb-agreement-20260917.json').read_text());assert x['candidate_count']>0 and x['pruned_partial_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert tuple(r['grammar_state']['adverb_pair']) in (('later','afterward'),('then','later'),('afterward','then'));assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_compatible_pair():
 assert 'earlier/later' in json.loads(Path('runs/cfg-before-paired-adverb-agreement-20260917.json').read_text())['next_repair']
