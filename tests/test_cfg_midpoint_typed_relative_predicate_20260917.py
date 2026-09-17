import json
from pathlib import Path
def test_relative_predicate_lane():
 x=json.loads(Path('runs/cfg-midpoint-typed-relative-predicate-20260917.json').read_text());assert x['candidate_count']==2304 and x['exact_count']==0
 for r in x['diagnostic_controls']:assert r['novelty_preflight']['typed_relative_predicate'] and r['grammar_state']['complete_obligations'] and r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_no_shortcuts():assert json.loads(Path('runs/cfg-midpoint-typed-relative-predicate-20260917.json').read_text())['admitted_renderings']==[]
