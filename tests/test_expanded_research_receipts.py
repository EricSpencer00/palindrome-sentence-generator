import hashlib,json,math
from pathlib import Path
from llm_palindrome.admission import normalize_letters
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'research/block-seams'
def test_expanded_manifest_keeps_proposal_failures_and_rounded_strata():
 d=json.loads((B/'expanded-representative-sample-001.json').read_text())
 assert d['proposal_occurrences']==23737 and len(d['sample'])==2378
 assert sum(s['denominator'] for s in d['strata'])==23737
 assert all(s['sample_count']==math.ceil(s['denominator']/10) for s in d['strata'])
 assert len({r['id'] for r in d['sample']})==2378
 assert all(hashlib.sha256((ROOT/s['path']).read_bytes()).hexdigest()==s['sha256'] for s in d['sources'])
 assert any(not r['exact'] for r in d['sample'])
 assert all(r['human_score'] is None for r in d['sample'])
def test_complete_constituent_output_equivalence_and_sample():
 d=json.loads((B/'complete-constituent-join-001.json').read_text());assert d['output_sets_equal']
 assert len(d['outputs'])==528 and len(d['stratified_sample'])==53
 assert all(normalize_letters(r['text'])==normalize_letters(r['text'])[::-1] for r in d['outputs'])
 assert sum(r['mechanically_eligible'] for r in d['outputs'])==335
 assert len({r['id'] for r in d['outputs']})==528
 assert all(r['human_score'] is None for r in d['outputs'])
def test_cold_profile_preserves_menus_and_matches_bounded_work_receipt():
 a=json.loads((B/'cold-position-profile-001-baseline.json').read_text());b=json.loads((B/'cold-position-profile-001-cached.json').read_text())
 assert a['menu_signatures']==b['menu_signatures'];assert b['position_cache_info']['hits']==31318;assert b['position_cache_info']['maxsize']==8192
 d=json.loads((B/'complete-toy-cache-comparison-001.json').read_text())
 for seed in (921,922,923):
  a,b=[r for r in d['rows'] if r['seed']==seed]
  assert a['outputs']==b['outputs'] and len(a['outputs'])==12
  assert a['attempted_actions']==b['attempted_actions']==812
  assert a['visited_states']==b['visited_states']==435
  assert not a['truncated'] and not b['truncated']
