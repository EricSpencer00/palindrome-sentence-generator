import json
from pathlib import Path
def test_lexical_trie_product_lane():
 x=json.loads(Path('runs/cfg-midpoint-lexical-trie-product-20260917.json').read_text());assert x['candidate_count']==3072 and x['exact_count']==0
 for r in x['diagnostic_controls']:
  assert r['novelty_preflight']['trie_intersection'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal'];assert r['grammar_state']['role_state']['agent']
def test_no_partial_admission():
 assert json.loads(Path('runs/cfg-midpoint-lexical-trie-product-20260917.json').read_text())['admitted_renderings']==[]
