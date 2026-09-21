import json, runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'experiments/relative_arc_twochar_trie_20260921.py'))
def test_twochar_index_and_rendered_controls():
    M['main'](); d=json.loads((ROOT/'runs/relative-arc-twochar-trie-20260921.json').read_text())
    assert d['candidate_count']==d['indexed_pair_count']
    assert d['candidate_count']>0
    assert d['exact_count']==0
    assert d['reader_eligible'] is False
    assert d['stats']['longest_letters']>38
    for row in d['rendered_candidates']:
        assert row['audit']['letters']>38
        assert row['audit']['forward_sha256']!=row['audit']['reverse_sha256']
        assert row['internal_predicate_index']['left_predicate_class']==row['internal_predicate_index']['right_predicate_class']
        assert row['internal_predicate_index']['agreement'] in {'singular','definite','plural'}
def test_independent_audit():
    assert M['audit']('A man, a plan, a canal.')['two_pointer_exact'] is False
