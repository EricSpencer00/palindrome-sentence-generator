import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('lane',Path(__file__).with_name('paragraph_abba_dialogue_trie_20260922.py'))
lane=importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)

def test_reverse_character_stream_crosses_word_boundary():
    # reverse("some m") == "memos"; this catches word-order-only symmetry.
    ok, trace=lane.online_pair('memos','some m')
    assert ok and len(trace)==5

def test_similar_phrase_does_not_fake_support():
    # reverse("some men") starts with n, so it must not be admitted as memos.
    ok, _=lane.online_pair('memos','some men')
    assert not ok

def test_outer_domain_condition_is_two_character_reverse_exposure():
    support=lane.outer_domain_support('An old cartographer marked the inlet.',
                                      'At dusk the keeper watched the arena.')
    assert support['compatible']
    assert support['left_exposed']=='an'
    assert support['right_reverse_exposed']=='an'
    rejected=lane.outer_domain_support('An old cartographer marked the inlet.',
                                       'Before sleep the navigator guarded the quiet cove.')
    assert not rejected['compatible']

def test_run_keeps_complete_prose_and_independent_hashes():
    result=lane.run()
    assert result['stats']['candidate_completions'] == 4
    assert all(row['rendered'].endswith('.') for row in result['rendered_candidates'])
    assert all(row['audit']['sha256_forward'] != row['audit']['sha256_reverse']
               for row in result['rendered_candidates'])
    assert result['stats']['exact_gt38'] == 0
    assert result['stats']['outer_pruned'] == 0
    assert all(row['outer_domain_support']['width'] == 10
               and row['outer_domain_support']['matched'] == 10
               for row in result['rendered_candidates'])
    assert result['fresh_residual_probe']['status'] == 'fresh_natural_scene_pair_found'
    assert result['fresh_residual_probe']['max_new_support_depth'] == 10
    assert 'three-character' not in result['method']
    assert any('red-laced insect collector' in row['rendered'] for row in result['rendered_candidates'])
    assert result['stats']['fresh_outer_pair_completions'] == 4
    assert result['stats']['fresh_outer_pair_support_depth'] == [10]
    assert all('A red-laced insect collector' in row['rendered'] and
               'wide caldera.' in row['rendered']
               for row in result['fresh_outer_pair_candidates'])
