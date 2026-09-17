import json
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
MOD=ROOT/'experiments/luna_lexicalized_reverse_trie_grammar_20260917.py'
spec=importlib.util.spec_from_file_location('lane',MOD); lane=importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)

def test_run_has_actual_long_prose_and_independent_audits():
    result=lane.run(); assert result['novelty_preflight']['status']=='passed'
    assert result['stats']['longest_letters']>80
    assert result['candidates']
    for row in result['candidates']:
        assert row['rendered'] and row['audit']['letters']>80
        assert row['audit']['independent_two_pointer_exact']==row['audit']['exact']
        assert row['audit']['sha256_forward'] != row['audit']['sha256_reverse']
        assert row['anti_shortcut_flags']['finished_tape_reversal'] is False
        assert row['trie_obligation']['matched_prefix_letters'] > 0

def test_provenance_and_repair_are_present():
    result=lane.run()
    assert result['provenance']['catalogue_text_imported'] is False
    assert len(result['provenance']['generator_sha256'])==64
    assert result['failure_and_repair']['concrete_next_repair']
