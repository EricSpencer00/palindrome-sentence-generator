import json
from pathlib import Path

def test_recursive_cfg_repair_artifact():
    x=json.loads(Path('runs/cfg-relative-clause-inflection-repair-20260917.json').read_text())
    assert x['control_count'] > 0 and x['repair_count'] > 0
    assert x['exact_count'] == 0
    for r in x['candidates']:
        assert r['anti_shortcut']['single_tree']
        assert not r['anti_shortcut']['word_order_only']
        assert r['audit']['two_pointer'] == r['audit']['reverse_sha256_equal']
        assert 'who ' in r['rendered']

def test_repair_is_explicit_and_held_out():
    x=json.loads(Path('runs/cfg-relative-clause-inflection-repair-20260917.json').read_text())
    assert 'second RC attachment site' in x['next_repair']
