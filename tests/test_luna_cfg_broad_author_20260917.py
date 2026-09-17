import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.luna_cfg_broad_author_20260917 import run, letters

def test_broad_lane_artifact_and_independent_audit():
    x=run()
    assert x['paths'] > 0 and x['search']['states'] > 0
    for row in x['exact_candidates']:
        assert row['audit']['exact']
        assert row['audit']['letters'] == len(letters(row['rendered']))
        assert row['provenance']['catalogue_lookup'] is False

def test_no_fabricated_lexical_units():
    from experiments.luna_cfg_broad_author_20260917 import LEX
    assert all(w.isalpha() for ws in LEX.values() for w in ws)
