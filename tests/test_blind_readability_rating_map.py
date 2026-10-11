"""Synthetic mapping tests; no artificial judgments enter research artifacts."""
import pytest
from experiments.verify_blind_readability_export_20261009 import fanout_ratings

def test_one_independent_rating_maps_to_all_method_seed_occurrences():
    occurrences=[dict(occurrence_id='runA#methodA',rating_id='text-A',normalized_text_key='sha256:A'),dict(occurrence_id='runB#methodB',rating_id='text-A',normalized_text_key='sha256:A')]
    unique=[dict(rating_id='text-A')]
    judgments=[dict(rating_id='text-A',grammar=3,readability=2,coherence=1,repetition=4,explanation='Synthetic fixture only.')]
    mapped=fanout_ratings(occurrences,unique,judgments)
    assert [r['occurrence_id'] for r in mapped]==['runA#methodA','runB#methodB']
    assert all(r['independent_judgment'] is judgments[0] for r in mapped)

def test_missing_duplicate_unknown_and_invalid_ratings_fail_closed():
    unique=[dict(rating_id='text-A')];occurrences=[dict(occurrence_id='run#A',rating_id='text-A',normalized_text_key='sha256:A')]
    good=dict(rating_id='text-A',grammar=3,readability=2,coherence=1,repetition=4,explanation='Synthetic fixture only.')
    for bad in [[],[good,good],[dict(good,rating_id='wrong')],[dict(good,grammar=True)],[dict(good,coherence=5)],[dict(good,explanation='')]]:
        with pytest.raises(AssertionError):fanout_ratings(occurrences,unique,bad)
