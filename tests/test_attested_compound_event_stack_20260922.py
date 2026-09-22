import hashlib
import json
from pathlib import Path

from experiments.attested_compound_event_stack_20260922 import (
    RESIDUAL,
    TRANSITIVE_FRAMES,
    cycle_equation,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "attested_compound_event_stack_20260922.py"
ARTIFACT = ROOT / "artifacts" / "attested-compound-event-stack-20260922" / "search.json"


def _payload():
    return json.loads(ARTIFACT.read_text())


def test_remote_search_is_source_identical_and_fixed_to_hst_bench():
    payload = _payload()
    assert payload["provenance"]["host"] == "hst-bench"
    assert payload["provenance"]["python"] == "3.12.3"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == payload["provenance"]["source_sha256"]
    assert payload["fixed_conditions"]["residual"] == RESIDUAL
    assert payload["fixed_conditions"]["stack_depth"] == 3
    assert payload["fixed_conditions"]["lexical_widening_after_run"] is False
    assert payload["fixed_conditions"]["bare_lemma_pairs"] is False


def test_compound_and_valency_domains_are_bounded_and_nonempty():
    payload = _payload()
    inventory = payload["inventory"]
    assert inventory["unique_noun_compounds"] == 14547
    assert inventory["unique_wordnet_licensed_verb_objects"] == 2092
    assert inventory["wordnet_valency_licensed_occurrences"] == 2356
    assert inventory["wordnet_valency_rejections"] == 13
    assert set(payload["fixed_conditions"]["wordnet_transitive_frames"]) == TRANSITIVE_FRAMES


def test_zero_result_records_the_first_exact_compound_cursor():
    payload = _payload()
    stats = payload["search_stats"]
    assert stats["residual_supported_verb_object_tapes"] == 251
    assert stats["compound_tape_misses"] == 251
    assert payload["exact_stack_count"] == 0
    assert payload["independently_audited_exact_stacks"] == []
    assert payload["survivors"] == []
    cursor = payload["first_compound_or_valency_obstruction"]
    required = cursor["required_compound_tape"]
    observed = cursor["observed_compound_tape"]
    stop = cursor["character_cursor"]
    assert cursor["grammar_phase"] == "noun_compound_return_lookup"
    assert required[:stop] == observed[:stop] == "strap"
    assert required[stop] != observed[stop]
    assert cursor["verb_object"]["words"] == ["sacrifice", "part"]
    assert set(cursor["verb_object"]["wordnet_frames"]).intersection(TRANSITIVE_FRAMES)


def test_cycle_equation_is_literal_not_a_finished_tape_reversal():
    row = cycle_equation("separateelectron", "snortceleetarape")
    assert row["holds"] is True
    assert row["left"] == "separateelectrons"
    assert row["right"] == "separateelectrons"
