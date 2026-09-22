import hashlib
import json
from pathlib import Path

from experiments.anaphoric_inflection_dependency_product_20260922 import (
    MORPHOLOGY_TRANSITIONS,
    REJECTED_MIGRATIONS,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "anaphoric_inflection_dependency_product_20260922.py"
ARTIFACT = ROOT / "runs" / "anaphoric-inflection-dependency-product-20260922.json"


def test_productive_s_migrates_only_across_licensed_boundaries():
    assert all(item.licensed for item in MORPHOLOGY_TRANSITIONS)
    assert not any(item.licensed for item in REJECTED_MIGRATIONS)
    assert {item.productive_s_sites for item in MORPHOLOGY_TRANSITIONS} == {
        ("possessive", "third_person_singular"),
        ("plural", "third_person_singular"),
    }
    assert all(item.finite_surface == "stops" for item in MORPHOLOGY_TRANSITIONS)


def test_clause_two_is_committed_before_equation_and_replaces_base_stop():
    payload = run()
    row = next(
        item for item in payload["audited_candidates"]
        if item["path"] == {
            "carrier": "speaker",
            "cycles": ("location-command",),
            "genitive": "plural-snoops-spoon",
            "morphology_transition": "plural-possessive-plus-3sg",
        }
    )
    assert row["dependency"]["clause_two_tokens_committed_before_equation"] == ("it", "stops")
    assert row["dependency"]["coreference_resolved"]
    assert row["dependency"]["predicate_agreement_resolved"]
    assert row["dependency"]["predicate_valency_resolved"]
    assert row["right_frontier"][:3] == ("spoon", "it", "stops")
    assert row["right_frontier"].count("stop") == 0
    assert payload["fixed_domain"]["post_hoc_insertion"] is False


def test_exact_first_inflection_cursor_obstruction_is_persisted():
    obstruction = run()["first_pronoun_inflection_cursor_obstruction"]
    assert obstruction["path"]["genitive"] == "plural-snoops-spoon"
    assert obstruction["dependency"]["clauses_finished"] == (True, True)
    assert obstruction["cursor"]["token"] == "stops"
    assert obstruction["cursor"]["expected_character"] == "p"
    assert obstruction["cursor"]["observed_character"] == "s"
    assert obstruction["cursor"]["right_cursor_before"] == 3
    assert obstruction["cursor"]["attempted_right_cursor"] == 4
    assert obstruction["terminal_register"]["owner"] == "L"
    assert obstruction["terminal_register"]["residual"] == "potsnoops"
    assert obstruction["terminal_register"]["closed"] is False


def test_freshness_resolves_plural_possessive_and_3sg_surfaces_to_lemmas():
    payload = run()
    row = next(
        item for item in payload["audited_candidates"]
        if item["path"]["genitive"] == "plural-stops-spot"
    )
    freshness = row["global_lemma_freshness"]
    assert freshness["morphological_surfaces_resolved"]["stops'"] == "stop"
    assert freshness["morphological_surfaces_resolved"]["stops"] == "stop"
    assert freshness["all_distinct"] is False


def test_bounded_operator_has_no_exact_connected_survivor_or_lexical_widening():
    payload = run()
    assert payload["stats"] == {
        "paths": 30,
        "clause_two_committed_paths": 30,
        "both_required_clauses_finished_paths": 1,
        "equation_closures": 0,
        "exact_paths_over_44": 0,
        "max_letters": 73,
        "survivors": 0,
    }
    assert payload["survivors"] == []
    assert payload["fixed_domain"]["lexical_widening"] is False
    assert payload["fixed_domain"]["changed_operator_only"] == "clause_2_anaphora_and_finite_inflection"
    assert payload["verdict"] == "bounded_pronoun_inflection_cursor_obstruction"


def test_artifact_audits_every_candidate_with_live_state_and_masks():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["provenance"]["host"] == "hst-bench"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == payload["provenance"]["source_sha256"]
    assert len(payload["audited_candidates"]) == payload["stats"]["paths"]
    for row in payload["audited_candidates"]:
        assert row["rendered"]
        assert row["dependency"]["observed_coreferent"] == "it"
        assert row["dependency"]["observed_finite_surface"] == "stops"
        assert row["character_register"]["owner"] in {"L", "R", ""}
        assert isinstance(row["character_register"]["residual"], str)
        assert row["character_register"]["left_cursor"] >= 0
        assert row["character_register"]["right_cursor"] >= 0
        assert all(step["masks_live"] for step in row["character_register"]["trace"])
        assert "all_distinct" in row["global_lemma_freshness"]
        assert "passes" in row["complementary_boundary_mask"]
        assert "passes" in row["proper_span_mask"]
