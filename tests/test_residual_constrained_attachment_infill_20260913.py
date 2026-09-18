from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.local_attachment_infill_validator_20260913 import (
    attachment_prefix, parse_sentence, render_sentence,
)
from experiments.residual_constrained_attachment_infill_20260913 import (
    FRAME, MAX_QUERIES, QueryLedger, attachments, certify_residual, finite_preflight,
    fringe_trace, prepare_query, query_preview, review_attachment, review_response,
    run, token_review, verify_residual,
)


@pytest.fixture(scope="module")
def certificate():
    return certify_residual()


def test_real_production_gap_has_six_actual_pairs_and_semantic_completion(certificate):
    assert certificate["production"]
    assert certificate["actual_pairs"] == 6
    assert certificate["matched_tape"] == "trader"
    left = "".join(certificate["frozen_left_tokens"])
    right = "".join(certificate["frozen_right_tokens"])
    assert left[:6] == right[::-1][:6]
    assert not certificate["left_boundary_at_frontier"]
    assert certificate["right_boundary_at_frontier"]
    assert certificate["right_remaining_frozen_tape"] == ""
    assert certificate["finite_semantic_completion_paths"] == 192
    assert certificate["finite_distinct_semantic_surfaces"] == 192
    assert all(r["next_pair_compatible"] for r in certificate["coaccessible_next_pairs"])
    witness = certificate["full_semantic_completion_witness"]
    assert any(p["semantic_relation_valid"] for p in parse_sentence(witness["words"]))
    assert verify_residual(certificate)


@pytest.mark.parametrize("key,value", [
    ("actual_pairs", 5), ("production", False), ("frame_id", "fixture"),
    ("matched_tape", "foobar"), ("certificate_sha256", "forged"),
    ("hole_role", "complete_sentence"), ("frozen_right_tokens", ["anything"]),
])
def test_forged_stale_shallow_or_toy_residual_cannot_form_query(certificate, key, value):
    changed = deepcopy(certificate)
    changed[key] = value
    assert not verify_residual(changed)
    with pytest.raises(ValueError):
        query_preview(changed)
    with pytest.raises(ValueError):
        review_response('{"attachments":[["artists"]]}', changed)


def test_source_and_validator_are_separate_and_grammar_reparses_structure():
    validator = Path("experiments/local_attachment_infill_validator_20260913.py").read_text()
    assert "from experiments" not in validator
    assert "SOURCE_OWNERS" not in validator and "FRAME" not in validator
    words = FRAME.left + ("local", "skilled", "artists") + FRAME.right
    parsed, = parse_sentence(words)
    assert parsed["attachment"]["surface_span"] == [len(FRAME.left), len(FRAME.left) + 3]
    assert parsed["attachment"]["relation"] == "possesses"
    assert parsed["attachment"]["patient_id"] == parsed["semantic_witness"]["defect_bearer_id"]
    assert parsed["semantic_relation_valid"]
    # Counterfactual syntax can parse without pretending it is a repair event.
    bad_event = tuple("worsen" if word == "repair" else word for word in words)
    assert parse_sentence(bad_event) and not parse_sentence(bad_event)[0]["semantic_relation_valid"]
    assert not parse_sentence(words + ("extra",))
    assert not parse_sentence(FRAME.left + ("paintings",) + FRAME.right)
    assert not parse_sentence(FRAME.left + ("artists", "local") + FRAME.right)
    # The validator also parses an unseen source sentence by its own grammar.
    assert parse_sentence("conservators patiently mend large rips in young donors blue artwork".split())[0]["semantic_relation_valid"]


def test_tokens_checked_incrementally_without_silent_deletion():
    assert attachment_prefix(("local",)) and not attachment_prefix(("local",))[0]["complete"]
    assert attachment_prefix(("local", "skilled", "artists"))[0]["complete"]
    assert not attachment_prefix(("local", "artists", "skilled"))
    row = review_attachment(["local", "unparseable", "artists"])
    assert len(row["token_reviews"]) == 2
    assert row["token_reviews"][0]["accepted_into_shadow"]
    assert not row["token_reviews"][1]["accepted_into_shadow"]
    assert not row["eligible_for_external_provenance"]
    assert not row["frozen_tape_mutated"]


@pytest.mark.parametrize("proposal,failed", [
    (["level"], "no_self_palindromic_word"),
    (["dogs", "dogs"], "distinct_words"),
    (["dog", "god"], "no_self_palindromic_proper_multiword_span"),
    (["regional", "artists"], "distinct_words"),
    (["zzqz"], "lexicon_words"),
])
def test_full_central_admission_preserves_shortcut_and_lexical_failures(proposal, failed):
    row = review_attachment(proposal)
    assert row["central_admission"][failed] is False
    assert failed in row["failures"]
    assert not row["promoted"] and not row["frozen_tape_mutated"]
    assert row["provenance"]["external_status"] == "not_checked"


def test_independent_token_gate_rejects_unknown_self_pal_and_context_repeat():
    assert not token_review((), "Traders")["checks"]["ascii_lowercase_token"]
    assert not token_review((), "level")["checks"]["not_self_palindromic_content"]
    assert not token_review((), "regional")["checks"]["distinct_content_in_context"]
    assert not token_review((), "zzqz")["checks"]["lexicon_word"]
    assert token_review(("local",), "artists")["accepted_into_shadow"]


@pytest.mark.parametrize("response", [
    'Traders repair some art.',
    '{"sentence":"Traders repair some art."}',
    '{"attachments":[["artists"]],"rewrite":"anything"}',
    '{"attachments":[["artists\'s"]]}',
    '{"attachments":[["local artists"]]}',
    '{"attachments":[["a","b","c","d"]]}',
    '{"attachments":["artists"]}',
    '{"attachments":[]}',
    '{"attachments":[["artists"],["artists"],["artists"],["artists"],["artists"]]}',
    'x' * 2049,
])
def test_schema_rejects_sentence_rewrites_punctuation_and_budget_overruns(certificate, response):
    before = deepcopy(certificate)
    result = review_response(response, certificate)
    assert not result["schema_accepted"]
    assert result["proposal_reviews"] == []
    assert not result["frozen_tape_mutated"]
    assert certificate == before


def test_response_ledger_keeps_every_proposal_and_exact_central_failures(certificate):
    raw = json.dumps({"attachments": [["local", "artists"], ["painters"], ["local", "artists"]]})
    result = review_response(raw, certificate)
    assert result["schema_accepted"]
    assert len(result["proposal_reviews"]) == 3  # No hidden deletion of duplicates.
    assert result["raw_response"] == raw and len(result["response_sha256"]) == 64
    assert all(not row["central_admission"]["exact_letter_palindrome"] for row in result["proposal_reviews"])
    assert all(not row["promoted"] for row in result["proposal_reviews"])
    rendered = result["proposal_reviews"][0]["rendered_diagnostic"]
    assert "local artists' red art." in rendered
    assert "".join(c for c in rendered.lower() if c.isalpha()) == "".join(FRAME.left + ("local", "artists") + FRAME.right)


@pytest.mark.parametrize("words,parity", [
    (("qwertyb", "uv", "ubytrewq"), "odd"),
    (("qwertyb", "uvv", "ubytrewq"), "even"),
])
def test_free_character_centers_only_as_nonproduction_mechanism_controls(words, parity):
    trace = fringe_trace(words)
    assert trace["termination"] == "closure" and trace["center_kind"] == parity
    assert "".join(words) == "".join(words)[::-1]


def test_online_island_guard_prunes_before_center():
    trace = fringe_trace(("abc", "dog", "god", "cba"))
    assert trace["termination"] == "online_proper_multiword_island"
    assert trace["actual_pairs"] == 3
    assert trace["remaining_span"] == [3, 9]


def test_exact_finite_accounting_and_all_reviews_preserved():
    report = finite_preflight()
    literal_paths = [FRAME.left + a + FRAME.right for a in attachments()]
    assert len(literal_paths) == report["source_attachment_derivations"] == 192
    assert len({render_sentence(w) for w in literal_paths}) == report["distinct_rendered_surfaces"] == 192
    assert len(report["reviews"]) == 192
    assert report["states_exhausted"]
    assert report["actual_pair_depth_distribution"] == {7: 192}
    assert report["exact_closures"] == 0 and report["closure_reviews"] == []
    assert report["central_and_semantic_survivors"] == 0
    assert all(row["fringe_trace"]["mismatch_pair"] == 8 and row["fringe_trace"]["left"] == "w" for row in report["reviews"])


def test_zero_preflight_blocks_queries_without_consuming_budget(certificate):
    ledger = QueryLedger()
    plan = prepare_query(certificate, ledger)
    assert not plan["permitted"]
    assert plan["reason"] == "exhaustive_allowed_attachment_domain_has_no_admissible_completion"
    assert ledger.reserved == ledger.executed == 0
    assert plan["prompt"]["operation"] == "fill_one_possessive_noun_phrase_only"
    assert not plan["transport_implemented"] and not plan["executed"]
    assert plan["residual_sha256"] == certificate["certificate_sha256"]


def test_query_reservations_have_hard_exact_budget():
    ledger = QueryLedger()
    assert [ledger.reserve() for _ in range(MAX_QUERIES)] == [1, 2]
    with pytest.raises(ValueError, match="budget exhausted"):
        ledger.reserve()
    assert ledger.reserved == 2 and ledger.executed == 0


def test_run_reports_no_model_transport_and_no_originality_claim():
    report = run()
    assert report["status"] == "finite_domain_exhausted_no_model_query"
    assert report["budget"]["maximum_queries"] == 2
    assert report["budget"]["reserved_queries"] == report["budget"]["executed_queries"] == 0
    assert report["provenance_design"]["external_searches_executed"] == 0
    assert not report["provenance_design"]["promotion_implemented"]
    assert report["promoted_candidates"] == []
