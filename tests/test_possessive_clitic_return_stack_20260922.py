import hashlib
import json
from pathlib import Path

from experiments.possessive_clitic_return_stack_20260922 import (
    ATTACHMENTS,
    CARRIERS,
    attachment_proper_span_mask,
    contraction_expansion_gate,
    proper_span_mask,
    run,
)
from llm_palindrome.admission import normalize_letters


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "possessive_clitic_return_stack_20260922.py"
ARTIFACT = ROOT / "runs" / "possessive-clitic-return-stack-20260922.json"


def test_all_open_pairs_satisfy_the_live_s_equation() -> None:
    assert all(pair.equation() for pair in CARRIERS)
    assert all(attachment.pair.equation() for attachment in ATTACHMENTS)


def test_apostrophe_does_not_hide_actual_proper_palindromic_spans() -> None:
    possessives = [a for a in ATTACHMENTS if a.construction.endswith("possessive")]
    assert possessives
    for attachment in possessives:
        mask = attachment_proper_span_mask(attachment)
        assert mask["passes"]
        assert not mask["two_pointer_exact"]
    # A genuinely local exact genitive is still caught after apostrophe removal.
    words = ("snoop's", "spoons", "nearby")
    mask = proper_span_mask(words)
    assert not mask["passes"]
    assert mask["exact_proper_spans"][0]["normalized_span"] == "snoopsspoons"


def test_is_and_has_expansions_are_explicit_letter_changing_rejections() -> None:
    contractions = [a for a in ATTACHMENTS if a.construction.startswith("contraction")]
    assert {a.expansion for a in contractions} == {("is",), ("has",)}
    for attachment in contractions:
        gate = contraction_expansion_gate(attachment)
        assert gate["applies"] and not gate["passes"]
        assert gate["surface_tape"] != gate["expanded_tape"]
        assert "changes letters" in gate["reason"]


def test_product_has_long_exact_structural_paths_but_no_admissible_survivor() -> None:
    payload = run()
    assert payload["stats"]["exact_structural_paths_over_44"] > 0
    assert payload["stats"]["max_structural_letters"] > 44
    assert payload["stats"]["survivors"] == 0
    assert payload["survivors"] == []
    assert "before LIFO return" in payload["obstruction"]["first_blocking_state"]
    assert payload["next_operator"]["name"] == "asynchronous cross-clause genitive dependency product"


def test_every_retained_certificate_is_exact_and_masks_are_live() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["provenance"]["host"] == "hst-bench"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == payload["provenance"]["source_sha256"]
    for row in payload["rejection_certificates"]:
        audit = row["whole_tape_audit"]
        assert audit["two_pointer_exact"]
        tape = audit["normalized_tape"]
        assert tape == tape[::-1]
        assert hashlib.sha256(tape.encode()).hexdigest() == audit["sha256_forward"]
        assert row["return_state"]["discipline"] == "strict_lifo"
        assert not row["return_state"]["pop_permitted"]
        assert row["equations"]["attachment"]
        assert row["semantic_roles"]
        assert row["valency"]
        assert row["agreement"]
        assert row["local_attachment_audit"]["normalized_span"]
        assert row["grammar_semantic_gate"]["failure"]


def test_apostrophes_do_not_hide_the_shortcut_from_normalization() -> None:
    singular = next(a for a in ATTACHMENTS if a.attachment_id == "singular-snoop-spoon")
    plural = next(a for a in ATTACHMENTS if a.attachment_id == "plural-snoops-spoon")
    assert normalize_letters(" ".join(singular.surface_words())) == "snoopsspoon"
    assert normalize_letters(" ".join(plural.surface_words())) == "snoopsspoon"
