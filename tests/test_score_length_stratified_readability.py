import json

import pytest

from experiments.score_length_stratified_readability import (
    DEFAULT_OUTPUT,
    candidate_sentences,
    manifest_label,
    select_heldout_span,
    split_fileids,
    verify_reference_report,
)


def test_brown_document_split_is_deterministic_and_disjoint():
    fileids = [f"news-{index:03d}" for index in range(100)]
    train, heldout = split_fileids(fileids)
    train_again, heldout_again = split_fileids(list(reversed(fileids)))
    assert train == train_again
    assert heldout == heldout_again
    assert set(train).isdisjoint(heldout)
    assert set(train) | set(heldout) == set(fileids)
    assert 10 <= len(heldout) <= 30


def test_manifest_label_normalizes_symlinked_tmp_paths(tmp_path):
    manifest = tmp_path / "selected-results.json"
    assert manifest_label(manifest) == "selected-results.json"


def test_candidate_sentence_splitter_keeps_quotes_with_terminal_sentence():
    assert candidate_sentences('A dog waits. “A cat sees!” Then both rest.') == [
        ["a", "dog", "waits"], ["a", "cat", "sees"], ["then", "both", "rest"]
    ]


def test_reference_report_pins_corpus_and_score_rows():
    reference = json.loads(DEFAULT_OUTPUT.read_text())
    verify_reference_report(reference, reference)
    changed = json.loads(json.dumps(reference))
    changed["method"]["brown_tokenized_sentence_stream_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reference Brown/method mismatch"):
        verify_reference_report(changed, reference)


def test_heldout_controls_are_complete_contiguous_and_nonoverlapping():
    documents = {
        "a": (
            [["one", "two"], ["three"], ["four", "five"], ["six"]],
            [(0, ["one", "two"]), (1, ["three"]),
             (2, ["four", "five"]), (3, ["six"])],
        ),
        "b": (
            [["red", "blue"], ["green"], ["gold", "white"]],
            [(0, ["red", "blue"]), (1, ["green"]),
             (2, ["gold", "white"])],
        ),
    }
    occupied = {}
    first = select_heldout_span(documents, 3, occupied, "control-a")
    fileid, start, end, count, surface = first
    occupied.setdefault(fileid, []).append((start, end))
    second = select_heldout_span(documents, 3, occupied, "control-b")
    fileid2, start2, end2, count2, _ = second
    assert count == 3
    assert count2 == 3
    assert fileid != fileid2 or end <= start2 or end2 <= start
    assert surface
