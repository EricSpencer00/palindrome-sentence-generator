import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.shared_tape_grammar import norm, segment, main


def test_decoder_only_changes_boundaries_on_frozen_tape():
    left = "fired lots action"
    right = "no it cast older if"
    assert norm(left)[::-1] == norm(right)
    assert norm(left + " " + right) == norm(left + " " + right)[::-1]
    assert len(norm(left + " " + right)) == 30
    assert len(set(left.split() + right.split())) == 8


def test_segmentations_are_exact_tape_decodings():
    tape = norm("fired lots action")[::-1]
    rows = segment(tape, {"no", "it", "cast", "older", "if"})
    assert "no it cast older if" in rows
    assert all(norm(row) == tape for row in rows)


def test_experiment_records_provenance_and_reader_gate():
    result = main(seed=7, trials=100)
    assert result["reader_gate"].startswith("Mechanical exactness")
    assert len(result["lexicon_sha256"]) == 64
