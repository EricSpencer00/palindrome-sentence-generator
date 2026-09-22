import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/diana_argument_continuation_20260929.py"
spec = importlib.util.spec_from_file_location("diana_cont", P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_online_diana_continuation_is_exact_and_longer_than_seed():
    result = mod.run()
    row = result["rendered_candidates"][0]
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["sha_equal"]
    assert row["audit"]["letters"] == 54
    assert row["audit"]["letters"] > len(mod.letters(mod.CENTER))
    # The streamed obligation is the reverse of the already-emitted left plus
    # center tape; the final eight letters are the rendered continuation.
    assert len(row["online_obligation"]["trace"]) == len(
        mod.letters(row["construction"]["left_clause"] + row["construction"]["center_discourse"])
    )


def test_provenance_and_reader_gate_are_explicit():
    row = mod.run()["rendered_candidates"][0]
    assert row["provenance"]["fresh_typed_continuation"]
    assert not row["provenance"]["complete_sentence_pair_sweep"]
    assert row["novelty_preflight"]["finished_tape_reversal"] is False
    assert row["provenance"]["reader_gate"].startswith("closed")
