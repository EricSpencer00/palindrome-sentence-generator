import hashlib
import re

from experiments.typed_phrase_graph_accumulate_20260930 import run


def test_accumulated_outer_edge_is_exact_and_independent():
    row = run()["candidate"]
    text = row["text"]
    tape = re.sub(r"[^a-z]", "", text.lower())
    assert len(tape) == 236
    assert tape == tape[::-1]
    assert row["audit"]["validator_exact"]
    assert hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(tape[::-1].encode()).hexdigest()
    assert row["provenance"]["reader_gate"].startswith("closed")
