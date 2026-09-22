import json
from pathlib import Path

from experiments.typed_phrase_graph_outer_question_answer_20260930 import audit, main


def test_outer_question_answer_is_exact_and_preserves_tape(monkeypatch):
    main()
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "runs/typed-phrase-graph-outer-question-answer-20260930.json").read_text())
    c = result["candidate"]
    assert c["audit"]["letters"] == 238
    assert c["audit"]["two_pointer_exact"]
    assert c["audit"]["validator_exact"]
    assert c["audit"]["sha_equal"]
    assert c["window_tape_reverse"]
    assert c["provenance"]["outside_tape_preserved"]
