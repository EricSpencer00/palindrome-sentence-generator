import json
from pathlib import Path

from experiments.typed_phrase_graph_seam_repair_20260930 import audit, main
from llm_palindrome.validator import normalize


def test_seam_repair_is_exact_and_keeps_outside(tmp_path):
    main()
    data = json.loads(Path("runs/typed-phrase-graph-seam-repair-20260930.json").read_text())
    cand = data["candidate"]
    assert cand["audit"]["letters"] == 228
    assert cand["audit"]["two_pointer_exact"]
    assert cand["audit"]["validator_exact"]
    assert cand["audit"]["sha_equal"]
    assert cand["provenance"]["outside_tape_preserved"]
    assert normalize(cand["rendered_window"]["left"]) == normalize(cand["rendered_window"]["right"])[::-1]
