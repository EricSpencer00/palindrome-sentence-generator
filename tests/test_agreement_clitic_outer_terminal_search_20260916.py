from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def tape(text: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", text)).lower()


def test_bounded_outer_terminal_search_keeps_complete_prose_and_audits_best_state():
    run = json.loads((ROOT / "runs/agreement-clitic-outer-terminal-search-20260916.json").read_text())
    assert run["novelty_preflight"]["status"] == "passed"
    assert run["novelty_preflight"]["duplicate_sweep"] is False
    assert run["candidate_count"] == 9
    assert run["outer_obligation_matches"] >= 1
    assert run["rendered_intact_scene"].endswith(".")
    best = run["best"]
    assert best["character_obligations"][0]["matched"] is True
    assert best["audit"]["exact"] is False
    assert best["audit"]["independent_two_pointer_exact"] is False
    normalized = tape(best["text"])
    assert best["audit"]["normalized_tape"] == normalized
    assert best["audit"]["sha256_forward"] == hashlib.sha256(normalized.encode()).hexdigest()
    assert best["audit"]["sha256_reverse"] == hashlib.sha256(normalized[::-1].encode()).hexdigest()
    assert best["audit"]["sha256_forward"] != best["audit"]["sha256_reverse"]
    assert all(value is False for value in best["anti_shortcut_flags"].values())
    assert run["provenance"]["catalogue_lookup"] is False
    assert run["next_repair"]["operator"]
