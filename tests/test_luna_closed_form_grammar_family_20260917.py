import json
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("luna_closed_form_grammar_family_20260917", ROOT / "experiments/luna_closed_form_grammar_family_20260917.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
BASE, EXTENSION, OUT, run, tape = MODULE.BASE, MODULE.EXTENSION, MODULE.OUT, MODULE.run, MODULE.tape


def test_base_and_one_extension_are_fresh_complete_prose_over_38_letters():
    data = run()
    base = data["candidates"]["base"]
    extended = data["candidates"]["one_extension"]
    assert base["letters"] > 38
    assert extended["letters"] > base["letters"]
    assert len(extended["clauses"]) == len(BASE.clauses) + 1
    assert all(item["complete"] for item in extended["clauses"])
    assert data["provenance"]["seed_used_in_output"] is False


def test_growth_changes_state_without_copying_a_finished_reverse_or_repeated_unit():
    data = run()
    row = data["candidates"]["one_extension"]
    assert row["pointer_audit"]["exact"] is False
    assert row["hash_audit"]["exact"] is False
    assert row["anti_shortcut"]["finished_tape_reversal"] is False
    assert row["anti_shortcut"]["word_order_mirror"] is False
    assert row["anti_shortcut"]["repeated_clause_unit"] is False
    assert row["obligation_ledger"][-1]["remaining_pair_debt"] > 0


def test_preflight_ledger_operator_and_saved_sha_are_reproducible():
    data = run()
    saved = json.loads(OUT.read_text())
    assert data["novelty_preflight"]["status"] == "passed"
    assert data["proof_obligation_ledger"]["closed_form_proved"] is False
    assert data["family"]["arbitrary_size"] is True
    assert data["next_test"]["operator"]
    assert saved["provenance"]["generator_sha256"] == data["provenance"]["generator_sha256"]
    assert tape(BASE.render()) != tape(BASE.extend(EXTENSION).render())
