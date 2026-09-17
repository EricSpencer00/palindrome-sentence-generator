import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "char_lm_obligation_beam_20260916",
    ROOT / "experiments/char_lm_obligation_beam_20260916.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_bounded_character_decoder_emits_intact_long_prose():
    result = MODULE.decode()
    row = result["best_candidate"]
    assert result["search"]["complete_realizations"] == 4096
    assert len(result["full_rendered_prose"]) > 100
    assert row["audit"]["letters"] > 100
    assert row["provenance"]["choices_before_rendering"] is True
    assert row["anti_shortcut"]["fixed_tape"] is False


def test_independent_pointer_and_sha_validation():
    result = MODULE.decode()
    row = result["best_candidate"]
    tape = MODULE.normalize(row["rendered"])
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    assert row["audit"]["independent_two_pointer_exact"] == (bool(tape) and i >= j)
    assert row["audit"]["sha256_forward"] == hashlib.sha256(tape.encode()).hexdigest()
    assert row["audit"]["sha256_reverse"] == hashlib.sha256(tape[::-1].encode()).hexdigest()
    assert row["audit"]["exact"] is False


def test_preflight_provenance_and_next_repair_are_persisted():
    result = MODULE.decode()
    saved = json.loads((ROOT / "runs/char-lm-obligation-beam-20260916.json").read_text())
    assert saved["novelty_preflight"]["status"] == "passed"
    assert saved["provenance"]["generator_sha256"] == result["provenance"]["generator_sha256"]
    assert saved["next_repair"]["operator"]
    assert saved["best_candidate"]["outside_in_obligation"]["first_open_obligation"]
