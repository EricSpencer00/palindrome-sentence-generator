import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "experiments/scene_lattice_live_character_equations_20260917.py"
RUN = ROOT / "runs/scene-lattice-live-character-equations-20260917.json"


def test_scene_lattice_contract_and_independent_audits():
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
    data = json.loads(RUN.read_text())
    assert data["novelty_preflight"]["passed"]
    assert data["summary"] == {"candidate_count": 3, "exact_count": 0, "reader_eligible_count": 0, "all_prose_gates_pass": True}
    for row in data["rows"]:
        assert row["lexical_equation"]["choices_considered"] == 9
        assert row["lexical_equation"]["ordinary_order"] is True
        assert row["lexical_equation"]["fixed_tape"] is False
        assert row["audit"]["exact_agreement"] is True
        assert row["audit"]["independent_two_pointer"]["exact"] is False
        assert row["audit"]["independent_slice"]["sha256_equal"] is False
        assert row["shortcut_rejection"]["intact_prose"] is True
        assert not any(row["shortcut_rejection"][key] for key in ("catalogue_text", "fragment", "repeated_span", "self_palindromic_span", "gibberish", "word_order_mirror"))
        assert row["next_reader_facing_repair"]
        assert row["provenance"]["source_sentences_copied"] is False
