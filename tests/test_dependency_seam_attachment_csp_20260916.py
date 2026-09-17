import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dependency_seam_attachment_csp_20260916",
    ROOT / "experiments/dependency_seam_attachment_csp_20260916.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ATTACHMENTS, EVENTS, SUBJECTS = MODULE.ATTACHMENTS, MODULE.EVENTS, MODULE.SUBJECTS
preflight, run = MODULE.preflight, MODULE.run


def test_dependency_attachment_csp_is_new_and_independently_checked():
    result = run()
    assert result["novelty_preflight"]["passed"]
    assert result["states_examined"] == 64
    assert result["exact_count"] == 0
    assert len(result["rendered_candidates"]) == 64
    for row in result["rendered_candidates"][:8]:
        assert row["rendered"]
        assert row["independent_reparse"]
        assert row["independent_exact_agreement"]
        assert row["letters"] > 100
        assert not row["anti_shortcut_flags"]["word_order_mirror"]
        assert not row["anti_shortcut_flags"]["repeated_palindromic_unit"]
        assert row["central_admission"]["distinct_words"]
        assert row["central_admission"]["no_repeated_nontrivial_unit"]
        assert row["anti_shortcut_flags"]["complete_dependency_constituent"]
        assert row["next_repair"]


def test_inventory_is_joint_subject_event_attachment_space():
    assert [len(x) for x in (SUBJECTS, EVENTS, ATTACHMENTS)] == [4, 4, 4]
    assert preflight()["collisions"] == []
