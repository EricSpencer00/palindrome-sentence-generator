from experiments.packed_editable_center_20260928 import INCUMBENT, audit, run
from experiments.packed_seam_grammar_20260927 import norm


def test_incumbent_has_independent_audit():
    checked = audit(INCUMBENT)
    assert checked["two_pointer_exact"]
    assert checked["sha_equal"]
    assert checked["letters"] == 38


def test_editable_solver_never_labels_inherited_seed_as_novel():
    result = run()
    assert not any(norm(row["rendered"]) == norm(INCUMBENT)
                   for row in result["exact_novel_candidates"])
    assert result["solver_stats"]["cap_reached"] is False


def test_accepting_witnesses_are_independently_exact():
    result = run()
    for row in result["all_accepting_witnesses"]:
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["sha_equal"]
