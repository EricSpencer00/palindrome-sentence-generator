from experiments.human_scene_edge_lattice_20260918 import audit, run, scene_bank


def test_scene_banks_are_independent_and_typed():
    left, right = scene_bank(0), scene_bank(3)
    assert len(left) == len(right) == 48
    assert left[0].__dict__ != right[0].__dict__
    assert left[0].text.endswith(".") and right[0].text.endswith(".")


def test_run_is_bounded_and_independently_audited():
    result = run(max_probes=40)
    assert result["stats"]["pair_worlds_checked"] == 40
    assert result["stats"]["edge_pruned"] == 40
    assert result["stats"]["exact"] == 0
    assert all("sha256_forward" in row["audit"] and "sha256_reverse" in row["audit"]
               for row in result["rendered_candidates_and_probes"])


def test_two_pointer_audit_has_true_control():
    assert audit("A man, a plan, a canal: Panama!")["two_pointer_exact"]
