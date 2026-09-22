import experiments.scene_lattice_live_equation_20260921 as experiment


def test_scene_lattice_records_outer_residual_without_certifying_readability():
    result = experiment.run()
    assert result["stats"]["cartesian_states"] == 128
    assert result["stats"]["rendered_controls"] == 64
    assert result["stats"]["live_residual_prunes"] == 64
    assert result["stats"]["exact_candidates"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(row["provenance"]["lexically_disjoint_scene_arms"]
               for row in result["diagnostic_controls"])
    assert all("grammar control only" in row["provenance"]["reader_status"]
               for row in result["diagnostic_controls"])

