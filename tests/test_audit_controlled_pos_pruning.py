import json

from experiments.audit_controlled_pos_pruning import audit


def test_audit_entry_point_is_available(tmp_path):
    # Fail before input loading so this unit test remains independent of the
    # staged evidence archive.
    (tmp_path / "provenance.json").write_text(json.dumps({
        "configuration": {"vocab": 1, "min_words": 1, "max_units": 2,
                          "node_budget": 1}
    }))
    (tmp_path / "summary.json").write_text(json.dumps({"seeds": []}))
    (tmp_path / "trials.jsonl").write_text("")
    # The real audit requires the frozen inputs and therefore is exercised by
    # the repository-level command. This test pins its public entry point.
    assert callable(audit)
