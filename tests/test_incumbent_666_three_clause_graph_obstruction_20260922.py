import json

from experiments.incumbent_666_three_clause_graph_obstruction_20260922 import (
    ATOM_LEFT,
    ATOM_RIGHT,
    GRAPH_LEFT,
    GRAPH_RIGHT,
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


def test_exact_27_letter_obstruction_and_insufficient_atom_are_persisted():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "three-clause-graph-no-admission-666")
    obstruction = row["primary_27_letter_obstruction"]

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is False
    assert obstruction["target_letters_per_side"] == 27
    assert obstruction["target_left"] == normalize("Nora sees Aram. Sara saw Noel live.")
    assert obstruction["target_right"] == normalize("Evil Leon was Aras. Mara sees Aron.")
    assert obstruction["target_left"] == obstruction["target_right"][::-1]
    assert obstruction["split_possibilities"]["left"]["clause_lengths"] == [12, 15]
    assert obstruction["split_possibilities"]["right"]["clause_lengths"] == [15, 12]
    assert len(obstruction["split_possibilities"]["other_split_cursors_rejected"]) == 24
    atom = obstruction["strongest_insufficient_atom"]
    assert atom["left"] == ATOM_LEFT
    assert atom["right"] == ATOM_RIGHT
    assert atom["letters_per_side"] == 12
    assert atom["total_atom_letters"] == 24
    assert atom["seam_deficit_per_side"] == 3
    assert atom["exact_reverse"] is True
    assert obstruction["obstruction"]["cursor"] == 27
    assert obstruction["obstruction"]["residual"] == ""


def test_widened_graph_closes_characters_but_is_rejected_by_full_parent_frame_gate():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "three-clause-graph-no-admission-666")
    graph = row["widened_graph_attempt"]
    child_audit = graph["candidate_independent_audit"]

    assert graph["normalized_windows"] == {"left": [156, 197], "right": [469, 510]}
    assert graph["raw_windows"] == {"left": [209, 263], "right": [644, 698]}
    assert normalize(GRAPH_LEFT) == normalize(GRAPH_RIGHT)[::-1]
    assert len(normalize(GRAPH_LEFT)) == len(normalize(GRAPH_RIGHT)) == 57
    assert graph["clause_boundary_cursors"] == {"left": [14, 29, 43, 57], "right": [14, 28, 43, 57]}
    assert graph["paired_cursors_after"] == [57, 57]
    assert graph["residuals"] == {"left": "", "right": ""}
    assert graph["left_stream"]["exact"] is True
    assert graph["right_stream"]["exact"] is True
    assert graph["gates"]["connected_three_clause_graph"] is True
    assert graph["gates"]["complete_natural_clauses"] is True
    assert graph["gates"]["neighbor_continuity_from_noel_left"] is True
    assert graph["gates"]["neighbor_continuity_from_leon_right"] is True
    assert graph["gates"]["global_clause_novelty"] is True
    assert graph["gates"]["global_frame_novelty"] is False
    assert graph["admission"]["accepted"] is False
    assert graph["admission"]["exact_character_closure"] is True
    assert graph["admission"]["exact_child_saved"] is False
    assert graph["admission"]["obstruction"]["cursor"] == 57
    assert graph["admission"]["obstruction"]["residual"] == ""
    assert graph["replaced_reused_frames"] == {"noel|stops": True, "evil leon|was": True}
    assert child_audit["normalized_letters"] == 698
    assert child_audit["two_pointer_exact"] is True
    assert row["growth_over_parent"] == 0
    assert graph["candidate_growth_over_parent"] == 32
