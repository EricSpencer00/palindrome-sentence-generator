import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
import experiments.dream_rsi_palindrome_20260917 as dream  # noqa: E402


def test_independent_audit_uses_a_letter_tape_and_two_pointer_mismatches():
    exact = dream.independent_audit("A man, a plan, a canal: Panama!")
    near = dream.independent_audit("A man, a plan, a canal: Panamx!")
    assert exact["exact"] is True
    assert exact["letters"] == 21
    assert exact["sha256_forward"] == exact["sha256_reverse"]
    assert near["exact"] is False
    assert near["mismatch_count"] == 1


def test_repeated_phrase_scaffold_is_not_admissible_without_metadata_flags():
    scaffold = {
        "rendered": "The nurse marks the calm chart beside the river beside the river beside the river.",
    }
    ordinary = {"rendered": "The nurse marks the calm chart beside the river before dusk."}
    assert dream._shortcut_free(scaffold) is False
    assert dream._shortcut_free(ordinary) is True


def test_self_palindromic_content_word_is_not_admissible_without_metadata_flags():
    assert dream._shortcut_free({"rendered": "The level rises beside the quiet garden."}) is False


def test_rare_fragmentary_exact_tape_is_not_admissible_without_metadata_flags():
    assert dream._shortcut_free({
        "rendered": "aardvark adrenocorticotropic civic — civic adrenocorticotropic aardvark."
    }) is False


def _node(text: str, parent: str | None = None) -> dream.Node:
    audit = dream.independent_audit(text)
    return dream.Node(
        node_id=hashlib.sha256(text.encode()).hexdigest()[:20],
        world="synthetic",
        source_path="runs/synthetic.json",
        source_location="$",
        parent_sha256=parent,
        forward_sha256=audit["sha256_forward"],
        action="two_region",
        rendered=text,
        letters=audit["letters"],
        exact=audit["exact"],
        mismatch_count=audit["mismatch_count"],
        mismatch_rate=audit["mismatch_rate"],
        length_ok=True,
        shortcut_free=True,
        intact_surface=True,
        reader_certified=False,
        failure_signature="edge:h>i",
    )


def test_tree_shape_counts_parent_edges_and_sibling_branches():
    root = _node("ordinary prose root")
    child = _node("ordinary prose child", root.forward_sha256)
    sibling = _node("ordinary prose sibling", root.forward_sha256)
    shape = dream.tree_shape([root, child, sibling])
    assert shape["parent_edges"] == 2
    assert shape["root_nodes"] == 1
    assert shape["branching_parents"] == 1
    assert shape["max_children"] == 2


def test_failure_repair_queue_preserves_actionable_trace_for_next_constructor():
    root = _node("ordinary prose root")
    child = _node("ordinary prose child", root.forward_sha256)
    queued = dream.failure_repair_queue([root, child])
    assert queued
    assert queued[0]["failure_signature"] == "edge:h>i"
    assert queued[0]["source_path"] == "runs/synthetic.json"
    assert "branch" in queued[0]["next_repair"]


def test_failure_repair_policy_is_distinct_and_actionable_nodes_score_higher():
    policy = next(p for p in dream.POLICIES if p["name"] == "failure_repair_first")
    node = _node("ordinary prose")
    score = dream._priority(node, policy, set(), set(), 0)[0]
    repeated = dream._priority(node, policy, {node.failure_signature}, set(), 0)[0]
    assert score > repeated


def test_malformed_mismatch_trace_is_not_promoted_as_seam_repair():
    assert dream._failure_signature({"mismatches": [[0, 9, None, None]]}, {}) == "unclassified"


def test_withheld_kernel_fixture_is_not_loaded_as_a_replay_candidate():
    worlds = dream.load_worlds()
    assert not any(node.source_path == "runs/cegar-role-product-20260917.json"
                   for node in worlds)


def test_replay_admission_count_exposes_the_rendered_row():
    exact = _node("A man, a plan, a canal: Panama!")
    report = dream.replay_world(
        [exact], next(p for p in dream.POLICIES if p["name"] == "fixed_mismatch_first"), 2
    )
    assert report["admissible_exact"] == 1
    assert report["admissible_exact_rows"][0]["rendered"] == exact.rendered
    assert report["admissible_exact_rows"][0]["source_path"] == exact.source_path


def test_word_order_only_exact_control_cannot_enter_admissible_tier():
    import json

    row = json.loads(
        (Path(__file__).parents[1] / "runs" / "dialogue-scene-semantic-palindrome-20260917.json").read_text()
    )["candidates"][2]
    assert row["shortcut_flags"]["word_order_only_symmetry"] is True
    assert dream._shortcut_free(row) is False
