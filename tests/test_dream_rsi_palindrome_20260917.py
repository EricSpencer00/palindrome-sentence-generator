import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
import dream_rsi_palindrome_20260917 as dream  # noqa: E402


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
