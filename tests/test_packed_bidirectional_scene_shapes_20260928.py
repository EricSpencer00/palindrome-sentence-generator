from experiments.packed_bidirectional_scene_shapes_20260928 import (
    SHAPES,
    audit,
    compile_shape,
    intersect,
    load_lexicon,
)


def test_independent_audit_is_exact_only_when_tapes_match():
    good = audit("A man, a plan, a canal: Panama")
    bad = audit("The pilot maps the chart")
    assert good["two_pointer_exact"] and good["sha_equal"]
    assert not bad["two_pointer_exact"] and not bad["sha_equal"]


def test_packed_product_walks_distinct_shapes_without_sentence_enumeration():
    lexicon = load_lexicon(limit=4)
    left = compile_shape(SHAPES[0], lexicon)
    right = compile_shape(SHAPES[1], lexicon)
    result = intersect(left, right, max_states=5000)
    assert result["states"] <= 5000
    assert result["transitions"] >= 0
