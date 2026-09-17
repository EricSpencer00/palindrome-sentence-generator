from experiments.bidirectional_boundary_beam import Trie, WORDS, segment, validate


def test_reverse_segmentation_is_exact_and_mechanically_verified():
    left = ["no", "evil"]
    right = ["live", "on"]
    result = validate(left, right)
    assert result["exact"] and result["reverse_tape"]
    assert result["sha256_forward"] != result["sha256_reverse"]
