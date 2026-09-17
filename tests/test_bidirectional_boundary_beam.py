from experiments.bidirectional_boundary_beam import Trie, WORDS, segment, validate
from experiments.bidirectional_boundary_beam import shape_ok


def test_reverse_segmentation_is_exact_and_mechanically_verified():
    left = ["no", "evil"]
    right = ["live", "on"]
    result = validate(left, right)
    assert result["exact"] and result["reverse_tape"]
    assert result["sha256_forward"] == result["sha256_reverse"]


def test_typed_clause_lattice_keeps_complete_surface():
    right = ["not", "set"]
    assert shape_ok(right)
    result = validate(["test", "on"], right)
    assert result["exact"] and result["reverse_tape"]
