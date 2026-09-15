from llm_palindrome.scalable import construct_exact


VOCAB = "a i an am as at be by do go he if in is it me my no of oh on or so to up us we".split()


def test_exact_length_core_handles_both_parities():
    for target in (7, 8, 17, 18):
        row = construct_exact(target, VOCAB, max_nodes=3000, candidate_limit=64)
        assert row["status"] == "exact"
        assert row["letters"] == target
        assert row["independent_exact_validation"]


def test_unreachable_strict_lexical_target_is_reported_not_fabricated():
    row = construct_exact(39, ["level"], max_nodes=100, candidate_limit=16)
    assert row["status"] in {"no_construction", "unreachable_center_parity", "node_budget"}
    assert "text" not in row


def test_fallback_is_explicitly_non_reader_evidence():
    row = construct_exact(39, ["level"], max_nodes=100, candidate_limit=16,
                          fallback_one_letters=True)
    assert row["status"] == "exact_fallback"
    assert row["fallback"] is True
    assert row["independent_exact_validation"]

