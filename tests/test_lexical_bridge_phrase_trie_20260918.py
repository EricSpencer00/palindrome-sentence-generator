from experiments.lexical_bridge_phrase_trie_20260918 import audit, segment

def test_independent_audit_detects_exact_tape():
    row = audit("Live on; no evil.")
    assert row["exact"] and row["independent_two_pointer"]
    assert row["forward_reverse_sha256"][0] == row["forward_reverse_sha256"][1]

def test_trie_resegments_reverse_tape_without_catalogue_lookup():
    assert ["no", "evil"] in segment("noevil")
    assert ["reward"] in segment("reward")
