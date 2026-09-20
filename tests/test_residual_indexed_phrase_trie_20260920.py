from residual_indexed_phrase_trie_20260920 import ReversePhraseTrie, audit, run, letters

def test_reverse_trie_indexes_terminal_and_respects_residual_prefix():
    t = ReversePhraseTrie(); t.add("returns with a map", "map")
    assert t.compatible(letters("pam"))[0][1] == 1
    assert t.compatible(letters("pam a htiw snruter"))

def test_run_retains_complete_prose_and_shortcut_gates():
    result = run(); rows = result["rendered_candidates"]
    assert result["stats"]["trie_nodes"] > 1
    assert rows and all(r["complete_prose"] for r in rows)
    assert all(not r["provenance"]["copied_or_reversed_tape"] for r in rows)
    assert all(len(r["audit"]["sha256_forward"]) == 64 for r in rows)

def test_audit_detects_nonpalindrome():
    assert not audit("A careful gardener finds the path.")["exact"]
