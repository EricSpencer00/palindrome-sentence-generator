from experiments.chunk_residual_intersection_20260914 import ChunkResidualTrie, audit, run

def test_staggered_chunk_boundaries_can_close_exactly():
    left = {"text": "ab c", "tape": "abc", "chunks": [{"text": "ab"}, {"text": "c"}]}
    right = {"text": "c ba", "tape": "cba", "chunks": [{"text": "c"}, {"text": "ba"}]}
    trie = ChunkResidualTrie([right])
    matched, partials = trie.match(left["tape"])
    assert matched == [right]
    assert partials >= 2
    assert audit("ab c; c ba")["exact"] is True

def test_bounded_chunk_residual_run_is_reader_gated():
    result = run(500)
    assert result["config"]["character_by_character"] is True
    assert result["config"]["nonterminal_chunk_boundaries"] is True
    assert result["candidate_count"] == 0
