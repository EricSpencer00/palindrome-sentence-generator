from experiments.typed_reverse_tape_search_20260914 import Trie, segment_reverse
import pytest


def test_trie_word_breaks_the_reversed_tape_without_character_shortcuts():
    trie = Trie(["no", "evil", "star", "rats", "live", "on"])
    zipfs = {word: 4.0 for word in ("no", "evil", "star", "rats", "live", "on")}
    paths = segment_reverse("noevilstar", trie, zipfs, top_k=8)
    assert any(path == ("no", "evil", "star") for _, path in paths)


def test_segmenter_never_emits_a_word_not_in_the_frozen_inventory():
    trie = Trie(["no", "evil", "star"])
    zipfs = {word: 4.0 for word in ("no", "evil", "star")}
    paths = segment_reverse("noevilstar", trie, zipfs, top_k=8)
    assert paths[0][1] == ("no", "evil", "star")
    assert paths[0][0] == pytest.approx(8.0)
