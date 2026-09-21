import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('lane',Path(__file__).with_name('paragraph_abba_dialogue_trie_20260922.py'))
lane=importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)

def test_reverse_character_stream_crosses_word_boundary():
    # reverse("some m") == "memos"; this catches word-order-only symmetry.
    ok, trace=lane.online_pair('memos','some m')
    assert ok and len(trace)==5

def test_similar_phrase_does_not_fake_support():
    # reverse("some men") starts with n, so it must not be admitted as memos.
    ok, _=lane.online_pair('memos','some men')
    assert not ok
