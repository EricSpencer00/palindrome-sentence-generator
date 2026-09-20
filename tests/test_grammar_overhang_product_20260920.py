from experiments import grammar_overhang_product_20260920 as lane


def _seed_bank():
    bank = {k: [] for k in lane.TEMPLATES[0]}
    # Include every tag used by the two seed frames; the deliberately tiny
    # bank makes this a structural regression rather than a lexical sweep.
    bank.update({
        "DET": ["an", "some"],
        "NOUN": ["aide", "memos", "men"],
        "VERB": ["rips", "inspire"],
        "NUM": ["nine"],
        "NAME": ["diana"],
    })
    return bank


def test_joint_word_offsets_recover_seed_without_tape_reversal(monkeypatch):
    monkeypatch.setattr(lane, "MIN_EXACT", 37)
    result = lane.search_pair(
        ("DET", "NOUN", "VERB", "NUM", "NOUN"),
        ("DET", "NOUN", "VERB", "NAME"),
        _seed_bank(),
        node_budget=1000,
    )
    rows = [row for row in result.rows
            if row["rendered"] == "an aide rips nine memos; some men inspire diana"]
    assert rows
    assert rows[0]["slot_provenance"][0] == {
        "side": "left", "index": 0, "tag": "DET", "word": "an"
    }
    assert rows[0]["slot_provenance"][-1] == {
        "side": "right", "index": 3, "tag": "NAME", "word": "diana"
    }


def test_terminal_frame_gate_rejects_malformed_exact_tape(monkeypatch):
    monkeypatch.setattr(lane, "MIN_EXACT", 0)
    bank = _seed_bank()
    bank["NOUN"].append("memos")
    # A singular determiner plus plural noun can still be made character-exact
    # in a POS-shaped bank; the independent terminal frame check must suppress
    # it rather than calling it a grammatical result.
    result = lane.search_pair(
        ("DET", "NOUN", "VERB", "NUM", "NOUN"),
        ("DET", "NOUN", "VERB", "NAME"),
        {**bank, "DET": ["a", "some"], "NOUN": ["memos", "men"],
         "VERB": ["rips", "inspire"], "NUM": ["nine"], "NAME": ["diana"]},
        node_budget=1000,
    )
    assert all(row["rendered"] != "a memos rips nine memos; some men inspire diana"
               for row in result.rows)
