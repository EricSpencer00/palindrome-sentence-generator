from experiments.paragraph_note_sentence_join_20260922 import (
    _pointer_audit,
    search,
)


def test_spots_stop_is_an_exact_small_oracle_below_length_gate():
    tagged = [
        [("spots", "NNS")],
        [("stop", "VB")],
    ]
    result = search(tagged, maximum_np_words=1, maximum_sentence_words=1,
                    minimum_letters=30)
    row = next(row for row in result["rows"]
               if row["x_object_phrase"] == "spots")
    assert row["rendered"] == (
        "No trace. Note spots. Stop. Set one carton."
    )
    assert row["independent_exact_audit"]["exact"] is True


def test_pointer_audit_rejects_near_miss():
    assert _pointer_audit("No lemon, no melon.")["exact"] is True
    assert _pointer_audit("No trace. Note marks. Stop. Set one carton.")["exact"] is False
