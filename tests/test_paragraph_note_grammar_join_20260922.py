from experiments.paragraph_note_grammar_join_20260922 import _sentences, search


def test_generated_sentences_are_typed_and_include_stop_oracle():
    sentences = _sentences()
    stop = next(row for row in sentences["stop"] if row["words"] == ("stop",))
    assert stop["frame"] == "imperative-intransitive"


def test_spots_stop_oracle_survives_joint_grammar_search():
    result = search([[("spots", "NNS")]], maximum_np_words=1,
                    minimum_letters=30)
    row = next(row for row in result["rows"]
               if row["x_object_phrase"] == "spots")
    assert row["y_generated_sentence"] == "stop"
    assert row["independent_exact_audit"]["exact"] is True
