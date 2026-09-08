from experiments.filter_sentence_pairs import candidate_pairs, parse_verdicts


def test_parse_verdicts_accepts_only_requested_boolean_ids():
    got = parse_verdicts('answer: {"a": true, "b": false, "x": 3}', {"a", "b"})
    assert got == {"a": True, "b": False}


def test_parse_verdicts_rejects_non_json_reply():
    assert parse_verdicts("I think they pass", {"a"}) == {}


def test_candidate_pairs_reads_a_sentence_plan_aggregate(tmp_path):
    path = tmp_path / "aggregate.json"
    path.write_text('{"planned_join0":{"pairs":['
                    '{"left":"items draw","right":"award items"}]}}')
    assert candidate_pairs(path) == [{"id": 0, "left": "items draw",
                                      "right": "award items",
                                      "origin": str(path)}]
