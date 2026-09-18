from experiments.guide_joint_dual_lexicalization import PROMPT, RESPONSE_PREFILL, parse_candidates


def candidate(index):
    return {"left": f"left {index}", "right": f"right {index}", "link": "scene"}


def test_parse_joint_candidates_requires_full_schema_and_count():
    raw = '{"candidates":' + str([candidate(i) for i in range(4)]).replace("'", '"') + '}'
    parsed, error = parse_candidates(raw)
    assert error is None
    assert len(parsed) == 4


def test_parser_accepts_the_model_continuation_after_its_json_prefill():
    rows = str([candidate(i) for i in range(4)]).replace("'", '"')
    continuation = rows[1:] + '}'
    parsed, error = parse_candidates(RESPONSE_PREFILL + continuation)
    assert error is None
    assert [row["left"] for row in parsed] == [f"left {i}" for i in range(4)]


def test_prompt_formats_its_literal_json_prefill():
    assert RESPONSE_PREFILL in PROMPT.format(intent="a small test scene")


def test_parse_joint_candidates_rejects_short_list():
    parsed, error = parse_candidates('{"candidates":[]}')
    assert parsed is None
    assert error == "need_exactly_4_candidates"
