from experiments.guide_joint_phrase_infill import parse_candidates


def test_parse_candidates_accepts_exact_clean_list():
    raw = '{"candidates":["clear path","small scene","quiet room","watch the door","tell a friend","move with care","a red light","one calm step"]}'
    parsed, error = parse_candidates(raw)
    assert error is None
    assert len(parsed) == 8


def test_parse_candidates_rejects_wrong_count_and_nonletters():
    parsed, error = parse_candidates('{"candidates":["one"]}')
    assert parsed is None
    assert error == "need_exactly_8_candidates"
    raw = '{"candidates":["a1","small scene","quiet room","watch the door","tell a friend","move with care","a red light","one calm step"]}'
    parsed, error = parse_candidates(raw)
    assert parsed is None
    assert error == "candidate_has_non_ascii_or_invalid_spacing"
