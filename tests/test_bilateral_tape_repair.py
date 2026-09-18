from experiments.bilateral_tape_repair import parse_choice, parse_phrases, proposal_checks, segmentations


def test_segmentations_only_emits_exact_word_breaks_of_the_fixed_tape():
    rows = segmentations("stepon", {"s", "te", "p", "on", "step", "no", "pets"},
                        min_words=2, max_words=4)
    assert "step on" in rows
    assert all("".join(row.split()) == "stepon" for row in rows)


def test_repair_choice_must_be_a_member_of_the_program_generated_lattice():
    assert parse_choice('{"chosen":"no pets"}', {"no pets"}) == ("no pets", None)
    assert parse_choice('{"chosen":"invented"}', {"no pets"}) == (None, "choice_not_in_lattice")


def test_proposal_parser_and_shape_gate_keep_invalid_material_visible():
    phrases, error = parse_phrases('{"phrases":["one", "two"]}', expected=2)
    assert error is None
    assert phrases == ["one", "two"]
    assert "letter_band" in proposal_checks("one two", {"one", "two"})
