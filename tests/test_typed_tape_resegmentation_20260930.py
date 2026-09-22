from experiments.typed_tape_resegmentation_20260930 import run


def test_fixed_tape_is_independently_exact():
    result = run()
    assert result["letters"] == 214
    assert result["exact_two_pointer"] is True
    assert result["validator"] is True
    assert result["sha256"] == result["independent_forward_reverse_sha256"]


def test_parser_consumes_typed_units_online_without_claiming_readability():
    result = run()
    assert result["typed_path"]
    assert all(item["online_match"] for item in result["typed_path"])
    assert result["provenance"]["punctuation_invented"] is False
    assert result["reader_gate"].startswith("not admitted")


def test_residual_is_explicit_and_next_repair_is_constructive():
    result = run()
    assert "residual" in result
    assert "replace one" in result["next_repair"]
