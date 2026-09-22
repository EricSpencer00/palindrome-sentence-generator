from experiments.authored_reverse_segmentation_dp_20260921 import audit, letters, run, segment

def test_dp_has_typed_slots_and_no_unjustified_exactness():
    result = run()
    assert result["method"].startswith("DP character-obligation")
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["left_templates"] == 3
    assert result["stats"]["exact_candidates"] == len(result["exact_candidates"])

def test_audit_is_independent_two_pointer_and_sha():
    a = audit("The harbor pilot marks a brass compass.")
    assert a["letters"] == len(letters("The harbor pilot marks a brass compass."))
    assert a["two_pointer_exact"] is False
    assert a["sha256_forward"] != a["sha256_reverse_obligation"]

def test_forward_control_does_not_parse_as_reverse_obligation():
    assert segment(letters("the harbor pilot marks a brass compass"), ("subject", "verb", "object")) == []

def test_targeted_expansion_records_exact_first_debt():
    result = run()
    expansion = result["targeted_domain_expansion"]
    assert expansion["new_slot"] == "subject"
    assert expansion["first_unsatisfied_obligations"][0]["first_obligation"].startswith("ss")
    assert expansion["parse_obtained"] is False
    assert "no ordinary English subject begins with 'ss'" in expansion["exact_obstruction"]
