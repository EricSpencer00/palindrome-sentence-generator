from experiments.exact_tape_lexical_resegmentation_20260917 import _phrase_bonus, audit, segment


def test_phrase_bonus_prefers_attested_join_and_rejects_unattested():
    assert _phrase_bonus("same", "tales") <= 0
    assert _phrase_bonus("of", "the") >= 0


def test_heldout_tape_segmentation_keeps_exact_certificate():
    text = "Levels same tales rows ties reversed ace. Demanded net tasks asks attended name decades. Reverse its worse late mass level."
    result = segment("".join(c for c in text.lower() if c.isalpha()), limit=5)
    assert result
    rendered = " ".join(result[0]["words"])
    certificate = audit(rendered)
    assert certificate["exact"]
    assert certificate["sha256_forward"] == certificate["sha256_reverse"]
