from experiments.slot_pair_character_search_20260919 import Slot, agreement_compatible

def test_word_feature_map_rejects_incompatible_pair():
    subject = Slot("subject", ("cat",), word_features=(("cat", "sing"),))
    verb = Slot("verb", ("run",), word_features=(("run", "plural"),))
    assert not agreement_compatible(subject, "cat", verb, "run")
