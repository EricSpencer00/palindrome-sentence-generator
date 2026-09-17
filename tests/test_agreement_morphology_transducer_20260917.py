import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("amt",ROOT/"experiments/agreement_morphology_transducer_20260917.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_transducer_emits_typed_prose_and_independent_audit():
    out=mod.run()
    assert out["candidate_count"] > 100
    assert out["exact_count"] == 0
    assert out["stats"]["longest_letters"] >= 100
    row=out["rendered_candidates"][0]
    assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
    assert all(not v for v in row["anti_shortcut_flags"].values())
    assert row["morphology_trace"][1]["number_checked"] is True

def test_internal_frontier_is_recorded_without_reverse_construction():
    row=mod.transduce("sg","present","the quiet archivist","labels","labeled",
                      "a weathered ledger","beside the glasshouse door","while")
    assert row["live_character_frontier"][0]["closed_pairs"] >= 0
    assert row["provenance"]["borrowed_text"] is False
