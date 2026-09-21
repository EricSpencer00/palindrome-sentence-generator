import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "experiments" / "centered_abba_pivot_20260921.py"
spec = importlib.util.spec_from_file_location("centered_abba", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_audit_independent_agreement_and_known_seed():
    seed = "An aide rips nine memos; some men inspire Diana."
    result = mod.audit(seed)
    assert result["exact"] is True
    assert result["independent_pointer_exact"] is True
    assert result["normalizers_agree"] is True
    assert result["letters"] == 38


def test_centered_abba_run_is_complete_prose_and_not_shortcut():
    result = mod.run()
    assert result["topology"] == "A1 B1 C B2 A2"
    assert result["candidate_count"] == 32
    assert result["max_length"] > 100
    assert result["exact_admitted_count"] == 0
    for row in result["records"]:
        assert row["provenance"]["center_independently_authored"]
        assert not row["provenance"]["reused_clause"]
        assert row["audit"]["normalizers_agree"]
        assert row["text"].endswith(".")
        assert row["text"].count(";") == 4
