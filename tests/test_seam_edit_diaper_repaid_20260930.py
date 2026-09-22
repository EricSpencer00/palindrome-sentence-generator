import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("seam_edit", Path("experiments/seam_edit_diaper_repaid_20260930.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_exact_independent_audits_and_window_only():
    assert mod.row["audit"]["two_pointer_exact"]
    assert mod.row["audit"]["sha_equal"]
    assert mod.row["audit"]["letters"] == 232
    assert mod.row["provenance"]["outside_tape_unchanged"]
    assert mod.NEW in mod.candidate
