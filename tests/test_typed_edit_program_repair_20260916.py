import importlib.util, sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("tep",Path(__file__).parents[1]/"experiments/typed_edit_program_repair_20260916.py")
m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
def test_typed_program_has_complete_prose_and_dual_audit():
 p=m.run(); assert p["states_examined"]==54 and p["exact_count"]==0
 assert min(r["letters"] for r in p["best_rendered_candidates"])>=60
 assert all(r["independent_exact_agreement"] for r in p["failed_attempts"])
 assert all(r["mismatch_certificate"]["first_mismatch"] for r in p["failed_attempts"])
 assert all(r["provenance"]["complete_prose"] for r in p["failed_attempts"])
def test_heldout_repair_is_concrete_and_not_wrapper():
 row=m.run()["failed_attempts"][0]; assert row["heldout_repair"]["operator"].startswith("held-out typed edit")
 assert row["provenance"]["catalogue_imported"] is False
