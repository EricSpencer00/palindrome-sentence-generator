from experiments.residual_terminal_repair_lattice_20260919 import audit, run

def test_seed_audit_is_independent():
 a=audit("An aide rips nine memos; some men inspire Diana.")
 assert a["two_pointer_exact"] and a["sha_equal"]

def test_only_typed_state_selected_rows_are_rendered():
 r=run()
 assert r["stats"]["selected_by_state"] >= r["stats"]["rendered"]
 assert r["stats"]["rendered"] > 0
 assert all(x["provenance"]["finished_tape_reversed"] is False for x in r["actual_candidates"])
 assert all(x["residual_state"]["valency"] for x in r["actual_candidates"])
