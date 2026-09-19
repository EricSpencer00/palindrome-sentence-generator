from experiments.residual_prefix2_attachment_lattice_20260919 import audit, run

def test_independent_seed_audit():
 a=audit("An aide rips nine memos; some men inspire Diana.")
 assert a["two_pointer_exact"] and a["sha_equal"]

def test_prefix_and_attachment_index_is_finite():
 r=run()
 assert r["stats"]["indexed_states"] >= 0
 assert set(r["index"]["attachments"]) == {"", " at dawn", " under the moon"}
 assert all(x["provenance"]["rlaif_used"] is False for x in r["actual_candidates"])
