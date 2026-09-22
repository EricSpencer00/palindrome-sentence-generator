from experiments.typed_phrase_graph_scene_repair_20260929 import run

def test_one_scene_edge_repair_is_exact_and_independently_audited():
    a = run()["candidate"]
    assert a["letters"] == 146
    assert a["exact_two_pointer"] and a["validator"]
    assert a["sha256"] == a["reverse_sha256"]
    assert a["novelty_preflight"]
    assert a["provenance"]["new_edge_type"] == "scene-subject-verb-object"
    assert not a["provenance"]["posthoc_repair"]
