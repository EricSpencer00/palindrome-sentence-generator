"""v4 exposes evidence and diagnostics while keeping generation gated."""
import hashlib

from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_v4_health_reports_evidence_gate():
    response = client.get("/api/v4/health")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "v4"
    assert body["gate"]["generation"] == "gated"
    assert body["gate"]["reader_evidence"] is False
    assert body["best_known_letters"] == 38
    assert body["optimization"]["objective_order"][2] == "longer rendered tape"


def test_v4_evidence_contains_actual_rendered_candidate_and_independent_audit():
    response = client.get("/api/v4/evidence")
    assert response.status_code == 200
    body = response.json()
    candidate = body["best_known"]
    tape = "anaideripsninememossomemeninspirediana"
    digest = hashlib.sha256(tape.encode("ascii")).hexdigest()
    assert candidate["rendered"] == "An aide rips nine memos; some men inspire Diana."
    assert candidate["audit"]["exact"] is True
    assert candidate["audit"]["independent_two_pointer"] is True
    assert candidate["audit"]["sha256_forward"] == digest
    assert candidate["audit"]["sha256_reverse"] == digest
    assert candidate["provenance"]["source"] == "project construction run; not catalogue text"
    assert candidate["provenance"]["run_id"] == "live-clause-pair-dfs-20260920-calibration"
    assert candidate["provenance"]["search_summary"]["pilot_lengths"] == "38–60; one independently recovered 38-letter anchor, no >38 closure"
    assert "longest 50 letters" in candidate["provenance"]["search_summary"]["latest_dream_rsi_repair"]
    assert candidate["promotion_status"] == "gated_pending_blinded_readers"
    assert candidate["rlaif"]["certifies_readability"] is False
    frontier = body["repair_frontier"]
    assert frontier[0]["letters"] == 50
    assert frontier[0]["exact"] is True
    assert frontier[0]["mechanically_admitted"] is True
    assert frontier[0]["reader_status"] == "not_run"
    assert frontier[1]["letters"] == 66
    assert frontier[1]["mechanically_admitted"] is False
    assert "hidden proper palindrome span" in frontier[1]["rejection"]


def test_v4_evaluate_returns_repair_feedback_without_certifying_readability():
    response = client.post("/api/v4/evaluate", json={"text": "An aide rips nine memos; some men inspire Diana."})
    assert response.status_code == 200
    body = response.json()
    assert body["candidate"]["audit"]["exact"] is True
    assert body["rlaif"]["status"] == "diagnostic_only"
    assert body["rlaif"]["human_evidence_required"] is True
    assert body["promotion"]["reader_status"] == "not_run"
    assert "Shakespearean" in body["rlaif"]["framework"]
    assert body["rlaif"]["scores"]["dramatic_cadence_diagnostic"] > 0
    assert body["rlaif"]["repairs"]


def test_v4_method_and_best_evaluation_are_explicitly_diagnostic():
    method = client.get("/api/v4/method")
    assert method.status_code == 200
    assert method.json()["status"] == "constructive_search_in_progress"
    assert method.json()["current_best"]["rendered"] == "An aide rips nine memos; some men inspire Diana."
    assert method.json()["optimization"]["current_search"] == "slot-pair-character-search-20260919"
    assert method.json()["optimization"]["search_history"][-1] == "slot-pair-character-search-20260919"
    assert method.json()["optimization"]["generation_policy"]["mode"] == "constructive_only"
    assert method.json()["optimization"]["generation_policy"]["posthoc_repair"] is False
    assert "retire the grammar family" in method.json()["optimization"]["generation_policy"]["failure_action"]
    assert method.json()["repair_frontier"][0]["letters"] == 50
    assert method.json()["method_runs"][0]["longest_rendered_letters"] == 183
    assert method.json()["method_runs"][1]["longest_exact_letters"] == 38
    assert method.json()["method_runs"][2]["exact_candidates"] == 0
    assert method.json()["method_runs"][3]["longest_rendered_letters"] == 66
    assert method.json()["method_runs"][4]["longest_rendered_letters"] == 67
    assert method.json()["method_runs"][5]["search_nodes"] == 12588
    assert method.json()["method_runs"][6]["search_nodes"] == 4042
    assert method.json()["method_runs"][7]["longest_rendered_letters"] == 77
    assert method.json()["method_runs"][8]["frontier_controls"] == 390
    assert method.json()["method_runs"][9]["longest_rendered_letters"] == 81
    assert method.json()["method_runs"][38]["run_id"] == "synchronous-lexical-centerout-20260919"
    assert method.json()["method_runs"][38]["frontier_states"] == 0
    assert method.json()["method_runs"][38]["constructive_closures"] == 0
    assert method.json()["method_runs"][39]["run_id"] == "phrase-boundary-indexed-centerout-20260920"
    assert method.json()["method_runs"][39]["frontier_states"] == 0
    assert method.json()["method_runs"][39]["constructive_closures"] == 0
    assert method.json()["method_runs"][40]["run_id"] == "brown-char-decoder-centerout-20260922"
    assert method.json()["method_runs"][40]["exact_candidates"] == 20
    assert method.json()["method_runs"][40]["reader_worthy_candidates"] == 0
    assert method.json()["method_runs"][41]["run_id"] == "right-boundary-wfsa-decoder-20260923"
    assert method.json()["method_runs"][41]["segmented_exact_candidates"] == 0
    assert method.json()["method_runs"][42]["run_id"] == "agreement-valency-wfsa-decoder-20260924"
    assert method.json()["method_runs"][42]["exact_candidates"] == 0
    assert method.json()["method_runs"][43]["run_id"] == "broad-lexical-boundary-wfsa-20260925"
    assert method.json()["method_runs"][43]["exact_lexical_closures"] == 0
    assert method.json()["method_runs"][44]["run_id"] == "variable-boundary-lattice-decoder-20260926"
    assert method.json()["method_runs"][44]["exact_candidates"] == 0
    assert method.json()["method_runs"][45]["run_id"] == "paired-clause-lattice-20260927"
    assert method.json()["method_runs"][45]["exact_candidates"] == 0
    assert method.json()["method_runs"][46]["run_id"] == "connector-clause-debt-lattice-20260928"
    assert method.json()["method_runs"][46]["exact_candidates"] == 0
    assert method.json()["method_runs"][47]["run_id"] == "semordnilap-grammar-intersection-20260929"
    assert method.json()["method_runs"][47]["exact_candidates"] == 20
    assert method.json()["method_runs"][48]["run_id"] == "semordnilap-agreement-clause-20260930"
    assert method.json()["method_runs"][48]["longest_exact_letters"] == 48
    assert method.json()["method_runs"][49]["run_id"] == "semordnilap-poetic-clause-20261001"
    assert method.json()["method_runs"][49]["longest_exact_letters"] == 56
    assert method.json()["method_runs"][49]["reader_worthy_candidates"] == 0
    assert method.json()["method_runs"][49]["reader_candidates_pending"] == 0
    assert method.json()["method_runs"][49]["status"] == "withdrawn_exact_diagnostic"
    assert method.json()["method_runs"][50]["run_id"] == "proper-name-scene-online-20260919"
    assert method.json()["method_runs"][50]["exact_candidates"] == 0
    assert method.json()["method_runs"][50]["longest_rendered_letters"] == 77
    assert method.json()["method_runs"][51]["run_id"] == "cross-boundary-exact-lattice-20260919-v2"
    assert method.json()["method_runs"][51]["exact_candidates"] == 0
    assert method.json()["method_runs"][52]["run_id"] == "cross-boundary-morphology-dp-20260919"
    assert method.json()["method_runs"][52]["bounded_template_trials"] == 4320
    assert method.json()["method_runs"][53]["run_id"] == "cfg-center-out-intersection-20260919"
    assert method.json()["method_runs"][53]["early_pruned_derivations"] == 10800
    assert method.json()["method_runs"][53]["exact_candidates"] == 0
    assert method.json()["method_runs"][54]["run_id"] == "slot-pair-character-search-20260919"
    assert method.json()["method_runs"][54]["pruned_states"] == 7
    assert method.json()["method_runs"][54]["exact_candidates"] == 0
    assert len(method.json()["method_runs"]) == 55
    assert method.json()["method_runs"][10]["expanded_orbit_states"] == 8
    assert method.json()["method_runs"][11]["search_nodes"] == 306725
    assert method.json()["method_runs"][12]["prior_exact_collisions"] == 10
    assert method.json()["method_runs"][13]["longest_retained_letters"] == 75
    assert method.json()["method_runs"][14]["retained_controls"] == 420
    assert method.json()["method_runs"][15]["longest_exact_letters"] == 30
    assert method.json()["method_runs"][16]["letters"] == 132
    assert method.json()["method_runs"][17]["edge_attempts"] == 2421192
    assert method.json()["method_runs"][18]["outer_equation_pruned"] == 15120
    assert method.json()["method_runs"][19]["expanded_product_states"] == 34
    assert method.json()["method_runs"][19]["intact_controls"] == 4
    assert method.json()["method_runs"][20]["expanded_product_states"] == 37
    assert method.json()["method_runs"][20]["intact_controls"] == 4
    assert method.json()["method_runs"][21]["bounded_assignments"] == 4
    assert method.json()["method_runs"][22]["complete_clauses"] == 2
    assert method.json()["method_runs"][23]["states"] == 12
    assert method.json()["method_runs"][24]["trie_nodes"] == 2781
    assert method.json()["method_runs"][25]["variants"] == 16
    assert method.json()["method_runs"][26]["visited_states"] == 4608
    assert method.json()["method_runs"][27]["live_states"] == 12
    assert method.json()["method_runs"][27]["exact_candidates"] == 0
    assert method.json()["method_runs"][28]["constructive_states_tested"] == 243
    assert method.json()["method_runs"][28]["constructive_closures"] == 0
    assert method.json()["method_runs"][29]["constructive_states_tested"] == 729
    assert method.json()["method_runs"][29]["constructive_closures"] == 0
    assert method.json()["method_runs"][30]["expanded_states"] == 36134
    assert method.json()["method_runs"][30]["exact_candidates"] == 0
    assert method.json()["method_runs"][31]["scene_paths"] == 8
    assert method.json()["method_runs"][31]["exact_candidates"] == 0
    assert method.json()["method_runs"][32]["fresh_frames_tested"] == 4
    assert method.json()["method_runs"][32]["exact_candidates"] == 0
    assert method.json()["method_runs"][33]["typed_templates"] == 2
    assert method.json()["method_runs"][33]["live_states"] == 88
    assert method.json()["method_runs"][34]["template_pairs"] == 961
    assert method.json()["method_runs"][34]["live_states"] == 179205
    assert method.json()["method_runs"][35]["constructive_states_tested"] == 200000
    assert method.json()["method_runs"][35]["constructive_closures"] == 0
    assert method.json()["method_runs"][36]["visited_transitions"] == 159
    assert method.json()["method_runs"][36]["exact_candidates_over_38"] == 0
    assert method.json()["method_runs"][37]["calibration_letters"] == 38
    assert method.json()["method_runs"][37]["long_form_exact_candidates"] == 0
    assert len(method.json()["rlaif_frontier"]) == 18


def test_v4_frontier_evaluation_keeps_rlaif_diagnostic_only():
    response = client.get("/api/v4/frontier-evaluation")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "diagnostic_only"
    assert body["certifies_readability"] is False
    assert body["human_evidence_required"] is True
    assert body["rows"][0]["rendered"] == "An aide rips nine memos; some men inspire Diana."
    assert body["rows"][0]["exact"] is True
    assert body["rows"][1]["letters"] == 50
    assert body["rows"][1]["rlaif"]["scores"]["shakespearean_image"] == 0.0
    assert body["rows"][2]["exact"] is True
    assert body["rows"][2]["mechanically_admitted"] is False
    assert body["ai_feedback_run"]["model"] == "gpt-oss:20b"
    assert body["ai_feedback_run"]["search_uses_feedback"] is False
    assert body["ai_feedback_run"]["scores"][0]["intact_english"] == 2
    assert body["ai_feedback_run"]["scores"][1]["intact_english"] == 0
    assert body["reader_package"]["status"] == "blinded_package_ready_human_ratings_pending"
    assert body["reader_package"]["randomized_blinded_order"] is True
    assert body["reader_package"]["answer_key_separated"] is True
    assert body["rows"][3]["role"] == "withdrawn_exact_diagnostic"
    assert body["rows"][3]["exact"] is True
    assert body["rows"][3]["mechanically_admitted"] is False
    assert body["rows"][4]["role"] == "longest_intact_control"
    assert body["rows"][4]["exact"] is False
    assert body["rows"][5]["role"] == "fragmentary_exact_diagnostic"
    assert body["rows"][5]["exact"] is True
    assert body["rows"][5]["mechanically_admitted"] is False
    assert body["rows"][6]["role"] == "short_intact_control"
    assert body["rows"][6]["exact"] is False
    assert body["rows"][7]["role"] == "withdrawn_anchor_embedded_frontier"
    assert body["rows"][7]["letters"] == 132
    assert body["rows"][7]["exact"] is True
    assert body["rows"][7]["mechanically_admitted"] is False
    assert body["rows"][8]["role"] == "agreement_carrying_intact_control"
    assert body["rows"][8]["exact"] is False
    assert body["rows"][8]["letters"] == 43
    assert body["rows"][9]["role"] == "finite_clause_intact_control"
    assert body["rows"][9]["exact"] is False
    assert body["rows"][9]["letters"] == 48
    assert body["rows"][10]["role"] == "center_out_intact_control"
    assert body["rows"][10]["exact"] is False
    assert body["rows"][10]["letters"] == 41
    assert body["rows"][11]["role"] == "center_out_setting_frame_control"
    assert body["rows"][11]["exact"] is False
    assert body["rows"][11]["letters"] == 59
    assert body["rows"][12]["role"] == "lexical_boundary_control"
    assert body["rows"][12]["exact"] is False
    assert body["rows"][12]["letters"] == 78
    assert body["rows"][13]["role"] == "cfg_character_intersection_control"
    assert body["rows"][13]["exact"] is False
    assert body["rows"][14]["role"] == "semantic_slot_scene_control"
    assert body["rows"][14]["exact"] is False
    assert body["rows"][15]["role"] == "large_lexicon_cfg_control"
    assert body["rows"][15]["exact"] is False
    assert body["rows"][16]["role"] == "morphology_orbit_control"
    assert body["rows"][16]["exact"] is False
    assert body["rows"][17]["role"] == "char_orbit_scene_control"
    assert body["rows"][17]["exact"] is False

    evaluation = client.get("/api/v4/best-evaluation")
    assert evaluation.status_code == 200
    assert evaluation.json()["candidate"]["audit"]["exact"] is True
    assert evaluation.json()["rlaif"]["certifies_readability"] is False


def test_v4_generation_is_fail_closed():
    for method in ("get", "post"):
        response = getattr(client, method)("/api/v4/generate")
        assert response.status_code == 503
        assert "blinded human-reader evidence" in response.json()["detail"]
