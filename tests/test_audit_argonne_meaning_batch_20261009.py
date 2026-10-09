import json

from experiments.audit_argonne_meaning_batch_20261009 import audit_text, build_audit, sha


def raw(name, payload):
    return name, json.dumps(payload).encode()


def test_independent_exact_checks_ignore_case_and_punctuation_but_reject_letters():
    result = audit_text("Madam? I'm Adam!")
    assert result["exact"] and result["checks_agree"]
    assert result["raw_outside_in"]["exact"] and result["normalized_reverse_exact"]
    assert not audit_text("Madam? I'm a dog!")["exact"]
    assert not audit_text("123?!")["exact"]
    unicode_result = audit_text("éMadam")
    assert unicode_result["raw_outside_in"]["exact"]
    assert not unicode_result["exact"] and unicode_result["unsupported_letters"] == ["é"]


def test_dedup_preserves_all_sources_and_reddit_credit_without_semantic_promotion():
    c = {"text": "Never odd or even.", "exact": True, "semantics": {"valid": True, "goal_reached": True}}
    files = [raw("connected-events-a.json", {"experiment": "argonne_connected_events_20261009", "candidates": [c]}),
             raw("connected-events-b.json", {"candidates": [{**c, "text": "Never odd, or even?"}]})]
    credit = {"author": "u/example", "source_url": "https://www.reddit.com/r/palindromes/example"}
    report = build_audit(files, {"exact": [{"text": c["text"], "sources": [credit]}]}, {"results": []})
    row, = report["candidates"]["candidates"]
    assert len(row["sources"]) == len(row["surfaces"]) == 2
    assert row["matches"]["reddit"][0]["sources"] == [credit]
    assert row["semantic_acceptance"] == "not_evaluated"
    assert report["summary"]["unique_exact_tapes"] == 1
    assert list(report["blind-packet"]["items"][0]) == ["id", "text"]


def test_typed_seed_only_excluded_and_extensions_grammar_not_promoted():
    seed = "Madam."
    generated = {"text": "A madam a.", "complete_clause_extensions": True, "whole_text_grammar_licensed": False}
    payload = {"seeds": {"s": seed}, "planned_jobs": 2, "results": [
        {"center_id": "s", "arm": "typed_context_debt", "status": "with_output", "selected": {"text": seed},
         "longest_exact": generated, "progress_examples": [generated], "whole_text_grammar_selected": None},
        {"center_id": "s", "arm": "typed_context_debt", "status": "deadline_no_output", "selected": None}]}
    report = build_audit([raw("typed-overhang-a.json", payload)], {"exact": []}, {"results": []})
    rows = report["candidates"]["candidates"]
    inherited = next(r for r in rows if r["inherited_seed_only"])
    assert not inherited["blind_eligible"]
    assert report["summary"]["blind_packet_items"] == 1
    assert report["sources"]["sources"][0]["source_accounting"]["recorded_jobs"] == 2
    extended = next(r for r in rows if not r["inherited_seed_only"])
    assert extended["sources"][0]["source_claims"]["whole_text_grammar_licensed"] is False
    assert extended["semantic_acceptance"] == "not_evaluated"


def test_source_failures_false_exactness_and_zero_candidate_denominators_remain():
    zero = raw("connected-events-zero.json", {"candidates": [], "scenes_planned": 21, "scenes_attempted": 3, "status": "completed_bounded_probe"})
    bad = raw("connected-events-bad.json", {"candidates": [{"text": "Cats eat fish.", "exact": True, "letters": 99}]})
    malformed = ("broken.json", b"{oops")
    report = build_audit([zero, bad, malformed], {"exact": []}, {"results": []})
    assert report["summary"]["raw_json_files"] == 3
    assert report["summary"]["blind_packet_items"] == 0
    codes = {e["code"] for e in report["errors"]["errors"]}
    assert {"invalid_json", "declared_length_mismatch", "declared_exactness_mismatch"} <= codes
    assert next(s for s in report["sources"]["sources"] if s["file"] == "broken.json")["sha256"] == sha(b"{oops")
    assert next(s for s in report["sources"]["sources"] if s["file"] == zero[0])["source_accounting"]["scenes_attempted"] == 3


def test_unsupported_spelling_does_not_merge_into_supported_tape():
    payload = {"candidates": [{"text": "Madam"}, {"text": "éMadam"}]}
    report = build_audit([raw("connected-events-a.json", payload)], {"exact": []}, {"results": []})
    assert report["summary"]["unique_tapes"] == 2
    assert report["summary"]["unique_exact_tapes"] == 1


def test_free_start_adapter_keeps_job_denominators_but_never_partial_prefix():
    candidate = {"text": "Never odd or even.", "whole_text_grammar_licensed": True}
    payload = {"experiment": "argonne_free_typed_clauses_20261009", "planned_jobs": 4,
               "results": [{"min_letters": 30, "status": "engine_returned", "candidates": [candidate],
                            "longest_exact": candidate, "deepest_prefix": {"text": "Madam."},
                            "counters": {"candidate_states": 200, "unique_admitted_exact_closures": 1}},
                           {"min_letters": 80, "status": "deadline", "candidates": [], "longest_exact": None}]}
    report = build_audit([raw("free-start.json", payload)], {"exact": []}, {"results": []})
    assert report["summary"]["candidate_observations"] == 2
    assert report["summary"]["unique_tapes"] == 1
    accounting = report["sources"]["sources"][0]["source_accounting"]
    assert accounting["recorded_jobs"] == 2 and accounting["planned_jobs"] == 4
    assert accounting["jobs"][0]["counters"]["candidate_states"] == 200
    assert report["candidates"]["candidates"][0]["semantic_acceptance"] == "not_evaluated"


def test_model_adapter_keeps_parse_failures_pending_checkpoints_and_reference_credits():
    credit = {"text": "Never odd or even.", "credit": "u/example", "url": "https://example.test/source"}
    base = {"arm": "direct", "proposal_stage": "initial_proposal", "candidate": None,
            "parse_error": None, "audit": None, "raw_response": "unfinished"}
    payload = {"experiment": "argonne_model_meaning_proposals_20261009", "status": "time_limit",
               "config": {"attempts": 9, "model": "fixture"}, "public_sources": [credit],
               "credited_demonstration": credit, "rows": [
                   {**base, "attempt": 0, "candidate": "Cats eat fish.", "raw_response": '{"text":"Cats eat fish."}',
                    "model_claims": {"plot": "Self-rated coherent"}, "audit": {"whole_text_exact": True}},
                   {**base, "attempt": 1, "parse_error": "invalid_json", "raw_response": "not JSON"},
                   {**base, "attempt": 2},
                   {**base, "attempt": 3, "candidate": "", "raw_response": '{"text":"","plot":""}'}]}
    report = build_audit([raw("gpu.json", payload)], {"exact": []}, {"results": []})
    assert report["summary"]["candidate_observations"] == 2
    assert report["summary"]["unique_exact_tapes"] == report["summary"]["blind_packet_items"] == 0
    accounting = report["sources"]["sources"][0]["source_accounting"]
    assert accounting["recorded_attempts"] == 4 and accounting["planned_attempts"] == 9
    assert accounting["unparsed_state_counts"] == {"parse_failure": 1, "unparsed_checkpoint": 1}
    assert accounting["unparsed_attempts"][0]["source_claims"]["raw_response"] == "not JSON"
    assert accounting["credited_demonstration"] == credit
    assert all(r["semantic_acceptance"] == "not_evaluated" for r in report["candidates"]["candidates"])
    assert {"proposal_parse_error", "declared_exactness_mismatch", "empty_text"} <= set(report["summary"]["source_error_counts"])


def test_cli_writes_review_artifacts_and_refuses_existing_directory(tmp_path, monkeypatch):
    import sys
    import pytest
    from experiments.audit_argonne_meaning_batch_20261009 import main
    inputs, output = tmp_path / "raw", tmp_path / "audit"
    inputs.mkdir()
    (inputs / "connected-events-a.json").write_text(json.dumps({"candidates": [{"text": "Never odd or even."}]}))
    reddit, current = tmp_path / "reddit.json", tmp_path / "current.json"
    reddit.write_text('{"exact": []}')
    current.write_text('{"results": []}')
    monkeypatch.setattr(sys, "argv", ["audit", "--results", str(inputs), "--out", str(output),
                                      "--reddit", str(reddit), "--current", str(current)])
    main()
    assert {p.name for p in output.iterdir()} == {
        "summary.json", "candidates.json", "sources.json", "errors.json", "schema.json",
        "blind-packet.json", "blind-key.json"}
    assert json.loads((output / "schema.json").read_text())["properties"]["schema_version"]["const"] == "1.0"
    assert json.loads((output / "summary.json").read_text())["lookup_sources"]["reddit"]["sha256"] == sha(reddit.read_bytes())
    with pytest.raises(FileExistsError):
        main()
