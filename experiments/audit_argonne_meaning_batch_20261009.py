"""Independent exactness and provenance audit for the Argonne meaning batch.

No model self-score or symbolic semantic assertion admits a text as meaningful.
Both exactness checks operate on original source text. Raw observations and
credits survive normalized-tape deduplication; a blinded packet hides claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text).lower()


def raw_outside_in(text: str) -> dict:
    """Compare original ASCII letters with two pointers; no normalized copy."""
    def letter(c):
        return "a" <= c <= "z" or "A" <= c <= "Z"
    left, right, pairs = 0, len(text) - 1, 0
    count = sum(letter(c) for c in text)
    while left < right:
        while left < right and not letter(text[left]):
            left += 1
        while left < right and not letter(text[right]):
            right -= 1
        if left < right:
            if text[left].lower() != text[right].lower():
                return {"exact": False, "letters": count, "compared_pairs": pairs,
                        "first_mismatch": {"raw_left_index": left, "raw_right_index": right,
                                           "left": text[left], "right": text[right]}}
            pairs += 1
            left += 1
            right -= 1
    return {"exact": count > 0, "letters": count, "compared_pairs": pairs, "first_mismatch": None}


def audit_text(text: str) -> dict:
    raw = raw_outside_in(text)
    normalized = normalize(text)
    unsupported = sorted({c for c in text if c.isalpha() and not c.isascii()})
    reverse_exact = bool(normalized) and normalized == normalized[::-1]
    agreement = raw["exact"] == reverse_exact and raw["letters"] == len(normalized)
    exact = agreement and raw["exact"] and reverse_exact and not unsupported
    return {"exact": exact, "raw_outside_in": raw,
            "normalized_reverse_exact": reverse_exact, "checks_agree": agreement,
            "supported_ascii_letters": not unsupported, "unsupported_letters": unsupported,
            "letters": len(normalized), "word_count": len(re.findall(r"[A-Za-z]+", text)),
            "normalized": normalized, "normalized_sha256": sha(normalized.encode("ascii")),
            "text_sha256": sha(text.encode("utf-8"))}


def add_error(errors, file, pointer, code, detail):
    errors.append({"file": file, "json_pointer": pointer, "code": code, "detail": detail})


def records(payload: dict, filename: str, errors: list) -> tuple[str, list[dict], dict]:
    """Adapters only enumerate documented candidate fields, never arbitrary prose."""
    experiment = str(payload.get("experiment", payload.get("experiment_id", "")))
    rows, accounting = [], {}
    if "connected_events" in experiment or filename.startswith("connected-events"):
        lane = "connected_events"
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            add_error(errors, filename, "/candidates", "schema_error", "expected a candidates array")
            return lane, rows, accounting
        for i, record in enumerate(candidates):
            rows.append({"record": record, "pointer": f"/candidates/{i}", "generated": True, "control": False})
        accounting = {k: payload.get(k) for k in ("status", "scenes_planned", "scenes_attempted", "scene_termination_counts", "overall_termination", "exact_candidates")}
    elif "typed-overhang" in experiment or "typed_overhang" in experiment or filename.startswith("typed-overhang"):
        lane = "typed_overhang"
        jobs = payload.get("results")
        if not isinstance(jobs, list):
            add_error(errors, filename, "/results", "schema_error", "expected a results array")
            return lane, rows, accounting
        seeds = payload.get("seeds", {})
        if not isinstance(seeds, dict):
            add_error(errors, filename, "/seeds", "schema_error", "expected a seed ID-to-text object")
            seeds = {}
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                add_error(errors, filename, f"/results/{i}", "schema_error", "expected a job object")
                continue
            lineage = {"center_id": job.get("center_id"), "seed_text": seeds.get(job.get("center_id")),
                       "arm": job.get("arm"), "search_seed": job.get("seed"), "status": job.get("status")}
            for field in ("selected", "longest_exact", "whole_text_grammar_selected"):
                if job.get(field) is not None:
                    rows.append({"record": job[field], "pointer": f"/results/{i}/{field}",
                                 "generated": True, "control": False, "lineage": lineage})
            progress = job.get("progress_examples", [])
            if not isinstance(progress, list):
                add_error(errors, filename, f"/results/{i}/progress_examples", "schema_error", "expected an array")
            else:
                for j, record in enumerate(progress):
                    rows.append({"record": record, "pointer": f"/results/{i}/progress_examples/{j}",
                                 "generated": True, "control": False, "lineage": lineage})
        accounting = {"status": payload.get("status"), "planned_jobs": payload.get("planned_jobs"),
                      "recorded_jobs": len(jobs), "terminal_status_counts": dict(Counter(j.get("status", "missing") for j in jobs if isinstance(j, dict))),
                      "source_summary": payload.get("summary")}
    elif "free_typed_clauses" in experiment:
        lane = "free_typed_clauses"
        jobs = payload.get("results")
        if not isinstance(jobs, list):
            add_error(errors, filename, "/results", "schema_error", "expected free-start results array")
            return lane, rows, accounting
        job_accounting = []
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                add_error(errors, filename, f"/results/{i}", "schema_error", "expected free-start job object")
                continue
            lineage = {k: job.get(k) for k in ("min_letters", "max_letters", "seed", "status", "max_overhang", "closed_stop_factor")}
            candidates = job.get("candidates", [])
            if not isinstance(candidates, list):
                add_error(errors, filename, f"/results/{i}/candidates", "schema_error", "expected candidates array")
                candidates = []
            for j, record in enumerate(candidates):
                rows.append({"record": record, "pointer": f"/results/{i}/candidates/{j}",
                             "generated": True, "control": False, "lineage": lineage})
            if job.get("longest_exact") is not None:
                rows.append({"record": job["longest_exact"], "pointer": f"/results/{i}/longest_exact",
                             "generated": True, "control": False, "lineage": lineage})
            job_accounting.append({"json_pointer": f"/results/{i}", **lineage,
                                   "counters": job.get("counters"), "sample_policy": job.get("sample_policy"),
                                   "seconds_budget": job.get("seconds_budget"), "elapsed_seconds": job.get("elapsed_seconds"),
                                   "error": job.get("error"), "sampled_candidate_records": len(candidates)})
        accounting = {"status": payload.get("status"), "planned_jobs": payload.get("planned_jobs"),
                      "recorded_jobs": len(jobs), "config": payload.get("config"), "jobs": job_accounting,
                      "terminal_status_counts": dict(Counter(j.get("status", "missing") for j in jobs if isinstance(j, dict)))}
    elif "model_meaning_proposals" in experiment:
        lane = "model_proposal"
        attempts = payload.get("rows")
        if not isinstance(attempts, list):
            add_error(errors, filename, "/rows", "schema_error", "expected a proposal rows array")
            return lane, rows, accounting
        unparsed = []
        for i, record in enumerate(attempts):
            pointer = f"/rows/{i}"
            if not isinstance(record, dict):
                add_error(errors, filename, pointer, "schema_error", "expected a proposal attempt object")
                continue
            if record.get("parse_error") is not None:
                add_error(errors, filename, pointer, "proposal_parse_error", str(record["parse_error"]))
            if isinstance(record.get("candidate"), str):
                rows.append({"record": record, "pointer": pointer, "generated": True, "control": False,
                             "lineage": {"reference": record.get("lineage"),
                                         "case_id": record.get("case_id"), "arm": record.get("arm"),
                                         "attempt": record.get("attempt"),
                                         "proposal_stage": record.get("proposal_stage"),
                                         "feedback_parent_attempt": record.get("feedback_parent_attempt"),
                                         "model_config": payload.get("config"), "runtime": payload.get("runtime")}})
            else:
                state = "parse_failure" if record.get("parse_error") is not None else "unparsed_checkpoint"
                if record.get("candidate") is not None:
                    state = "invalid_candidate_type"
                    add_error(errors, filename, pointer, "schema_error", "candidate must be a string or null")
                unparsed.append({"json_pointer": pointer, "state": state, "source_claims": record})
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        accounting = {"status": payload.get("status"), "planned_attempts": config.get("attempts"),
                      "recorded_attempts": len(attempts), "parsed_candidate_observations": len(rows),
                      "unparsed_attempts": unparsed,
                      "unparsed_state_counts": dict(Counter(r["state"] for r in unparsed)),
                      "arm_counts": dict(Counter(r.get("arm", "missing") for r in attempts if isinstance(r, dict))),
                      "stage_counts": dict(Counter(r.get("proposal_stage", "missing") for r in attempts if isinstance(r, dict))),
                      "source_stats": payload.get("stats"), "config": payload.get("config"),
                      "runtime": payload.get("runtime"), "public_sources": payload.get("public_sources"),
                      "credited_demonstration": payload.get("credited_demonstration")}
    elif any(term in experiment.lower() or term in filename.lower() for term in ("model-proposal", "model_proposal", "gpu-proposal", "gpu_proposal", "proposal_control")):
        lane = "model_proposal"
        candidates = payload.get("candidates", payload.get("results", payload.get("attempts")))
        if not isinstance(candidates, list):
            add_error(errors, filename, "", "schema_error", "model payload requires candidates/results/attempts array")
            return lane, rows, accounting
        for i, record in enumerate(candidates):
            if not isinstance(record, dict):
                add_error(errors, filename, f"/candidates/{i}", "schema_error", "expected proposal object")
                continue
            control = bool(record.get("control")) or "control" in str(record.get("kind", "")).lower()
            rows.append({"record": record, "pointer": f"/{'candidates' if 'candidates' in payload else 'results' if 'results' in payload else 'attempts'}/{i}",
                         "generated": not control, "control": control,
                         "lineage": {"model": payload.get("model"), "model_record": record.get("model")}})
        accounting = {k: payload.get(k) for k in ("status", "planned_attempts", "attempted", "summary", "model")}
    else:
        return "unrecognized", [], {"status": payload.get("status"), "experiment": experiment}
    return lane, rows, accounting


def build_audit(raw_files: list[tuple[str, bytes]], reddit: dict, current: dict) -> dict:
    errors, sources, grouped = [], [], {}
    public_reddit = defaultdict(list)
    public_current = defaultdict(list)
    seed_tapes = set()
    for row in reddit.get("exact", []):
        public_reddit[normalize(row["text"])].append(row)
    for row in current.get("results", []):
        text = row.get("surface", row.get("text", ""))
        public_current[normalize(text)].append(row)
        if row.get("id") in ("38-control", "54-np-candidate"):
            seed_tapes.add(normalize(text))
    parsed = []
    for filename, raw in sorted(raw_files):
        source = {"file": filename, "sha256": sha(raw), "bytes": len(raw)}
        sources.append(source)
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            source["status"] = "invalid_json"
            add_error(errors, filename, "", "invalid_json", str(exc))
            continue
        if not isinstance(payload, dict):
            source["status"] = "invalid_root"
            add_error(errors, filename, "", "schema_error", "expected object root")
            continue
        if isinstance(payload.get("seeds"), dict):
            seed_tapes.update(normalize(v) for v in payload["seeds"].values() if isinstance(v, str))
        lane, rows, accounting = records(payload, filename, errors)
        source.update(lane=lane, status="enumerated" if lane != "unrecognized" else "recorded_without_candidate_adapter",
                      source_accounting=accounting, candidate_observations_enumerated=len(rows))
        parsed.append((source, rows))
    for source, rows in parsed:
        for row in rows:
            record, pointer = row["record"], row["pointer"]
            if not isinstance(record, dict):
                add_error(errors, source["file"], pointer, "schema_error", "expected a candidate object")
                continue
            text = record.get("text", record.get("surface", record.get("candidate")))
            if not isinstance(text, str):
                add_error(errors, source["file"], pointer, "missing_text", "candidate must contain a nonempty text string")
                continue
            if not text.strip():
                add_error(errors, source["file"], pointer, "empty_text", "parsed candidate text is empty")
            independent = audit_text(text)
            norm = independent["normalized"]
            # Unsupported spellings cannot merge into an ASCII candidate that
            # would otherwise pass after their letters were silently removed.
            identity = (norm, tuple(independent["unsupported_letters"]))
            if identity not in grouped:
                identifier = independent["normalized_sha256"] if not identity[1] else sha(json.dumps(identity, ensure_ascii=False).encode())
                grouped[identity] = {"id": identifier, "text": text, "audit": independent,
                                     "surfaces": [], "sources": [],
                                     "matches": {"reddit": public_reddit.get(norm, []) if independent["supported_ascii_letters"] else [],
                                                 "paper_week_results": public_current.get(norm, []) if independent["supported_ascii_letters"] else []},
                                     "inherited_seed_only": bool(norm) and norm in seed_tapes,
                                     "semantic_acceptance": "not_evaluated"}
            candidate = grouped[identity]
            if not any(s["text"] == text for s in candidate["surfaces"]):
                candidate["surfaces"].append({"text": text, "audit": independent})
            claimed_audit = record.get("audit") if isinstance(record.get("audit"), dict) else {}
            claimed_letters = record.get("letters", claimed_audit.get("letters"))
            if claimed_letters is not None and claimed_letters != independent["letters"]:
                add_error(errors, source["file"], pointer, "declared_length_mismatch", f"declared {claimed_letters}; independently counted {independent['letters']}")
            claimed_hash = record.get("normalized_sha256", claimed_audit.get("normalized_sha256"))
            if claimed_hash is not None and claimed_hash != independent["normalized_sha256"]:
                add_error(errors, source["file"], pointer, "declared_tape_hash_mismatch", "declared normalized hash disagrees with raw text")
            claims_exact = (any(record.get(k) is True for k in ("exact", "independent_exact", "independently_exact"))
                            or claimed_audit.get("exact") is True or claimed_audit.get("whole_text_exact") is True)
            if claims_exact and not independent["exact"]:
                add_error(errors, source["file"], pointer, "declared_exactness_mismatch", "source claims exact but independent checks reject")
            if not independent["checks_agree"]:
                add_error(errors, source["file"], pointer, "independent_check_disagreement", "raw-pointer and normalized-reversal checks disagree")
            lineage = row.get("lineage", {})
            candidate["sources"].append({"file": source["file"], "file_sha256": source["sha256"],
                                         "json_pointer": pointer, "lane": source["lane"],
                                         "generated_claim": row["generated"], "control": row["control"],
                                         "lineage": lineage, "source_claims": record})
    candidates = sorted(grouped.values(), key=lambda r: (-r["audit"]["letters"], r["id"]))
    for row in candidates:
        row["blind_eligible"] = (row["audit"]["exact"] and not row["inherited_seed_only"]
                                 and any(s["generated_claim"] and not s["control"] for s in row["sources"]))
        row["lookup_scope"] = "Exact normalized matches to these finite snapshots only; absence is not a global originality claim."
    eligible = [r for r in candidates if r["blind_eligible"]]
    random.Random(20261009).shuffle(eligible)
    retained = eligible[:200]
    packet = {"study": "argonne-meaning-batch-20261009", "status": "unrated_blind_machine_or_human_review_packet",
              "instructions": "Rate each complete text as written; do not infer meaning from its palindrome structure. No prior ratings are supplied.",
              "items": [{"id": f"B{i:03d}", "text": row["text"]} for i, row in enumerate(retained, 1)]}
    key = {"study": packet["study"], "selection": "fixed-seed random order; first 200 unique exact generated tapes; inherited seed-only records excluded",
           "eligible_unique_candidates": len(eligible), "items": [{"id": f"B{i:03d}", "candidate_id": row["id"]} for i, row in enumerate(retained, 1)]}
    observations = [s for row in candidates for s in row["sources"]]
    summary = {
        "status": "completed_with_source_errors" if errors else "completed_independent_audit",
        "raw_json_files": len(sources), "recognized_files": sum(s.get("lane") not in (None, "unrecognized") for s in sources),
        "candidate_observations": len(observations), "unique_tapes": len(candidates),
        "unique_exact_tapes": sum(r["audit"]["exact"] for r in candidates),
        "unique_nonexact_or_unsupported_tapes": sum(not r["audit"]["exact"] for r in candidates),
        "inherited_seed_only_tapes": sum(r["inherited_seed_only"] for r in candidates),
        "exact_generated_blind_eligible": len(eligible), "blind_packet_items": len(retained),
        "blind_packet_omitted_by_cap": max(0, len(eligible) - 200),
        "source_error_count": len(errors), "source_error_counts": dict(Counter(r["code"] for r in errors)),
        "lane_observation_counts": dict(Counter(r["lane"] for r in observations)),
        "meaning_evidence": "not_evaluated; symbolic roles, world replay, grammar flags and model scores remain source assertions",
        "limits": ["Independent exactness is necessary, not sufficient for readable or meaningful prose.",
                   "Reference seeds, prior-paper matches, source credits, source assertions and all original punctuation variants are retained.",
                   "Search attempt denominators and failure statuses remain in sources.json, including zero-candidate files.",
                   "Whole-text grammar flags are distinct from extension-only flags and never promoted to human acceptance.",
                   "The blind packet contains no model scores, symbolic claims, provenance or novelty labels; its key is separate."],
    }
    return {"summary": summary, "candidates": {"schema_version": "1.0", "candidates": candidates},
            "sources": {"sources": sources}, "errors": {"errors": errors}, "blind-packet": packet, "blind-key": key}


OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "urn:palindrome:argonne-meaning-candidates:1",
    "title": "Independent Argonne candidate audit", "type": "object", "required": ["schema_version", "candidates"],
    "properties": {"schema_version": {"const": "1.0"}, "candidates": {"type": "array", "items": {
        "type": "object", "required": ["id", "text", "audit", "surfaces", "sources", "matches", "inherited_seed_only", "blind_eligible", "semantic_acceptance"],
        "properties": {"id": {"type": "string", "pattern": "^[a-f0-9]{64}$"}, "text": {"type": "string"},
                       "audit": {"type": "object", "required": ["exact", "raw_outside_in", "normalized_reverse_exact", "checks_agree", "unsupported_letters", "letters", "normalized_sha256"]},
                       "sources": {"type": "array", "minItems": 1}, "surfaces": {"type": "array", "minItems": 1},
                       "blind_eligible": {"type": "boolean"}, "semantic_acceptance": {"const": "not_evaluated"}}}}},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reddit", type=Path, default=ROOT / "data/reddit-palindromes/2026-10-09/audit/audit.json")
    parser.add_argument("--current", type=Path, default=ROOT / "paper/week_results.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite an existing audit directory")
    if not args.results.is_dir():
        raise NotADirectoryError(args.results)
    raw = [(str(p.relative_to(args.results)), p.read_bytes()) for p in sorted(args.results.rglob("*.json"))]
    if not raw:
        raise ValueError("results directory contains no JSON files")
    lookup_bytes = {"reddit": args.reddit.read_bytes(), "paper_week_results": args.current.read_bytes()}
    result = build_audit(raw, json.loads(lookup_bytes["reddit"]), json.loads(lookup_bytes["paper_week_results"]))
    result["summary"]["lookup_sources"] = {name: {"path": str(args.reddit if name == "reddit" else args.current), "sha256": sha(data)} for name, data in lookup_bytes.items()}
    result["summary"]["script_sha256"] = sha(Path(__file__).read_bytes())
    result["schema"] = OUTPUT_SCHEMA
    args.out.mkdir(parents=True, exist_ok=False)
    for name, payload in result.items():
        (args.out / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
