"""Finite, residual-constrained local lexical infill; no model transport.

The only writable proposal field is a 1–3-token human-possessor attachment.
The whole sentence is independently parsed and centrally checked in a shadow
review.  Neither a model response nor this module can mutate the frozen tape
or promote a candidate.  A complete finite zero suppresses model queries.

The art endpoint is reused explicitly as a live six-pair control, not claimed
as a new endpoint discovery.  The repair changes an argument attachment, not
the outer source tape, center, or word order.  No source tape is reflected.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.local_attachment_infill_validator_20260913 import (
    attachment_prefix, parse_sentence, render_sentence,
)
from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS, has_only_ordinary_short_words, is_lexical_word,
    mechanical_admission_checks, normalize_letters,
)

MIN_PAIRS = 6
MAX_QUERIES = 2
MAX_PROPOSALS_PER_QUERY = 4
MAX_TOKENS_PER_PROPOSAL = 3
MAX_RESPONSE_BYTES = 2048
MIN_LETTERS, MAX_LETTERS = 100, 240


@dataclass(frozen=True)
class Frame:
    id: str
    left: tuple[str, ...]
    right: tuple[str, ...]
    hole_role: str = "human_plural_possessor_of_artwork"


FRAME = Frame(
    "repair-owned-art-v1",
    tuple("traders who inspect damaged paintings borrowed from regional museums carefully repair small tears in".split()),
    ("red", "art"),
)
# Source production inventories: the independent validator imports none of them.
SOURCE_ORIGINS = ((), ("local",), ("regional",), ("foreign",))
SOURCE_QUALIFICATIONS = ((), ("young",), ("retired",), ("skilled",))
SOURCE_OWNERS = tuple("artists painters curators collectors restorers sculptors framers dealers donors patrons residents farmers".split())


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def attachments():
    return tuple(a + b + (c,) for a, b, c in product(SOURCE_ORIGINS, SOURCE_QUALIFICATIONS, SOURCE_OWNERS))


def file_manifest():
    names = (
        "experiments/residual_constrained_attachment_infill_20260913.py",
        "experiments/local_attachment_infill_validator_20260913.py",
        "llm_palindrome/admission.py", "data/lexicon.txt", "data/known_palindromes.json",
    )
    return {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def boundary_depths(words):
    cuts, depth = set(), 0
    for word in words:
        depth += len(word)
        cuts.add(depth)
    return cuts


def fringe_trace(words):
    """Actual outside-in traversal, free parity, with online interior-island guard."""
    words = tuple(words)
    tape = "".join(words)
    cuts = boundary_depths(words)
    for depth in range(len(tape) // 2):
        if tape[depth] != tape[-depth - 1]:
            return {"actual_pairs": depth, "termination": "outer_mismatch",
                    "mismatch_pair": depth + 1, "left": tape[depth], "right": tape[-depth - 1]}
        matched = depth + 1
        end = len(tape) - matched
        if matched in cuts and end in cuts and matched < end:
            inside_cuts = [cut for cut in cuts if matched < cut < end]
            if inside_cuts:
                return {"actual_pairs": matched, "termination": "online_proper_multiword_island",
                        "remaining_span": [matched, end], "internal_word_cuts": sorted(inside_cuts)}
    return {"actual_pairs": len(tape) // 2, "termination": "closure",
            "center_kind": "odd" if len(tape) % 2 else "even",
            "center_characters": tape[len(tape) // 2] if len(tape) % 2 else ""}


def token_review(prefix, token, frozen_words=FRAME.left + FRAME.right):
    """Check one token before it may enter an immutable shadow proposal."""
    shape = isinstance(token, str) and bool(re.fullmatch("[a-z]+", token))
    trial = tuple(prefix) + (token,) if shape else tuple(prefix)
    content_before = {w for w in (*frozen_words, *prefix) if w not in REPEATABLE_FUNCTION_WORDS}
    checks = {
        "ascii_lowercase_token": shape,
        "lexicon_word": shape and is_lexical_word(token),
        "ordinary_short_word": shape and has_only_ordinary_short_words((token,)),
        "not_self_palindromic_content": shape and (token != token[::-1] or token in REPEATABLE_FUNCTION_WORDS),
        "distinct_content_in_context": shape and (token in REPEATABLE_FUNCTION_WORDS or token not in content_before),
        "independent_attachment_prefix": shape and bool(attachment_prefix(trial)),
        "bounded_attachment": len(trial) <= MAX_TOKENS_PER_PROPOSAL,
    }
    return {"token": token, "prefix_before": list(prefix), "checks": checks,
            "accepted_into_shadow": all(checks.values())}


def review_attachment(proposal):
    """Diagnostics are not committed tape; all central failures are preserved."""
    shape = isinstance(proposal, list) and 1 <= len(proposal) <= MAX_TOKENS_PER_PROPOSAL
    token_rows, accepted = [], ()
    if shape:
        for token in proposal:
            row = token_review(accepted, token)
            token_rows.append(row)
            if row["accepted_into_shadow"]:
                accepted += (token,)
            else:
                # Later tokens are not tried against a silently deleted token.
                break
    all_tokens = shape and len(token_rows) == len(proposal) and all(r["accepted_into_shadow"] for r in token_rows)
    safe_tokens = shape and all(isinstance(t, str) and re.fullmatch("[a-z]+", t) for t in proposal)
    words = FRAME.left + tuple(proposal) + FRAME.right if safe_tokens else ()
    rendered = render_sentence(words) if words else ""
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    parses = parse_sentence(words) if words else []
    semantics = any(p["complete"] and p["semantic_relation_valid"] for p in parses)
    role = any(p["attachment"]["surface_span"] == [len(FRAME.left), len(FRAME.left) + len(proposal)]
               and p["attachment"]["patient_id"] == "artwork" for p in parses) if safe_tokens else False
    trace = fringe_trace(words) if words else None
    eligible = all_tokens and semantics and role and all(checks.values()) and trace["termination"] == "closure"
    failures = [name for name, value in checks.items() if not value]
    failures.extend(name for name, value in (("proposal_schema", shape), ("all_token_checks", all_tokens),
                                             ("independent_whole_sentence_semantics", semantics),
                                             ("structural_attachment_span", role)) if not value)
    return {"proposal": proposal, "token_reviews": token_rows, "rendered_diagnostic": rendered,
            "normalized": normalize_letters(rendered), "letters": len(normalize_letters(rendered)),
            "central_admission": checks, "independent_parses": parses, "fringe_trace": trace,
            "failures": failures, "eligible_for_external_provenance": eligible,
            "provenance": {"external_status": "not_checked", "originality_claim": False},
            "frozen_tape_mutated": False, "promoted": False}


def finite_preflight():
    """Exhaust all derivations and deduplicate surfaces independently of count."""
    rows = [review_attachment(list(a)) for a in attachments()]
    surfaces = {r["rendered_diagnostic"] for r in rows}
    exact = [r for r in rows if r["central_admission"]["exact_letter_palindrome"]]
    failures = Counter(check for row in rows for check in row["failures"])
    return {"source_attachment_derivations": len(rows), "distinct_rendered_surfaces": len(surfaces),
            "states_exhausted": True, "reviews": rows,
            "actual_pair_depth_distribution": dict(sorted(Counter(r["fringe_trace"]["actual_pairs"] for r in rows).items())),
            "failure_counts": dict(sorted(failures.items())), "exact_closures": len(exact),
            "closure_reviews": exact,
            "central_and_semantic_survivors": sum(r["eligible_for_external_provenance"] for r in rows)}


def certify_residual():
    """Bind a production gap to >=6 real frozen pairs and full semantic paths."""
    left, right = "".join(FRAME.left), "".join(FRAME.right)
    depth = 0
    while depth < min(len(left), len(right)) and left[depth] == right[-depth - 1]:
        depth += 1
    completions = attachments()
    semantic_paths = []
    for attachment in completions:
        words = FRAME.left + attachment + FRAME.right
        parses = parse_sentence(words)
        if any(p["semantic_relation_valid"] for p in parses):
            semantic_paths.append(words)
    if depth < MIN_PAIRS or not semantic_paths:
        raise ValueError("no live production residual at six actual frozen outer pairs")
    # The first unconsumed right character belongs to the declared gap.  This
    # is more than a compatible endpoint elsewhere in a disconnected grammar.
    if depth != len(right) or len(left) <= depth:
        raise ValueError("the missing attachment is not at the live frontier")
    rows = [{"attachment": list(a), "next_right_letter": a[-1][-1],
             "next_left_letter": left[depth], "next_pair_compatible": a[-1][-1] == left[depth]}
            for a in completions]
    payload = {"production": True, "frame_id": FRAME.id, "hole_role": FRAME.hole_role,
               "frozen_left_tokens": list(FRAME.left), "frozen_right_tokens": list(FRAME.right),
               "actual_pairs": depth, "matched_tape": left[:depth],
               "left_remaining_frozen_tape": left[depth:], "right_remaining_frozen_tape": "",
               "left_boundary_at_frontier": depth in boundary_depths(FRAME.left),
               "right_boundary_at_frontier": True,
               "finite_semantic_completion_paths": len(semantic_paths),
               "finite_distinct_semantic_surfaces": len(set(semantic_paths)),
               "attachment_domain_sha256": digest(completions), "file_manifest": file_manifest(),
               "coaccessible_next_pairs": rows,
               "full_semantic_completion_witness": {"words": list(semantic_paths[0]),
                                                     "parses": parse_sentence(semantic_paths[0])}}
    return {**payload, "certificate_sha256": digest(payload)}


def verify_residual(certificate):
    # A caller's production flag, depth, or source hash is never authoritative.
    return isinstance(certificate, dict) and certificate == certify_residual()


@dataclass
class QueryLedger:
    reserved: int = 0
    executed: int = 0

    def reserve(self):
        if self.reserved >= MAX_QUERIES:
            raise ValueError("two-query hard budget exhausted")
        self.reserved += 1
        return self.reserved


def query_preview(certificate):
    if not verify_residual(certificate):
        raise ValueError("untrusted, stale, toy, or insufficient-depth residual")
    prompt = {
        "operation": "fill_one_possessive_noun_phrase_only",
        "semantic_context": "A human plural owner possesses the red artwork whose tears the traders repair.",
        "frozen_before": list(FRAME.left), "frozen_after": list(FRAME.right),
        "required_role": FRAME.hole_role,
        "outer_constraint": {"real_pairs_already_matched": certificate["actual_pairs"],
                             "next_left_letter": certificate["left_remaining_frozen_tape"][0],
                             "next_right_letter_must_be_last_letter_of_attachment": True},
        "allowed_tokens": sorted({w for a in attachments() for w in a}),
        "grammar": "ORIGIN? QUALIFICATION? HUMAN_PLURAL; no more than three tokens",
        "response_schema": {"attachments": [["local", "artists"]]},
        "limits": {"alternatives": MAX_PROPOSALS_PER_QUERY, "tokens_each": MAX_TOKENS_PER_PROPOSAL,
                   "response_bytes": MAX_RESPONSE_BYTES},
        "prohibitions": "Do not rewrite context, generate a sentence, add punctuation, quote a palindrome, or claim provenance.",
    }
    return {"prompt": prompt, "prompt_sha256": digest(prompt),
            "residual_sha256": certificate["certificate_sha256"],
            "transport_implemented": False, "executed": False}


def prepare_query(certificate, ledger):
    preview = query_preview(certificate)
    preflight = finite_preflight()
    if preflight["states_exhausted"] and not preflight["central_and_semantic_survivors"]:
        return {**preview, "permitted": False, "reason": "exhaustive_allowed_attachment_domain_has_no_admissible_completion",
                "query_number": None}
    return {**preview, "permitted": True, "query_number": ledger.reserve()}


def review_response(raw_response, certificate):
    if not verify_residual(certificate):
        raise ValueError("response has no current production residual certificate")
    if not isinstance(raw_response, str) or len(raw_response.encode()) > MAX_RESPONSE_BYTES:
        return {"schema_accepted": False, "reason": "response_type_or_byte_budget", "proposal_reviews": [], "frozen_tape_mutated": False}
    result = {"response_sha256": sha256(raw_response.encode()).hexdigest(), "raw_response": raw_response,
              "residual_sha256": certificate["certificate_sha256"], "frozen_tape_mutated": False}
    try:
        value = json.loads(raw_response)
    except json.JSONDecodeError:
        return {**result, "schema_accepted": False, "reason": "invalid_json", "proposal_reviews": []}
    valid = (isinstance(value, dict) and set(value) == {"attachments"}
             and isinstance(value["attachments"], list)
             and 1 <= len(value["attachments"]) <= MAX_PROPOSALS_PER_QUERY
             and all(isinstance(a, list) and 1 <= len(a) <= MAX_TOKENS_PER_PROPOSAL
                     and all(isinstance(w, str) and re.fullmatch("[a-z]+", w) for w in a)
                     for a in value["attachments"]))
    if not valid:
        return {**result, "schema_accepted": False, "reason": "only_bounded_attachment_token_arrays_are_allowed", "proposal_reviews": []}
    return {**result, "schema_accepted": True,
            "proposal_reviews": [review_attachment(a) for a in value["attachments"]]}


def run():
    certificate = certify_residual()
    ledger = QueryLedger()
    query = prepare_query(certificate, ledger)
    preflight = finite_preflight()
    mismatch_counts = Counter((r["fringe_trace"].get("mismatch_pair"), r["fringe_trace"].get("left"),
                              r["fringe_trace"].get("right")) for r in preflight["reviews"])
    return {"experiment": "residual-constrained-local-attachment-infill-v1",
            "status": "finite_domain_exhausted_no_model_query", "residual_certificate": certificate,
            "budget": {"maximum_queries": MAX_QUERIES, "maximum_proposals_per_query": MAX_PROPOSALS_PER_QUERY,
                       "maximum_tokens_per_proposal": MAX_TOKENS_PER_PROPOSAL, "maximum_response_bytes": MAX_RESPONSE_BYTES,
                       "reserved_queries": ledger.reserved, "executed_queries": ledger.executed},
            "query_preview_and_decision": query, "finite_preflight": preflight,
            "outer_mismatches": [{"pair": pair, "left": left, "right": right, "surfaces": count}
                                 for (pair, left, right), count in sorted(mismatch_counts.items())],
            "provenance_design": {"generator": "fixed locally authored grammar; not model-generated",
                                  "chain": "source/validator/admission/lexicon/catalogue hashes -> residual hash -> prompt hash -> response hash -> full per-proposal checks",
                                  "external_requirement": "Every exact centrally admitted and independently parsed surface requires a separate exact-phrase, normalized-fragment, endpoint-family and catalogue provenance review with URLs and dates; local absence never establishes originality.",
                                  "external_searches_executed": 0, "promotion_implemented": False},
            "next_repair": "Compile an argument-order alternation that moves the possessor out of the pre-art gap; this fixed trader/w frontier proves that changing only its bounded owner vocabulary cannot repair the next mismatch.",
            "promoted_candidates": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "budget": report["budget"],
                      "source_derivations": report["finite_preflight"]["source_attachment_derivations"],
                      "surfaces": report["finite_preflight"]["distinct_rendered_surfaces"],
                      "depths": report["finite_preflight"]["actual_pair_depth_distribution"],
                      "exact_closures": report["finite_preflight"]["exact_closures"],
                      "outer_mismatches": report["outer_mismatches"]}, indent=2))
