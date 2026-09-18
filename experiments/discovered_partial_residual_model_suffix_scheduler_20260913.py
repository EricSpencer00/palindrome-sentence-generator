"""Discover partial-residual model prompts from a typed semantic inventory.

This wrapper performs no model call.  It first enumerates typed imperative
contexts and discovers their actual opening/reverse-tail residuals.  Only
discovered contexts are eligible for a frozen local-model proposal.  A frozen
response contains role alternatives, never a sentence or letter tape; the
Python evaluator independently validates roles, assembles text, audits exact
symmetry, and applies the central admission gates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 30, 160
WORD_RE = re.compile(r"[a-z]+")
RESPONSE_FIELDS = frozenset({"proposal_id", "context", "required_reverse_prefix", "locations", "purposes"})


@dataclass(frozen=True)
class LocationSpec:
    place_det: str
    place_adj: str
    place: str


@dataclass(frozen=True)
class PurposeSpec:
    purpose: str
    purpose_object: str
    purpose_type: str
    purpose_adjectives: tuple[str, ...]


@dataclass(frozen=True)
class FrameSpec:
    frame_id: str
    action: str
    object: str
    object_type: str
    object_adjectives: tuple[str, ...]
    locations: tuple[LocationSpec, ...]
    purposes: tuple[PurposeSpec, ...]


@dataclass(frozen=True)
class ContextSpec:
    frame_id: str
    action: str
    object: str
    object_det: str
    object_adj: str


@dataclass(frozen=True)
class ProposalLocation:
    place_det: str
    place_adj: str
    place: str


@dataclass(frozen=True)
class ProposalPurpose:
    purpose: str
    purpose_det: str
    purpose_adj: str
    purpose_object: str


@dataclass(frozen=True)
class FrozenResponse:
    raw_response: str
    response_sha256: str
    payload: dict[str, Any]


class ProposalSchemaError(ValueError):
    pass


# Broad authoring inventory.  No endpoint pair is selected here: discovery
# computes residuals over every typed action/object/attachment combination.
AUTHORING_INVENTORY: tuple[FrameSpec, ...] = (
    FrameSpec("canvas-display", "paint", "canvas", "artifact", ("bright", "clean", "useful"),
              (LocationSpec("a", "quiet", "studio"), LocationSpec("a", "public", "gallery")),
              (PurposeSpec("display", "map", "artifact", ("clear", "useful")),
               PurposeSpec("frame", "map", "artifact", ("simple", "clear")),
               PurposeSpec("store", "canvas", "artifact", ("clean", "useful")))),
    FrameSpec("book-paper", "read", "book", "artifact", ("new", "useful", "thick"),
              (LocationSpec("a", "quiet", "office"), LocationSpec("a", "small", "library")),
              (PurposeSpec("file", "paper", "artifact", ("final", "clear")),
               PurposeSpec("scan", "paper", "artifact", ("clean", "useful")),
               PurposeSpec("cite", "paper", "artifact", ("final", "clear")))),
    FrameSpec("model-diagram", "make", "model", "artifact", ("useful", "simple", "detailed"),
              (LocationSpec("a", "remote", "workshop"), LocationSpec("a", "public", "studio")),
              (PurposeSpec("sketch", "diagram", "artifact", ("clear", "simple")),
               PurposeSpec("show", "model", "artifact", ("useful", "new")),
               PurposeSpec("test", "model", "artifact", ("simple", "detailed")))),
    FrameSpec("project-help", "plan", "project", "artifact", ("new", "large", "useful"),
              (LocationSpec("a", "quiet", "office"), LocationSpec("a", "public", "studio")),
              (PurposeSpec("seek", "help", "support", ("local", "useful")),
               PurposeSpec("fund", "project", "artifact", ("new", "large")),
               PurposeSpec("start", "project", "artifact", ("new", "useful")))),
)

# Independent parser inventory: it is declared separately and does not read
# FrameSpec, proposal metadata, or the response's semantic labels.
PARSE_FRAMES = {
    ("paint", "canvas", "display"): {"map": "artifact"},
    ("paint", "canvas", "frame"): {"map": "artifact"},
    ("paint", "canvas", "store"): {"canvas": "artifact"},
    ("read", "book", "file"): {"paper": "artifact"},
    ("read", "book", "scan"): {"paper": "artifact"},
    ("read", "book", "cite"): {"paper": "artifact"},
    ("make", "model", "sketch"): {"diagram": "artifact"},
    ("make", "model", "show"): {"model": "artifact"},
    ("make", "model", "test"): {"model": "artifact"},
    ("plan", "project", "seek"): {"help": "support"},
    ("plan", "project", "fund"): {"project": "artifact"},
    ("plan", "project", "start"): {"project": "artifact"},
}
PARSE_ACTION_OBJECTS = {"paint": {"canvas"}, "read": {"book"}, "make": {"model"}, "plan": {"project"}}
PARSE_OBJECT_ADJECTIVES = {"bright", "clean", "useful", "new", "thick", "simple", "detailed", "large"}
PARSE_PLACE_ADJECTIVES = {"quiet", "public", "small", "remote"}
PARSE_PLACES = {"studio", "gallery", "office", "library", "workshop"}
PARSE_PURPOSE_ADJECTIVES = {"clear", "useful", "simple", "clean", "final", "new", "detailed", "local", "large"}
PARSE_OBJECT_TYPES = {"canvas": "artifact", "map": "artifact", "paper": "artifact", "book": "artifact",
                      "model": "artifact", "diagram": "artifact", "project": "artifact", "help": "support"}


def article(adjective: str) -> str:
    return "an" if adjective[:1].lower() in "aeiou" else "a"


def _surface_words(context: ContextSpec, location: ProposalLocation, purpose: ProposalPurpose) -> tuple[str, ...]:
    return (context.action, context.object_det, context.object_adj, context.object, "in",
            location.place_det, location.place_adj, location.place, "to", purpose.purpose,
            purpose.purpose_det, purpose.purpose_adj, purpose.purpose_object)


def _render(words: Iterable[str]) -> str:
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def _reverse_tail(context: ContextSpec, purpose: ProposalPurpose) -> str:
    tail = ("to", purpose.purpose, purpose.purpose_det, purpose.purpose_adj, purpose.purpose_object)
    return normalize_letters("".join(tail))[::-1]


def discover_partial_contexts(inventory: tuple[FrameSpec, ...] = AUTHORING_INVENTORY,
                              minimum_pairs: int = 2) -> tuple[dict[str, Any], ...]:
    """Enumerate residuals; no action/object or endpoint is preselected."""
    discoveries = []
    for frame in inventory:
        for object_adj in frame.object_adjectives:
            context = ContextSpec(frame.frame_id, frame.action, frame.object, article(object_adj), object_adj)
            opening = normalize_letters("".join((context.action, context.object_det, context.object_adj, context.object)))
            for purpose in frame.purposes:
                for purpose_adj in purpose.purpose_adjectives:
                    purpose_det = article(purpose_adj)
                    reverse_tail = normalize_letters("".join(("to", purpose.purpose, purpose_det,
                                                              purpose_adj, purpose.purpose_object)))[::-1]
                    depth = 0
                    for left, right in zip(opening, reverse_tail):
                        if left != right:
                            break
                        depth += 1
                    if depth < minimum_pairs:
                        continue
                    discoveries.append({
                        "frame_id": frame.frame_id, "context": asdict(context),
                        "required_reverse_prefix": opening[:depth], "matched_pairs": depth,
                        "opening_letters": opening, "reverse_tail_letters": reverse_tail,
                        "discovered_from_attachment": {"purpose": purpose.purpose,
                                                       "purpose_object": purpose.purpose_object,
                                                       "purpose_adj": purpose_adj},
                        "semantic_frame": [frame.action, frame.object, purpose.purpose, purpose.purpose_object],
                    })
    return tuple(discoveries)


def group_discoveries(discoveries: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in discoveries:
        context = row["context"]
        key = json.dumps({"frame_id": row["frame_id"], "action": context["action"],
                          "object": context["object"], "object_adj": context["object_adj"],
                          "required_reverse_prefix": row["required_reverse_prefix"]}, sort_keys=True)
        grouped.setdefault(key, []).append(row)
    return grouped


def proposal_prompt(discovery: dict[str, Any]) -> str:
    context = discovery["context"]
    return ("Return JSON typed alternatives for the missing location and purpose attachments only. "
            f"Context action={context['action']}, object={context['object']}, modifier={context['object_adj']}; "
            f"required reverse prefix={discovery['required_reverse_prefix']}. "
            "Return no complete sentence, surface, character tape, or mirrored text; Python assembles and verifies it.")


def freeze_model_response(raw_response: str) -> FrozenResponse:
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ProposalSchemaError("model response must be nonempty JSON text")
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ProposalSchemaError("model response is not JSON") from exc
    if not isinstance(payload, dict):
        raise ProposalSchemaError("model response must be an object")
    unknown = sorted(set(payload) - RESPONSE_FIELDS)
    missing = sorted(RESPONSE_FIELDS - set(payload))
    if unknown or missing:
        raise ProposalSchemaError(f"proposal schema mismatch unknown={unknown} missing={missing}")
    return FrozenResponse(raw_response, sha256(raw_response.encode()).hexdigest(), payload)


def _parse_proposal(frozen: FrozenResponse) -> tuple[ContextSpec, tuple[ProposalLocation, ...], tuple[ProposalPurpose, ...]]:
    payload = frozen.payload
    c = payload["context"]
    if not isinstance(c, dict) or set(c) != {"frame_id", "action", "object", "object_det", "object_adj"}:
        raise ProposalSchemaError("proposal context must identify the discovered typed context")
    if not all(isinstance(c[key], str) for key in c):
        raise ProposalSchemaError("proposal context values must be strings")
    locations, purposes = [], []
    for row in payload["locations"]:
        if not isinstance(row, dict) or set(row) != {"place_det", "place_adj", "place"} or not all(isinstance(row[k], str) for k in row):
            raise ProposalSchemaError("invalid typed location option")
        locations.append(ProposalLocation(row["place_det"], row["place_adj"], row["place"]))
    for row in payload["purposes"]:
        if not isinstance(row, dict) or set(row) != {"purpose", "purpose_det", "purpose_adj", "purpose_object"} or not all(isinstance(row[k], str) for k in row):
            raise ProposalSchemaError("invalid typed purpose option")
        purposes.append(ProposalPurpose(row["purpose"], row["purpose_det"], row["purpose_adj"], row["purpose_object"]))
    return ContextSpec(c["frame_id"], c["action"], c["object"], c["object_det"], c["object_adj"]), tuple(locations), tuple(purposes)


def independent_parse(text: str) -> dict[str, Any]:
    tokens = tuple(WORD_RE.findall(text.lower()))
    if len(tokens) != 13:
        return {"ok": False, "reason": "wrong_role_arity", "tokens": list(tokens)}
    action, od, oa, obj, prep, pd, pa, place, to, purpose, pod, poa, pobj = tokens
    article_ok = od == article(oa) and pd == article(pa) and pod == article(poa)
    frame = PARSE_FRAMES.get((action, obj, purpose), {})
    role_ok = (action in PARSE_ACTION_OBJECTS and obj in PARSE_ACTION_OBJECTS[action]
               and oa in PARSE_OBJECT_ADJECTIVES and prep == "in" and pa in PARSE_PLACE_ADJECTIVES
               and place in PARSE_PLACES and to == "to" and poa in PARSE_PURPOSE_ADJECTIVES
               and frame.get(pobj) == PARSE_OBJECT_TYPES.get(pobj))
    return {"ok": article_ok and role_ok, "agreement_ok": article_ok, "valency_ok": role_ok,
            "tokens": list(tokens), "independent_inventory": True}


def outside_in_ledger(text: str) -> dict[str, Any]:
    tape = normalize_letters(text)
    events = []
    for pair in range(len(tape) // 2):
        right = len(tape) - pair - 1
        event = {"pair": pair + 1, "left": tape[pair], "right": tape[right], "equal": tape[pair] == tape[right]}
        events.append(event)
        if not event["equal"]:
            return {"exact": False, "letters": len(tape), "events": events, "first_mismatch": event,
                    "normalized_sha256": sha256(tape.encode()).hexdigest()}
    return {"exact": bool(tape), "letters": len(tape), "events": events, "first_mismatch": None,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def _validate_and_evaluate(frozen: FrozenResponse, discovery: dict[str, Any], frame: FrameSpec) -> dict[str, Any]:
    base = {"record_kind": "discovered_partial_residual_frozen_proposal", "frozen_response": True,
            "response_sha256": frozen.response_sha256, "model_output_is_proposal_only": True,
            "proposal_id": frozen.payload.get("proposal_id"), "rendered": [], "records": [], "exact_candidates": []}
    try:
        context, locations, purposes = _parse_proposal(frozen)
    except ProposalSchemaError as exc:
        base.update({"preconstruction": {"accepted": False, "reasons": [str(exc)]},
                     "construction_started": False})
        return base
    expected = ContextSpec(**discovery["context"])
    reasons = []
    if context != expected:
        reasons.append("context_not_discovered_or_does_not_match")
    if not isinstance(frozen.payload["required_reverse_prefix"], str) or normalize_letters(frozen.payload["required_reverse_prefix"]) != normalize_letters(discovery["required_reverse_prefix"]):
        reasons.append("residual_not_discovered_for_context")
    if len(locations) < 2 or len(purposes) < 2:
        reasons.append("multiple_alternatives_required")
    valid_locations = {(row.place_det, row.place_adj, row.place) for row in frame.locations}
    valid_purposes = {(p.purpose, p.purpose_object, adj) for p in frame.purposes for adj in p.purpose_adjectives}
    for location in locations:
        if (location.place_det, location.place_adj, location.place) not in valid_locations or location.place_det != article(location.place_adj):
            reasons.append("location_not_in_typed_inventory")
    for purpose in purposes:
        if (purpose.purpose, purpose.purpose_object, purpose.purpose_adj) not in valid_purposes or purpose.purpose_det != article(purpose.purpose_adj):
            reasons.append("purpose_not_in_typed_inventory")
    base["preconstruction"] = {"accepted": not reasons, "reasons": sorted(set(reasons)),
                               "discovered_context": True, "location_count": len(locations), "purpose_count": len(purposes)}
    if reasons:
        base["construction_started"] = False
        return base
    required = normalize_letters(discovery["required_reverse_prefix"])
    for location in locations:
        for purpose in purposes:
            reverse_tail = normalize_letters("".join(("to", purpose.purpose, purpose.purpose_det,
                                                       purpose.purpose_adj, purpose.purpose_object)))[::-1]
            if not reverse_tail.startswith(required):
                base["records"].append({"location": asdict(location), "purpose": asdict(purpose),
                                        "reverse_tail_prefix": reverse_tail,
                                        "rejection": "partial_residual_not_met", "construction_started": False})
                continue
            text = _render(_surface_words(context, location, purpose))
            ledger = outside_in_ledger(text)
            parsed = independent_parse(text)
            central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            codes = [key for key, value in central.items() if not value]
            if not ledger["exact"]: codes.append("outside_in_ledger_mismatch")
            if not parsed["ok"]: codes.append("independent_parse_failed")
            row = {"location": asdict(location), "purpose": asdict(purpose), "rendered": text,
                   "reverse_tail_prefix": reverse_tail, "outside_in_ledger": ledger,
                   "independent_parse": parsed, "central_admission": central,
                   "mechanically_admitted": not codes, "rejection_codes": codes,
                   "construction_started": True,
                   "reader_status": "human-unreviewed; programmatic checks do not certify readability"}
            base["records"].append(row); base["rendered"].append(text)
            if row["mechanically_admitted"] and ledger["exact"]:
                base["exact_candidates"].append(row)
    base["construction_started"] = bool(base["records"])
    return base


def run(raw_responses_by_key: Mapping[str, tuple[str, ...]] | None = None,
        inventory: tuple[FrameSpec, ...] = AUTHORING_INVENTORY, minimum_pairs: int = 2) -> dict[str, Any]:
    discoveries = discover_partial_contexts(inventory, minimum_pairs)
    grouped = group_discoveries(discoveries)
    responses = raw_responses_by_key or {}
    frame_by_id = {frame.frame_id: frame for frame in inventory}
    frozen_records, evaluations = [], []
    for key, rows in grouped.items():
        # A response may be submitted only to a context discovered by this run.
        for raw in responses.get(key, ()):
            try:
                frozen = freeze_model_response(raw)
            except ProposalSchemaError as exc:
                digest = sha256(raw.encode()).hexdigest()
                frozen_records.append({"response_sha256": digest, "raw_response": raw, "parse_error": str(exc)})
                evaluations.append({"record_kind": "frozen_response_rejection", "frozen_response": True,
                                    "model_output_is_proposal_only": True, "construction_started": False,
                                    "error": str(exc), "raw_response_sha256": digest})
                continue
            frozen_records.append({"response_sha256": frozen.response_sha256, "raw_response": frozen.raw_response})
            evaluations.append(_validate_and_evaluate(frozen, rows[0], frame_by_id[rows[0]["frame_id"]]))
    exact = [row for result in evaluations for row in result.get("exact_candidates", [])]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {"status": "discovered_partial_residual_model_suffix_scheduler_no_model_call",
            "config": {"model_calls_enabled": False, "broad_inventory_discovery": True,
                       "prompts_only_for_discovered_contexts": True, "proposal_prompt_is_partial_only": True,
                       "frozen_external_responses": True, "python_owns_surface_assembly": True,
                       "python_owns_outside_in_ledger": True, "independent_complete_reparse": True,
                       "central_admission_before_reader_study": True, "corpus_generation": False,
                       "minimum_discovered_pairs": minimum_pairs},
            "discovery_count": len(discoveries), "prompt_group_count": len(grouped),
            "discovered_contexts": discoveries,
            "prompt_groups": {key: {"count": len(rows), "prompt": proposal_prompt(rows[0])} for key, rows in grouped.items()},
            "frozen_responses": frozen_records, "response_results": evaluations,
            "exact_candidates": exact, "admitted_candidates": admitted,
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "external_model_calls": 0, "external_output_role": "typed proposal only",
                           "surface_source": "Python assembly after discovered residual and independent parse"},
            "reader_facing_next_test": "Only an admitted exact surface may enter randomized blinded intact-prose versus shuffled-control reading; programmatic checks do not certify readability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--responses", type=Path, help="JSON object mapping discovered prompt keys to response arrays")
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    supplied = json.loads(args.responses.read_text()) if args.responses else None
    result = run(supplied)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "discoveries": result["discovery_count"],
                      "groups": result["prompt_group_count"], "exact": len(result["exact_candidates"]),
                      "admitted": len(result["admitted_candidates"])}, indent=2))


if __name__ == "__main__": main()
