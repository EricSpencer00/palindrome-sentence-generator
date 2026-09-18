"""Model-proposed semantic suffixes constrained by a live character residual.

This is an interface experiment, not a model call.  A local model may return
only typed location and purpose alternatives for an already selected
action/object context plus the required reverse prefix.  The response is
frozen verbatim and hashed; Python rejects unknown fields, validates the
semantic roles independently, assembles every surface, and owns the exact
outside-in ledger.  No model-provided sentence, letter tape, or mirrored text
can enter construction.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 30, 160
WORD_RE = re.compile(r"[a-z]+")
RESPONSE_FIELDS = frozenset({"proposal_id", "context", "required_reverse_prefix", "locations", "purposes"})


@dataclass(frozen=True)
class ActionObjectContext:
    action: str
    object: str
    object_det: str = "a"
    object_adj: str = "clear"


@dataclass(frozen=True)
class Attachment:
    purpose: str
    purpose_det: str
    purpose_adj: str
    purpose_object: str


@dataclass(frozen=True)
class Location:
    place_det: str
    place_adj: str
    place: str


@dataclass(frozen=True)
class FrozenModelResponse:
    raw_response: str
    response_sha256: str
    payload: dict[str, Any]


class ProposalSchemaError(ValueError):
    pass


# Authoring inventory: the model can select among these ordinary typed roles,
# but cannot introduce an untyped word or a full surface.  The options are
# intentionally wider than one attachment so the residual performs selection.
AUTHOR_ACTION_OBJECTS = {"draw": {"map"}}
AUTHOR_OBJECT_ADJECTIVES = {"map": {"clear", "detailed", "simple"}}
AUTHOR_LOCATIONS = {
    "studio": {"quiet", "public"},
    "gallery": {"quiet", "public"},
    "office": {"small", "remote"},
}
AUTHOR_PURPOSES = {
    ("draw", "map"): {
        ("protect", "ward"): "person",
        ("label", "map"): "artifact",
        ("display", "map"): "artifact",
    }
}
AUTHOR_PURPOSE_ADJECTIVES = {
    "ward": {"kind", "young", "local"},
    "map": {"clear", "simple", "useful"},
}
AUTHOR_PURPOSE_VERB_TYPES = {"protect": "person", "label": "artifact", "display": "artifact"}

# Deliberately separate parse inventory: no response metadata or authoring
# dataclass is consulted by the completed-surface parser.
PARSE_ACTION_OBJECTS = {"draw": {"map"}}
PARSE_OBJECT_TYPES = {"map": "artifact", "ward": "person"}
PARSE_OBJECT_ADJECTIVES = {"clear", "detailed", "simple"}
PARSE_LOCATION_ADJECTIVES = {"quiet", "public", "small", "remote"}
PARSE_LOCATION_NOUNS = {"studio", "gallery", "office"}
PARSE_PURPOSE_FRAMES = {
    ("draw", "map", "protect"): {"ward": "person"},
    ("draw", "map", "label"): {"map": "artifact"},
    ("draw", "map", "display"): {"map": "artifact"},
}
PARSE_PURPOSE_ADJECTIVES = {"kind", "young", "local", "clear", "simple", "useful"}


def article(adjective: str) -> str:
    return "an" if adjective[:1].lower() in "aeiou" else "a"


def proposal_prompt(context: ActionObjectContext, required_reverse_prefix: str) -> str:
    """Build a partial-residual request; it never asks for a palindrome/tape."""
    return (
        "Return JSON only with typed alternatives for the missing location and "
        "purpose attachments. Context action=" + context.action +
        ", object=" + context.object +
        ", required reverse prefix=" + required_reverse_prefix + ". "
        "Each alternative must be an ordinary role value with a semantic type. "
        "Do not return a complete sentence, a candidate surface, or a character string; "
        "Python will assemble and verify the text."
    )


def freeze_model_response(raw_response: str) -> FrozenModelResponse:
    """Freeze external output before any semantic or character processing."""
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ProposalSchemaError("model response must be nonempty JSON text")
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ProposalSchemaError("model response is not JSON") from exc
    if not isinstance(payload, dict):
        raise ProposalSchemaError("model response must be a JSON object")
    unknown = sorted(set(payload) - RESPONSE_FIELDS)
    missing = sorted(RESPONSE_FIELDS - set(payload))
    if unknown:
        raise ProposalSchemaError(f"model response contains non-proposal fields: {unknown}")
    if missing:
        raise ProposalSchemaError(f"model response misses proposal fields: {missing}")
    # Preserve the exact external bytes and immutable digest in the run ledger.
    return FrozenModelResponse(raw_response, sha256(raw_response.encode()).hexdigest(), payload)


def _context(payload: dict[str, Any]) -> ActionObjectContext:
    value = payload.get("context")
    if not isinstance(value, dict) or set(value) - {"action", "object"}:
        raise ProposalSchemaError("context must contain only action and object")
    if not isinstance(value.get("action"), str) or not isinstance(value.get("object"), str):
        raise ProposalSchemaError("context action/object must be strings")
    return ActionObjectContext(value["action"], value["object"])


def _as_options(payload: dict[str, Any]) -> tuple[tuple[Location, ...], tuple[Attachment, ...]]:
    locations, purposes = [], []
    for row in payload["locations"]:
        if not isinstance(row, dict) or set(row) != {"place_det", "place_adj", "place"}:
            raise ProposalSchemaError("location options must be typed triples")
        if not all(isinstance(row[key], str) for key in ("place_det", "place_adj", "place")):
            raise ProposalSchemaError("location option values must be strings")
        locations.append(Location(row["place_det"], row["place_adj"], row["place"]))
    for row in payload["purposes"]:
        if not isinstance(row, dict) or set(row) != {"purpose", "purpose_det", "purpose_adj", "purpose_object"}:
            raise ProposalSchemaError("purpose options must be typed quadruples")
        if not all(isinstance(row[key], str) for key in ("purpose", "purpose_det", "purpose_adj", "purpose_object")):
            raise ProposalSchemaError("purpose option values must be strings")
        purposes.append(Attachment(row["purpose"], row["purpose_det"], row["purpose_adj"], row["purpose_object"]))
    return tuple(locations), tuple(purposes)


def validate_proposal(frozen: FrozenModelResponse, context: ActionObjectContext,
                      required_reverse_prefix: str) -> dict[str, Any]:
    """Validate roles without constructing a surface from model output."""
    payload = frozen.payload
    reasons = []
    try:
        submitted_context = _context(payload)
        locations, purposes = _as_options(payload)
    except ProposalSchemaError as exc:
        return {"accepted": False, "construction_started": False, "reasons": [str(exc)],
                "response_sha256": frozen.response_sha256}
    if not isinstance(payload["proposal_id"], str):
        reasons.append("proposal_id_not_string")
    if submitted_context != context:
        reasons.append("context_does_not_match_request")
    if not isinstance(payload["required_reverse_prefix"], str):
        reasons.append("required_reverse_prefix_not_string")
    elif normalize_letters(payload["required_reverse_prefix"]) != normalize_letters(required_reverse_prefix):
        reasons.append("required_residual_does_not_match_request")
    if len(locations) < 2 or len(purposes) < 2:
        reasons.append("multiple_alternatives_required")
    for location in locations:
        if location.place_det != article(location.place_adj) or location.place not in AUTHOR_LOCATIONS:
            reasons.append("invalid_location_article_or_noun")
        elif location.place_adj not in AUTHOR_LOCATIONS[location.place]:
            reasons.append("invalid_location_modifier")
    for attachment in purposes:
        key = (context.action, context.object)
        frame = AUTHOR_PURPOSES.get(key, {})
        if (attachment.purpose, attachment.purpose_object) not in frame:
            reasons.append("invalid_purpose_frame")
        elif attachment.purpose_adj not in AUTHOR_PURPOSE_ADJECTIVES.get(attachment.purpose_object, set()):
            reasons.append("invalid_purpose_modifier")
        if attachment.purpose_det != article(attachment.purpose_adj):
            reasons.append("invalid_purpose_article")
    return {"accepted": not reasons, "construction_started": False, "reasons": sorted(set(reasons)),
            "response_sha256": frozen.response_sha256, "location_count": len(locations),
            "purpose_count": len(purposes)}


def assemble_surface(context: ActionObjectContext, location: Location, attachment: Attachment) -> str:
    """Python-owned construction from validated typed slots."""
    words = (context.action, context.object_det, context.object_adj, context.object,
             "in", location.place_det, location.place_adj, location.place,
             "to", attachment.purpose, attachment.purpose_det,
             attachment.purpose_adj, attachment.purpose_object)
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def outside_in_ledger(text: str) -> dict[str, Any]:
    tape = normalize_letters(text)
    events = []
    for pair in range(len(tape) // 2):
        right = len(tape) - 1 - pair
        event = {"pair": pair + 1, "left_index": pair, "right_index": right,
                 "left": tape[pair], "right": tape[right], "equal": tape[pair] == tape[right]}
        events.append(event)
        if not event["equal"]:
            return {"exact": False, "letters": len(tape), "events": events,
                    "first_mismatch": event, "normalized_sha256": sha256(tape.encode()).hexdigest()}
    return {"exact": bool(tape), "letters": len(tape), "events": events,
            "first_mismatch": None, "normalized_sha256": sha256(tape.encode()).hexdigest()}


def independent_parse(text: str) -> dict[str, Any]:
    """Reparse the assembled 13-token imperative using only parse inventory."""
    tokens = tuple(WORD_RE.findall(text.lower()))
    if len(tokens) != 13:
        return {"ok": False, "reason": "wrong_role_arity", "tokens": list(tokens)}
    action, od, oa, obj, prep, pd, pa, place, to, purpose, pod, poa, pobj = tokens
    article_ok = od == article(oa) and pd == article(pa) and pod == article(poa)
    frame = PARSE_PURPOSE_FRAMES.get((action, obj, purpose), {})
    role_ok = (action in PARSE_ACTION_OBJECTS and obj in PARSE_ACTION_OBJECTS[action]
               and oa in PARSE_OBJECT_ADJECTIVES and prep == "in" and place in PARSE_LOCATION_NOUNS
               and pa in PARSE_LOCATION_ADJECTIVES and to == "to" and poa in PARSE_PURPOSE_ADJECTIVES
               and pobj in PARSE_OBJECT_TYPES and frame.get(pobj) == PARSE_OBJECT_TYPES[pobj])
    return {"ok": article_ok and role_ok, "agreement_ok": article_ok, "valency_ok": role_ok,
            "tokens": list(tokens), "independent_inventory": True}


def reverse_tail_prefix(context: ActionObjectContext, location: Location, attachment: Attachment) -> str:
    words = ("to", attachment.purpose, attachment.purpose_det,
             attachment.purpose_adj, attachment.purpose_object)
    return normalize_letters("".join(words))[::-1]


def matched_pairs(context: ActionObjectContext, location: Location, attachment: Attachment) -> int:
    opening = normalize_letters("".join((context.action, context.object_det, context.object_adj, context.object)))
    ending = reverse_tail_prefix(context, location, attachment)
    return next((i for i, (left, right) in enumerate(zip(opening, ending)) if left != right), min(len(opening), len(ending)))


def evaluate_frozen_response(frozen: FrozenModelResponse, context: ActionObjectContext,
                             required_reverse_prefix: str) -> dict[str, Any]:
    screen = validate_proposal(frozen, context, required_reverse_prefix)
    base = {"record_kind": "partial_residual_model_suffix_proposal", "proposal_id": frozen.payload.get("proposal_id"),
            "response_sha256": frozen.response_sha256, "frozen_response": True,
            "model_output_is_proposal_only": True, "preconstruction": screen,
            "rendered": None, "construction_started": False, "records": [], "exact_candidates": []}
    if not screen["accepted"]:
        return base
    locations, purposes = _as_options(frozen.payload)
    for location in locations:
        for attachment in purposes:
            reverse_prefix = reverse_tail_prefix(context, location, attachment)
            if not reverse_prefix.startswith(normalize_letters(required_reverse_prefix)):
                base["records"].append({"location": location.__dict__, "attachment": attachment.__dict__,
                                        "reverse_tail_prefix": reverse_prefix,
                                        "rejection": "required_partial_residual_not_met",
                                        "construction_started": False})
                continue
            text = assemble_surface(context, location, attachment)
            ledger = outside_in_ledger(text)
            parsed = independent_parse(text)
            central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            codes = [key for key, value in central.items() if not value]
            if not ledger["exact"]:
                codes.append("outside_in_ledger_mismatch")
            if not parsed["ok"]:
                codes.append("independent_parse_failed")
            row = {"location": location.__dict__, "attachment": attachment.__dict__,
                   "rendered": text, "matched_pairs": matched_pairs(context, location, attachment),
                   "reverse_tail_prefix": reverse_prefix, "outside_in_ledger": ledger,
                   "independent_parse": parsed, "central_admission": central,
                   "mechanically_admitted": not codes, "rejection_codes": codes,
                   "construction_started": True,
                   "reader_status": "human-unreviewed; programmatic checks do not certify readability"}
            base["records"].append(row)
            if row["mechanically_admitted"] and ledger["exact"]:
                base["exact_candidates"].append(row)
    base["construction_started"] = bool(base["records"])
    base["rendered"] = [row["rendered"] for row in base["records"] if row.get("rendered")]
    return base


def run(raw_responses: tuple[str, ...] = (), context: ActionObjectContext = ActionObjectContext("draw", "map"),
        required_reverse_prefix: str = "draw") -> dict[str, Any]:
    frozen, results = [], []
    for raw in raw_responses:
        try:
            item = freeze_model_response(raw)
        except ProposalSchemaError as exc:
            digest = sha256(raw.encode()).hexdigest()
            frozen.append({"response_sha256": digest, "raw_response": raw, "parse_error": str(exc)})
            results.append({"record_kind": "frozen_response_rejection", "frozen_response": True,
                            "model_output_is_proposal_only": True, "construction_started": False,
                            "error": str(exc), "raw_response_sha256": digest})
            continue
        frozen.append({"response_sha256": item.response_sha256, "raw_response": item.raw_response})
        results.append(evaluate_frozen_response(item, context, required_reverse_prefix))
    exact = [row for result in results for row in result.get("exact_candidates", [])]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {"status": "partial_residual_model_suffix_proposal_no_model_call",
            "config": {"model_calls_enabled": False, "proposal_prompt_is_partial_only": True,
                       "frozen_external_responses": True, "python_owns_surface_assembly": True,
                       "python_owns_outside_in_ledger": True, "independent_complete_reparse": True,
                       "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                       "central_admission_before_reader_study": True, "corpus_generation": False},
            "request": {"context": context.__dict__, "required_reverse_prefix": required_reverse_prefix,
                        "prompt": proposal_prompt(context, required_reverse_prefix)},
            "frozen_responses": frozen, "response_results": results,
            "exact_candidates": exact, "admitted_candidates": admitted,
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "external_model_calls": 0, "external_output_role": "proposal_only",
                           "surface_source": "Python assembly from independently validated typed slots"},
            "reader_facing_next_test": "Only an admitted exact surface may enter randomized blinded intact-prose versus shuffled-control reading; programmatic checks do not certify readability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--responses", type=Path, help="JSON array of frozen local-model response strings; no model is called")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    raw = tuple(json.loads(args.responses.read_text())) if args.responses else ()
    result = run(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "responses": len(result["response_results"]),
                      "exact": len(result["exact_candidates"]), "admitted": len(result["admitted_candidates"])}, indent=2))


if __name__ == "__main__":
    main()
