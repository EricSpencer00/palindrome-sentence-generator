"""Brown/word-frequency registry for model proposals under a live residual.

The local corpus supplies only lexical evidence: Brown POS counts and
wordfreq frequency.  A separate authored semantic registry assigns explicit
object/place types and verb valency, then the Brown/wordfreq intersection
provides a broad role vocabulary.  Discovery enumerates that vocabulary and
retains only contexts with at least five actual outside-in endpoint matches.
Only those discovered context/residual groups may receive frozen model
proposals.  A proposal contains role values, never a sentence or character
tape; Python assembles and independently reparses every surface.

This module intentionally does not call a local model.  Its CLI consumes only
already frozen response strings supplied by a caller.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.safe_vocab import is_allowed

MIN_LETTERS, MAX_LETTERS = 30, 160
MIN_DISCOVERED_PAIRS = 5
WORD_RE = re.compile(r"[a-z]+")
RESPONSE_FIELDS = frozenset({"proposal_id", "context", "required_reverse_prefix", "locations", "purposes"})
BROWN_TAGS = {
    "noun": frozenset({"NN", "NNS", "NNP", "NNPS"}),
    "verb": frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}),
    "adjective": frozenset({"JJ", "JJR", "JJS"}),
}


@dataclass(frozen=True)
class Lexeme:
    word: str
    roles: tuple[str, ...]
    semantic_types: tuple[str, ...]
    brown_count: int
    zipf_frequency: float


@dataclass(frozen=True)
class VerbFrame:
    verb: str
    object_type: str
    objects: tuple[str, ...]


@dataclass(frozen=True)
class FrameSpec:
    frame_id: str
    action: str
    object: str
    object_type: str
    purpose_frames: tuple[VerbFrame, ...]


@dataclass(frozen=True)
class ContextSpec:
    frame_id: str
    action: str
    object: str
    object_det: str
    object_adj: str


@dataclass(frozen=True)
class LocationProposal:
    place_det: str
    place_adj: str
    place: str


@dataclass(frozen=True)
class PurposeProposal:
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


# Explicit semantic typing and valency.  Brown/wordfreq may only attest or
# rank these words; neither source is allowed to invent a semantic relation.
SEMANTIC_TYPES: dict[str, frozenset[str]] = {
    "artifact": frozenset({
        "bag", "book", "box", "boxes", "canvas", "card", "cart", "code", "diagram",
        "file", "letter", "map", "model", "paper", "plan", "pots", "project", "report",
        "record", "room", "table", "tool", "work", "yard",
    }),
    "person": frozenset({"artist", "child", "clerk", "guard", "teacher", "worker", "writer"}),
    "place": frozenset({
        "field", "gallery", "garden", "hall", "home", "library", "market", "office",
        "park", "room", "school", "shop", "station", "studio", "workshop", "yard",
    }),
    "support": frozenset({"help"}),
}
SEMANTIC_WORD_TYPES = {word: frozenset(kind for kind, words in SEMANTIC_TYPES.items() if word in words)
                       for words in SEMANTIC_TYPES.values() for word in words}

# These are role inventories, not endpoint pairs.  Discovery below tries all
# compatible frames and all Brown/wordfreq-qualified modifiers.
FRAME_REGISTRY: tuple[FrameSpec, ...] = (
    FrameSpec("stop-cart", "stop", "cart", "artifact", (
        VerbFrame("store", "artifact", ("pots", "boxes")),
        VerbFrame("load", "artifact", ("boxes", "bag")),
        VerbFrame("move", "artifact", ("cart", "box")),
    )),
    FrameSpec("paint-canvas", "paint", "canvas", "artifact", (
        VerbFrame("display", "artifact", ("map", "canvas")),
        VerbFrame("frame", "artifact", ("map", "canvas")),
        VerbFrame("store", "artifact", ("canvas", "box")),
    )),
    FrameSpec("read-book", "read", "book", "artifact", (
        VerbFrame("file", "artifact", ("paper", "record")),
        VerbFrame("cite", "artifact", ("paper", "work")),
        VerbFrame("scan", "artifact", ("paper", "map")),
    )),
    FrameSpec("make-model", "make", "model", "artifact", (
        VerbFrame("sketch", "artifact", ("diagram", "map")),
        VerbFrame("show", "artifact", ("model", "work")),
        VerbFrame("test", "artifact", ("model", "tool")),
    )),
    FrameSpec("plan-project", "plan", "project", "artifact", (
        VerbFrame("seek", "support", ("help",)),
        VerbFrame("fund", "artifact", ("project", "work")),
        VerbFrame("start", "artifact", ("project", "plan")),
    )),
)

# Modifier candidates have explicit syntactic roles.  Their Brown tag and
# wordfreq intersection is computed at runtime, so this is not a tiny fixed
# output menu and unobserved/non-English strings cannot enter proposals.
MODIFIER_WORDS = frozenset({
    "annual", "bright", "clean", "clear", "detailed", "empty", "extra", "final", "full",
    "heavy", "kind", "large", "local", "new", "plain", "public", "quiet", "red", "remote",
    "safe", "sharp", "simple", "small", "thick", "useful", "young",
})
_BROWN_COUNTS: dict[str, Counter[str]] | None = None


def build_registry(*, min_zipf: float = 2.2, brown_words: Iterable[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Build a broad role registry from local Brown counts × wordfreq.

    Brown and wordfreq are lexical filters/rankers only. Semantic type and
    verb valency remain explicit in ``SEMANTIC_TYPES`` and ``FRAME_REGISTRY``.
    """
    global _BROWN_COUNTS
    if brown_words is None and _BROWN_COUNTS is not None:
        counts = {role: Counter(table) for role, table in _BROWN_COUNTS.items()}
    else:
        if brown_words is None:
            from nltk.corpus import brown
            brown_words = brown.tagged_words()
        counts = {role: Counter() for role in BROWN_TAGS}
        for raw, tag in brown_words:
            word = str(raw).casefold()
            if not word.isascii() or not word.isalpha() or not is_allowed(word):
                continue
            for role, tags in BROWN_TAGS.items():
                if tag in tags:
                    counts[role][word] += 1
        if _BROWN_COUNTS is None:
            _BROWN_COUNTS = {role: Counter(table) for role, table in counts.items()}
    from wordfreq import zipf_frequency
    frame_words = set()
    for frame in FRAME_REGISTRY:
        frame_words.update((frame.action, frame.object))
        for verb_frame in frame.purpose_frames:
            frame_words.add(verb_frame.verb)
            frame_words.update(verb_frame.objects)
    semantic_words = set(SEMANTIC_WORD_TYPES) | set(MODIFIER_WORDS) | frame_words
    entries: dict[str, Lexeme] = {}
    for word in semantic_words:
        freq = float(zipf_frequency(word, "en"))
        roles = tuple(sorted(role for role, table in counts.items() if table[word]))
        if freq < min_zipf or not roles:
            continue
        entries[word] = Lexeme(word, roles, tuple(sorted(SEMANTIC_WORD_TYPES.get(word, ()))),
                               max((counts[role][word] for role in roles), default=0), freq)
    return {"entries": entries, "frames": FRAME_REGISTRY, "min_zipf": min_zipf,
            "brown_role_counts": {role: sum(1 for word in table if word in entries) for role, table in counts.items()}}


def _has_role(registry: dict[str, Any], word: str, role: str) -> bool:
    return role in registry["entries"].get(word, Lexeme(word, (), (), 0, 0.0)).roles


def _has_type(registry: dict[str, Any], word: str, semantic_type: str) -> bool:
    return semantic_type in registry["entries"].get(word, Lexeme(word, (), (), 0, 0.0)).semantic_types


def discover_partial_contexts(registry: dict[str, Any], minimum_pairs: int = MIN_DISCOVERED_PAIRS) -> tuple[dict[str, Any], ...]:
    discoveries = []
    adjectives = sorted(word for word in MODIFIER_WORDS if _has_role(registry, word, "adjective"))
    for frame in registry["frames"]:
        if not (_has_role(registry, frame.action, "verb") and _has_role(registry, frame.object, "noun")
                and _has_type(registry, frame.object, frame.object_type)):
            continue
        for object_adj in adjectives:
            context = ContextSpec(frame.frame_id, frame.action, frame.object, article(object_adj), object_adj)
            opening = normalize_letters("".join((context.action, context.object_det, context.object_adj, context.object)))
            for verb_frame in frame.purpose_frames:
                if not _has_role(registry, verb_frame.verb, "verb"):
                    continue
                for purpose_object in verb_frame.objects:
                    if not (_has_role(registry, purpose_object, "noun") and _has_type(registry, purpose_object, verb_frame.object_type)):
                        continue
                    for purpose_adj in adjectives:
                        # ``the`` licenses both count and mass/plural objects;
                        # its use is legal independent of adjective onset.
                        reverse_tail = normalize_letters("".join(("to", verb_frame.verb, "the",
                                                                  purpose_adj, purpose_object)))[::-1]
                        depth = 0
                        for left, right in zip(opening, reverse_tail):
                            if left != right: break
                            depth += 1
                        if depth < minimum_pairs: continue
                        discoveries.append({"frame_id": frame.frame_id, "context": asdict(context),
                            "required_reverse_prefix": opening[:depth], "matched_pairs": depth,
                            "opening_letters": opening, "reverse_tail_letters": reverse_tail,
                            "semantic_frame": [frame.action, frame.object, verb_frame.verb, purpose_object],
                            "discovered_from": {"purpose_adj": purpose_adj, "purpose_object": purpose_object}})
    return tuple(discoveries)


def group_discoveries(discoveries: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in discoveries:
        c = row["context"]
        key = json.dumps({"frame_id": row["frame_id"], "action": c["action"], "object": c["object"],
                          "object_det": c["object_det"], "object_adj": c["object_adj"],
                          "required_reverse_prefix": row["required_reverse_prefix"]}, sort_keys=True)
        groups.setdefault(key, []).append(row)
    return groups


def proposal_prompt(discovery: dict[str, Any], registry: dict[str, Any]) -> str:
    c = discovery["context"]
    role_counts = {role: sum(role in entry.roles for entry in registry["entries"].values())
                   for role in ("noun", "verb", "adjective")}
    return ("Return JSON typed alternatives for location and purpose roles only. "
            f"Context action={c['action']}, object={c['object']}, modifier={c['object_adj']}; "
            f"required reverse prefix={discovery['required_reverse_prefix']}. "
            f"The registry has {role_counts['noun']} noun, {role_counts['verb']} verb, and {role_counts['adjective']} adjective entries. "
            "Use only registry words and explicit semantic relations. Return no complete sentence, surface, character tape, or mirrored text; Python assembles and verifies it.")


def freeze_model_response(raw_response: str) -> FrozenResponse:
    if not isinstance(raw_response, str) or not raw_response.strip(): raise ProposalSchemaError("empty response")
    try: payload = json.loads(raw_response)
    except json.JSONDecodeError as exc: raise ProposalSchemaError("response is not JSON") from exc
    if not isinstance(payload, dict): raise ProposalSchemaError("response must be an object")
    unknown, missing = sorted(set(payload) - RESPONSE_FIELDS), sorted(RESPONSE_FIELDS - set(payload))
    if unknown or missing: raise ProposalSchemaError(f"proposal schema mismatch unknown={unknown} missing={missing}")
    return FrozenResponse(raw_response, sha256(raw_response.encode()).hexdigest(), payload)


def article(adjective: str) -> str:
    return "an" if adjective[:1].lower() in "aeiou" else "a"


def _parse_payload(frozen: FrozenResponse) -> tuple[ContextSpec, tuple[LocationProposal, ...], tuple[PurposeProposal, ...]]:
    c = frozen.payload["context"]
    expected = {"frame_id", "action", "object", "object_det", "object_adj"}
    if not isinstance(c, dict) or set(c) != expected or not all(isinstance(c[k], str) for k in c):
        raise ProposalSchemaError("invalid discovered context")
    locations, purposes = [], []
    for row in frozen.payload["locations"]:
        if not isinstance(row, dict) or set(row) != {"place_det", "place_adj", "place"} or not all(isinstance(row[k], str) for k in row):
            raise ProposalSchemaError("invalid typed location")
        locations.append(LocationProposal(**row))
    for row in frozen.payload["purposes"]:
        if not isinstance(row, dict) or set(row) != {"purpose", "purpose_det", "purpose_adj", "purpose_object"} or not all(isinstance(row[k], str) for k in row):
            raise ProposalSchemaError("invalid typed purpose")
        purposes.append(PurposeProposal(**row))
    return ContextSpec(**c), tuple(locations), tuple(purposes)


def _surface(context: ContextSpec, location: LocationProposal, purpose: PurposeProposal) -> str:
    words = (context.action, context.object_det, context.object_adj, context.object, "in",
             location.place_det, location.place_adj, location.place, "to", purpose.purpose,
             purpose.purpose_det, purpose.purpose_adj, purpose.purpose_object)
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def independent_parse(text: str) -> dict[str, Any]:
    """Separate feature/valency parse; it does not consult proposal metadata."""
    tokens = tuple(WORD_RE.findall(text.lower()))
    if len(tokens) != 13: return {"ok": False, "reason": "wrong_role_arity", "tokens": list(tokens)}
    action, od, oa, obj, prep, pd, pa, place, to, purpose, pod, poa, pobj = tokens
    parse_actions = {"stop": {"cart"}, "paint": {"canvas"}, "read": {"book"},
                     "make": {"model"}, "plan": {"project"}}
    parse_frames = {
        ("stop", "cart", "store"): {"pots": "artifact", "boxes": "artifact"},
        ("stop", "cart", "load"): {"boxes": "artifact", "bag": "artifact"},
        ("stop", "cart", "move"): {"cart": "artifact", "box": "artifact"},
        ("paint", "canvas", "display"): {"map": "artifact", "canvas": "artifact"},
        ("paint", "canvas", "frame"): {"map": "artifact", "canvas": "artifact"},
        ("paint", "canvas", "store"): {"canvas": "artifact", "box": "artifact"},
        ("read", "book", "file"): {"paper": "artifact", "record": "artifact"},
        ("read", "book", "cite"): {"paper": "artifact", "work": "artifact"},
        ("read", "book", "scan"): {"paper": "artifact", "map": "artifact"},
        ("make", "model", "sketch"): {"diagram": "artifact", "map": "artifact"},
        ("make", "model", "show"): {"model": "artifact", "work": "artifact"},
        ("make", "model", "test"): {"model": "artifact", "tool": "artifact"},
        ("plan", "project", "seek"): {"help": "support"},
        ("plan", "project", "fund"): {"project": "artifact", "work": "artifact"},
        ("plan", "project", "start"): {"project": "artifact", "plan": "artifact"},
    }
    adjectives = {"annual", "bright", "clean", "clear", "detailed", "empty", "extra", "final", "full",
                  "heavy", "large", "local", "new", "plain", "public", "quiet", "red", "remote", "safe",
                  "sharp", "simple", "small", "thick", "useful", "young"}
    places = {"field", "gallery", "garden", "hall", "home", "library", "market", "office", "park", "room",
              "school", "shop", "station", "studio", "workshop", "yard"}
    det_ok = lambda det, adj: det == "the" or det == article(adj)
    article_ok = det_ok(od, oa) and det_ok(pd, pa) and det_ok(pod, poa)
    role_ok = (action in parse_actions and obj in parse_actions[action] and oa in adjectives and prep == "in"
               and pa in adjectives and place in places and to == "to" and poa in adjectives
               and pobj in parse_frames.get((action, obj, purpose), {}))
    return {"ok": article_ok and role_ok, "agreement_ok": article_ok, "valency_ok": role_ok,
            "tokens": list(tokens), "independent_inventory": True}


def outside_in_ledger(text: str) -> dict[str, Any]:
    tape = normalize_letters(text); events = []
    for pair in range(len(tape) // 2):
        right = len(tape) - pair - 1
        event = {"pair": pair + 1, "left": tape[pair], "right": tape[right], "equal": tape[pair] == tape[right]}
        events.append(event)
        if not event["equal"]:
            return {"exact": False, "letters": len(tape), "events": events, "first_mismatch": event,
                    "normalized_sha256": sha256(tape.encode()).hexdigest()}
    return {"exact": bool(tape), "letters": len(tape), "events": events, "first_mismatch": None,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def _evaluate(frozen: FrozenResponse, discovery: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    base = {"record_kind": "brown_wordfreq_registry_frozen_suffix_proposal", "frozen_response": True,
            "model_output_is_proposal_only": True, "response_sha256": frozen.response_sha256,
            "proposal_id": frozen.payload.get("proposal_id"), "rendered": [], "records": [], "exact_candidates": []}
    try: context, locations, purposes = _parse_payload(frozen)
    except ProposalSchemaError as exc:
        base.update({"preconstruction": {"accepted": False, "reasons": [str(exc)]}, "construction_started": False}); return base
    expected = ContextSpec(**discovery["context"]); reasons = []
    if context != expected: reasons.append("context_not_discovered")
    if not isinstance(frozen.payload["required_reverse_prefix"], str) or normalize_letters(frozen.payload["required_reverse_prefix"]) != normalize_letters(discovery["required_reverse_prefix"]):
        reasons.append("residual_not_discovered")
    if len(locations) < 2 or len(purposes) < 2: reasons.append("multiple_alternatives_required")
    frame = next(frame for frame in registry["frames"] if frame.frame_id == discovery["frame_id"])
    allowed_locations = {(det, adj, place) for adj in MODIFIER_WORDS for place in SEMANTIC_TYPES["place"]
                         for det in (article(adj), "the")}
    allowed_purposes = {(vf.verb, adj, obj) for vf in frame.purpose_frames for obj in vf.objects
                        for adj in MODIFIER_WORDS}
    for loc in locations:
        if (loc.place_det, loc.place_adj, loc.place) not in allowed_locations or not (_has_role(registry, loc.place_adj, "adjective") and _has_type(registry, loc.place, "place")):
            reasons.append("location_not_in_external_registry")
    for pur in purposes:
        determiner_ok = pur.purpose_det == "the" or pur.purpose_det == article(pur.purpose_adj)
        if (pur.purpose, pur.purpose_adj, pur.purpose_object) not in allowed_purposes or not determiner_ok or not (_has_role(registry, pur.purpose, "verb") and _has_role(registry, pur.purpose_adj, "adjective") and _has_role(registry, pur.purpose_object, "noun")):
            reasons.append("purpose_not_in_external_registry")
    base["preconstruction"] = {"accepted": not reasons, "reasons": sorted(set(reasons)), "discovered_context": True}
    if reasons: base["construction_started"] = False; return base
    required = normalize_letters(discovery["required_reverse_prefix"])
    for loc in locations:
        for pur in purposes:
            reverse_tail = normalize_letters("".join(("to", pur.purpose, pur.purpose_det, pur.purpose_adj, pur.purpose_object)))[::-1]
            if not reverse_tail.startswith(required):
                base["records"].append({"location": asdict(loc), "purpose": asdict(pur), "reverse_tail_prefix": reverse_tail,
                                        "rejection": "partial_residual_not_met", "construction_started": False}); continue
            text = _surface(context, loc, pur); ledger = outside_in_ledger(text); parsed = independent_parse(text)
            central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            codes = [key for key, value in central.items() if not value]
            if not ledger["exact"]: codes.append("outside_in_ledger_mismatch")
            if not parsed["ok"]: codes.append("independent_parse_failed")
            row = {"location": asdict(loc), "purpose": asdict(pur), "rendered": text,
                   "matched_pairs": discovery["matched_pairs"], "reverse_tail_prefix": reverse_tail,
                   "outside_in_ledger": ledger, "independent_parse": parsed, "central_admission": central,
                   "mechanically_admitted": not codes, "rejection_codes": codes, "construction_started": True,
                   "reader_status": "human-unreviewed; programmatic checks do not certify readability"}
            base["records"].append(row); base["rendered"].append(text)
            if row["mechanically_admitted"] and ledger["exact"]: base["exact_candidates"].append(row)
    base["construction_started"] = bool(base["records"]); return base


def run(raw_responses_by_key: Mapping[str, tuple[str, ...]] | None = None,
        *, min_zipf: float = 2.2, minimum_pairs: int = MIN_DISCOVERED_PAIRS,
        brown_words: Iterable[tuple[str, str]] | None = None) -> dict[str, Any]:
    registry = build_registry(min_zipf=min_zipf, brown_words=brown_words)
    discoveries = discover_partial_contexts(registry, minimum_pairs); groups = group_discoveries(discoveries)
    supplied = raw_responses_by_key or {}; frozen, results = [], []
    for key, rows in groups.items():
        for raw in supplied.get(key, ()):
            try: item = freeze_model_response(raw)
            except ProposalSchemaError as exc:
                digest = sha256(raw.encode()).hexdigest(); frozen.append({"response_sha256": digest, "raw_response": raw, "parse_error": str(exc)})
                results.append({"record_kind": "frozen_response_rejection", "frozen_response": True, "model_output_is_proposal_only": True, "construction_started": False, "error": str(exc), "raw_response_sha256": digest}); continue
            frozen.append({"response_sha256": item.response_sha256, "raw_response": item.raw_response}); results.append(_evaluate(item, rows[0], registry))
    exact = [row for result in results for row in result.get("exact_candidates", [])]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {"status": "brown_wordfreq_registry_partial_residual_scheduler_no_model_call",
            "config": {"model_calls_enabled": False, "brown_pos_lexical_source": True, "wordfreq_lexical_source": True,
                       "explicit_semantic_types": True, "explicit_verb_valency": True, "proposal_values_must_be_registry_backed": True,
                       "prompts_only_for_discovered_contexts": True, "minimum_discovered_pairs": minimum_pairs,
                       "frozen_external_responses": True, "python_owns_surface_assembly": True,
                       "python_owns_outside_in_ledger": True, "independent_complete_reparse": True,
                       "corpus_generation": False, "central_admission_before_reader_study": True},
            "registry": {"entry_count": len(registry["entries"]), "brown_role_counts": registry["brown_role_counts"], "min_zipf": min_zipf,
                         "semantic_type_counts": {kind: sum(word in registry["entries"] for word in words) for kind, words in SEMANTIC_TYPES.items()}},
            "discovery_count": len(discoveries), "prompt_group_count": len(groups), "prompted_contexts": discoveries,
            "prompt_groups": {key: {"count": len(rows), "prompt": proposal_prompt(rows[0], registry)} for key, rows in groups.items()},
            "frozen_responses": frozen, "response_results": results, "exact_candidates": exact, "admitted_candidates": admitted,
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "external_model_calls": 0,
                           "semantic_registry": "explicit typed frames; Brown/POS and wordfreq only lexical filters/rankers",
                           "surface_source": "Python assembly from registry-backed frozen role proposals"},
            "reader_facing_next_test": "Only an admitted exact surface may enter randomized blinded intact-prose versus shuffled-control reading; programmatic checks do not certify readability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--responses", type=Path, help="JSON object mapping discovered prompt keys to response arrays; no model call")
    parser.add_argument("--min-zipf", type=float, default=2.2); args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    supplied = json.loads(args.responses.read_text()) if args.responses else None
    result = run(supplied, min_zipf=args.min_zipf); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "registry": result["registry"]["entry_count"], "discoveries": result["discovery_count"], "groups": result["prompt_group_count"], "exact": len(result["exact_candidates"]), "admitted": len(result["admitted_candidates"])}, indent=2))


if __name__ == "__main__": main()
