"""Online two-event discourse search with a shared-entity completion seam.

The product is assembled from typed event slots, not from completed sentence
strings.  Each event records its introduced entities and the second event may
only use a typed reference after completion of the first.  The packed
character intersection is the exact gate; the semantic bookkeeping is kept in
the provenance trace so a future lane can widen one event without changing
the matcher.
"""
from collections import deque
import hashlib
import json
from pathlib import Path

from experiments.packed_seam_grammar_20260927 import Grammar, SEED, audit, intersect, norm
from llm_palindrome.validator import is_palindrome

ROOT = Path(__file__).resolve().parents[1]
ID = "shared-entity-event-completion-20260930"


EVENTS = {
    "make_mural": {
        "introduces": {"artist", "mural"},
        "slots": (
            ("subject:singular:introduce:artist", ("an artist", "a painter", "the artist")),
            ("verb:singular:make", ("paints", "draws", "sketches")),
            ("object:singular:introduce:mural", ("a mural", "the mural")),
        ),
    },
    "praise_mural": {
        "requires": {"artist", "mural"},
        "slots": (
            ("subject:singular:reference:artist", ("the artist", "she", "the painter")),
            ("verb:singular:human", ("praises", "thanks", "greets")),
            ("object:singular:reference:mural", ("the mural", "it")),
        ),
    },
}


def compile_event_graph(order):
    """Compile one typed event order; no complete sentence is materialized."""
    introduced = set()
    g = Grammar()
    trace = []
    for name in order:
        event = EVENTS[name]
        if not event.get("requires", set()) <= introduced:
            return None, trace
        before = introduced.copy()
        for role, alternatives in event["slots"]:
            g.slot(alternatives, role)
        introduced.update(event.get("introduces", set()))
        trace.append(dict(event=name, required=sorted(event.get("requires", set())),
                          introduced_before=sorted(before),
                          introduced_after=sorted(introduced),
                          completion_boundary="epsilon"))
    return g, trace


def seed_control():
    return dict(rendered=SEED, audit=audit(SEED), exact_validator=is_palindrome(SEED),
                seed_control=True, human_readability_evidence="historically reader-admitted")


def run():
    conditions = []
    for order in (("make_mural", "praise_mural"), ("praise_mural", "make_mural")):
        grammar, trace = compile_event_graph(order)
        if grammar is None:
            conditions.append(dict(order=order, valid_typed_order=False, candidates=[],
                                   rejection="reference event precedes introduction"))
            continue
        result = intersect(grammar, max_letters=180, cap=100000)
        for row in result["candidates"]:
            row["audit"]["independent_validator_exact"] = is_palindrome(row["rendered"])
            row["forward_sha256"] = row["audit"]["sha256"]
            row["reverse_sha256"] = row["audit"]["reverse_sha256"]
            row["novel_relative_to_seed"] = norm(row["rendered"]) != norm(SEED)
            row["readability_gate"] = "not collected; mechanical candidate only"
        result.update(order=order, valid_typed_order=True, event_trace=trace)
        conditions.append(result)
    return dict(
        experiment_id=ID,
        method="typed shared-entity event completion with online character intersection",
        conditions=conditions,
        controls=[seed_control()],
        provenance=dict(
            complete_sentence_enumeration=False,
            generated_sentence_count_before_intersection=0,
            catalogue_replay=False,
            semordnilap_only_bank=False,
            repeated_unit=False,
            posthoc_repair=False,
            per_candidate_rlaif=False,
            source="fresh ordinary-English typed event slots authored for this lane",
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
        novelty_preflight=dict(
            novel_algorithm_claim=False,
            representation_change="event completion carries introduced artist/mural entities into typed reference slots",
            distinction="the reference event is rejected before compilation when its antecedent is unavailable",
        ),
        reader_test=dict(
            status="not collected",
            protocol="Novel exact outputs require randomized blinded ratings with intact and shuffled controls.",
        ),
        next_repair=dict(
            operator="widen the shared mural reference with ordinary plural agreement while retaining online event completion",
            reason="current reference vocabulary has no exact closure beyond controls; keep semantic state and vary only one boundary",
        ),
    )


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": ID, "conditions": [
        {"order": c.get("order"), "valid": c.get("valid_typed_order"),
         "states": c.get("states"), "exact": len(c.get("candidates", [])),
         "longest": max((r["audit"]["letters"] for r in c.get("candidates", [])), default=0)}
        for c in result["conditions"]]}))
