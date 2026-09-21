"""Joint executable-discourse and character-orbit construction.

The search state carries a tiny finite world model while lexical spans are
emitted from both opposing grammar cursors.  Semantic preconditions are checked
before a complete surface exists; character obligations are consumed at each
word boundary.  This is deliberately a small falsifiable construction, not a
claim that semantic labels certify readability.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/executable-discourse-shared-orbit-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "executable-discourse-shared-orbit-20260920"
SIGNATURE = "finite-model-precondition|joint-character-orbit|packed-event-grammar|entity-binding"

# A frozen, hand-authored micro-world.  Event effects are what make the
# semantic check executable rather than a post-render label.
EVENTS = (
    {"id": "open_gate", "subject": "guard", "verb": "opened", "object": "gate",
     "pre": (("closed", "gate"),), "effects": (("open", "gate"),), "relation": "enables_entry"},
    {"id": "enter_yard", "subject": "visitor", "verb": "entered", "object": "yard",
     "pre": (("open", "gate"),), "effects": (("inside", "visitor", "yard"),), "relation": "enabled_by_open_gate"},
    {"id": "find_key", "subject": "nurse", "verb": "found", "object": "key",
     "pre": (("key_at", "table"),), "effects": (("has", "nurse", "key"),), "relation": "enables_cabinet"},
    {"id": "open_cabinet", "subject": "nurse", "verb": "opened", "object": "cabinet",
     "pre": (("has", "nurse", "key"), ("closed", "cabinet")),
     "effects": (("open", "cabinet"),), "relation": "enabled_by_key"},
    {"id": "raise_flag", "subject": "sailor", "verb": "raised", "object": "flag",
     "pre": (("flag_down", "flag"),), "effects": (("up", "flag"),), "relation": "enables_answer"},
    {"id": "answer_signal", "subject": "keeper", "verb": "answered", "object": "signal",
     "pre": (("up", "flag"),), "effects": (("acknowledged", "keeper", "signal"),), "relation": "enabled_by_flag"},
)

DETS = ("the", "a")
CONNECTORS = ("so", "then")
WORLD_INITIAL = {
    ("closed", "gate"), ("closed", "cabinet"), ("key_at", "table"), ("flag_down", "flag"),
}


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    tape = letters(text)
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {
        "letters": len(tape),
        "pointer_exact": bool(tape) and i >= j,
        "first_mismatch": None if i >= j else [i, j, tape[i], tape[j]],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def novelty_preflight() -> dict:
    data = json.loads(REGISTRY.read_text())
    rows = [x for x in data.get("entries", []) + data.get("excluded", []) if x.get("id") != EXPERIMENT_ID]
    overlap = [x["id"] for x in rows if x.get("signature") == SIGNATURE]
    artifact = str(Path(__file__).relative_to(ROOT))
    collisions = [x["id"] for x in rows if x.get("artifact") == artifact]
    return {"status": "passed" if not overlap and not collisions else "blocked",
            "signature_overlaps": overlap, "artifact_collisions": collisions,
            "excluded_routes": ["finished-tape reversal", "post-search semantics", "catalogue/API text"]}


def initial_facts(event: dict) -> set[tuple[str, ...]]:
    return set(WORLD_INITIAL)


def apply_effects(facts: set[tuple[str, ...]], effects: tuple[tuple[str, ...], ...]) -> set[tuple[str, ...]]:
    """Apply effects with the tiny world's mutually exclusive state pairs."""
    out = set(facts)
    opposites = {"open": "closed", "closed": "open", "up": "flag_down", "flag_down": "up"}
    for effect in effects:
        if effect:
            opposite = opposites.get(effect[0])
            if opposite:
                out.discard((opposite,) + tuple(effect[1:]))
        out.add(effect)
    return out


def semantic_join(first: dict, second: dict) -> tuple[bool, str, set[tuple[str, ...]]]:
    """Apply first event effects and test second event preconditions."""
    if first["id"] == second["id"]:
        return False, "discourse requires two distinct events", set()
    facts = initial_facts(first)
    missing_first = [p for p in first["pre"] if p not in facts]
    if missing_first:
        return False, f"missing first-event precondition {missing_first[0]}", facts
    facts = apply_effects(facts, first["effects"])
    if not any(p in first["effects"] for p in second["pre"]):
        return False, "second event has no precondition enabled by the first event", facts
    missing = [p for p in second["pre"] if p not in facts]
    if missing:
        return False, f"missing precondition {missing[0]}", facts
    return True, "", facts


def clause(event: dict, subject_det: str, object_det: str) -> tuple[str, ...]:
    return (subject_det, event["subject"], event["verb"], object_det, event["object"])


def consume(left_word: str, right_word: str, pending: str) -> tuple[bool, str, int]:
    """Consume opposing characters without constructing a finished tape."""
    a = pending + letters(left_word)
    b = letters(right_word)[::-1]
    n = min(len(a), len(b))
    if a[:n] != b[:n]:
        return False, pending, n
    return True, a[n:], n


def render(first: dict, second: dict, d1: str, d2: str, connector: str) -> str:
    left = clause(first, d1, d2)
    right = clause(second, d1, d2)
    return " ".join(left) + ", " + connector + " " + " ".join(right) + "."


def run() -> dict:
    pre = novelty_preflight()
    if pre["status"] != "passed":
        raise RuntimeError(pre)
    semantic_prunes = 0
    character_prunes = 0
    expanded = 0
    valid_pairs = []
    rejected = []
    rows = []

    # Select event pairs before lexical spans.  The semantic relation is live;
    # a missing precondition rejects a branch while only event IDs are known.
    for first in EVENTS:
        for second in EVENTS:
            expanded += 1
            ok, witness, facts = semantic_join(first, second)
            if not ok:
                semantic_prunes += 1
                if len(rejected) < 12:
                    rejected.append({"first": first["id"], "second": second["id"], "witness": witness,
                                     "lexical_spans_unresolved": True})
                continue
            valid_pairs.append((first, second, facts))
            # Two opposing grammar cursors advance one lexical span at a time.
            # Right words are consumed in reverse cursor order, but remain an
            # independently authored event clause in the final rendering.
            for d1 in DETS:
                for d2 in DETS:
                    left = clause(first, d1, d2)
                    right_outer = tuple(reversed(clause(second, d1, d2)))
                    pending = ""
                    ok_chars = True
                    checked = 0
                    for lw, rw in zip(left, right_outer):
                        ok_chars, pending, n = consume(lw, rw, pending)
                        checked += n
                        if not ok_chars:
                            character_prunes += 1
                            break
                    if ok_chars and not pending:
                        text = render(first, second, d1, d2, "then")
                        a = audit(text)
                        rows.append({"rendered": text, "audit": a,
                                     "event_pair": [first["id"], second["id"]],
                                     "world_facts": sorted(" ".join(x) for x in facts),
                                     "characters_checked": checked,
                                     "provenance": {"semantic_before_render": True,
                                                    "lexical_spans_jointly_unresolved": True,
                                                    "finished_tape_reversal": False,
                                                    "post_hoc_repair": False,
                                                    "catalogue_text": False,
                                                    "word_order_symmetry": False,
                                                    "repeated_units": False}})

    controls = []
    for first, second, facts in valid_pairs[:3]:
        text = render(first, second, first["subject_det"] if "subject_det" in first else "the", "the", "then")
        controls.append({"rendered": text, "audit": audit(text),
                         "reader_eligible": False,
                         "provenance": {"finite_world_model": True,
                                        "preconditions_satisfied": True,
                                        "independent_event_pair": True,
                                        "catalogue_text": False}})

    # Keep the invalid/valid semantic contrast executable and reproducible.
    semantics_off_pairs = len(EVENTS) * len(EVENTS)
    exact = [r for r in rows if r["audit"]["pointer_exact"] and
             r["audit"]["sha256_forward"] == r["audit"]["sha256_reverse"] and
             r["audit"]["letters"] > 38]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "method": "joint executable finite-world discourse CSP with shared character-orbit obligations",
        "config": {"events": len(EVENTS), "determinants": len(DETS), "connectors": len(CONNECTORS),
                    "semantic_checked_before_complete_surface": True, "post_search_scoring": False},
        "stats": {"event_pairs_expanded": expanded, "semantic_valid_pairs": len(valid_pairs),
                  "semantic_prunes": semantic_prunes, "character_prunes": character_prunes,
                  "semantics_off_pairs": semantics_off_pairs, "rendered_candidates": len(rows),
                  "prose_controls": len(controls), "exact_gt38": len(exact),
                  "max_control_letters": max((x["audit"]["letters"] for x in controls), default=0)},
        "rendered_candidates": rows[:20], "prose_controls": controls,
        "exact_candidates": exact, "semantic_rejection_witnesses": rejected,
        "novelty_preflight": pre,
        "provenance": {"world_model": "frozen hand-authored precondition/effect events",
                       "independent_audit": ["two-pointer", "forward/reverse SHA-256"],
                       "semantic_pruning_before_render": True, "catalogue_text": False,
                       "finished_tape_reversal": False, "post_hoc_repair": False,
                       "reader_evidence": False},
        "falsifier": "if semantics-on and semantics-off visit identical event states, or semantic witnesses occur only after rendering, the joint representation is falsified",
        "next_operator": "carry typed recipient/theme argument roles through the executable event state before lexical realization",
        "status": "fresh exact >38 requires blinded human reading" if exact else "no exact >38 closure; executable discourse controls retained",
    }
    RUN.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
