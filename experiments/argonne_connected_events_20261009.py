"""Exact event-language intersection with executable ownership and signing.

This is a bounded expansion of the earlier world-preimage experiment, not a
claim to invent semantic palindrome search. Repeated entities are allowed;
repeated propositions are not. No loose-word beam or language-model score is
used. A zero result describes this finite event language only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.search import WordTries, consume


def tape(text):
    if any(c.isalpha() and not c.isascii() for c in text):
        raise ValueError("unsupported non-ASCII letter")
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


@dataclass(frozen=True)
class Operation:
    kind: str
    item: int
    actor: int
    recipient: int = -1


@dataclass(frozen=True)
class Event:
    text: str
    operation: Operation
    template: str


class World:
    """Document owners, current-holder reading, and completed signatures.

    Handover invalidates the current-holder read bit. Signing requires that
    the current holder has read the document. All documents start with actor
    zero and must eventually be signed. Signing is irreversible in this model.
    """
    def __init__(self, actors=4, items=3):
        self.actors, self.items = actors, items
        self.owner_states = actors ** items
        self.mask = (1 << items) - 1
        self.states = self.owner_states * (1 << (2 * items))
        self.initial = 0
        self._preimage = {}
        self._transitions = {}
        self.preimage_calls = 0
        self.preimage_cache_hits = 0
        self.goal_states = sum(1 << state for state in range(self.states) if self.goal(state))

    def decode(self, state):
        owners = state % self.owner_states
        flags = state // self.owner_states
        return owners, flags & self.mask, flags >> self.items

    def goal(self, state):
        return self.decode(state)[2] == self.mask

    def apply(self, state, operation):
        owners, read, signed = self.decode(state)
        place = self.actors ** operation.item
        owner = (owners // place) % self.actors
        bit = 1 << operation.item
        if owner != operation.actor:
            return None
        if operation.kind == "handover":
            if operation.recipient == owner:
                return None
            owners += (operation.recipient - owner) * place
            read &= ~bit
        elif operation.kind == "read":
            if read & bit:
                return None
            read |= bit
        elif operation.kind == "sign":
            if not read & bit or signed & bit:
                return None
            signed |= bit
        else:
            raise ValueError(f"unknown event kind: {operation.kind}")
        return owners + self.owner_states * (read + (signed << self.items))

    def preimage(self, operation, allowed_after):
        self.preimage_calls += 1
        key = operation, allowed_after
        if key in self._preimage:
            self.preimage_cache_hits += 1
        if key not in self._preimage:
            if operation not in self._transitions:
                self._transitions[operation] = tuple(
                    (before, after) for before in range(self.states)
                    if (after := self.apply(before, operation)) is not None)
            value = 0
            for before, after in self._transitions[operation]:
                if allowed_after & (1 << after):
                    value |= 1 << before
            self._preimage[key] = value
        return self._preimage[key]


def build_events(actors, objects):
    events = []
    for item, noun in enumerate(objects):
        obj = f"the {noun}"
        for actor, person in enumerate(actors):
            for kind, forms in (
                ("read", [(f"{person} reads {obj}", "active_read"),
                          (f"{person} reads through {obj}", "read_through"),
                          (f"{obj} is read by {person}", "passive_read")]),
                ("sign", [(f"{person} signs {obj}", "active_sign"),
                          (f"{person} puts a signature on {obj}", "signature_on"),
                          (f"{person} adds a signature to {obj}", "signature_to"),
                          (f"{obj} is signed by {person}", "passive_sign")]),
            ):
                operation = Operation(kind, item, actor)
                events.extend(Event(text, operation, template) for text, template in forms)
            for recipient, other in enumerate(actors):
                if recipient == actor:
                    continue
                operation = Operation("handover", item, actor, recipient)
                for text, template in (
                    (f"{person} gives {obj} to {other}", "gives_to"),
                    (f"{person} hands {obj} to {other}", "hands_to"),
                    (f"{person} passes {obj} to {other}", "passes_to"),
                    (f"{other} gets {obj} from {person}", "gets_from"),
                    (f"{other} receives {obj} from {person}", "receives_from"),
                    (f"{obj} passes from {person} to {other}", "object_passes"),
                ):
                    events.append(Event(text, operation, template))
    return events


def replay(events, world):
    state = world.initial
    seen, traces, links, read_provider, owner_provider = set(), [], [], {}, {}
    entity_sets = []
    for index, event in enumerate(events):
        operation = event.operation
        if operation in seen:
            return {"valid": False, "reason": "repeated_proposition", "at": index}
        after = world.apply(state, operation)
        if after is None:
            return {"valid": False, "reason": "unmet_precondition", "at": index}
        seen.add(operation)
        if operation.item in owner_provider:
            links.append({"from": owner_provider[operation.item], "to": index, "kind": "ownership_precondition"})
        if operation.kind == "handover":
            owner_provider[operation.item] = index
            read_provider.pop(operation.item, None)
        elif operation.kind == "read":
            read_provider[operation.item] = index
        else:
            links.append({"from": read_provider[operation.item], "to": index, "kind": "read_before_sign"})
        participants = {("actor", operation.actor), ("item", operation.item)}
        if operation.recipient >= 0:
            participants.add(("actor", operation.recipient))
        entity_sets.append(participants)
        traces.append({"kind": operation.kind, "actor": operation.actor,
                       "item": operation.item, "recipient": operation.recipient,
                       "world_before": state, "world_after": after})
        state = after
    connected = not events
    if events:
        reached, entities = {0}, set(entity_sets[0])
        while True:
            added = {i for i, values in enumerate(entity_sets) if i not in reached and values & entities}
            if not added:
                break
            reached.update(added)
            for i in added:
                entities.update(entity_sets[i])
        connected = len(reached) == len(events)
    return {"valid": True, "goal_reached": world.goal(state), "connected_events": connected,
            "no_repeated_proposition": True, "trace": traces, "causal_links": links,
            "readability_evidence": "not_evaluated"}


def phrase_reuse_audit(events, world):
    """Ground repeated clause-internal phrases in this declared event grammar.

    Predicate markers cover the fixed grammar, not arbitrary English. Reusing
    syntax is allowed when its occurrences have distinct operations/bindings
    and each changes the relevant resource. References carry no novelty credit.
    """
    predicate_markers = {
        "read": {"read", "reads"},
        "sign": {"signs", "signed", "puts", "adds", "signature"},
        "handover": {"gives", "hands", "passes", "gets", "receives"},
    }
    state, evidence, occurrences, counts = world.initial, [], defaultdict(list), Counter()
    invalid = []
    for index, event in enumerate(events):
        op = event.operation
        counts[op] += 1
        after = world.apply(state, op)
        def local(value):
            owners, read, signed = world.decode(value)
            return {"owner": (owners // (world.actors ** op.item)) % world.actors,
                    "read_by_current_holder": bool(read & (1 << op.item)),
                    "signed": bool(signed & (1 << op.item))}
        before_local = local(state)
        after_local = local(after) if after is not None else None
        if after is None:
            invalid.append(index)
        evidence.append({"event_index": index, "operation": asdict(op),
                         "role_bindings": {"agent": op.actor, "resource": op.item,
                                           "recipient": op.recipient if op.recipient >= 0 else None},
                         "effect": {"before": before_local, "after": after_local},
                         "valid_transition": after is not None,
                         "changes_resource": after_local is not None and before_local != after_local})
        words = re.findall(r"[a-z]+", event.text.lower())
        markers = predicate_markers.get(op.kind, set())
        for start in range(len(words)):
            for end in range(start + 2, len(words) + 1):
                phrase = words[start:end]
                occurrences[" ".join(phrase)].append({
                    "event_index": index, "word_start": start, "word_end": end,
                    "contains_predicate": bool(set(phrase) & markers)})
        if after is not None:
            state = after
    repeated, references = [], []
    for phrase, uses in sorted(occurrences.items()):
        if len(uses) < 2:
            continue
        indices = [use["event_index"] for use in uses]
        if not any(use["contains_predicate"] for use in uses):
            references.append({"phrase": phrase, "event_indices": indices,
                               "license": "reference_or_function_words; no progress credit"})
            continue
        distinct = len({events[i].operation for i in indices}) == len(indices)
        grounded = all(evidence[i]["valid_transition"] and evidence[i]["changes_resource"] for i in indices)
        repeated.append({"phrase": phrase, "uses": uses, "licensed": distinct and grounded,
                         "distinct_operation_bindings": distinct,
                         "reason": "distinct bound operations with resource effects" if distinct and grounded
                         else "repeated operation or unsupported resource effect"})
    duplicate_ops = [{"operation": asdict(op), "count": count}
                     for op, count in counts.items() if count > 1]
    return {"passed": not duplicate_ops and not invalid and all(row["licensed"] for row in repeated),
            "scope": "All clause-internal multiword phrases in the fixed event grammar; not a general English semantic parser.",
            "event_groundings": evidence, "repeated_predicate_phrases": repeated,
            "allowed_reference_phrases": references, "repeated_propositions": duplicate_ops,
            "invalid_transition_indices": invalid,
            "progress_units": "Distinct grounded operations, never wording variants or reference repetitions.",
            "grounded_operation_count": len({events[i].operation for i, row in enumerate(evidence)
                                              if row["valid_transition"] and row["changes_resource"]})}


def candidate_record(events, world):
    sentences = [event.text[:1].upper() + event.text[1:] + "." for event in events]
    text = " ".join(sentences)
    letters = tape(text)
    semantics = replay(events, world)
    sentence_tapes = [tape(sentence) for sentence in sentences]
    paired = [{"left": i, "right": j} for i in range(len(sentences))
              for j in range(i + 1, len(sentences)) if sentence_tapes[i] == sentence_tapes[j][::-1]]
    words = text.lower().replace(".", "").split()
    return {
        "text": text, "letters": len(letters), "sentences": sentences,
        "exact": bool(letters) and letters == letters[::-1],
        "normalized_sha256": hashlib.sha256(letters.encode()).hexdigest(),
        "semantics": semantics,
        "phrase_reuse_audit": phrase_reuse_audit(events, world),
        "construction_flags": {
            "complete_sentence_reverse_pairs": paired,
            "self_palindromic_sentences": [i for i, value in enumerate(sentence_tapes) if value == value[::-1]],
            "repeated_sentences": len(set(sentence_tapes)) != len(sentence_tapes),
            "repeated_templates": {k: v for k, v in Counter(e.template for e in events).items() if v > 1},
            "repeated_words": {k: v for k, v in Counter(words).items() if v > 1},
            "boundary_aligned_word_mirror": words == [w[::-1] for w in reversed(words)],
        },
        "novelty": "Generated from stated event grammar; no corpus-absence or global originality claim.",
        "paper_strict_admission": "not_applied; current experiment permits repeated entities and reports construction shortcuts",
        "human_readability": "not_evaluated",
    }


def closest_boundary_events(events, residual, side, limit=3):
    """Explain character conflicts without pretending these events are legal.

    This diagnostic compares lexical surfaces only. Runtime state or repeated
    proposition constraints may still reject the listed alternatives.
    """
    scored = []
    for event in events:
        value = tape(event.text)
        exposed = value[::-1] if side == "right" else value
        common = 0
        while common < min(len(residual), len(exposed)) and residual[common] == exposed[common]:
            common += 1
        mismatch = None
        if common < min(len(residual), len(exposed)):
            mismatch = {"offset": common, "required": residual[common], "proposed": exposed[common]}
        scored.append({"event": event.text, "template": event.template,
                       "exposed_edge": "reversed_suffix" if side == "right" else "forward_prefix",
                       "exposed_letters": exposed, "common_prefix_letters": common,
                       "first_mismatch": mismatch, "semantic_eligibility": "not_checked"})
    scored.sort(key=lambda row: (-row["common_prefix_letters"], len(row["exposed_letters"]), row["event"]))
    selected, templates = [], set()
    for row in scored:
        if row["template"] not in templates:
            selected.append(row)
            templates.add(row["template"])
        if len(selected) == limit:
            break
    return selected


def search(events, world, *, min_letters, max_letters, deadline, seed=0,
           max_states=200000, max_results=12):
    """Match entire event macros with residuals across independent boundaries.

    Left transitions execute forward. Right transitions prepend an event to
    the suffix by taking the exact preimage of its allowable world states.
    Neither clause nor word boundaries have to align with their reflection.
    A palindromic residual can close inside an event.
    """
    by_surface = defaultdict(list)
    for index, event in enumerate(events):
        by_surface[event.text.lower()].append(index)
    tries = WordTries(list(by_surface))
    operations = {operation: index for index, operation in enumerate(dict.fromkeys(e.operation for e in events))}
    # Priority, unique serial, L, R, debt, owner, world, suffix-preimage, used, letters.
    queue = [(0, 0, (), (), "", "", world.initial, world.goal_states, 0, 0)]
    seen, candidates, tapes = set(), [], set()
    counter, serial, frontier = Counter(), 0, []
    item_masks = [sum(1 << index for operation, index in operations.items() if operation.item == item)
                  for item in range(world.items)]
    coverage = {"max_events_in_partial_state": 0, "max_distinct_documents_in_partial_state": 0,
                "max_forward_signed_documents": 0, "max_forward_read_documents": 0,
                "max_left_events": 0, "max_right_events": 0, "deepest_matched_letters_per_side": 0}
    forward_states, event_kinds = set(), set()
    preimage_start, cache_start = world.preimage_calls, world.preimage_cache_hits
    queue_peak = 1
    rng = random.Random(seed)
    while queue and len(seen) < max_states and len(candidates) < max_results and time.monotonic() < deadline:
        _, _, left, right, debt, owner, forward, suffix, used, letters = heapq.heappop(queue)
        counter["states_popped"] += 1
        key = debt, owner, forward, suffix, used, letters
        if key in seen:
            counter["duplicate_state_pops"] += 1
            continue
        seen.add(key)
        matched = (letters - len(debt)) // 2
        _, read_mask, signed_mask = world.decode(forward)
        for field, value in (
            ("max_events_in_partial_state", used.bit_count()),
            ("max_distinct_documents_in_partial_state", sum(bool(used & mask) for mask in item_masks)),
            ("max_forward_signed_documents", signed_mask.bit_count()),
            ("max_forward_read_documents", read_mask.bit_count()),
            ("max_left_events", len(left)), ("max_right_events", len(right)),
            ("deepest_matched_letters_per_side", matched),
        ):
            coverage[field] = max(coverage[field], value)
        forward_states.add(forward)
        if min_letters <= letters <= max_letters and debt == debt[::-1] and suffix & (1 << forward):
            row = candidate_record([events[i] for i in left + right], world)
            norm = tape(row["text"])
            if not row["exact"] or not row["semantics"]["valid"] or not row["semantics"]["goal_reached"]:
                raise AssertionError("construction failed independent replay")
            if row["semantics"]["connected_events"] and row["phrase_reuse_audit"]["passed"] and norm not in tapes:
                tapes.add(norm)
                candidates.append(row)
        if letters >= max_letters:
            continue
        side = "right" if owner == "left" else "left"
        options = (tries.right_candidates(debt, len(by_surface)) if side == "right"
                   else tries.left_candidates(debt, len(by_surface)))
        advanced = 0
        expansion_complete = True
        local_prunes = Counter()
        if not options:
            local_prunes["no_letter_compatible_event"] += 1
        for unit in options:
            if time.monotonic() >= deadline:
                expansion_complete = False
                local_prunes["deadline_incomplete_expansion"] += 1
                break
            unit_tape = tape(unit)
            exposed = unit_tape[::-1] if side == "right" else unit_tape
            match = consume(exposed, debt)
            if match is None:
                continue
            remaining, flipped = match
            next_owner = side if flipped else owner
            if not remaining:
                next_owner = ""
            for index in by_surface[unit]:
                operation = events[index].operation
                bit = 1 << operations[operation]
                counter["letter_compatible_extensions"] += 1
                if used & bit:
                    counter["repeated_proposition_prunes"] += 1
                    local_prunes["repeated_proposition"] += 1
                    continue
                next_length = letters + len(unit_tape)
                if next_length > max_letters:
                    counter["length_prunes"] += 1
                    local_prunes["length_cap"] += 1
                    continue
                if side == "left":
                    next_forward = world.apply(forward, operation)
                    next_suffix = suffix
                    if next_forward is None:
                        counter["forward_precondition_prunes"] += 1
                        local_prunes["forward_precondition"] += 1
                        continue
                else:
                    next_forward = forward
                    next_suffix = world.preimage(operation, suffix)
                    if not next_suffix:
                        counter["backward_preimage_prunes"] += 1
                        local_prunes["backward_preimage"] += 1
                        continue
                next_left = left + (index,) if side == "left" else left
                next_right = (index,) + right if side == "right" else right
                serial += 1
                # Length is an exploration order over exact grammar states,
                # never a replacement for semantic admission or reader quality.
                heapq.heappush(queue, (-next_length + rng.random(), serial, next_left, next_right,
                                      remaining, next_owner, next_forward, next_suffix, used | bit, next_length))
                advanced += 1
                event_kinds.add(operation.kind)
                counter["enqueued"] += 1
                queue_peak = max(queue_peak, len(queue))
        if not advanced:
            frontier.append({"matched_letters_per_side": matched, "unmatched_letters": debt,
                             "debt_owner": owner, "left_events": [events[i].text for i in left],
                             "right_events": [events[i].text for i in right],
                             "left_templates": [events[i].template for i in left],
                             "right_templates": [events[i].template for i in right],
                             "required_side": side, "expansion_complete": expansion_complete,
                             "prune_reasons": dict(local_prunes)})
            frontier.sort(key=lambda item: -item["matched_letters_per_side"])
            del frontier[8:]
    timed_out = time.monotonic() >= deadline
    # Explain only the retained deepest failures, after bounded search. This
    # costs at most 8 x the small event inventory, not one inventory scan per
    # expanded state. It can add a short diagnostic overhead after the cutoff.
    for row in frontier:
        row["closest_boundary_events"] = closest_boundary_events(events, row["unmatched_letters"], row["required_side"])
    coverage.update(forward_world_states_reached=len(forward_states),
                    event_kinds_reached=sorted(event_kinds),
                    scope="Partial supported construction states; these counts do not certify a closed connected paragraph.")
    return {"candidates": candidates, "states": len(seen), "counters": dict(counter),
            "event_coverage": coverage, "queue_peak": queue_peak,
            "backward_preimage_calls": world.preimage_calls - preimage_start,
            "backward_preimage_cache_hits": world.preimage_cache_hits - cache_start,
            "frontier": frontier, "remaining_queue": len(queue),
            "termination": "deadline" if timed_out else "state_limit" if queue and len(seen) >= max_states
            else "result_limit" if queue else "frontier_exhausted",
            "memoization": "Equivalent continuation states retain one witness; not exhaustive surface enumeration."}


ACTOR_GROUPS = [
    ("Nora", "Aron", "Leon", "Noel"), ("Pam", "Mira", "Ada", "Otto"),
    ("Diana", "Nadia", "Ari", "Ira"), ("Hannah", "Eve", "Anna", "Otto"),
    ("Liam", "Maya", "Evan", "Ava"), ("Dan", "Ed", "Ray", "Ivy"),
    ("Mara", "Aram", "Iris", "Aidan"),
]
OBJECT_GROUPS = [("letter", "form", "report"), ("note", "order", "permit"),
                 ("map", "record", "form")]


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(seconds=120, seed=20261009, min_letters=200, max_letters=1500, max_states=200000, checkpoint=None):
    started = time.monotonic()
    world = World()
    scenes = [(actors, objects) for actors in ACTOR_GROUPS for objects in OBJECT_GROUPS]
    random.Random(seed).shuffle(scenes)
    result = {
        "experiment": "argonne_connected_events_20261009", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"seconds": seconds, "seed": seed, "min_letters": min_letters,
                   "max_letters": max_letters, "max_states_per_scene": max_states},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prior_work": ["experiments/world_preimage_character_product_20260921.py",
                       "experiments/first_person_discourse_abba_20261002.py",
                       "experiments/hierarchical_paragraph_abba_20261002.py"],
        "intervention": "Four actors and three resources; forward ownership/read/sign transitions, exact suffix preimages, no repeated proposition, active/passive event alternatives, independently matching clause boundaries.",
        "difference_from_prior": "Expands the earlier tiny world-preimage domain to unbounded-by-clause-count event traces within length/state/time caps; permits natural repeated entities; replaces shared-narrator association with explicit read-before-sign and ownership obligations. This is a controlled domain expansion, not a novel algorithm claim.",
        "goal": "All three documents signed by a holder who read them; every event in one connected participant/resource graph.",
        "prior_failure_avoided": "No sees/spots/stops chain, I-saw/X-was-I pair bank, frozen inherited seed, copied palindrome, or loose-word frequency score.",
        "semantic_limit": "The small symbolic world certifies its specified transitions, not naturalness, an interesting story, or real-world plausibility of every permitted route.",
        "phrase_reuse_policy": "Repeated entity/function references and syntax are allowed. Predicate-containing phrase reuse requires distinct grounded operation bindings and resource effects. Identical operations remain forbidden across paraphrases; extra wording earns no semantic progress.",
        "scenes": [], "candidates": [], "readability_evidence": "not_evaluated",
    }
    for index, (actors, objects) in enumerate(scenes):
        if time.monotonic() >= started + seconds:
            break
        events = build_events(actors, objects)
        found = search(events, world, min_letters=min_letters, max_letters=max_letters,
                       deadline=started + seconds, seed=seed + index, max_states=max_states)
        scene = {"actors": actors, "objects": objects, "event_surfaces": len(events),
                 "distinct_propositions": len({event.operation for event in events}), **found}
        result["scenes"].append(scene)
        result["candidates"].extend({"scene_index": index, **row} for row in found["candidates"])
        result["elapsed_seconds"] = time.monotonic() - started
        result["status"] = "searching"
        if checkpoint:
            checkpoint(result)
    result.update(status="completed_bounded_probe", elapsed_seconds=time.monotonic() - started,
                  scenes_planned=len(scenes), scenes_attempted=len(result["scenes"]),
                  exact_candidates=len(result["candidates"]),
                  longest_exact=max((row["letters"] for row in result["candidates"]), default=0))
    result["scene_termination_counts"] = dict(Counter(scene["termination"] for scene in result["scenes"]))
    result["overall_termination"] = ("deadline" if time.monotonic() >= started + seconds
                                     else "all_planned_scenes_attempted")
    result["deepest_matched_letters_per_side"] = max(
        (scene["event_coverage"]["deepest_matched_letters_per_side"] for scene in result["scenes"]), default=0)
    result["maximum_partial_event_count"] = max(
        (scene["event_coverage"]["max_events_in_partial_state"] for scene in result["scenes"]), default=0)
    result["budget_accounting"] = "Global cooperative cutoff includes world setup and search; at most eight small lexical-conflict scans per attempted scene can finish after the cutoff. States are unique continuation keys; popped duplicates and queue peak are separate."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--min-letters", type=int, default=200)
    parser.add_argument("--max-letters", type=int, default=1500)
    parser.add_argument("--max-states", type=int, default=200000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to replace an existing result")
    if args.seconds <= 0 or not 1 <= args.min_letters <= args.max_letters or args.max_states <= 0:
        parser.error("positive time/state limits and an ordered positive length range are required")
    result = run(args.seconds, args.seed, args.min_letters, args.max_letters, args.max_states,
                 checkpoint=lambda value: atomic_write(args.out, value))
    atomic_write(args.out, result)
    print(json.dumps({key: result[key] for key in ("status", "exact_candidates", "longest_exact", "scenes_attempted", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
