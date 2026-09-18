"""Two-sided partial-word transducer for typed reason/explanation prose.

The transducer operates on one connected eight-slot discourse plan.  It keeps
forward and reversed lexical-trie nodes for the active words and emits one
equal character pair at a time.  A syntax role advances only when its current
word reaches a trie terminal.  Complete surfaces are independently parsed and
then checked by the central admission gate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 30, 60


@dataclass(frozen=True)
class Symbol:
    name: str
    features: tuple[tuple[str, str], ...] = ()
    def feature(self, key: str, default: str = "") -> str:
        return dict(self.features).get(key, default)


def sym(name: str, **features: str) -> Symbol:
    return Symbol(name, tuple(sorted(features.items())))


@dataclass(frozen=True)
class ParseNode:
    symbol: Symbol
    terminal: str = ""
    production: str = ""
    children: tuple["ParseNode", ...] = ()


@dataclass(frozen=True)
class EventFrame:
    name: str
    subject: str
    predicate: str
    cause_subject: str
    cause_adjective: str


EVENTS = (
    EventFrame("lamp_brightness", "lamp", "glows", "room", "bright"),
    EventFrame("plant_moisture", "plant", "grows", "soil", "moist"),
    EventFrame("fire_fuel", "fire", "burns", "wood", "dry"),
    EventFrame("bell_wind", "bell", "rings", "wind", "strong"),
)


@dataclass(frozen=True)
class Slot:
    role: str
    words: tuple[str, ...]
    type: str
    number: str


class Trie:
    """Tiny immutable-addressed trie used from both surface directions."""
    def __init__(self, words: tuple[str, ...], *, reverse: bool = False):
        children = [{}]
        terminal = [False]
        self.words = words
        for word in words:
            sequence = word[::-1] if reverse else word
            node = 0
            for char in sequence:
                nxt = children[node].get(char)
                if nxt is None:
                    nxt = len(children); children[node][char] = nxt
                    children.append({}); terminal.append(False)
                node = nxt
            terminal[node] = True
        self.children = tuple(dict(row) for row in children)
        self.terminal = tuple(terminal)

    def next_chars(self, node: int) -> tuple[str, ...]:
        return tuple(sorted(self.children[node]))

    def advance(self, node: int, char: str) -> int | None:
        return self.children[node].get(char)


@dataclass(frozen=True)
class State:
    event_index: int
    slots: tuple[Slot, ...]
    left_index: int
    right_index: int
    left_words: tuple[str, ...]
    right_words: tuple[str, ...]
    left_prefix: str
    right_reverse_prefix: str
    left_node: int
    right_node: int
    length: int
    pair_trace: tuple[tuple[int, str, str], ...]
    syntax_trace: tuple[str, ...]


def frame_slots(frame: EventFrame) -> tuple[Slot, ...]:
    return (
        Slot("subject_det", ("the", "a"), "determiner", "sing"),
        Slot("subject", (frame.subject,), "personified_subject", "sing"),
        Slot("predicate", (frame.predicate,), "intransitive_predicate", "sing"),
        Slot("connector", ("because",), "causal_connector", "none"),
        Slot("cause_det", ("the", "a"), "determiner", "sing"),
        Slot("cause_subject", (frame.cause_subject,), "cause", "sing"),
        Slot("copula", ("is",), "copula", "sing"),
        Slot("cause_adjective", (frame.cause_adjective,), "cause_property", "sing"),
    )


def plan_words(frame: EventFrame, *, first: bool = True) -> tuple[str, ...]:
    """Generate a normal surface from the grammar, never from a fixed string."""
    return tuple(slot.words[0] for slot in frame_slots(frame))


def parse_complete(text: str):
    """Independent backtracking parse over every authored event frame."""
    if text != text.strip() or re.sub(r"[a-z ,;.!?]", "", text.lower()):
        return None
    tokens = tuple(re.findall(r"[a-z]+", text.lower()))
    memo = {}

    def parses(event_index: int, slot_index: int, offset: int):
        key = (event_index, slot_index, offset)
        if key in memo:
            return memo[key]
        slots = frame_slots(EVENTS[event_index])
        if slot_index == len(slots):
            answer = ((ParseNode(sym("RELATION", event=EVENTS[event_index].name),
                                  production=f"RELATION:{EVENTS[event_index].name}"), offset),)
            memo[key] = answer
            return answer
        answers = []
        slot = slots[slot_index]
        for word in slot.words:
            if offset >= len(tokens) or tokens[offset] != word:
                continue
            for child, end in parses(event_index, slot_index + 1, offset + 1):
                node = ParseNode(sym("SLOT", role=slot.role, type=slot.type,
                                     number=slot.number, form=word), terminal=word,
                                 production=f"SLOT:{slot.role}:{word}")
                answers.append((ParseNode(sym("RELATION_PREFIX", event=EVENTS[event_index].name),
                                          production=f"PREFIX:{slot_index}", children=(node, child)), end))
        memo[key] = tuple(answers)
        return memo[key]

    for event_index in range(len(EVENTS)):
        for tree, end in parses(event_index, 0, 0):
            if end == len(tokens):
                return tree
    return None


def exact_audit(text: str) -> dict:
    tape = "".join(c for c in text.lower() if "a" <= c <= "z")
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2)
                  if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape),
            "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def semantic_witness(tree) -> dict:
    relation = ""
    cursor = tree
    while cursor:
        if cursor.symbol.name == "RELATION":
            relation = cursor.symbol.feature("event")
            break
        cursor = cursor.children[-1] if cursor.children else None
    words = []
    node = tree
    while node and node.children:
        child = node.children[0]
        if child.terminal:
            words.append(child.terminal)
        node = node.children[1] if len(node.children) > 1 else None
    frame = next((item for item in EVENTS if item.name == relation), None)
    roles_ok = bool(frame and len(words) == 8 and words[3] == "because" and words[6] == "is")
    return {"discourse_function": "causal_explanation", "event": relation,
            "antecedent_subject": frame.subject if frame else "",
            "cause_subject": frame.cause_subject if frame else "",
            "agreement_ok": bool(roles_ok and words[1] == frame.subject and words[2] == frame.predicate),
            "valency_ok": bool(roles_ok and words[5] == frame.cause_subject and words[7] == frame.cause_adjective),
            "subject_action_ok": roles_ok, "complete_tree": bool(tree)}


def audit(text: str, kind: str, trace=()):
    tree = parse_complete(text)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    witness = semantic_witness(tree)
    codes = [key for key, value in central.items() if not value]
    if tree is None: codes.append("independent_complete_reparse_failed")
    for key, code in (("agreement_ok", "agreement_failure"), ("valency_ok", "valency_failure"),
                      ("subject_action_ok", "subject_action_semantics_failure")):
        if not witness[key]: codes.append(code)
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact_audit(text),
            "independent_parse": tree is not None, "feature_witness": witness,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "pair_trace": [list(item) for item in trace],
            "reader_status": "unreviewed; programmatic checks do not certify readability"}


def trie_for(slot: Slot, *, reverse=False) -> Trie:
    return Trie(slot.words, reverse=reverse)


def render_partial(state: State) -> str:
    left = list(state.left_words)
    right = list(state.right_words)
    if state.left_prefix: left.append(state.left_prefix)
    if state.right_reverse_prefix: right.insert(0, state.right_reverse_prefix[::-1])
    left_text, right_text = " ".join(left), " ".join(right)
    if left_text and right_text: return f"{left_text} … {right_text}"
    return left_text or right_text


def render_final(state: State) -> str:
    return " ".join(state.left_words + tuple(reversed(state.right_words)))


def start_states() -> tuple[State, ...]:
    return tuple(State(i, frame_slots(frame), 0, 7, (), (), "", "", 0, 0, 0, (),
                       (f"S:relation:{frame.name}",)) for i, frame in enumerate(EVENTS))


def step(state: State) -> tuple[State, ...]:
    """Advance both active lexical tries by exactly one equal character."""
    if state.left_index >= state.right_index:
        return ()
    left_slot, right_slot = state.slots[state.left_index], state.slots[state.right_index]
    left_trie = trie_for(left_slot)
    right_trie = trie_for(right_slot, reverse=True)
    left_chars = left_trie.next_chars(state.left_node)
    right_chars = right_trie.next_chars(state.right_node)
    outputs = []
    for char in sorted(set(left_chars) & set(right_chars)):
        left_node = left_trie.advance(state.left_node, char)
        right_node = right_trie.advance(state.right_node, char)
        if left_node is None or right_node is None:
            continue
        left_prefix = state.left_prefix + char
        right_prefix = state.right_reverse_prefix + char
        left_finishes = left_trie.terminal[left_node]
        right_finishes = right_trie.terminal[right_node]
        left_choices = (False, True) if left_finishes else (False,)
        right_choices = (False, True) if right_finishes else (False,)
        for finish_left in left_choices:
            for finish_right in right_choices:
                next_left_index, next_right_index = state.left_index, state.right_index
                next_left_words, next_right_words = state.left_words, state.right_words
                next_left_prefix, next_right_prefix = left_prefix, right_prefix
                next_left_node, next_right_node = left_node, right_node
                syntax = state.syntax_trace
                if finish_left:
                    next_left_words += (left_prefix,); next_left_index += 1
                    next_left_prefix, next_left_node = "", 0
                    syntax += (f"left_complete:{left_slot.role}:{left_prefix}",)
                if finish_right:
                    next_right_words += (right_prefix[::-1],); next_right_index -= 1
                    next_right_prefix, next_right_node = "", 0
                    syntax += (f"right_complete:{right_slot.role}:{right_prefix[::-1]}",)
                # With an even slot plan, the two pointers must cross on the
                # same paired character; entering the same slot would assign
                # one connected lexical constituent from both sides.
                if next_left_index > next_right_index or next_left_index < next_right_index:
                    outputs.append(State(state.event_index, state.slots, next_left_index, next_right_index,
                                         next_left_words, next_right_words, next_left_prefix, next_right_prefix,
                                         next_left_node, next_right_node, state.length + 1,
                                         state.pair_trace + ((state.length + 1, char, char),), syntax))
    return tuple(outputs)


def attempts(state):
    if state.left_index >= state.right_index:
        return {"available": False, "reason": "pointers_crossed_or_met"}
    left_slot, right_slot = state.slots[state.left_index], state.slots[state.right_index]
    return {"available": True, "left_role": left_slot.role, "right_role": right_slot.role,
            "left_prefix": state.left_prefix, "right_reverse_prefix": state.right_reverse_prefix,
            "left_next": list(trie_for(left_slot).next_chars(state.left_node)),
            "right_next_reversed": list(trie_for(right_slot, reverse=True).next_chars(state.right_node)),
            "matching_next": sorted(set(trie_for(left_slot).next_chars(state.left_node)) &
                                    set(trie_for(right_slot, reverse=True).next_chars(state.right_node)))}


def transduce(max_states=100000):
    queue, seen = deque(start_states()), set(); stats = {"states": 0, "character_pairs": 0,
        "word_completions": 0, "complete_trees": 0, "exact_closures": 0, "dead_char_frontiers": 0}
    deepest = None; deepest_len = -1; first_viable = None; first_rejection = None; exact = []
    while queue and stats["states"] < max_states:
        state = queue.pop(); key = (state.event_index, state.left_index, state.right_index,
            state.left_words, state.right_words, state.left_prefix, state.right_reverse_prefix,
            state.left_node, state.right_node, state.length)
        if key in seen: continue
        seen.add(key); stats["states"] += 1
        offered = attempts(state)
        ledger = {"event": EVENTS[state.event_index].name, "rendered_partial": render_partial(state),
                  "left_index": state.left_index, "right_index": state.right_index,
                  "left_prefix": state.left_prefix, "right_reverse_prefix": state.right_reverse_prefix,
                  "length": state.length, "pair_trace": [list(item) for item in state.pair_trace],
                  "syntax_trace": list(state.syntax_trace), "attempts": offered}
        if state.length > deepest_len: deepest_len, deepest = state.length, ledger
        next_states = step(state)
        if state.length and next_states and first_viable is None: first_viable = ledger
        if state.length and not next_states and first_rejection is None:
            first_rejection = {**ledger, "rejection": "no equal next trie character or safe word-boundary transition"}
        if not next_states: stats["dead_char_frontiers"] += 1
        stats["character_pairs"] += len(next_states)
        for next_state in next_states:
            if next_state.length > state.length:
                completed = len(next_state.syntax_trace) - len(state.syntax_trace)
                stats["word_completions"] += max(0, completed)
            if next_state.left_index > next_state.right_index:
                stats["complete_trees"] += 1
                text = render_final(next_state); row = audit(text, "complete_partial_word_transducer", next_state.pair_trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact.append(row)
            else:
                queue.append(next_state)
    return {"stats": stats, "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": exact, "deepest_state": deepest,
            "first_viable_state": first_viable, "first_rejection": first_rejection}


def run(max_states=100000):
    result = transduce(max_states)
    return {"status": "partial_word_reason_transducer_intersection",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_states": max_states,
                       "single_connected_discourse_plan": True, "partial_word_transducer": True,
                       "forward_and_reverse_lexical_tries": True, "equal_character_pair_emission": True,
                       "advance_role_only_on_word_completion": True, "independent_complete_reparse": True,
                       "endpoint_scaffold_gate": True, "corpus_generation": False,
                       "preauthored_full_control": False},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "grammar_sha256": sha256(json.dumps([frame.__dict__ for frame in EVENTS], sort_keys=True).encode()).hexdigest(),
                           "material": "authored typed reason/explanation frames; no catalogue or known palindrome material"},
            "rendered_candidates": result["exact_closures"], **result,
            "reader_facing_next_operator": "Use the recorded trie-boundary mismatch to author a new ordinary cause-property frame, then rerun the same partial-word transducer.",
            "scope": "Zero-closure transducer run is diagnostic; no reader-facing palindrome is claimed."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
