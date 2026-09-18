"""Typed boundary-role relexicalization with a partial-word zipper.

This is a fresh construction.  It first enumerates semantically compatible
outer roles (subject determiner/subject and final object complement), then
traverses one connected causal explanation plan with forward and reversed
lexical tries.  Characters are emitted in equal pairs; a role pointer moves
only after its current word reaches a trie terminal.
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
class Frame:
    name: str
    subject: str
    predicate: str
    cause: str
    cause_verb: str
    object: str
    subject_det: str
    object_det: str
    object_number: str


FRAMES = (
    Frame("signal_detection", "signal", "flashes", "sensor", "detect", "camera", "a", "a", "sing"),
    Frame("beacon_recording", "beacon", "glows", "monitor", "record", "data", "a", "the", "sing"),
    Frame("alarm_notice", "alarm", "rings", "guard", "notice", "smoke", "an", "the", "sing"),
    Frame("plant_wind", "plant", "bends", "wind", "move", "leaves", "a", "the", "plur"),
    Frame("tablet_data", "tablet", "works", "user", "read", "data", "a", "the", "sing"),
)


@dataclass(frozen=True)
class Slot:
    role: str
    words: tuple[str, ...]
    type: str
    number: str


def slots(frame: Frame) -> tuple[Slot, ...]:
    # Ten slots keep the explanation plan two-sided without a hidden central
    # word.  The modal is typed as an ability relation, not padding.
    return (
        Slot("subject_det", (frame.subject_det, "the"), "determiner", "sing"),
        Slot("subject", (frame.subject,), "event_subject", "sing"),
        Slot("predicate", (frame.predicate,), "intransitive_predicate", "sing"),
        Slot("connector", ("because",), "causal_connector", "none"),
        Slot("cause_det", ("the",), "determiner", "sing"),
        Slot("cause_subject", (frame.cause,), "cause_agent", "sing"),
        Slot("modal", ("can",), "ability_modal", "none"),
        Slot("cause_verb", (frame.cause_verb,), "transitive_predicate", "sing"),
        Slot("object_det", (frame.object_det,), "determiner", "sing"),
        Slot("object", (frame.object,), "event_object", frame.object_number),
    )


class Trie:
    def __init__(self, words: tuple[str, ...], reverse: bool = False):
        children, terminal = [{}], [False]
        for word in words:
            node = 0
            for char in (word[::-1] if reverse else word):
                nxt = children[node].get(char)
                if nxt is None:
                    nxt = len(children); children[node][char] = nxt
                    children.append({}); terminal.append(False)
                node = nxt
            terminal[node] = True
        self.children, self.terminal = tuple(children), tuple(terminal)
    def chars(self, node): return tuple(sorted(self.children[node]))
    def advance(self, node, char): return self.children[node].get(char)


@dataclass(frozen=True)
class State:
    frame_index: int
    slot_rows: tuple[Slot, ...]
    left_index: int
    right_index: int
    left_words: tuple[str, ...]
    right_words: tuple[str, ...]
    left_prefix: str
    right_reverse_prefix: str
    left_node: int
    right_node: int
    length: int
    pairs: tuple[tuple[int, str], ...]
    role_trace: tuple[str, ...]


def generated_surface(frame: Frame) -> str:
    return " ".join(slot.words[0] for slot in slots(frame))


def parse_complete(text: str):
    if text != text.strip() or re.sub(r"[a-z ,;.!?]", "", text.lower()): return None
    tokens = tuple(re.findall(r"[a-z]+", text.lower()))
    for index, frame in enumerate(FRAMES):
        frame_slots = slots(frame); memo = {}
        def parse_slot(slot_index, offset):
            key = (slot_index, offset)
            if key in memo: return memo[key]
            if slot_index == len(frame_slots):
                answer = ((ParseNode(sym("RELATION", event=frame.name), production=f"RELATION:{frame.name}"), offset),)
                memo[key] = answer; return answer
            slot = frame_slots[slot_index]; answers = []
            for word in slot.words:
                if offset >= len(tokens) or tokens[offset] != word: continue
                for tail, end in parse_slot(slot_index + 1, offset + 1):
                    leaf = ParseNode(sym("SLOT", role=slot.role, type=slot.type, number=slot.number, form=word), terminal=word, production=f"SLOT:{slot.role}:{word}")
                    answers.append((ParseNode(sym("PREFIX", event=frame.name), production=f"PREFIX:{slot_index}", children=(leaf, tail)), end))
            memo[key] = tuple(answers); return memo[key]
        for tree, end in parse_slot(0, 0):
            if end == len(tokens): return tree
    return None


def semantic_witness(tree):
    event = ""; words = []; cursor = tree
    while cursor:
        if cursor.symbol.name == "RELATION": event = cursor.symbol.feature("event")
        if cursor.children:
            child = cursor.children[0]
            if child.terminal: words.append(child.terminal)
            cursor = cursor.children[-1]
        else: break
    frame = next((item for item in FRAMES if item.name == event), None)
    okay = bool(frame and len(words) == 10 and words[3] == "because" and words[6] == "can")
    return {"discourse_function": "causal_explanation", "event": event,
            "agreement_ok": bool(okay and words[1] == frame.subject),
            "valency_ok": bool(okay and words[2] == frame.predicate and words[5] == frame.cause and words[7] == frame.cause_verb and words[9] == frame.object),
            "subject_action_ok": bool(okay), "complete_tree": bool(tree)}


def exact_audit(text):
    tape = "".join(c for c in text.lower() if "a" <= c <= "z")
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest()}


def audit(text, kind, pairs=()):
    tree = parse_complete(text); witness = semantic_witness(tree)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if tree is None: codes.append("independent_complete_reparse_failed")
    for key, code in (("agreement_ok", "agreement_failure"), ("valency_ok", "valency_failure"), ("subject_action_ok", "subject_action_semantics_failure")):
        if not witness[key]: codes.append(code)
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact_audit(text), "independent_parse": tree is not None,
            "feature_witness": witness, "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "pair_trace": [list(item) for item in pairs],
            "reader_status": "unreviewed; programmatic checks do not certify readability"}


def outer_role_compatibilities():
    """Jointly enumerate legal outer roles before any interior traversal."""
    rows = []
    for frame in FRAMES:
        frame_slots = slots(frame); left = frame_slots[0]; right = frame_slots[-1]
        for left_word in left.words:
            for right_word in right.words:
                forward, reverse = Trie((left_word,)), Trie((right_word,), reverse=True)
                node_l = node_r = 0; prefix = ""
                while True:
                    common = set(forward.chars(node_l)) & set(reverse.chars(node_r))
                    if not common: break
                    char = sorted(common)[0]; node_l = forward.advance(node_l, char); node_r = reverse.advance(node_r, char); prefix += char
                rows.append({"event": frame.name, "left_role": left.role, "left_word": left_word,
                             "right_role": right.role, "right_word": right_word,
                             "compatible_prefix": prefix, "prefix_letters": len(prefix),
                             "left_word_complete": forward.terminal[node_l], "right_word_complete": reverse.terminal[node_r]})
    return rows


def render_partial(state):
    left, right = list(state.left_words), list(state.right_words)
    if state.left_prefix: left.append(state.left_prefix)
    if state.right_reverse_prefix: right.insert(0, state.right_reverse_prefix[::-1])
    left_text, right_text = " ".join(left), " ".join(right)
    return f"{left_text} … {right_text}" if left_text and right_text else left_text or right_text


def step(state):
    if state.left_index >= state.right_index: return ()
    left_slot, right_slot = state.slot_rows[state.left_index], state.slot_rows[state.right_index]
    left_trie, right_trie = Trie(left_slot.words), Trie(right_slot.words, reverse=True)
    common = set(left_trie.chars(state.left_node)) & set(right_trie.chars(state.right_node)); output = []
    for char in sorted(common):
        ln, rn = left_trie.advance(state.left_node, char), right_trie.advance(state.right_node, char)
        lp, rp = state.left_prefix + char, state.right_reverse_prefix + char
        left_finish = left_trie.terminal[ln]; right_finish = right_trie.terminal[rn]
        for finish_l in ((False, True) if left_finish else (False,)):
            for finish_r in ((False, True) if right_finish else (False,)):
                li, ri = state.left_index, state.right_index; lw, rw = state.left_words, state.right_words; lpx, rpx, lnode, rnode = lp, rp, ln, rn; trace = state.role_trace
                if finish_l:
                    lw += (lp,); li += 1; lpx, lnode = "", 0; trace += (f"left_complete:{left_slot.role}:{lp}",)
                if finish_r:
                    rw += (rp[::-1],); ri -= 1; rpx, rnode = "", 0; trace += (f"right_complete:{right_slot.role}:{rp[::-1]}",)
                if li == ri: continue
                output.append(State(state.frame_index, state.slot_rows, li, ri, lw, rw, lpx, rpx, lnode, rnode,
                                    state.length + 1, state.pairs + ((state.length + 1, char),), trace))
    return tuple(output)


def transduce(max_states=100000):
    queue = deque(State(i, slots(frame), 0, 9, (), (), "", "", 0, 0, 0, (), (f"S:explanation:{frame.name}",)) for i, frame in enumerate(FRAMES)); seen = set()
    stats = {"states": 0, "character_pairs": 0, "word_completions": 0, "complete_trees": 0, "exact_closures": 0, "dead_frontiers": 0}
    deepest = None; deepest_len = -1; contradiction = None; exact = []
    while queue and stats["states"] < max_states:
        state = queue.pop(); key = (state.frame_index, state.left_index, state.right_index, state.left_words, state.right_words, state.left_prefix, state.right_reverse_prefix, state.left_node, state.right_node, state.length)
        if key in seen: continue
        seen.add(key); stats["states"] += 1; next_states = step(state)
        ledger = {"event": FRAMES[state.frame_index].name, "rendered_partial": render_partial(state), "left_index": state.left_index, "right_index": state.right_index, "left_prefix": state.left_prefix, "right_reverse_prefix": state.right_reverse_prefix, "length": state.length, "pair_trace": [list(item) for item in state.pairs], "role_trace": list(state.role_trace), "next_pair_count": len(next_states)}
        if state.length > deepest_len: deepest_len, deepest = state.length, ledger
        if not next_states:
            stats["dead_frontiers"] += 1; contradiction = contradiction or {**ledger, "rejection": "compatibility automaton has no equal next character"}
        stats["character_pairs"] += len(next_states)
        for nxt in next_states:
            completed = len(nxt.role_trace) - len(state.role_trace); stats["word_completions"] += max(0, completed)
            if nxt.left_index > nxt.right_index:
                stats["complete_trees"] += 1; text = " ".join(nxt.left_words + tuple(reversed(nxt.right_words))); row = audit(text, "complete_boundary_relexicalized_transducer", nxt.pairs)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact.append(row)
            else: queue.append(nxt)
    return {"stats": stats, "states_exhausted": not queue and stats["states"] < max_states, "exact_closures": exact, "deepest_state_ledger": deepest, "deepest_contradiction": contradiction}


def run(max_states=100000):
    result = transduce(max_states)
    ordinary_reports = [dict(audit(generated_surface(frame), "grammar_generated_diagnostic"), promotion_status="not_promoted_without_transducer_closure") for frame in FRAMES]
    return {"status": "typed_boundary_relexicalization_reason_transducer", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_states": max_states, "single_connected_discourse_plan": True, "partial_word_transducer": True, "forward_and_reverse_lexical_tries": True, "boundary_role_compatibility_automaton": True, "equal_character_pair_emission": True, "advance_role_only_on_word_completion": True, "independent_complete_reparse": True, "endpoint_scaffold_gate": True, "preauthored_full_control": False, "corpus_generation": False}, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": sha256(json.dumps([frame.__dict__ for frame in FRAMES], sort_keys=True).encode()).hexdigest(), "material": "authored productive causal explanation frames; no catalogue or known palindrome material", "known_tape_check": "every generated report is normalized and checked against the shared exclusion catalogue"}, "outer_role_compatibilities": outer_role_compatibilities(), "ordinary_generated_reports": ordinary_reports, "rendered_candidates": result["exact_closures"], **result, "reader_facing_next_operator": "Author a new semantically licensed cause-object role whose full reversed trie prefix is compatible with the subject boundary, then rerun this transducer.", "scope": "Zero-closure finite transducer run is diagnostic; no reader-facing palindrome is claimed."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
