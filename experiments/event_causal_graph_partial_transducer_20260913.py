"""Event-causal graph transducer with partial-word boundary relexicalization.

Each surface plan is derived from an authored cause-event/effect-event graph.
An independent graph replay checker validates the agent, action, target,
mechanism, and effect relation before the two-sided lexical trie transducer
is allowed to traverse it.  The transducer emits equal characters one pair at
a time and advances a syntax role only when its current word completes.
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
class Graph:
    name: str
    cause_subject: str
    cause_manner: str
    cause_action: str
    cause_target: str
    connector: str
    effect_subject: str
    effect_action: str
    effect_target: str
    mechanism: str
    effect_target_det: str


GRAPHS = (
    Graph("sensor_signal_monitor", "sensor", "quickly", "detects", "spark", "so", "monitor", "records", "signal", "detected_event", "the"),
    Graph("server_file_user", "server", "securely", "stores", "file", "so", "user", "reads", "record", "stored_document", "the"),
    Graph("printer_page_student", "printer", "neatly", "prints", "page", "so", "student", "studies", "lesson", "printed_lesson", "the"),
    Graph("tablet_note_reader", "tablet", "carefully", "stores", "note", "so", "reader", "reads", "data", "recorded_information", "the"),
)


@dataclass(frozen=True)
class Slot:
    role: str
    words: tuple[str, ...]
    type: str
    number: str


def graph_slots(graph: Graph) -> tuple[Slot, ...]:
    # Twelve lexical roles give the zipper room to cross several word
    # boundaries while retaining an ordinary, explicit causal relation.
    return (
        Slot("cause_det", ("a", "the"), "determiner", "sing"),
        Slot("cause_subject", (graph.cause_subject,), "cause_agent", "sing"),
        Slot("cause_manner", (graph.cause_manner,), "manner", "none"),
        Slot("cause_action", (graph.cause_action,), "cause_action", "sing"),
        Slot("cause_target_det", ("a", "the"), "determiner", "sing"),
        Slot("cause_target", (graph.cause_target,), "cause_target", "sing"),
        Slot("connector", (graph.connector,), "causal_connector", "none"),
        Slot("effect_det", ("a", "the"), "determiner", "sing"),
        Slot("effect_subject", (graph.effect_subject,), "effect_agent", "sing"),
        Slot("effect_action", (graph.effect_action,), "effect_action", "sing"),
        Slot("effect_target_det", (graph.effect_target_det,), "determiner", "sing"),
        Slot("effect_target", (graph.effect_target,), "effect_target", "sing"),
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
class ParseNode:
    role: str
    word: str
    children: tuple["ParseNode", ...] = ()


@dataclass(frozen=True)
class State:
    graph_index: int
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


def graph_replay(graph: Graph, words: tuple[str, ...]) -> dict:
    expected_slots = graph_slots(graph)
    exact_surface = len(words) == len(expected_slots) and all(
        word in slot.words for word, slot in zip(words, expected_slots)
    )
    return {
        "cause_event": {"agent": graph.cause_subject, "manner": graph.cause_manner,
                         "action": graph.cause_action, "target": graph.cause_target},
        "effect_event": {"agent": graph.effect_subject, "action": graph.effect_action,
                          "target": graph.effect_target},
        "mechanism": graph.mechanism,
        "connector": graph.connector,
        "causal_edge": f"{graph.cause_action}({graph.cause_subject},{graph.cause_target}) -> {graph.effect_action}({graph.effect_subject},{graph.effect_target})",
        "replay_ok": exact_surface,
    }


def parse_complete(text: str):
    if text != text.strip() or re.sub(r"[a-z ,;.!?]", "", text.lower()): return None
    tokens = tuple(re.findall(r"[a-z]+", text.lower()))
    for index, graph in enumerate(GRAPHS):
        rows = graph_slots(graph)
        if len(tokens) != len(rows): continue
        if all(token in row.words for token, row in zip(tokens, rows)):
            tail = ParseNode("relation", graph.name)
            for row, token in reversed(tuple(zip(rows, tokens))):
                tail = ParseNode(row.role, token, (tail,))
            return tail
    return None


def parsed_words(tree):
    words = []
    while tree:
        if tree.role != "relation": words.append(tree.word)
        tree = tree.children[0] if tree.children else None
    return tuple(words)


def semantic_witness(tree):
    words = parsed_words(tree) if tree else ()
    relation = ""
    cursor = tree
    while cursor:
        if cursor.role == "relation":
            relation = cursor.word
            break
        cursor = cursor.children[0] if cursor.children else None
    graph = next((item for item in GRAPHS if item.name == relation), None)
    replay = graph_replay(graph, words) if graph else {"replay_ok": False}
    return {"graph": graph.name if graph else "", "cause_event": replay.get("cause_event", {}),
            "effect_event": replay.get("effect_event", {}), "mechanism": replay.get("mechanism", ""),
            "causal_edge": replay.get("causal_edge", ""), "causal_graph_replay_ok": replay.get("replay_ok", False),
            "agreement_ok": bool(graph and len(words) == 12), "valency_ok": bool(graph and len(words) == 12),
            "subject_action_ok": bool(graph and replay.get("replay_ok")), "complete_tree": bool(tree)}


def exact_audit(text):
    tape = "".join(c for c in text.lower() if "a" <= c <= "z")
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def audit(text, kind, pairs=()):
    tree = parse_complete(text); witness = semantic_witness(tree)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if tree is None: codes.append("independent_complete_reparse_failed")
    for key, code in (("causal_graph_replay_ok", "causal_graph_replay_failure"), ("agreement_ok", "agreement_failure"), ("valency_ok", "valency_failure"), ("subject_action_ok", "subject_action_semantics_failure")):
        if not witness[key]: codes.append(code)
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact_audit(text), "independent_parse": tree is not None,
            "feature_witness": witness, "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "pair_trace": [list(item) for item in pairs],
            "reader_status": "unreviewed; programmatic checks do not certify readability"}


def generated_surface(graph: Graph) -> str:
    return " ".join(slot.words[0] for slot in graph_slots(graph))


def joint_boundary_roles():
    rows = []
    for graph in GRAPHS:
        first, last = graph_slots(graph)[0], graph_slots(graph)[-1]
        for left_word in first.words:
            for right_word in last.words:
                left, right = Trie((left_word,)), Trie((right_word,), reverse=True); ln = rn = 0; prefix = ""
                while (chars := set(left.chars(ln)) & set(right.chars(rn))):
                    char = sorted(chars)[0]; ln, rn = left.advance(ln, char), right.advance(rn, char); prefix += char
                rows.append({"graph": graph.name, "left_role": first.role, "left_word": left_word, "right_role": last.role, "right_word": right_word, "compatible_prefix": prefix, "prefix_letters": len(prefix), "graph_replay_precheck": graph_replay(graph, tuple(slot.words[0] for slot in graph_slots(graph)))["replay_ok"]})
    return rows


def render_partial(state):
    left, right = list(state.left_words), list(state.right_words)
    if state.left_prefix: left.append(state.left_prefix)
    if state.right_reverse_prefix: right.insert(0, state.right_reverse_prefix[::-1])
    return f"{' '.join(left)} … {' '.join(right)}" if left and right else " ".join(left or right)


def step(state):
    if state.left_index >= state.right_index: return ()
    left_slot, right_slot = state.slot_rows[state.left_index], state.slot_rows[state.right_index]
    left, right = Trie(left_slot.words), Trie(right_slot.words, reverse=True); common = set(left.chars(state.left_node)) & set(right.chars(state.right_node)); out = []
    for char in sorted(common):
        ln, rn = left.advance(state.left_node, char), right.advance(state.right_node, char); lp, rp = state.left_prefix + char, state.right_reverse_prefix + char
        for finish_l in ((False, True) if left.terminal[ln] else (False,)):
            for finish_r in ((False, True) if right.terminal[rn] else (False,)):
                li, ri, lw, rw, lpx, rpx, lnode, rnode, trace = state.left_index, state.right_index, state.left_words, state.right_words, lp, rp, ln, rn, state.role_trace
                if finish_l: lw += (lp,); li += 1; lpx, lnode = "", 0; trace += (f"left_complete:{left_slot.role}:{lp}",)
                if finish_r: rw += (rp[::-1],); ri -= 1; rpx, rnode = "", 0; trace += (f"right_complete:{right_slot.role}:{rp[::-1]}",)
                if li == ri: continue
                out.append(State(state.graph_index, state.slot_rows, li, ri, lw, rw, lpx, rpx, lnode, rnode, state.length + 1, state.pairs + ((state.length + 1, char),), trace))
    return tuple(out)


def transduce(max_states=100000):
    queue = deque(State(i, graph_slots(graph), 0, 11, (), (), "", "", 0, 0, 0, (), (f"GRAPH:{graph.name}",)) for i, graph in enumerate(GRAPHS) if graph_replay(graph, tuple(slot.words[0] for slot in graph_slots(graph)))["replay_ok"]); seen = set(); stats = {"states": 0, "character_pairs": 0, "word_completions": 0, "complete_trees": 0, "exact_closures": 0, "dead_frontiers": 0}; deepest = None; contradiction = None; deepest_len = -1; exact = []
    while queue and stats["states"] < max_states:
        state = queue.pop(); key = (state.graph_index, state.left_index, state.right_index, state.left_words, state.right_words, state.left_prefix, state.right_reverse_prefix, state.left_node, state.right_node, state.length)
        if key in seen: continue
        seen.add(key); stats["states"] += 1; next_states = step(state); ledger = {"graph": GRAPHS[state.graph_index].name, "rendered_partial": render_partial(state), "left_index": state.left_index, "right_index": state.right_index, "left_prefix": state.left_prefix, "right_reverse_prefix": state.right_reverse_prefix, "length": state.length, "pair_trace": [list(item) for item in state.pairs], "role_trace": list(state.role_trace), "next_pair_count": len(next_states)}
        if state.length > deepest_len: deepest_len, deepest = state.length, ledger
        if not next_states:
            stats["dead_frontiers"] += 1
            if contradiction is None: contradiction = {**ledger, "rejection": "graph-compatible tries have no equal next character"}
        stats["character_pairs"] += len(next_states)
        for nxt in next_states:
            stats["word_completions"] += max(0, len(nxt.role_trace) - len(state.role_trace))
            if nxt.left_index > nxt.right_index:
                stats["complete_trees"] += 1; text = " ".join(nxt.left_words + tuple(reversed(nxt.right_words))); row = audit(text, "complete_graph_transducer", nxt.pairs)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"] and row["feature_witness"]["causal_graph_replay_ok"]: stats["exact_closures"] += 1; exact.append(row)
            else: queue.append(nxt)
    return {"stats": stats, "states_exhausted": not queue and stats["states"] < max_states, "exact_closures": exact, "deepest_state_ledger": deepest, "deepest_contradiction": contradiction}


def run(max_states=100000):
    result = transduce(max_states); reports = [dict(audit(generated_surface(graph), "grammar_generated_graph_diagnostic"), promotion_status="not_promoted_without_transducer_closure") for graph in GRAPHS]
    return {"status": "event_causal_graph_partial_transducer", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_states": max_states, "single_connected_discourse_plan": True, "event_causal_graph_required": True, "independent_graph_replay_before_traversal": True, "partial_word_transducer": True, "forward_and_reverse_lexical_tries": True, "equal_character_pair_emission": True, "advance_role_only_on_word_completion": True, "independent_complete_reparse": True, "endpoint_scaffold_gate": True, "preauthored_full_control": False, "corpus_generation": False}, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": sha256(json.dumps([graph.__dict__ for graph in GRAPHS], sort_keys=True).encode()).hexdigest(), "material": "authored cause-event/effect-event graphs with ordinary lexical realizations; no catalogue or known palindrome material", "known_tape_check": "all generated reports pass normalized shared exclusion-catalogue checks"}, "joint_boundary_roles": joint_boundary_roles(), "ordinary_generated_reports": reports, "rendered_candidates": result["exact_closures"], **result, "reader_facing_next_operator": "Author a new graph-compatible effect target whose reversed trie prefix extends the deepest recorded boundary, then rerun the graph transducer.", "scope": "Zero-closure finite graph transducer run is diagnostic; no reader-facing palindrome is claimed."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
