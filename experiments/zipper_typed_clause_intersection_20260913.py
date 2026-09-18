"""A fresh bidirectional zipper over one typed English clause tree.

Unlike a normal sentence-first search, this experiment does not lexicalize the
whole clause before checking its ends.  It expands the single connected tree,
then alternates exposed left/right lexical slots.  Whenever one side leaves a
character residual, the opposite lexical production is filtered against that
live character before it is committed.  Every closure is independently
reparsed and sent through the central admission checks.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sys

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
class Production:
    identifier: str
    lhs: Symbol
    rhs: tuple[Symbol, ...]


@dataclass(frozen=True)
class Node:
    identifier: int
    symbol: Symbol
    terminal: str = ""
    production: str = ""
    children: tuple[int, ...] = ()


@dataclass(frozen=True)
class Leaf:
    identifier: int
    node: int
    word: str
    label: str
    left: int = 0
    right: int = 0


@dataclass(frozen=True)
class State:
    frontier: tuple[int, ...]
    nodes: tuple[Node, ...]
    leaves: tuple[Leaf, ...]
    residual: str
    owner: int
    turn: int
    length: int
    trace: tuple[tuple[int, str], ...]
    zipper_choices: tuple[tuple[int, str, str, str], ...] = ()


@dataclass(frozen=True)
class ParseNode:
    symbol: Symbol
    terminal: str = ""
    production: str = ""
    children: tuple["ParseNode", ...] = ()


# Each event has a small authored semantic frame.  Every lexical option is a
# normal word, but the frame prevents a broad noun menu from creating arbitrary
# subject/object combinations.
EVENTS = {
    "repair": {
        "verb": "repairs",
        "agents": ("mechanic", "technician", "worker"),
        "objects": ("camera", "radio", "tablet"),
        "subject_adjectives": ("red", "calm", "careful"),
        "object_adjectives": ("blue", "clean", "small"),
    },
    "move": {
        "verb": "moves",
        "agents": ("porter", "worker"),
        "objects": ("chair", "table", "piano"),
        "subject_adjectives": ("brisk", "calm", "careful"),
        "object_adjectives": ("round", "plain", "small"),
    },
    "paint": {
        "verb": "paints",
        "agents": ("artist", "painter"),
        "objects": ("mural", "canvas", "panel"),
        "subject_adjectives": ("young", "calm", "careful"),
        "object_adjectives": ("bright", "plain", "large"),
    },
    "load": {
        "verb": "loads",
        "agents": ("porter", "worker"),
        "objects": ("crate", "basket", "parcel"),
        "subject_adjectives": ("brisk", "strong", "careful"),
        "object_adjectives": ("heavy", "plain", "small"),
    },
}


class ZipperGrammar:
    """One clause tree with event, agreement, and valency features."""

    def start(self) -> Symbol:
        return sym("S")

    def terminal(self, symbol: Symbol) -> str | None:
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs: Symbol) -> tuple[Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "S":
            return (Production("S:single_clause", lhs, (sym("CLAUSE"),)),)
        if lhs.name == "CLAUSE":
            return tuple(
                Production(
                    f"CLAUSE:{event_name}", lhs,
                    (sym("NP", role="subject", event=event_name, type="person", number="sing"),
                     sym("V", event=event_name, subject_number="sing"),
                     sym("NP", role="object", event=event_name, type="object", number="sing")),
                ) for event_name in EVENTS
            )
        if lhs.name == "NP":
            event = EVENTS.get(f.get("event"))
            if event is None:
                return ()
            role = f.get("role")
            if role == "subject":
                return (Production(
                    f"NP:subject:{f['event']}", lhs,
                    (sym("DET", role="subject", number="sing", form="a"),
                     sym("ADJ", role="subject", event=f["event"]),
                     sym("AGENT", event=f["event"], type="person", number="sing")),
                ),)
            if role == "object":
                return (Production(
                    f"NP:object:{f['event']}", lhs,
                    (sym("DET", role="object", number="sing", form="a"),
                     sym("ADJ", role="object", event=f["event"]),
                     sym("PATIENT", event=f["event"], type="object", number="sing")),
                ),)
            return ()
        if lhs.name == "DET":
            return (Production("DET:a", lhs, (sym("T", label="det", form="a"),)),)
        if lhs.name == "ADJ":
            event = EVENTS.get(f.get("event"))
            if event is None:
                return ()
            words = event["subject_adjectives"] if f.get("role") == "subject" else event["object_adjectives"]
            return tuple(Production(
                f"ADJ:{f['role']}:{f['event']}:{word}", lhs,
                (sym("T", label=f"adj_{f['role']}", form=word),),
            ) for word in words)
        if lhs.name == "AGENT":
            event = EVENTS.get(f.get("event"))
            if event is None:
                return ()
            return tuple(Production(
                f"AGENT:{f['event']}:{word}", lhs,
                (sym("T", label="agent", form=word),),
            ) for word in event["agents"])
        if lhs.name == "PATIENT":
            event = EVENTS.get(f.get("event"))
            if event is None:
                return ()
            return tuple(Production(
                f"PATIENT:{f['event']}:{word}", lhs,
                (sym("T", label="patient", form=word),),
            ) for word in event["objects"])
        if lhs.name == "V":
            event = EVENTS.get(f.get("event"))
            if event is None or f.get("subject_number") != "sing":
                return ()
            return (Production(
                f"V:{f['event']}:{event['verb']}", lhs,
                (sym("T", label="verb", form=event["verb"]),),
            ),)
        return ()

    def digest(self) -> str:
        seen, queue, rows = set(), deque([self.start()]), []
        while queue:
            symbol = queue.popleft()
            if symbol in seen:
                continue
            seen.add(symbol)
            productions = self.productions(symbol)
            rows.append((symbol.name, symbol.features, [
                (p.identifier, [child.name + str(child.features) for child in p.rhs])
                for p in productions
            ]))
            queue.extend(child for p in productions for child in p.rhs
                         if self.terminal(child) is None)
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def node_map(state: State) -> dict[int, Node]:
    return {node.identifier: node for node in state.nodes}


def leaf_map(state: State) -> dict[int, Leaf]:
    return {leaf.identifier: leaf for leaf in state.leaves}


def expand(grammar: ZipperGrammar, state: State, frontier_index: int) -> tuple[State, ...]:
    nodes = node_map(state)
    ref = state.frontier[frontier_index]
    target = nodes[ref]
    answers = []
    for production in grammar.productions(target.symbol):
        next_id = max(nodes, default=-1) + 1
        child_ids, next_nodes, next_leaves = [], dict(nodes), list(state.leaves)
        for child in production.rhs:
            child_id = next_id
            next_id += 1
            word = grammar.terminal(child)
            next_nodes[child_id] = Node(child_id, child, terminal=word or "")
            child_ids.append(child_id)
            if word is not None:
                next_leaves.append(Leaf(child_id, child_id, word,
                                        child.feature("label", child.name)))
        next_nodes[ref] = Node(ref, target.symbol, production=production.identifier,
                               children=tuple(child_ids))
        next_frontier = (state.frontier[:frontier_index] + tuple(child_ids) +
                         state.frontier[frontier_index + 1:])
        answers.append(State(
            next_frontier, tuple(next_nodes.values()), tuple(next_leaves),
            state.residual, state.owner, state.turn, state.length,
            state.trace + ((ref, production.identifier),), state.zipper_choices,
        ))
    return tuple(answers)


def ordered_leaves(state: State) -> tuple[Leaf, ...]:
    nodes, leaves = node_map(state), leaf_map(state)
    output = []

    def visit(ref: int):
        node = nodes[ref]
        if node.terminal:
            leaf = leaves.get(ref)
            if leaf:
                output.append(leaf)
            return
        for child in node.children:
            visit(child)

    visit(0)
    return tuple(output)


def exposed(state: State, side: int) -> Leaf | None:
    nodes, leaves = node_map(state), leaf_map(state)
    active = [ref for ref in (state.frontier if side == 1 else reversed(state.frontier))
              if nodes[ref].terminal]
    return leaves[active[0]] if active else None


def emit(state: State, side: int) -> State | None:
    leaf = exposed(state, side)
    if leaf is None or leaf.left + leaf.right >= len(leaf.word):
        return None
    character = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
    new_leaf = Leaf(leaf.identifier, leaf.node, leaf.word, leaf.label,
                    leaf.left + 1 if side == 1 else leaf.left,
                    leaf.right + 1 if side == -1 else leaf.right)
    leaves = tuple(new_leaf if item.identifier == leaf.identifier else item
                   for item in state.leaves)
    residual, owner, turn = state.residual, state.owner, state.turn
    if not residual:
        residual, owner, turn = character, side, side
    elif owner == side:
        residual += character
    elif residual[0] != character:
        return None
    else:
        prior_owner = owner
        residual = residual[1:]
        owner = prior_owner if residual else 0
        turn = side if residual else -side
    frontier = state.frontier
    if new_leaf.left + new_leaf.right == len(new_leaf.word):
        frontier = tuple(ref for ref in frontier if ref != leaf.identifier)
    return State(frontier, state.nodes, leaves, residual, owner, turn,
                 state.length + 1, state.trace, state.zipper_choices)


def complete(state: State) -> bool:
    return not state.frontier and not state.residual and all(
        leaf.left + leaf.right == len(leaf.word) for leaf in state.leaves
    )


def render(state: State) -> str:
    return " ".join(leaf.word for leaf in ordered_leaves(state))


def parse_complete(grammar: ZipperGrammar, text: str):
    """Independent token parser with complete backtracking over all rules."""
    if text != text.strip() or re.sub(r"[a-z ,;.!?]", "", text.lower()):
        return None
    tokens = tuple(re.findall(r"[a-z]+", text.lower()))
    memo = {}

    def parses(symbol, offset):
        key = (symbol, offset)
        if key in memo:
            return memo[key]
        term = grammar.terminal(symbol)
        if term is not None:
            result = ((ParseNode(symbol, terminal=term), offset + 1),) \
                if offset < len(tokens) and tokens[offset] == term else ()
            memo[key] = result
            return result
        answers = []
        for production in grammar.productions(symbol):
            partials = [([], offset)]
            for child in production.rhs:
                next_partials = []
                for children, cursor in partials:
                    next_partials.extend(
                        (children + [tree], end)
                        for tree, end in parses(child, cursor)
                    )
                partials = next_partials
                if not partials:
                    break
            answers.extend(
                (ParseNode(symbol, production=production.identifier,
                           children=tuple(children)), end)
                for children, end in partials
            )
        memo[key] = tuple(answers)
        return memo[key]

    return next((tree for tree, end in parses(grammar.start(), 0)
                 if end == len(tokens)), None)


def exact_audit(text: str) -> dict:
    tape = "".join(character for character in text.lower()
                    if "a" <= character <= "z")
    mismatches = [(i, len(tape) - i - 1)
                  for i in range(len(tape) // 2)
                  if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape),
            "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def semantic_witness(tree) -> dict:
    complete = bool(tree and tree.symbol.name == "S" and len(tree.children) == 1)
    clause = tree.children[0] if complete else None
    complete = bool(complete and clause.symbol.name == "CLAUSE" and len(clause.children) == 3)
    subject, verb, patient = clause.children if complete else (None, None, None)
    complete = bool(complete and subject.symbol.name == "NP"
                    and verb.symbol.name == "V" and patient.symbol.name == "NP")
    subject_ok = bool(subject and dict(subject.symbol.features).get("role") == "subject"
                      and dict(subject.symbol.features).get("type") == "person"
                      and dict(subject.symbol.features).get("number") == "sing")
    patient_ok = bool(patient and dict(patient.symbol.features).get("role") == "object"
                      and dict(patient.symbol.features).get("type") == "object"
                      and dict(patient.symbol.features).get("number") == "sing")
    event = dict(verb.symbol.features).get("event") if verb else ""
    event_ok = event in EVENTS and verb.symbol.feature("subject_number") == "sing"
    return {
        "subject_role": "person",
        "subject_number": "sing",
        "verb_frame": "transitive",
        "object_role": "object",
        "event": event,
        "agreement_ok": bool(complete and subject_ok and event_ok),
        "valency_ok": bool(complete and patient_ok and event_ok),
        "subject_action_ok": bool(complete and subject_ok and patient_ok and event_ok),
        "complete_tree": complete,
    }


def audit(grammar, text: str, kind: str, trace) -> dict:
    tree = parse_complete(grammar, text)
    exact = exact_audit(text)
    witness = semantic_witness(tree)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS,
                                          max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if tree is None:
        codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]:
        codes.append("agreement_failure")
    if not witness["valency_ok"]:
        codes.append("valency_failure")
    if not witness["subject_action_ok"]:
        codes.append("subject_action_semantics_failure")
    return {
        "record_kind": kind, "rendered": text,
        "independent_exact_audit": exact,
        "independent_parse": tree is not None,
        "feature_witness": witness, "central_admission": central,
        "mechanically_admitted": not codes, "rejection_codes": codes,
        "shared_tree_trace": list(trace),
        "reader_status": "unreviewed; programmatic checks do not certify readability",
    }


def explicit_control(grammar: ZipperGrammar) -> State:
    state = State((0,), (Node(0, grammar.start()),), (), "", 0, 1, 0, ())
    desired = {
        "CLAUSE": "CLAUSE:repair", "NP:subject": "NP:subject:repair",
        "NP:object": "NP:object:repair", "ADJ:subject": "ADJ:subject:repair:red",
        "AGENT": "AGENT:repair:mechanic", "V": "V:repair:repairs",
        "ADJ:object": "ADJ:object:repair:blue", "PATIENT": "PATIENT:repair:camera",
        "DET": "DET:a",
    }
    while True:
        nodes = node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier)
                      if not nodes[ref].terminal]
        if not unresolved:
            return state
        index = unresolved[0]
        symbol = nodes[state.frontier[index]].symbol
        if symbol.name == "S":
            identifier = "S:single_clause"
        elif symbol.name == "CLAUSE":
            identifier = desired["CLAUSE"]
        elif symbol.name == "NP":
            identifier = desired[f"NP:{symbol.feature('role')}"]
        elif symbol.name in {"ADJ", "AGENT", "V", "PATIENT", "DET"}:
            identifier = desired[symbol.name if symbol.name in {"V", "PATIENT", "AGENT", "DET"}
                                   else f"ADJ:{symbol.feature('role')}"]
        else:
            raise AssertionError(symbol.name)
        options = [candidate for candidate in expand(grammar, state, index)
                   if candidate.trace[-1][1] == identifier]
        if len(options) != 1:
            raise AssertionError(f"nonunique control path {symbol.name} {identifier}")
        state = options[0]


def choice_character(candidate: State, side: int) -> str | None:
    leaf = exposed(candidate, side)
    if leaf is None:
        return None
    return leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]


def zipper_expand(grammar: ZipperGrammar, state: State, index: int, side: int) -> tuple[State, ...]:
    """Expand one edge, selecting lexical alternatives against live residual."""
    expected = state.residual[0] if state.residual else ""
    answers = []
    for candidate in expand(grammar, state, index):
        if expected and choice_character(candidate, side) != expected:
            continue
        symbol = node_map(state)[state.frontier[index]].symbol
        production = candidate.trace[-1][1]
        choices = candidate.zipper_choices + ((side, symbol.name, production, expected),)
        answers.append(State(candidate.frontier, candidate.nodes, candidate.leaves,
                             candidate.residual, candidate.owner, candidate.turn,
                             candidate.length, candidate.trace, choices))
    return tuple(answers)


def attempts(state: State):
    sides = (-state.owner,) if state.residual else (state.turn, -state.turn)
    output = []
    for side in sides:
        leaf = exposed(state, side)
        if leaf is None:
            output.append({"side": side, "available": False})
            continue
        character = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        output.append({"side": side, "available": True, "slot": leaf.label,
                       "word": leaf.word, "character": character,
                       "accepted": emit(state, side) is not None})
    return output


def zipper_solver(grammar: ZipperGrammar, max_states: int = 100000) -> dict:
    initial = State((0,), (Node(0, grammar.start()),), (), "", 0, 1, 0, ())
    queue, seen = deque([initial]), set()
    exact_rows, admitted = {}, {}
    stats = Counter(states=0, expansions=0, emissions=0,
                    residual_contradictions=0, complete_trees=0, exact_closures=0)
    deepest = None
    deepest_contradiction = None
    deepest_length = deepest_contradiction_length = -1

    while queue and stats["states"] < max_states:
        state = queue.pop()
        key = (state.frontier, state.nodes, state.leaves, state.residual,
               state.owner, state.turn, state.length)
        if key in seen:
            continue
        seen.add(key)
        stats["states"] += 1
        offered = attempts(state)
        ledger = {
            "frontier": list(state.frontier),
            "rendered_tree": render(state),
            "residual": state.residual, "owner": state.owner,
            "turn": state.turn, "length": state.length,
            "trace": [list(item) for item in state.trace],
            "zipper_choices": [list(item) for item in state.zipper_choices],
            "attempts": offered,
        }
        if state.length > deepest_length:
            deepest_length, deepest = state.length, ledger
        if any(row.get("available") and not row.get("accepted") for row in offered):
            if state.length > deepest_contradiction_length:
                deepest_contradiction_length, deepest_contradiction = state.length, ledger
        if complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS:
                row = audit(grammar, render(state), "complete_zipper_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1
                    exact_rows.setdefault(row["rendered"], row)
                    if row["mechanically_admitted"]:
                        admitted.setdefault(row["rendered"], row)
            continue

        nodes = node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier)
                      if not nodes[ref].terminal]
        index, side = None, None
        if state.residual:
            side = -state.owner
            edge = 0 if side == 1 else len(state.frontier) - 1
            if edge >= 0 and not nodes[state.frontier[edge]].terminal:
                index = edge
        elif unresolved:
            side = state.turn
            candidate_indices = unresolved if side == 1 else list(reversed(unresolved))
            index = candidate_indices[0]
        if index is not None:
            for next_state in zipper_expand(grammar, state, index, side):
                queue.append(next_state)
                stats["expansions"] += 1
        for emit_side in ((-state.owner,) if state.residual else (state.turn, -state.turn)):
            if exposed(state, emit_side) is None:
                continue
            next_state = emit(state, emit_side)
            if next_state is not None:
                queue.append(next_state)
                stats["emissions"] += 1
            else:
                stats["residual_contradictions"] += 1
    return {
        "stats": dict(stats),
        "states_exhausted": not queue and stats["states"] < max_states,
        "exact_closures": list(exact_rows.values()),
        "mechanically_admitted_closures": list(admitted.values()),
        "deepest_state_ledger": {
            "deepest_state": deepest,
            "deepest_contradiction_state": deepest_contradiction,
        },
    }


def zipper_preflight(grammar: ZipperGrammar) -> dict:
    """Use the real zipper scheduler on the selected ordinary control tree."""
    state = explicit_control(grammar)
    events = []
    failed = None
    for step in range(1, 31):
        sides = (-state.owner,) if state.residual else (state.turn,)
        side = sides[0]
        leaf = exposed(state, side)
        if leaf is None:
            break
        character = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        next_state = emit(state, side)
        if next_state is None:
            failed = {"step": step, "side": side, "slot": leaf.label,
                      "word": leaf.word, "character": character,
                      "emitter_rejected": True}
            break
        state = next_state
        events.append({"step": step, "side": side, "slot": leaf.label,
                       "word": leaf.word, "character": character,
                       "residual_after": state.residual})
    return {"rendered": render(state), "emitter_events": events,
            "failed_attempt": failed,
            "independent_parse": parse_complete(grammar, render(state)) is not None,
            "diagnostic_only": True}


def run(max_states: int = 100000) -> dict:
    grammar = ZipperGrammar()
    result = zipper_solver(grammar, max_states)
    control_state = explicit_control(grammar)
    control_text = render(control_state)
    control = audit(grammar, control_text, "complete_zipper_control", control_state.trace)
    control.update({
        "diagnostic_only": True,
        "grammar_tree_fully_expanded": True,
        "provenance": {"construction": "authored transitive event frame: mechanic repairs camera",
                        "shared_tree": True, "seed_phrase": False},
        "reader_status": "grammar control only; not a palindrome candidate or readability evidence",
    })
    return {
        "status": "zipper_typed_clause_intersection",
        "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                   "max_states": max_states, "single_shared_derivation_tree": True,
                   "bidirectional_zipper": True,
                   "lexical_choice_requires_live_residual": True,
                   "closure_requires_all_leaves_consumed": True,
                   "independent_complete_reparse": True,
                   "endpoint_scaffold_gate": True,
                   "corpus_generation": False},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grammar_sha256": grammar.digest(),
                       "material": "authored typed event frames; no catalogue or known palindrome material"},
        "rendered_candidates": [control],
        "complete_zipper_control": control,
        **result,
        "zipper_boundary_preflight": zipper_preflight(grammar),
        "reader_facing_next_operator": "Replace the exposed subject-adjective/agent seam with a newly authored semantically licensed agent whose next letters match the actual residual, then rerun the full zipper.",
        "scope": "Short bounded zipper run is diagnostic; no reader-facing palindrome is claimed.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"],
                      "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__":
    main()
