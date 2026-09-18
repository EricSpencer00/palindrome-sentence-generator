"""Bidirectional zipper for a connected two-clause English relation.

The zipper chooses lexical constituents at the exposed ends of one relation
tree.  It never creates two sentence halves: both clauses, their subjects,
predicates, and typed objects are descendants of one root.  A lexical choice
on the opposite edge is admitted only when its currently exposed character
consumes the live reversal residual.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, deque
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
class ParseNode:
    symbol: Symbol
    terminal: str = ""
    production: str = ""
    children: tuple["ParseNode", ...] = ()


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


# Each relation is an authored, coherent two-clause event frame.  These are
# small semantic alternatives, not a phrase catalogue: every complete tree
# has a plural first-person subject in clause one and an independently typed
# plural agent/object relation in clause two.
RELATIONS = {
    "record_use": {
        "left_verb": "record", "left_object": "survey",
        "right_subject": "clerks", "right_adjective": "skilled",
        "right_verb": "use", "right_object": "screw",
        "right_object_adjective": "plain",
    },
    "review_sort": {
        "left_verb": "review", "left_object": "report",
        "right_subject": "aides", "right_adjective": "quiet",
        "right_verb": "sort", "right_object": "file",
        "right_object_adjective": "small",
    },
    "draft_pack": {
        "left_verb": "draft", "left_object": "plan",
        "right_subject": "workers", "right_adjective": "strong",
        "right_verb": "pack", "right_object": "crate",
        "right_object_adjective": "heavy",
    },
}


class TwoClauseGrammar:
    """One connected relation tree with typed agents, actions, and objects."""

    def start(self):
        return sym("S")

    def terminal(self, symbol):
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs):
        f = dict(lhs.features)
        if lhs.name == "S":
            return (Production("S:relation", lhs, (sym("RELATION"),)),)
        if lhs.name == "RELATION":
            rows = []
            for relation in RELATIONS:
                rows.append(Production(
                    f"RELATION:{relation}", lhs,
                    (sym("CLAUSE", side="left", relation=relation),
                     sym("CONNECTOR", form="while"),
                     sym("CLAUSE", side="right", relation=relation)),
                ))
            return tuple(rows)
        if lhs.name == "CLAUSE":
            relation = RELATIONS.get(f.get("relation"))
            if relation is None:
                return ()
            side = f.get("side")
            if side == "left":
                return (Production(
                    f"CLAUSE:left:{f['relation']}", lhs,
                    (sym("SUBJECT", side="left", relation=f["relation"],
                         type="person", number="plur"),
                     sym("VERB", side="left", relation=f["relation"],
                         subject_number="plur"),
                     sym("OBJECT", side="left", relation=f["relation"],
                         type="document", number="sing")),
                ),)
            return (Production(
                f"CLAUSE:right:{f['relation']}", lhs,
                (sym("SUBJECT", side="right", relation=f["relation"],
                     type="person", number="plur"),
                 sym("VERB", side="right", relation=f["relation"],
                     subject_number="plur"),
                 sym("OBJECT", side="right", relation=f["relation"],
                     type="implement", number="sing")),
            ),)
        if lhs.name == "CONNECTOR":
            return tuple(Production(f"CONNECTOR:{word}", lhs,
                                    (sym("T", label="connector", form=word),))
                         for word in ("while", "and"))
        if lhs.name == "DET":
            identifier = (f"DET:{f['role']}:a" if f.get("role") != "right_subject"
                          else "DET:right_subject:the")
            return (Production(identifier, lhs,
                               (sym("T", label="det", form=f["form"]),)),)
        if lhs.name == "ADJ":
            return (Production(f"ADJ:{f['role']}:{f['form']}", lhs,
                               (sym("T", label="adj", form=f["form"]),)),)
        relation = RELATIONS.get(f.get("relation"))
        if relation is None:
            return ()
        if lhs.name == "SUBJECT":
            if f.get("side") == "left":
                return (Production("SUBJECT:left:we", lhs,
                                    (sym("T", label="left_subject", form="we"),)),)
            return (Production(
                f"SUBJECT:right:{f['relation']}", lhs,
                (sym("DET", form="the", role="right_subject"),
                 sym("ADJ", form=relation["right_adjective"], role="right_subject"),
                 sym("T", label="right_subject", form=relation["right_subject"])),
            ),)
        if lhs.name == "VERB":
            verb = relation["left_verb"] if f.get("side") == "left" else relation["right_verb"]
            return (Production(f"VERB:{f['side']}:{f['relation']}:{verb}", lhs,
                               (sym("T", label=f"{f['side']}_verb", form=verb),)),)
        if lhs.name == "OBJECT":
            if f.get("side") == "left":
                noun, adjective = relation["left_object"], "brief"
            else:
                noun, adjective = relation["right_object"], relation["right_object_adjective"]
            return (Production(
                f"OBJECT:{f['side']}:{f['relation']}:{noun}", lhs,
                (sym("DET", form="a", role=f"{f['side']}_object"),
                 sym("ADJ", form=adjective, role=f"{f['side']}_object"),
                 sym("T", label=f"{f['side']}_object", form=noun)),
            ),)
        return ()

    def digest(self):
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


def node_map(state):
    return {node.identifier: node for node in state.nodes}


def leaf_map(state):
    return {leaf.identifier: leaf for leaf in state.leaves}


def expand(grammar, state, index):
    nodes = node_map(state)
    ref = state.frontier[index]
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
        frontier = (state.frontier[:index] + tuple(child_ids) +
                    state.frontier[index + 1:])
        answers.append(State(frontier, tuple(next_nodes.values()), tuple(next_leaves),
                             state.residual, state.owner, state.turn, state.length,
                             state.trace + ((ref, production.identifier),),
                             state.zipper_choices))
    return tuple(answers)


def ordered_leaves(state):
    nodes, leaves, output = node_map(state), leaf_map(state), []

    def visit(ref):
        node = nodes[ref]
        if node.terminal:
            if ref in leaves:
                output.append(leaves[ref])
            return
        for child in node.children:
            visit(child)

    visit(0)
    return tuple(output)


def render(state):
    return " ".join(leaf.word for leaf in ordered_leaves(state))


def exposed(state, side):
    nodes, leaves = node_map(state), leaf_map(state)
    refs = [ref for ref in (state.frontier if side == 1 else reversed(state.frontier))
            if nodes[ref].terminal]
    return leaves[refs[0]] if refs else None


def emit(state, side):
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
        residual = residual[1:]
        if not residual:
            owner, turn = 0, -side
    frontier = state.frontier
    if new_leaf.left + new_leaf.right == len(new_leaf.word):
        frontier = tuple(ref for ref in frontier if ref != leaf.identifier)
    return State(frontier, state.nodes, leaves, residual, owner, turn,
                 state.length + 1, state.trace, state.zipper_choices)


def complete(state):
    return not state.frontier and not state.residual and all(
        leaf.left + leaf.right == len(leaf.word) for leaf in state.leaves
    )


def parse_complete(grammar, text):
    if text != text.strip() or re.sub(r"[a-z ,;.!?]", "", text.lower()):
        return None
    tokens, memo = tuple(re.findall(r"[a-z]+", text.lower())), {}

    def parses(symbol, offset):
        key = (symbol, offset)
        if key in memo:
            return memo[key]
        term = grammar.terminal(symbol)
        if term is not None:
            answer = ((ParseNode(symbol, terminal=term), offset + 1),) \
                if offset < len(tokens) and tokens[offset] == term else ()
            memo[key] = answer
            return answer
        answers = []
        for production in grammar.productions(symbol):
            partials = [([], offset)]
            for child in production.rhs:
                next_partials = []
                for children, cursor in partials:
                    next_partials.extend((children + [tree], end)
                                         for tree, end in parses(child, cursor))
                partials = next_partials
                if not partials:
                    break
            answers.extend((ParseNode(symbol, production=production.identifier,
                                      children=tuple(children)), end)
                           for children, end in partials)
        memo[key] = tuple(answers)
        return memo[key]

    return next((tree for tree, end in parses(grammar.start(), 0)
                 if end == len(tokens)), None)


def exact_audit(text):
    tape = "".join(character for character in text.lower()
                    if "a" <= character <= "z")
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2)
                  if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape),
            "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def semantic_witness(tree):
    relation = ""
    agreement = valency = action = complete_tree = False
    if tree and tree.symbol.name == "S" and len(tree.children) == 1:
        root = tree.children[0]
        if root.symbol.name == "RELATION" and len(root.children) == 3:
            relation = root.production.split(":", 1)[-1]
            left, connector, right = root.children
            left_ok = left.symbol.name == "CLAUSE" and left.symbol.feature("side") == "left"
            right_ok = right.symbol.name == "CLAUSE" and right.symbol.feature("side") == "right"
            connector_ok = (connector.symbol.name == "CONNECTOR"
                            and connector.symbol.feature("form") in {"while", "and"})
            if left_ok and right_ok and connector_ok and len(left.children) == 3 and len(right.children) == 3:
                lsub, lverb, lobj = left.children
                rsub, rverb, robj = right.children
                agreement = all(node.symbol.feature("number") in {"plur", ""}
                                for node in (lsub, rsub))
                valency = (lobj.symbol.feature("type") == "document"
                           and robj.symbol.feature("type") == "implement")
                action = (lverb.symbol.feature("subject_number") == "plur"
                          and rverb.symbol.feature("subject_number") == "plur")
                complete_tree = True
    return {"relation": relation, "agreement_ok": agreement,
            "valency_ok": valency, "subject_action_ok": action,
            "left_subject_type": "person", "right_subject_type": "person",
            "left_object_type": "document", "right_object_type": "implement",
            "complete_tree": complete_tree}


def audit(grammar, text, kind, trace):
    tree = parse_complete(grammar, text)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS,
                                          max_letters=MAX_LETTERS)
    witness = semantic_witness(tree)
    codes = [key for key, value in central.items() if not value]
    if tree is None:
        codes.append("independent_complete_reparse_failed")
    for key, code in (("agreement_ok", "agreement_failure"),
                      ("valency_ok", "valency_failure"),
                      ("subject_action_ok", "subject_action_semantics_failure")):
        if not witness[key]:
            codes.append(code)
    return {"record_kind": kind, "rendered": text,
            "independent_exact_audit": exact_audit(text),
            "independent_parse": tree is not None, "feature_witness": witness,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "shared_tree_trace": list(trace),
            "reader_status": "unreviewed; programmatic checks do not certify readability"}


def explicit_control(grammar):
    state = State((0,), (Node(0, grammar.start()),), (), "", 0, 1, 0, ())
    wanted = {
        "S": "S:relation", "RELATION": "RELATION:record_use",
        "CLAUSE:left": "CLAUSE:left:record_use", "CLAUSE:right": "CLAUSE:right:record_use",
        "CONNECTOR": "CONNECTOR:while",
        "SUBJECT:left": "SUBJECT:left:we", "SUBJECT:right": "SUBJECT:right:record_use",
        "VERB:left": "VERB:left:record_use:record", "VERB:right": "VERB:right:record_use:use",
        "OBJECT:left": "OBJECT:left:record_use:survey", "OBJECT:right": "OBJECT:right:record_use:screw",
        "DET:left_object": "DET:left_object:a", "DET:right_object": "DET:right_object:a",
        "DET:right_subject": "DET:right_subject:the", "ADJ:right_subject": "ADJ:right_subject:skilled",
        "ADJ:left_object": "ADJ:left_object:brief", "ADJ:right_object": "ADJ:right_object:plain",
    }
    while True:
        nodes = node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved:
            return state
        index = unresolved[0]
        symbol = nodes[state.frontier[index]].symbol
        if symbol.name in {"S", "RELATION"}:
            identifier = wanted[symbol.name]
        elif symbol.name == "CLAUSE":
            identifier = wanted[f"CLAUSE:{symbol.feature('side')}"]
        elif symbol.name == "CONNECTOR":
            identifier = wanted["CONNECTOR"]
        elif symbol.name in {"SUBJECT", "VERB", "OBJECT"}:
            identifier = wanted[f"{symbol.name}:{symbol.feature('side')}"]
        elif symbol.name == "DET":
            identifier = wanted[f"DET:{symbol.feature('role')}"]
        elif symbol.name == "ADJ":
            identifier = wanted[f"ADJ:{symbol.feature('role')}"]
        else:
            raise AssertionError(symbol.name)
        options = [candidate for candidate in expand(grammar, state, index)
                   if candidate.trace[-1][1] == identifier]
        if len(options) != 1:
            raise AssertionError(f"nonunique control path {symbol.name}: {identifier}")
        state = options[0]


def live_character(candidate, side):
    leaf = exposed(candidate, side)
    if leaf is None:
        return None
    return leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]


def zipper_expand(grammar, state, index, side):
    expected = state.residual[0] if state.residual else ""
    symbol = node_map(state)[state.frontier[index]].symbol
    answers = []
    for candidate in expand(grammar, state, index):
        if expected and live_character(candidate, side) != expected:
            continue
        choices = candidate.zipper_choices + ((side, symbol.name,
                                               candidate.trace[-1][1], expected),)
        answers.append(State(candidate.frontier, candidate.nodes, candidate.leaves,
                             candidate.residual, candidate.owner, candidate.turn,
                             candidate.length, candidate.trace, choices))
    return tuple(answers)


def attempts(state):
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


def zipper_solver(grammar, max_states=100000):
    initial = State((0,), (Node(0, grammar.start()),), (), "", 0, 1, 0, ())
    queue, seen = deque([initial]), set()
    exact_rows, admitted = {}, {}
    stats = Counter(states=0, expansions=0, emissions=0,
                    residual_contradictions=0, complete_trees=0, exact_closures=0)
    deepest = deepest_contradiction = None
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
        ledger = {"frontier": list(state.frontier), "rendered_tree": render(state),
                  "residual": state.residual, "owner": state.owner,
                  "turn": state.turn, "length": state.length,
                  "trace": [list(item) for item in state.trace],
                  "zipper_choices": [list(item) for item in state.zipper_choices],
                  "attempts": offered}
        if state.length > deepest_length:
            deepest_length, deepest = state.length, ledger
        if any(row.get("available") and not row.get("accepted") for row in offered):
            if state.length > deepest_contradiction_length:
                deepest_contradiction_length, deepest_contradiction = state.length, ledger
        if complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS:
                row = audit(grammar, render(state), "complete_zipper_relation", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1
                    exact_rows.setdefault(row["rendered"], row)
                    if row["mechanically_admitted"]:
                        admitted.setdefault(row["rendered"], row)
            continue
        nodes = node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        index = None
        if state.residual:
            side = -state.owner
            edge = 0 if side == 1 else len(state.frontier) - 1
            if edge >= 0 and not nodes[state.frontier[edge]].terminal:
                index = edge
        elif unresolved:
            side = state.turn
            index = (unresolved[0] if side == 1 else unresolved[-1])
        if index is not None:
            for next_state in zipper_expand(grammar, state, index, side):
                queue.append(next_state)
                stats["expansions"] += 1
        for side in ((-state.owner,) if state.residual else (state.turn, -state.turn)):
            if exposed(state, side) is None:
                continue
            next_state = emit(state, side)
            if next_state is not None:
                queue.append(next_state)
                stats["emissions"] += 1
            else:
                stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": list(exact_rows.values()),
            "mechanically_admitted_closures": list(admitted.values()),
            "deepest_state_ledger": {"deepest_state": deepest,
                                     "deepest_contradiction_state": deepest_contradiction}}


def boundary_preflight(grammar):
    state = explicit_control(grammar)
    events, failed = [], None
    for step in range(1, 31):
        side = (-state.owner,) if state.residual else (state.turn,)
        side = side[0]
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
            "failed_attempt": failed, "independent_parse": parse_complete(grammar, render(state)) is not None,
            "diagnostic_only": True}


def run(max_states=100000):
    grammar = TwoClauseGrammar()
    result = zipper_solver(grammar, max_states)
    control_state = explicit_control(grammar)
    control = audit(grammar, render(control_state), "complete_two_clause_relation_control", control_state.trace)
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True,
                    "provenance": {"construction": "authored two-clause record/use relation",
                                    "shared_tree": True, "seed_phrase": False},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "zipper_two_clause_relation_intersection",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                       "max_states": max_states, "single_shared_derivation_tree": True,
                       "bidirectional_zipper": True, "central_relation_jointly_lexicalized": True,
                       "outer_constituents_jointly_lexicalized": True,
                       "lexical_choice_requires_live_residual": True,
                       "closure_requires_all_leaves_consumed": True,
                       "independent_complete_reparse": True,
                       "endpoint_scaffold_gate": True, "corpus_generation": False},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "grammar_sha256": grammar.digest(),
                           "material": "authored typed two-clause relations; no catalogue or known palindrome material"},
            "rendered_candidates": [control], "complete_two_clause_relation_control": control,
            **result, "joint_boundary_preflight": boundary_preflight(grammar),
            "reader_facing_next_operator": "Replace the actual exposed clause-boundary constituent with a newly authored typed relation whose live reverse prefix is verified by the zipper, then rerun the full scheduler.",
            "scope": "Short bounded zipper relation run is diagnostic; no reader-facing palindrome is claimed."}


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
