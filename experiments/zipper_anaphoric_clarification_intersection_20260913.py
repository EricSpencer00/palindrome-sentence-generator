"""Residual-driven zipper for an anaphoric clarification relation.

The grammar is one connected tree: an assertion introduces a map and a second
clause uses ``it`` to clarify that map's function.  Lexical constituents at
the two exposed edges are selected by the live character residual, while all
roles and the anaphoric link remain grammar features.
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


class AnaphoricGrammar:
    """An assertion plus an independently parsed map clarification."""

    def start(self):
        return sym("S")

    def terminal(self, symbol):
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs):
        f = dict(lhs.features)
        if lhs.name == "S":
            return (Production("S:discourse", lhs, (sym("DISC"),)),)
        if lhs.name == "DISC":
            return (Production("DISC:map-clarification", lhs, (
                sym("ASSERTION"), sym("CONNECTOR", form="and"), sym("CLARIFICATION"),
            )),)
        if lhs.name == "ASSERTION":
            return (Production("ASSERTION:expert-found-map", lhs, (
                sym("SUBJECT", type="person", number="sing"),
                sym("VERB", event="find", subject_number="sing"),
                sym("OBJECT", role="introduced", type="artifact", number="sing"),
            )),)
        if lhs.name == "CLARIFICATION":
            return (Production("CLARIFICATION:map-explains-arena", lhs, (
                sym("PRONOUN", form="it", antecedent="map", number="sing"),
                sym("VERB", event="explain", subject_number="sing"),
                sym("OBJECT", role="explained", type="place", number="sing"),
            )),)
        if lhs.name == "CONNECTOR":
            return (Production("CONNECTOR:and", lhs, (sym("T", label="connector", form="and"),)),)
        if lhs.name == "SUBJECT":
            return (Production("SUBJECT:an-expert", lhs, (
                sym("DET", form="an", role="subject"),
                sym("AGENT", form="expert", type="person", number="sing"),
            )),)
        if lhs.name == "PRONOUN":
            return (Production("PRONOUN:it", lhs, (sym("T", label="pronoun", form="it"),)),)
        if lhs.name == "VERB":
            verb = "found" if f.get("event") == "find" else "explains"
            return (Production(f"VERB:{f['event']}:{verb}", lhs,
                               (sym("T", label="verb", form=verb),)),)
        if lhs.name == "OBJECT":
            if f.get("role") == "introduced":
                return (Production("OBJECT:introduced:map", lhs, (
                    sym("DET", form="a", role="introduced"),
                    sym("ADJ", form="useful", role="introduced"),
                    sym("PATIENT", form="map", type="artifact", number="sing"),
                )),)
            return (Production("OBJECT:explained:arena", lhs, (
                sym("DET", form="the", role="explained"),
                sym("PATIENT", form="arena", type="place", number="sing"),
            )),)
        if lhs.name == "DET":
            return (Production(f"DET:{f['role']}:{f['form']}", lhs,
                               (sym("T", label="det", form=f["form"]),)),)
        if lhs.name == "ADJ":
            return (Production(f"ADJ:{f['role']}:{f['form']}", lhs,
                               (sym("T", label="adj", form=f["form"]),)),)
        if lhs.name == "AGENT":
            return (Production("AGENT:expert", lhs, (sym("T", label="agent", form="expert"),)),)
        if lhs.name == "PATIENT":
            return (Production(f"PATIENT:{f['type']}:{f['form']}", lhs,
                               (sym("T", label="patient", form=f["form"]),)),)
        return ()

    def digest(self):
        seen, queue, rows = set(), deque([self.start()]), []
        while queue:
            symbol = queue.popleft()
            if symbol in seen:
                continue
            seen.add(symbol)
            prods = self.productions(symbol)
            rows.append((symbol.name, symbol.features, [
                (p.identifier, [child.name + str(child.features) for child in p.rhs])
                for p in prods
            ]))
            queue.extend(child for p in prods for child in p.rhs
                         if self.terminal(child) is None)
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def node_map(state): return {node.identifier: node for node in state.nodes}
def leaf_map(state): return {leaf.identifier: leaf for leaf in state.leaves}


def expand(grammar, state, index):
    nodes, ref = node_map(state), state.frontier[index]
    answers = []
    for production in grammar.productions(nodes[ref].symbol):
        next_id, child_ids, next_nodes, next_leaves = max(nodes, default=-1) + 1, [], dict(nodes), list(state.leaves)
        for child in production.rhs:
            cid, next_id = next_id, next_id + 1
            word = grammar.terminal(child)
            next_nodes[cid] = Node(cid, child, terminal=word or "")
            child_ids.append(cid)
            if word is not None:
                next_leaves.append(Leaf(cid, cid, word, child.feature("label", child.name)))
        next_nodes[ref] = Node(ref, nodes[ref].symbol, production=production.identifier, children=tuple(child_ids))
        frontier = state.frontier[:index] + tuple(child_ids) + state.frontier[index + 1:]
        answers.append(State(frontier, tuple(next_nodes.values()), tuple(next_leaves), state.residual,
                             state.owner, state.turn, state.length, state.trace + ((ref, production.identifier),), state.zipper_choices))
    return tuple(answers)


def ordered_leaves(state):
    nodes, leaves, result = node_map(state), leaf_map(state), []
    def visit(ref):
        node = nodes[ref]
        if node.terminal:
            if ref in leaves: result.append(leaves[ref])
            return
        for child in node.children: visit(child)
    visit(0)
    return tuple(result)


def render(state): return " ".join(leaf.word for leaf in ordered_leaves(state))


def exposed(state, side):
    nodes, leaves = node_map(state), leaf_map(state)
    refs = [ref for ref in (state.frontier if side == 1 else reversed(state.frontier)) if nodes[ref].terminal]
    return leaves[refs[0]] if refs else None


def emit(state, side):
    leaf = exposed(state, side)
    if leaf is None or leaf.left + leaf.right >= len(leaf.word): return None
    char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
    new_leaf = Leaf(leaf.identifier, leaf.node, leaf.word, leaf.label,
                    leaf.left + 1 if side == 1 else leaf.left,
                    leaf.right + 1 if side == -1 else leaf.right)
    leaves = tuple(new_leaf if item.identifier == leaf.identifier else item for item in state.leaves)
    residual, owner, turn = state.residual, state.owner, state.turn
    if not residual: residual, owner, turn = char, side, side
    elif owner == side: residual += char
    elif residual[0] != char: return None
    else:
        residual = residual[1:]
        if not residual: owner, turn = 0, -side
    frontier = state.frontier
    if new_leaf.left + new_leaf.right == len(new_leaf.word):
        frontier = tuple(ref for ref in frontier if ref != leaf.identifier)
    return State(frontier, state.nodes, leaves, residual, owner, turn, state.length + 1,
                 state.trace, state.zipper_choices)


def complete(state):
    return not state.frontier and not state.residual and all(
        leaf.left + leaf.right == len(leaf.word) for leaf in state.leaves)


def parse_complete(grammar, text):
    if text != text.strip() or re.sub(r"[a-z ,;.!?]", "", text.lower()): return None
    tokens, memo = tuple(re.findall(r"[a-z]+", text.lower())), {}
    def parses(symbol, offset):
        key = (symbol, offset)
        if key in memo: return memo[key]
        term = grammar.terminal(symbol)
        if term is not None:
            ans = ((ParseNode(symbol, terminal=term), offset + 1),) if offset < len(tokens) and tokens[offset] == term else ()
            memo[key] = ans; return ans
        answers = []
        for production in grammar.productions(symbol):
            partials = [([], offset)]
            for child in production.rhs:
                nxt = []
                for children, cursor in partials:
                    nxt.extend((children + [tree], end) for tree, end in parses(child, cursor))
                partials = nxt
                if not partials: break
            answers.extend((ParseNode(symbol, production=production.identifier, children=tuple(children)), end)
                           for children, end in partials)
        memo[key] = tuple(answers); return memo[key]
    return next((tree for tree, end in parses(grammar.start(), 0) if end == len(tokens)), None)


def exact_audit(text):
    tape = "".join(c for c in text.lower() if "a" <= c <= "z")
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def semantic_witness(tree):
    complete_tree = bool(tree and tree.symbol.name == "S" and len(tree.children) == 1)
    disc = tree.children[0] if complete_tree else None
    complete_tree = bool(complete_tree and disc.symbol.name == "DISC" and len(disc.children) == 3)
    assertion, connector, clarification = disc.children if complete_tree else (None, None, None)
    complete_tree = bool(complete_tree and assertion.symbol.name == "ASSERTION" and clarification.symbol.name == "CLARIFICATION")
    if complete_tree:
        asub, averb, aobj = assertion.children
        cpron, cverb, cobj = clarification.children
        subject_ok = asub.symbol.feature("type") == "person" and asub.symbol.feature("number") == "sing"
        found_ok = averb.symbol.feature("event") == "find"
        map_ok = aobj.symbol.feature("type") == "artifact" and aobj.symbol.feature("role") == "introduced"
        pronoun_ok = cpron.symbol.feature("antecedent") == "map" and cpron.symbol.feature("form") == "it"
        explain_ok = cverb.symbol.feature("event") == "explain"
        arena_ok = cobj.symbol.feature("type") == "place" and cobj.symbol.feature("role") == "explained"
    else:
        subject_ok = found_ok = map_ok = pronoun_ok = explain_ok = arena_ok = False
    return {"discourse_function": "anaphoric_clarification", "antecedent": "map",
            "agreement_ok": bool(complete_tree and subject_ok and pronoun_ok),
            "valency_ok": bool(complete_tree and found_ok and explain_ok and map_ok and arena_ok),
            "subject_action_ok": bool(complete_tree and subject_ok and found_ok and explain_ok and map_ok and arena_ok),
            "coreference_ok": bool(complete_tree and pronoun_ok), "complete_tree": complete_tree}


def audit(grammar, text, kind, trace):
    tree = parse_complete(grammar, text); central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    witness = semantic_witness(tree); codes = [key for key, value in central.items() if not value]
    if tree is None: codes.append("independent_complete_reparse_failed")
    for key, code in (("agreement_ok", "agreement_failure"), ("valency_ok", "valency_failure"),
                      ("subject_action_ok", "subject_action_semantics_failure"), ("coreference_ok", "coreference_failure")):
        if not witness[key]: codes.append(code)
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact_audit(text),
            "independent_parse": tree is not None, "feature_witness": witness, "central_admission": central,
            "mechanically_admitted": not codes, "rejection_codes": codes, "shared_tree_trace": list(trace),
            "reader_status": "unreviewed; programmatic checks do not certify readability"}


def explicit_control(grammar):
    state = State((0,), (Node(0, grammar.start()),), (), "", 0, 1, 0, ())
    wanted = {
        "S": "S:discourse", "DISC": "DISC:map-clarification", "ASSERTION": "ASSERTION:expert-found-map",
        "CLARIFICATION": "CLARIFICATION:map-explains-arena", "CONNECTOR": "CONNECTOR:and",
        "SUBJECT": "SUBJECT:an-expert", "PRONOUN": "PRONOUN:it", "VERB:find": "VERB:find:found",
        "VERB:explain": "VERB:explain:explains", "OBJECT:introduced": "OBJECT:introduced:map",
        "OBJECT:explained": "OBJECT:explained:arena", "DET:subject": "DET:subject:an",
        "DET:introduced": "DET:introduced:a", "DET:explained": "DET:explained:the",
        "ADJ:introduced": "ADJ:introduced:useful", "AGENT": "AGENT:expert",
        "PATIENT:artifact": "PATIENT:artifact:map", "PATIENT:place": "PATIENT:place:arena",
    }
    while True:
        nodes = node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved: return state
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol
        if symbol.name in {"S", "DISC", "ASSERTION", "CLARIFICATION", "CONNECTOR", "SUBJECT", "PRONOUN", "AGENT"}:
            identifier = wanted[symbol.name]
        elif symbol.name == "VERB": identifier = wanted[f"VERB:{symbol.feature('event')}"]
        elif symbol.name == "OBJECT": identifier = wanted[f"OBJECT:{symbol.feature('role')}"]
        elif symbol.name == "DET": identifier = wanted[f"DET:{symbol.feature('role')}"]
        elif symbol.name == "ADJ": identifier = wanted[f"ADJ:{symbol.feature('role')}"]
        elif symbol.name == "PATIENT": identifier = wanted[f"PATIENT:{symbol.feature('type')}"]
        else: raise AssertionError(symbol.name)
        options = [candidate for candidate in expand(grammar, state, index) if candidate.trace[-1][1] == identifier]
        if len(options) != 1: raise AssertionError(f"nonunique control path {symbol.name}: {identifier}")
        state = options[0]


def live_character(state, side):
    leaf = exposed(state, side)
    return None if leaf is None else (leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right])


def zipper_expand(grammar, state, index, side):
    expected = state.residual[0] if state.residual else ""; symbol = node_map(state)[state.frontier[index]].symbol; output = []
    for candidate in expand(grammar, state, index):
        if expected and live_character(candidate, side) != expected: continue
        choices = candidate.zipper_choices + ((side, symbol.name, candidate.trace[-1][1], expected),)
        output.append(State(candidate.frontier, candidate.nodes, candidate.leaves, candidate.residual,
                            candidate.owner, candidate.turn, candidate.length, candidate.trace, choices))
    return tuple(output)


def attempts(state):
    sides = (-state.owner,) if state.residual else (state.turn, -state.turn); output = []
    for side in sides:
        leaf = exposed(state, side)
        if leaf is None: output.append({"side": side, "available": False}); continue
        char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        output.append({"side": side, "available": True, "slot": leaf.label, "word": leaf.word,
                       "character": char, "accepted": emit(state, side) is not None})
    return output


def zipper_solver(grammar, max_states=100000):
    initial = State((0,), (Node(0, grammar.start()),), (), "", 0, 1, 0, ()); queue = deque([initial]); seen = set()
    exact_rows, admitted = {}, {}; stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0, complete_trees=0, exact_closures=0)
    deepest = contradiction = None; deepest_length = contradiction_length = -1
    while queue and stats["states"] < max_states:
        state = queue.pop(); key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.turn, state.length)
        if key in seen: continue
        seen.add(key); stats["states"] += 1; offered = attempts(state)
        ledger = {"frontier": list(state.frontier), "rendered_tree": render(state), "residual": state.residual,
                  "owner": state.owner, "turn": state.turn, "length": state.length,
                  "trace": [list(item) for item in state.trace], "zipper_choices": [list(item) for item in state.zipper_choices], "attempts": offered}
        if state.length > deepest_length: deepest_length, deepest = state.length, ledger
        if any(row.get("available") and not row.get("accepted") for row in offered) and state.length > contradiction_length:
            contradiction_length, contradiction = state.length, ledger
        if complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS:
                row = audit(grammar, render(state), "complete_anaphoric_zipper", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(row["rendered"], row)
                    if row["mechanically_admitted"]: admitted.setdefault(row["rendered"], row)
            continue
        nodes = node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]; index = None
        if state.residual:
            side = -state.owner; edge = 0 if side == 1 else len(state.frontier) - 1
            if edge >= 0 and not nodes[state.frontier[edge]].terminal: index = edge
        elif unresolved:
            side = state.turn; index = unresolved[0] if side == 1 else unresolved[-1]
        if index is not None:
            for next_state in zipper_expand(grammar, state, index, side): queue.append(next_state); stats["expansions"] += 1
        for side in ((-state.owner,) if state.residual else (state.turn, -state.turn)):
            if exposed(state, side) is None: continue
            next_state = emit(state, side)
            if next_state is not None: queue.append(next_state); stats["emissions"] += 1
            else: stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()),
            "deepest_state_ledger": {"deepest_state": deepest, "deepest_contradiction_state": contradiction}}


def boundary_preflight(grammar):
    state = explicit_control(grammar); events, failed = [], None
    for step in range(1, 31):
        side = (-state.owner,) if state.residual else (state.turn,); side = side[0]; leaf = exposed(state, side)
        if leaf is None: break
        char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]; next_state = emit(state, side)
        if next_state is None:
            failed = {"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "emitter_rejected": True}; break
        state = next_state; events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "residual_after": state.residual})
    return {"rendered": render(state), "emitter_events": events, "failed_attempt": failed,
            "independent_parse": parse_complete(grammar, render(state)) is not None, "diagnostic_only": True}


def run(max_states=100000):
    grammar = AnaphoricGrammar(); result = zipper_solver(grammar, max_states); control_state = explicit_control(grammar)
    control = audit(grammar, render(control_state), "complete_anaphoric_clarification_control", control_state.trace)
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True,
                    "provenance": {"construction": "authored assertion plus map anaphoric clarification", "shared_tree": True, "seed_phrase": False},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "zipper_anaphoric_clarification_intersection",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_states": max_states,
                       "single_shared_derivation_tree": True, "bidirectional_zipper": True,
                       "central_discourse_seam_jointly_lexicalized": True, "outer_constituents_jointly_lexicalized": True,
                       "lexical_choice_requires_live_residual": True, "anaphoric_coreference_required": True,
                       "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True,
                       "endpoint_scaffold_gate": True, "corpus_generation": False},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                           "material": "authored typed assertion/map clarification; no catalogue or known palindrome material"},
            "rendered_candidates": [control], "complete_anaphoric_clarification_control": control, **result,
            "joint_boundary_preflight": boundary_preflight(grammar),
            "reader_facing_next_operator": "Replace the actual exposed anaphoric object boundary with a newly authored coreferential relation, then rerun the full zipper.",
            "scope": "Short bounded anaphoric zipper run is diagnostic; no reader-facing palindrome is claimed."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
