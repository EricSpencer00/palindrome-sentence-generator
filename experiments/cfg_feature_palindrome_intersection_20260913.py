"""Single-tree CFG/feature-grammar intersection with exact palindromes.

The search maintains one ordered derivation frontier, not two independently
generated halves. Every grammar expansion replaces one node in that shared
frontier; left/right emission advances terminal leaves of the same tree while
consuming a character-level reversal residual. A closure is recorded only
when every leaf is consumed and an independent complete reparse succeeds.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
WORD = re.compile(r"[a-z]+")


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
class Event:
    name: str
    agent_type: str
    patient_type: str
    singular_verb: str
    plural_verb: str


EVENTS = (
    Event("guide", "person", "person", "guides", "guide"),
    Event("teach", "person", "person", "teaches", "teach"),
    Event("carry", "person", "object", "carries", "carry"),
    Event("place", "person", "object", "places", "place"),
    Event("watch", "person", "object", "watches", "watch"),
)
NOUNS = {
    ("person", "sing"): ("artist", "baker", "doctor", "poet", "teacher", "guard"),
    ("person", "plur"): ("artists", "bakers", "doctors", "poets", "teachers", "guards"),
    ("object", "sing"): ("book", "map", "parcel", "painting", "letter", "basket"),
    ("object", "plur"): ("books", "maps", "parcels", "paintings", "letters", "baskets"),
}
ADJECTIVES = ("calm", "careful", "kind", "quiet", "brave", "patient", "bright", "young")
LOCATIONS = ("garden", "studio", "school", "market", "harbor", "office")


class FeatureGrammar:
    """A finite-depth context-free grammar with feature-unified productions."""

    def __init__(self, max_clauses: int = 5):
        self.max_clauses = max_clauses

    def start(self) -> Symbol:
        return sym("S", depth=str(self.max_clauses))

    def terminal(self, symbol: Symbol) -> str | None:
        return "and" if symbol.name == "CONJ" else (symbol.feature("form") if symbol.name == "T" else None)

    def productions(self, lhs: Symbol) -> tuple[Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "S":
            d = int(f["depth"])
            one = Production(f"S:{d}:one", lhs, (sym("CLAUSE"),))
            if d == 1:
                return (one,)
            return (one, Production(f"S:{d}:coord", lhs,
                                    (sym("CLAUSE"), sym("CONJ"), sym("S", depth=str(d - 1)))))
        if lhs.name == "CLAUSE":
            return tuple(Production(f"CLAUSE:{e.name}:{n}", lhs,
                                    (sym("NP", role="agent", type=e.agent_type, number=n),
                                     sym("VP", event=e.name, agent_number=n)))
                           for e in EVENTS for n in ("sing", "plur"))
        if lhs.name == "VP":
            e = next((x for x in EVENTS if x.name == f["event"]), None)
            if e is None:
                return ()
            number = f["agent_number"]
            verb = e.singular_verb if number == "sing" else e.plural_verb
            patient = sym("NP", role="patient", type=e.patient_type, number="sing")
            v = sym("V", event=e.name, number=number, form=verb)
            return (Production(f"VP:{e.name}:{number}:bare", lhs, (v, patient)),
                    Production(f"VP:{e.name}:{number}:setting", lhs,
                               (v, patient, sym("PP", role="setting"))))
        if lhs.name == "NP":
            role, typ, n = f["role"], f["type"], f["number"]
            det = "a" if n == "sing" else "the"
            rows = []
            for adjective in ("",) + ADJECTIVES:
                for noun in NOUNS[(typ, n)]:
                    rhs = [sym("DET", number=n, form=det)]
                    if adjective:
                        rhs.append(sym("ADJ", form=adjective))
                    rhs.append(sym("N", role=role, type=typ, number=n, form=noun))
                    rows.append(Production(f"NP:{role}:{typ}:{n}:{adjective}:{noun}", lhs, tuple(rhs)))
            return tuple(rows)
        if lhs.name == "PP":
            return tuple(Production(f"PP:{place}", lhs,
                                    (sym("PREP", form="in"), sym("DET", number="sing", form="the"),
                                     sym("LOC", form=place))) for place in LOCATIONS)
        if lhs.name in {"DET", "ADJ", "N", "V", "PREP", "LOC"}:
            return (Production(f"{lhs.name}:{lhs.feature('form')}", lhs,
                                (sym("T", label=lhs.name, form=lhs.feature("form")),)),)
        return ()

    def digest(self) -> str:
        seen, queue, rows = set(), deque([self.start()]), []
        while queue:
            lhs = queue.popleft()
            if lhs in seen:
                continue
            seen.add(lhs)
            prods = self.productions(lhs)
            rows.append((lhs.name, lhs.features, [(p.identifier, [x.name + str(x.features) for x in p.rhs]) for p in prods]))
            queue.extend(x for p in prods for x in p.rhs if self.terminal(x) is None)
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


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
    length: int
    trace: tuple[tuple[int, str], ...]


def node_map(state: State) -> dict[int, Node]:
    return {n.identifier: n for n in state.nodes}


def leaf_map(state: State) -> dict[int, Leaf]:
    return {x.identifier: x for x in state.leaves}


def expand(grammar: FeatureGrammar, state: State, frontier_index: int) -> tuple[State, ...]:
    nodes, leaves = node_map(state), leaf_map(state)
    ref = state.frontier[frontier_index]
    target = nodes[ref]
    answers = []
    for prod in grammar.productions(target.symbol):
        next_id = max(nodes, default=-1) + 1
        child_ids, next_leaves, next_nodes = [], list(state.leaves), dict(nodes)
        for child in prod.rhs:
            cid = next_id
            next_id += 1
            word = grammar.terminal(child)
            next_nodes[cid] = Node(cid, child, terminal=word or "")
            child_ids.append(cid)
            if word is not None:
                next_leaves.append(Leaf(cid, cid, word, child.feature("label", child.name)))
        next_nodes[ref] = Node(ref, target.symbol, production=prod.identifier, children=tuple(child_ids))
        frontier = state.frontier[:frontier_index] + tuple(child_ids) + state.frontier[frontier_index + 1:]
        answers.append(State(frontier, tuple(next_nodes.values()), tuple(next_leaves), state.residual,
                             state.owner, state.length, state.trace + ((ref, prod.identifier),)))
    return tuple(answers)


def ordered_leaves(state: State) -> tuple[Leaf, ...]:
    nodes = node_map(state)
    result = []
    def visit(ref: int) -> None:
        node = nodes[ref]
        if node.terminal:
            leaf = leaf_map(state).get(ref)
            if leaf:
                result.append(leaf)
            return
        for child in node.children:
            visit(child)
    visit(0)
    return tuple(result)


def emit(state: State, side: int) -> State | None:
    nodes = node_map(state)
    active = [ref for ref in state.frontier if nodes[ref].terminal]
    if not active:
        return None
    ref = active[0] if side == 1 else active[-1]
    leaf = leaf_map(state)[ref]
    if leaf.left + leaf.right >= len(leaf.word):
        return None
    char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
    new_leaf = Leaf(leaf.identifier, leaf.node, leaf.word, leaf.label,
                    leaf.left + 1 if side == 1 else leaf.left,
                    leaf.right + 1 if side == -1 else leaf.right)
    if side == 1:
        next_leaves = tuple(new_leaf if x.identifier == leaf.identifier else x for x in state.leaves)
    else:
        next_leaves = tuple(new_leaf if x.identifier == leaf.identifier else x for x in state.leaves)
    residual, owner = state.residual, state.owner
    if not residual:
        residual, owner = char, side
    elif owner == side:
        residual += char
    elif residual[0] != char:
        return None
    else:
        # Remaining debt still belongs to the side that produced it.  The
        # consuming side is not the owner of the unmatched continuation.
        prior_owner = owner
        residual = residual[1:]
        owner = prior_owner if residual else 0
    next_frontier = state.frontier
    if new_leaf.left + new_leaf.right == len(new_leaf.word):
        next_frontier = tuple(x for x in state.frontier if x != leaf.identifier)
    return State(next_frontier, state.nodes, next_leaves, residual, owner, state.length + 1, state.trace)


def complete(state: State) -> bool:
    return not state.frontier and all(x.left + x.right == len(x.word) for x in state.leaves)


def render(state: State) -> str:
    return " ".join(x.word for x in ordered_leaves(state))


def finish_one_tree(grammar: FeatureGrammar, state: State, limit: int = 1000) -> State:
    """Complete one remaining grammar tree for an independently auditable rejection."""
    for _ in range(limit):
        nodes = node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved:
            return state
        target_index = next((i for i in unresolved
                             if nodes[state.frontier[i]].symbol.name == "S"
                             and nodes[state.frontier[i]].symbol.feature("depth") == "2"), unresolved[0])
        options = expand(grammar, state, target_index)
        if not options:
            return state
        # Prefer a four-clause bounded coordination (the depth-two node takes
        # its one-clause branch), then longest lexical realizations. This keeps
        # retained diagnostics in the 100--180-letter search band.
        state = options[0] if target_index != unresolved[0] or nodes[state.frontier[target_index]].symbol.name == "S" and nodes[state.frontier[target_index]].symbol.feature("depth") == "2" else options[-1]
    return state


@dataclass(frozen=True)
class Tree:
    symbol: Symbol
    production: str
    children: tuple["Tree", ...]
    terminal: str = ""


def parse_tree(grammar: FeatureGrammar, text: str) -> Tree | None:
    if text != text.strip() or re.sub(r"[a-z ]", "", text.lower()):
        return None
    tokens = tuple(WORD.findall(text.lower()))
    memo: dict[tuple[Symbol, int], tuple[tuple[Tree, int], ...]] = {}
    def parse_all(symbol: Symbol, offset: int) -> tuple[tuple[Tree, int], ...]:
        key = (symbol, offset)
        if key in memo:
            return memo[key]
        terminal = grammar.terminal(symbol)
        if terminal is not None:
            out = ((Tree(symbol, f"lex:{terminal}", (), terminal), offset + 1),) if offset < len(tokens) and tokens[offset] == terminal else ()
            memo[key] = out
            return out
        answers = []
        for prod in grammar.productions(symbol):
            partials: list[tuple[list[Tree], int]] = [([], offset)]
            for child in prod.rhs:
                next_partials = []
                for children, cursor in partials:
                    for node, end in parse_all(child, cursor):
                        next_partials.append((children + [node], end))
                partials = next_partials
                if not partials:
                    break
            answers.extend((Tree(symbol, prod.identifier, tuple(children)), cursor)
                           for children, cursor in partials)
        memo[key] = tuple(answers)
        return memo[key]
    return next((tree for tree, end in parse_all(grammar.start(), 0) if end == len(tokens)), None)


def features(tree: Tree) -> dict[str, object]:
    roles, events, agreement, valency = [], [], True, True
    def visit(node: Tree) -> None:
        nonlocal agreement, valency
        if node.symbol.name == "NP":
            roles.append(dict(node.symbol.features))
        if node.symbol.name == "VP":
            e = next((x for x in EVENTS if x.name == node.symbol.feature("event")), None)
            if e is not None:
                events.append(e.name)
            if e is None or len(node.children) not in (2, 3) or node.children[0].symbol.name != "V" or node.children[1].symbol.name != "NP":
                valency = False
            elif node.children[1].symbol.feature("type") != e.patient_type:
                valency = False
            if e and node.children[0].symbol.feature("number") != node.symbol.feature("agent_number"):
                agreement = False
        for child in node.children:
            visit(child)
    visit(tree)
    return {"semantic_roles": roles, "events": events, "agreement_ok": agreement, "valency_ok": valency, "complete_tree": tree.symbol.name == "S"}


def exact_audit(text: str) -> dict[str, object]:
    tape = "".join(c for c in text.lower() if "a" <= c <= "z")
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]
    bounds, cursor = [], 0
    for word in WORD.findall(text.lower()):
        cursor += len(word); bounds.append(cursor)
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest(),
            "shifted_word_boundaries": sorted(set(bounds[:-1]) - {len(tape) - x for x in bounds[:-1]})}


def audit(grammar: FeatureGrammar, text: str, kind: str, trace: tuple[tuple[int, str], ...], reason: str | None = None) -> dict[str, object]:
    exact, tree = exact_audit(text), parse_tree(grammar, text)
    witness = features(tree) if tree else {"complete_tree": False, "agreement_ok": False, "valency_ok": False}
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [k for k, value in central.items() if not value]
    if tree is None: codes.append("independent_complete_reparse_failed")
    if not witness.get("agreement_ok"): codes.append("agreement_failure")
    if not witness.get("valency_ok"): codes.append("valency_failure")
    if reason and reason not in codes: codes.append(reason)
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact,
            "independent_parse": tree is not None, "feature_witness": witness,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "shared_tree_trace": list(trace),
            "reader_status": "unreviewed; programmatic checks do not certify readability"}


def intersect(grammar: FeatureGrammar, *, max_states: int = 50000, max_records: int = 100) -> dict[str, object]:
    root = Node(0, grammar.start())
    initial = State((0,), (root,), (), "", 0, 0, ())
    long_control = finish_one_tree(grammar, initial)
    long_control_base = audit(grammar, render(long_control), "complete_shared_tree_diagnostic", long_control.trace)
    queue, seen, closures, exact_rows, rejects, tree_rejects = deque([initial]), set(), {}, {}, {}, {}
    stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0,
                    complete_trees=0, exact_closures=0, complete_reparse_rejections=0,
                    complete_tree_rejections=0)

    def retain_tree_rejection(kind: str, reason: str, origin: State) -> None:
        if len(tree_rejects) >= max_records:
            return
        row = dict(long_control_base)
        row["record_kind"] = kind
        row["rejection_codes"] = list(long_control_base["rejection_codes"])
        if reason not in row["rejection_codes"]:
            row["rejection_codes"].append(reason)
        row["diagnostic_origin_trace"] = list(origin.trace)
        if row["rendered"] not in tree_rejects:
            tree_rejects[row["rendered"]] = row
            stats["complete_tree_rejections"] += 1
    while queue and stats["states"] < max_states:
        # Depth-first frontier completion reaches fully expanded trees quickly;
        # palindrome residuals still prune every character contradiction.
        state = queue.pop(); stats["states"] += 1
        key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.length)
        if key in seen: continue
        seen.add(key)
        if complete(state):
            stats["complete_trees"] += 1
            text = render(state)
            if MIN_LETTERS <= state.length <= MAX_LETTERS:
                if state.residual == state.residual[::-1]:
                    row = audit(grammar, text, "complete_shared_tree_closure", state.trace)
                    if row["independent_exact_audit"]["exact"]:
                        stats["exact_closures"] += 1
                        exact_rows.setdefault(text, row)
                        if row["mechanically_admitted"]: closures.setdefault(text, row)
                        elif len(rejects) < max_records:
                            rejects.setdefault(text, row); stats["complete_reparse_rejections"] += 1
                elif len(tree_rejects) < max_records:
                    # This is a complete, independently reparsable grammar
                    # tree whose outer characters matched until the remaining
                    # centre failed; retain it as evidence, never as a result.
                    row = audit(grammar, text, "complete_tree_central_residual_rejection", state.trace,
                                "central_residual_not_palindromic")
                    tree_rejects.setdefault(text, row); stats["complete_tree_rejections"] += 1
            continue
        # Expand only one nonterminal at the current outer frontier. This edits
        # the single tree; no second grammar is ever stitched to it.
        nodes_now = node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes_now[ref].terminal]
        if unresolved:
            # Finish the exposed left edge first, then the exposed right edge,
            # and only then move inward. One expansion at a time avoids a
            # Cartesian product of unrelated interior choices.
            if not nodes_now[state.frontier[0]].terminal:
                expansion_indices = (0,)
            elif not nodes_now[state.frontier[-1]].terminal:
                expansion_indices = (len(state.frontier) - 1,)
            else:
                expansion_indices = (unresolved[0],)
        else:
            expansion_indices = ()
        for i in expansion_indices:
            for next_state in expand(grammar, state, i):
                queue.append(next_state); stats["expansions"] += 1
        left_leaf = any(node_map(state)[ref].terminal for ref in state.frontier[:1])
        right_leaf = any(node_map(state)[ref].terminal for ref in state.frontier[-1:])
        if not state.residual and left_leaf and right_leaf:
            left_ref, right_ref = state.frontier[0], state.frontier[-1]
            left_word = leaf_map(state)[left_ref].word
            right_leaf_state = leaf_map(state)[right_ref]
            right_word = right_leaf_state.word
            if left_word[0] != right_word[-1]:
                # The one-tree frontier already exposes incompatible outer
                # characters; do not spend states expanding its interior.
                if len(tree_rejects) < max_records:
                    retain_tree_rejection("complete_shared_tree_outer_character_rejection",
                                          "outer_character_contradiction", state)
                stats["residual_contradictions"] += 1
                continue
        sides = (-state.owner,) if state.residual else (1, -1)
        for side in sides:
            if (side == 1 and not left_leaf) or (side == -1 and not right_leaf): continue
            next_state = emit(state, side)
            if next_state is not None:
                queue.append(next_state); stats["emissions"] += 1
            elif state.residual:
                stats["residual_contradictions"] += 1
                if len(tree_rejects) < max_records:
                    retain_tree_rejection("complete_shared_tree_residual_rejection",
                                          "character_residual_contradiction", state)
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(closures.values()),
            "complete_reparse_rejections": list(rejects.values()),
            "complete_tree_rejections": list(tree_rejects.values()),
            "grammar_derivation_digest": grammar.digest()}


def run(*, max_states: int = 50000) -> dict[str, object]:
    grammar = FeatureGrammar(5); result = intersect(grammar, max_states=max_states)
    return {"status": "single_shared_tree_cfg_feature_palindrome_intersection", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_clauses": grammar.max_clauses,
        "max_states": max_states, "single_shared_derivation_tree": True,
        "character_residual_during_derivation": True, "closure_requires_all_leaves_consumed": True,
        "independent_complete_reparse": True, "corpus_generation": False,
        "catalogue_used_only_for_central_exclusion": True},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                       "material": "task-authored feature grammar terminals only; no corpus or catalogue construction material"},
        "grammar": {"events": [asdict(x) for x in EVENTS], "noun_features": {str(k): v for k, v in NOUNS.items()},
                    "adjectives": ADJECTIVES, "locations": LOCATIONS}, **result,
        "reader_facing_next_operator": "Add a typed lexical substitution transducer over this same shared tree while retaining full-leaf closure and independent reparse.",
        "scope": "This is evidence about one finite grammar; exactness and feature parsing do not certify human readability."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=50000)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "closures": len(result["exact_closures"]), "complete_reparse_rejections": len(result["complete_reparse_rejections"])}, indent=2))


if __name__ == "__main__": main()
