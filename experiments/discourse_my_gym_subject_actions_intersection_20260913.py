"""Full one-tree search for a subject-action ``my ... gym`` discourse.

This is a fresh endpoint topology after the rejected ``my/acronym`` route.
The grammar licenses one ordinary subject and four explicit events.  Noun
phrases are typed object patients, and every character is emitted from the
same fully expanded tree while the exact reversal residual is maintained.
The endpoint is an authored ``gym`` location, not a catalogue phrase or a
construction seed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, deque
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "experiments/discourse_now_unwon_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("my_gym_engine_20260913", ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = ENGINE
spec.loader.exec_module(ENGINE)
BASE = ENGINE.BASE
from llm_palindrome.admission import mechanical_admission_checks


MIN_LETTERS, MAX_LETTERS = 100, 180

# These are authored, role-specific alternatives.  They do not come from a
# sentence catalogue: all combinations retain the same subject/action/object
# semantics and are independently reparsed before any closure is admitted.
SUBJECTS = (("gardeners", "plural"),)
EVENTS = (
    ("tend", "plants", ("healthy", "young", "flowering")),
    ("water", "beds", ("clean", "narrow", "garden")),
    ("prune", "trees", ("tall", "old", "fruitful")),
    ("carry", "tools", ("heavy", "useful", "wooden")),
)
ENGINE.FIXED_LABELS.update({
    "fixed_my", "fixed_gardeners", "fixed_tend", "fixed_water", "fixed_prune",
    "fixed_carry", "fixed_to", "fixed_the", "fixed_and", "fixed_gym",
    "fixed_plants", "fixed_beds", "fixed_trees", "fixed_tools",
    "fixed_healthy", "fixed_young", "fixed_flowering", "fixed_clean",
    "fixed_narrow", "fixed_garden", "fixed_tall", "fixed_old",
    "fixed_fruitful", "fixed_heavy", "fixed_useful", "fixed_wooden",
})


class SubjectActionGrammar:
    """A finite typed CFG whose complete trees are ordinary coordinated prose."""

    def __init__(self, max_depth: int = 0):
        self.max_depth = max_depth

    def start(self):
        return BASE.sym("D", depth=str(self.max_depth))

    def terminal(self, symbol):
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs):
        f = dict(lhs.features)
        if lhs.name == "D":
            rhs = [
                BASE.sym("T", label="fixed_my", form="my"),
                BASE.sym("T", label="fixed_gardeners", form="gardeners"),
            ]
            for index, (verb, noun, adjectives) in enumerate(EVENTS):
                if index:
                    rhs.append(BASE.sym("T", label=f"fixed_and_{index}", form="and"))
                rhs.extend((
                    BASE.sym("T", label=f"fixed_{verb}", form=verb),
                    BASE.sym("NP", role=f"patient_{index + 1}", type="object",
                             frame=noun, number="plural", depth="0"),
                ))
            rhs.extend((
                BASE.sym("T", label="fixed_to", form="to"),
                BASE.sym("T", label="fixed_the", form="the"),
                BASE.sym("T", label="fixed_gym", form="gym"),
            ))
            return (BASE.Production(
                "D:my-gardeners-four-subject-action-events-to-gym", lhs, tuple(rhs)),)
        if lhs.name == "NP":
            noun = f["frame"]
            row = next((event for event in EVENTS if event[1] == noun), None)
            if row is None:
                return ()
            adjectives = row[2]
            # The variants differ only in ordinary descriptive vocabulary; the
            # object role and plural agreement remain fixed in every production.
            return tuple(
                BASE.Production(
                    f"NP:{f['role']}:{noun}:{adj}:base", lhs,
                    (BASE.sym("T", label="fixed_the", form="the"),
                     BASE.sym("T", label=f"fixed_{adj}", form=adj),
                     BASE.sym("T", label=f"fixed_{noun}", form=noun)),
                ) for adj in adjectives
            )
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
                (p.identifier, [x.name + str(x.features) for x in p.rhs])
                for p in productions
            ]))
            queue.extend(x for p in productions for x in p.rhs if self.terminal(x) is None)
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def parse_complete(grammar, text):
    """Independently reparse all grammar productions, with backtracking."""
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
            answers = ((BASE.Tree(symbol, f"lex:{term}", (), term), offset + 1),) \
                if offset < len(tokens) and tokens[offset] == term else ()
            memo[key] = answers
            return answers
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
                (BASE.Tree(symbol, production.identifier, tuple(children)), end)
                for children, end in partials
            )
        memo[key] = tuple(answers)
        return memo[key]

    return next((tree for tree, end in parses(grammar.start(), 0)
                 if end == len(tokens)), None)


def exact_audit(text):
    tape = "".join(c for c in text.lower() if "a" <= c <= "z")
    mismatches = [
        (i, len(tape) - i - 1)
        for i in range(len(tape) // 2)
        if tape[i] != tape[-i - 1]
    ]
    return {
        "exact": bool(tape) and not mismatches,
        "letters": len(tape),
        "mismatches": mismatches,
        "normalized_sha256": sha256(tape.encode()).hexdigest(),
    }


def semantic_witness(tree):
    roles, frames = [], []
    agreement_ok = True
    valency_ok = True

    def visit(node):
        nonlocal agreement_ok, valency_ok
        if node.symbol.name == "NP":
            roles.append(dict(node.symbol.features))
            frames.append(node.symbol.feature("frame"))
            if node.symbol.feature("type") != "object" or node.symbol.feature("number") != "plural":
                agreement_ok = False
        for child in node.children:
            visit(child)

    if tree:
        visit(tree)
    expected_frames = [event[1] for event in EVENTS]
    if frames != expected_frames:
        valency_ok = False
    root_ok = bool(
        tree and tree.symbol.name == "D" and len(tree.children) >= 3
        and tree.children[0].terminal == "my"
        and tree.children[1].terminal == "gardeners"
        and tree.children[-1].terminal == "gym"
    )
    if not root_ok:
        agreement_ok = False
    subject_action_map = {verb: noun for verb, noun, _ in EVENTS}
    return {
        "semantic_roles": roles,
        "event_frames": frames,
        "subject": "gardeners",
        "subject_number": "plural",
        "subject_action_map": subject_action_map,
        "subject_action_ok": bool(root_ok and subject_action_map == {
            "tend": "plants", "water": "beds", "prune": "trees", "carry": "tools"
        }),
        "relative_count": 0,
        "agreement_ok": agreement_ok,
        "valency_ok": valency_ok,
        "complete_tree": bool(tree and tree.symbol.name == "D"),
    }


def audit(grammar, text, kind, trace):
    tree = parse_complete(grammar, text)
    exact = exact_audit(text)
    witness = semantic_witness(tree)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
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
        "record_kind": kind,
        "rendered": text,
        "independent_exact_audit": exact,
        "independent_parse": tree is not None,
        "feature_witness": witness,
        "central_admission": central,
        "mechanically_admitted": not codes,
        "rejection_codes": codes,
        "shared_tree_trace": list(trace),
        "reader_status": "unreviewed; programmatic checks do not certify readability",
    }


def explicit_control(grammar):
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    choices = {event[1]: event[2][0] for event in EVENTS}
    while True:
        nodes = BASE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved:
            return state
        index = unresolved[0]
        symbol = nodes[state.frontier[index]].symbol
        f = dict(symbol.features)
        if symbol.name == "D":
            identifier = "D:my-gardeners-four-subject-action-events-to-gym"
        elif symbol.name == "NP":
            noun = f["frame"]
            identifier = f"NP:{f['role']}:{noun}:{choices[noun]}:base"
        else:
            raise AssertionError(symbol.name)
        options = [candidate for candidate in BASE.expand(grammar, state, index)
                   if candidate.trace[-1][1] == identifier]
        if len(options) != 1:
            raise AssertionError(f"nonunique control path {symbol.name} {identifier}")
        state = options[0]


def boundary_preflight():
    """Replay the first unequal boundary with the real emitter."""
    grammar = SubjectActionGrammar(0)
    state = explicit_control(grammar)
    events, failed = [], None
    for step in range(1, 31):
        side = 1 if step % 2 else -1
        nodes, leaves = BASE.node_map(state), BASE.leaf_map(state)
        active = [ref for ref in (state.frontier if side == 1 else reversed(state.frontier))
                  if nodes[ref].terminal]
        if not active:
            break
        leaf = leaves[active[0]]
        char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        next_state = BASE.emit(state, side)
        if next_state is None:
            failed = {
                "step": step, "side": side, "slot": leaf.label,
                "word": leaf.word, "character": char, "emitter_rejected": True,
            }
            break
        state = next_state
        events.append({
            "step": step, "side": side, "slot": leaf.label,
            "word": leaf.word, "character": char, "residual_after": state.residual,
        })
    rendered = BASE.render(state)
    return {
        "rendered": rendered,
        "emitter_events": events,
        "failed_attempt": failed,
        "independent_parse": parse_complete(grammar, rendered) is not None,
        "diagnostic_only": True,
    }


def solver(grammar, max_states=100000):
    prior = ENGINE.audit
    ENGINE.audit = audit
    try:
        return ENGINE.solver(grammar, max_states=max_states)
    finally:
        ENGINE.audit = prior


def run(max_states=100000):
    grammar = SubjectActionGrammar(0)
    result = solver(grammar, max_states)
    control_state = explicit_control(grammar)
    control_text = BASE.render(control_state)
    control = audit(grammar, control_text, "complete_subject_action_control", control_state.trace)
    control.update({
        "diagnostic_only": True,
        "grammar_tree_fully_expanded": True,
        "provenance": {
            "construction": "four explicit ordinary subject-action/object events",
            "shared_tree": True,
            "endpoint": "authored lexicon-backed gym location",
        },
        "reader_status": "grammar control only; not a palindrome candidate or readability evidence",
    })
    return {
        "status": "discourse_my_gym_subject_actions_intersection",
        "config": {
            "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
            "max_depth": grammar.max_depth, "max_states": max_states,
            "single_shared_derivation_tree": True,
            "character_residual_during_derivation": True,
            "outer_boundary_crossing": "my is matched through longer gym and into gardeners",
            "explicit_clause_count": 4,
            "recursive_padding": False,
            "closure_requires_all_leaves_consumed": True,
            "independent_complete_reparse": True,
            "corpus_generation": False,
            "endpoint_scaffold_gate": True,
        },
        "provenance": {
            "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "grammar_sha256": grammar.digest(),
            "material": "authored subject-action grammar and typed ordinary object frames; no catalogue or known palindrome material",
        },
        "complete_subject_action_control": control,
        **result,
        "joint_boundary_preflight": boundary_preflight(),
        "reader_facing_next_operator": "Use the replayed deepest contradiction to replace the exact exposed action/object boundary with a semantically licensed frame, then rerun the full scheduler.",
        "scope": "Zero-closure bounded run is diagnostic; no reader-facing palindrome is claimed.",
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
    print(json.dumps({
        "out": str(args.out),
        "states": result["stats"]["states"],
        "exact_closures": len(result["exact_closures"]),
    }, indent=2))


if __name__ == "__main__":
    main()
