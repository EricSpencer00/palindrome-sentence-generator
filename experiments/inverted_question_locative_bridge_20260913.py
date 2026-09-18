"""Original full bridge through a typed locative relative.

This branch does not import any known palindrome or catalogue text.  Its
authored bridge is the ordinary locative relative ``a temple ... met at``:
the terminal stream ``met at`` reverses to ``t-a-t-e-m``, matching the fixed
``t`` from ``it`` plus the initial ``a tem`` of the variable NP.  The probe is
not itself a palindrome; it only preflights this local seam before the
>=100-letter connected-tree search.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/inverted_question_edge_unit_tree_20260913.py"
spec = importlib.util.spec_from_file_location("edge_unit_for_locative_bridge_20260913", SOURCE)
EDGE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = EDGE
spec.loader.exec_module(EDGE)
BASE, QUESTION = EDGE.BASE, EDGE.QUESTION
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = EDGE.MIN_LETTERS, EDGE.MAX_LETTERS
FIXED_LABELS = EDGE.FIXED_LABELS
TEMPLE = EDGE.EdgeLexeme("temple", "noun", "location", units=("tem", "ple"), rank=1)
MET_AT = EDGE.EdgeLexeme("met", "verb", "location", units=("met",), rank=1)
MORPHOLOGY = tuple(x for x in EDGE.MORPHOLOGY if x.form != "met") + (TEMPLE, MET_AT)
LEXEME_BY_FORM = dict(EDGE.LEXEME_BY_FORM, temple=TEMPLE, met=MET_AT)


class LocativeBridgeGrammar(EDGE.EdgeUnitGrammar):
    """Feature grammar with a location-headed ``V ... at`` relative."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "Q":
            return (BASE.Production("Q:was-it-location-i-saw", lhs, (
                BASE.sym("T", label="fixed_was", form="was"), BASE.sym("T", label="fixed_it", form="it"),
                BASE.sym("NP", role="complement", type="location", number="sing", depth=f["depth"]),
                BASE.sym("T", label="fixed_i", form="i"), BASE.sym("T", label="fixed_saw", form="saw"))),)
        if lhs.name == "REL" and f["head_type"] == "location":
            depth = int(f["depth"])
            subject = BASE.sym("NP", role="relative_agent", type="person", number="sing", depth=str(depth))
            return (BASE.Production(f"REL:location:{depth}:locative", lhs, (
                BASE.sym("T", label="relative_that", form="that"), subject,
                BASE.sym("V", event="locative_gap", patient_type="location", subject_number="sing"),
                BASE.sym("T", label="relative_prep", form="at"))),)
        if lhs.name == "V" and f.get("patient_type") == "location":
            return (BASE.Production("V:locative_met", lhs, (
                BASE.sym("LEX_SLOT", category="verb", patient_type="location", role="relative_verb"),)),)
        if lhs.name == "LEX_SLOT" and f.get("category") == "noun" and f.get("type") == "location":
            return (BASE.Production("LEX_SLOT:temple", lhs, (
                BASE.sym("T", label=f"noun_{f['role']}", form="temple"),)),)
        if lhs.name == "LEX_SLOT" and f.get("category") == "verb" and f.get("patient_type") == "location":
            return (BASE.Production("LEX_SLOT:met", lhs, (
                BASE.sym("T", label="relative_verb", form="met"),)),)
        return super().productions(lhs)


def feature_witness(tree: BASE.Tree | None) -> dict[str, object]:
    roles, relatives, agreement, valency = [], 0, True, True
    def visit(node: BASE.Tree) -> None:
        nonlocal relatives, agreement, valency
        if node.symbol.name == "NP": roles.append(dict(node.symbol.features))
        if node.symbol.name == "REL":
            relatives += 1
            if len(node.children) == 4:
                if node.children[0].terminal != "that" or node.children[2].symbol.name != "V" or node.children[3].terminal != "at":
                    valency = False
                elif node.children[2].symbol.feature("patient_type") != node.symbol.feature("head_type"):
                    valency = False
                elif node.children[1].symbol.feature("number") != node.children[2].symbol.feature("subject_number"):
                    agreement = False
            elif len(node.children) == 3:
                if node.children[0].terminal != "that" or node.children[2].symbol.name != "V": valency = False
                elif node.children[2].symbol.feature("patient_type") != node.symbol.feature("head_type"): valency = False
                elif node.children[1].symbol.feature("number") != node.children[2].symbol.feature("subject_number"): agreement = False
            else: valency = False
        for child in node.children: visit(child)
    if tree: visit(tree)
    return {"semantic_roles": roles, "relative_count": relatives, "agreement_ok": agreement,
            "valency_ok": valency, "complete_tree": bool(tree and tree.symbol.name == "Q")}


def audit(grammar: LocativeBridgeGrammar, text: str, kind: str, trace: tuple[tuple[int, str], ...]) -> dict[str, object]:
    exact, tree = QUESTION.exact_audit(text), QUESTION.parse_tree(grammar, text)
    witness = feature_witness(tree)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if tree is None: codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]: codes.append("agreement_failure")
    if not witness["valency_ok"]: codes.append("valency_failure")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact,
            "independent_parse": tree is not None, "feature_witness": witness,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "shared_tree_trace": list(trace),
            "reader_status": "unreviewed; programmatic measures do not certify readability"}


def content_words(state: BASE.State) -> tuple[str, ...]:
    return tuple(x.word for x in BASE.ordered_leaves(state) if x.label not in FIXED_LABELS)


def lexical_unit_accepts(state: BASE.State, side: int, added: list[BASE.Leaf]) -> bool:
    if not state.residual or state.owner == side: return True
    expected = state.residual[0]
    for leaf in added:
        if leaf.label in FIXED_LABELS: continue
        lexeme = LEXEME_BY_FORM.get(leaf.word)
        if lexeme is None: continue
        unit = EDGE.active_unit(lexeme, leaf, side)
        if unit[0 if side == 1 else -1] != expected: return False
    return True


def expand_edge(grammar: LocativeBridgeGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = set(content_words(state)); previous = BASE.leaf_map(state); output = []
    for candidate in BASE.expand(grammar, state, index):
        added = [x for x in candidate.leaves if x.identifier not in previous]
        if not lexical_unit_accepts(state, side, added): continue
        if any(x.word in used or x.word == x.word[::-1] for x in added if x.label not in FIXED_LABELS): continue
        output.append(candidate)
    return tuple(output)


def solver(grammar: LocativeBridgeGrammar, *, max_states: int = 100000) -> dict[str, object]:
    root = BASE.Node(0, grammar.start()); initial = BASE.State((0,), (root,), (), "", 0, 0, ())
    queue, seen, exact_rows, admitted, rejected = deque([initial]), set(), {}, {}, {}
    stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0,
                    complete_trees=0, exact_closures=0, independent_reparse_rejections=0)
    while queue and stats["states"] < max_states:
        state = queue.pop(); stats["states"] += 1
        key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.length)
        if key in seen: continue
        seen.add(key)
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS and state.residual == state.residual[::-1]:
                text = BASE.render(state); row = audit(grammar, text, "complete_locative_bridge_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]: admitted.setdefault(text, row)
                    elif len(rejected) < 100: rejected.setdefault(text, row); stats["independent_reparse_rejections"] += 1
            continue
        nodes = BASE.node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        index = None
        if state.residual:
            side = -state.owner; edge = 0 if side == 1 else len(state.frontier) - 1
            if not nodes[state.frontier[edge]].terminal: index = edge
        elif unresolved:
            if not nodes[state.frontier[0]].terminal: index, side = 0, 1
            elif not nodes[state.frontier[-1]].terminal: index, side = len(state.frontier) - 1, -1
            else: index, side = unresolved[0], 1
        if index is not None:
            for next_state in expand_edge(grammar, state, index, side): queue.append(next_state); stats["expansions"] += 1
        left_leaf, right_leaf = bool(nodes[state.frontier[0]].terminal), bool(nodes[state.frontier[-1]].terminal)
        sides = (-state.owner,) if state.residual else (1, -1)
        for side in sides:
            if (side == 1 and not left_leaf) or (side == -1 and not right_leaf): continue
            next_state = BASE.emit(state, side)
            if next_state is not None: queue.append(next_state); stats["emissions"] += 1
            else: stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()),
            "independent_reparse_rejections": list(rejected.values())}


def bridge_preflight(grammar: LocativeBridgeGrammar) -> dict[str, object]:
    text = "was it a temple that a teacher met at i saw"
    normalized = "".join(c for c in text if "a" <= c <= "z")
    seam_pairs = [(normalized[i], normalized[-1 - i]) for i in range(4, 9)]
    return {"rendered": text, "normalized": normalized, "bridge_stream": "tatem",
            "terminal_reverse_stream": "tatem", "seam_pairs": seam_pairs,
            "full_multi_character_match": seam_pairs == [("t", "t"), ("a", "a"), ("t", "t"), ("e", "e"), ("m", "m")],
            "independent_parse": QUESTION.parse_tree(grammar, text) is not None,
            "exact_audit": QUESTION.exact_audit(text), "construction_material": "authored temple/met-at locative grammar only"}


def choose(grammar: LocativeBridgeGrammar, state: BASE.State, index: int, identifier: str) -> BASE.State:
    options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == identifier]
    if len(options) != 1: raise AssertionError(f"nonunique control production {identifier}")
    return options[0]


def explicit_control(grammar: LocativeBridgeGrammar) -> BASE.State:
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    adjectives = iter(("quiet", "careful", "patient", "brave", "young", "gentle", "steady"))
    people = iter(("teacher", "guard", "writer", "singer", "farmer", "nurse"))
    verbs = iter(("greeted", "trusted", "followed", "helped", "joined"))
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved: break
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "Q": ident = "Q:was-it-location-i-saw"
        elif symbol.name == "NP":
            d = int(f["depth"]); ident = f"NP:{f['role']}:{f['type']}:{d}:{'modified-relative' if d > 0 else 'modified'}"
        elif symbol.name == "DET_SLOT": ident = "DET_SLOT:a"
        elif symbol.name == "ADJ_SLOT": ident = f"ADJ_SLOT:{next(adjectives)}"
        elif symbol.name == "LEX_SLOT":
            if f["category"] == "noun": form = "temple" if f["type"] == "location" else next(people)
            else: form = "met" if f["patient_type"] == "location" else next(verbs)
            ident = f"LEX_SLOT:{form}"
        elif symbol.name == "REL": ident = f"REL:location:{f['depth']}:locative" if f["head_type"] == "location" else f"REL:{f['head_type']}:{f['depth']}"
        elif symbol.name == "V": ident = "V:locative_met" if f.get("patient_type") == "location" else "V:edge_unit"
        else: raise AssertionError(symbol.name)
        state = choose(grammar, state, index, ident)
    if any(not BASE.node_map(state)[ref].terminal for ref in state.frontier): raise AssertionError("control not fully expanded")
    return state


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = LocativeBridgeGrammar(6); result = solver(grammar, max_states=max_states)
    probe = bridge_preflight(grammar)
    control_state = explicit_control(grammar); control_text = BASE.render(control_state)
    control = audit(grammar, control_text, "complete_locative_recursive_control", control_state.trace)
    content = [x.word for x in BASE.ordered_leaves(control_state) if x.label not in FIXED_LABELS]
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "content_word_forms": content,
                    "content_word_forms_unique": len(content) == len(set(content)),
                    "provenance": {"construction": "explicit six-level locative relative tree",
                                   "lexical_source": "hand-authored edge-unit inventory only", "shared_tree": True},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "inverted_question_locative_full_bridge_shared_tree", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth,
        "max_states": max_states, "envelope": "Was it ... I saw?", "single_shared_derivation_tree": True,
        "character_residual_during_derivation": True, "full_terminal_to_prefix_preflight": True,
        "locative_gap_valency": True, "edge_unit_filter_at_exposed_leaf": True,
        "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                       "material": "authored temple/met-at edge-unit bridge; no catalogue or known palindrome material"},
        "full_boundary_preflight": probe, "complete_recursive_control": control,
        "edge_unit_inventory": {"entries": [asdict(x) for x in MORPHOLOGY]}, **result,
        "reader_facing_next_operator": "Add another independently authored typed locative or transitive edge sequence matching a preflighted multi-character seam.",
        "scope": "The bridge probe is not a palindrome candidate or readability evidence; only fresh >=100-letter exact closures could enter reader evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
