"""Full two-character envelope bridge over the connected edge-unit tree.

The preceding edge-unit run checked only the first outer seam character.
This wrapper makes the complete local bridge explicit: after ``wasi`` the
outer relative verb must expose ``t`` and the following article ``a`` must
meet the verb's preceding ``a``.  The authored transitive past-tense verb
``beat`` (units ``be`` + ``at``) satisfies that typed bridge.  Search still
uses one grammar tree and live character residuals; the bridge is a
feasibility constraint, never a frozen half or a generated phrase.
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
spec = importlib.util.spec_from_file_location("edge_unit_tree_for_full_bridge_20260913", SOURCE)
EDGE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = EDGE
spec.loader.exec_module(EDGE)
BASE, QUESTION = EDGE.BASE, EDGE.QUESTION
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = EDGE.MIN_LETTERS, EDGE.MAX_LETTERS
FIXED_LABELS = EDGE.FIXED_LABELS
BRIDGE_LEXEME = EDGE.EdgeLexeme("beat", "verb", patient_type="object", units=("be", "at"), rank=7)
LEXEME_BY_FORM = dict(EDGE.LEXEME_BY_FORM, beat=BRIDGE_LEXEME)
MORPHOLOGY = EDGE.MORPHOLOGY + (BRIDGE_LEXEME,)
UNIT_INDEX = dict(EDGE.UNIT_INDEX)
for side, char in ((1, BRIDGE_LEXEME.first_unit[0]), (-1, BRIDGE_LEXEME.last_unit[-1])):
    key = (BRIDGE_LEXEME.category, side, char)
    UNIT_INDEX[key] = UNIT_INDEX.get(key, ()) + (BRIDGE_LEXEME,)


class FullBridgeGrammar(EDGE.EdgeUnitGrammar):
    """The edge-unit grammar with one additional typed bridge lexeme."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        if lhs.name == "LEX_SLOT" and lhs.feature("category") == "verb" and lhs.feature("patient_type") == "object":
            base = super().productions(lhs)
            extra = BASE.Production("LEX_SLOT:beat", lhs,
                                    (BASE.sym("T", label="relative_verb", form="beat"),))
            return base + (extra,)
        return super().productions(lhs)


def boundary_feasibility(category: str, side: int, expected: str, *, type: str = "") -> tuple[str, ...]:
    return tuple(x.form for x in UNIT_INDEX.get((category, side, expected), ())
                 if not type or x.type == type or x.patient_type == type)


def full_boundary_bridge() -> dict[str, object]:
    """Audit both required characters at the first variable envelope seam."""
    suffix = BRIDGE_LEXEME.units[-1]
    return {"fixed_prefix_after_wasi": "t",
            "required_core_suffix": "at",
            "required_core_final": suffix[-1],
            "required_core_penultimate": suffix[-2],
            "typed_object_verbs_matching_final_t": boundary_feasibility("verb", -1, "t", type="object"),
            "typed_object_verbs_matching_full_at": tuple(x.form for x in MORPHOLOGY
                                                           if x.category == "verb" and x.patient_type == "object"
                                                           and x.form.endswith("at")),
            "satisfiable": suffix == "at" and suffix[-1] == "t" and suffix[-2] == "a"
                           and "beat" in boundary_feasibility("verb", -1, "t", type="object")}


def content_words(state: BASE.State) -> tuple[str, ...]:
    return tuple(x.word for x in BASE.ordered_leaves(state) if x.label not in FIXED_LABELS)


def lexical_unit_accepts(state: BASE.State, side: int, added: list[BASE.Leaf]) -> bool:
    if not state.residual or state.owner == side:
        return True
    expected = state.residual[0]
    for leaf in added:
        if leaf.label in FIXED_LABELS:
            continue
        lexeme = LEXEME_BY_FORM.get(leaf.word)
        if lexeme is None:
            continue
        unit = EDGE.active_unit(lexeme, leaf, side)
        key = (lexeme.category, side, expected)
        if lexeme not in UNIT_INDEX.get(key, ()) or unit[0 if side == 1 else -1] != expected:
            return False
    return True


def expand_edge(grammar: FullBridgeGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = set(content_words(state)); previous = BASE.leaf_map(state); output = []
    for candidate in BASE.expand(grammar, state, index):
        added = [x for x in candidate.leaves if x.identifier not in previous]
        if not lexical_unit_accepts(state, side, added):
            continue
        if any(x.word in used or x.word == x.word[::-1] for x in added if x.label not in FIXED_LABELS):
            continue
        output.append(candidate)
    return tuple(output)


def audit(grammar: FullBridgeGrammar, text: str, kind: str, trace: tuple[tuple[int, str], ...]) -> dict[str, object]:
    exact, tree = QUESTION.exact_audit(text), QUESTION.parse_tree(grammar, text)
    witness = QUESTION.feature_witness(tree)
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


def solver(grammar: FullBridgeGrammar, *, max_states: int = 100000) -> dict[str, object]:
    root = BASE.Node(0, grammar.start())
    initial = BASE.State((0,), (root,), (), "", 0, 0, ())
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
                text = BASE.render(state); row = audit(grammar, text, "complete_full_bridge_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]: admitted.setdefault(text, row)
                    elif len(rejected) < 100:
                        rejected.setdefault(text, row); stats["independent_reparse_rejections"] += 1
            continue
        nodes = BASE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        index = None
        if state.residual:
            side = -state.owner; edge = 0 if side == 1 else len(state.frontier) - 1
            if not nodes[state.frontier[edge]].terminal: index = edge
        elif unresolved:
            if not nodes[state.frontier[0]].terminal: index, side = 0, 1
            elif not nodes[state.frontier[-1]].terminal: index, side = len(state.frontier) - 1, -1
            else: index, side = unresolved[0], 1
        if index is not None:
            for next_state in expand_edge(grammar, state, index, side):
                queue.append(next_state); stats["expansions"] += 1
        left_leaf, right_leaf = bool(nodes[state.frontier[0]].terminal), bool(nodes[state.frontier[-1]].terminal)
        sides = (-state.owner,) if state.residual else (1, -1)
        for side in sides:
            if (side == 1 and not left_leaf) or (side == -1 and not right_leaf): continue
            next_state = BASE.emit(state, side)
            if next_state is not None:
                queue.append(next_state); stats["emissions"] += 1
            else: stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()),
            "independent_reparse_rejections": list(rejected.values())}


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = FullBridgeGrammar(6); result = solver(grammar, max_states=max_states)
    control_state = EDGE.explicit_control(grammar); control_text = BASE.render(control_state)
    control = audit(grammar, control_text, "complete_full_bridge_grammar_control", control_state.trace)
    content = [x.word for x in BASE.ordered_leaves(control_state) if x.label not in FIXED_LABELS]
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True,
                    "content_word_forms": content,
                    "content_word_forms_unique": len(content) == len(set(content)),
                    "provenance": {"construction": "explicit six-level object-gap relative tree",
                                   "lexical_source": "hand-authored edge-unit inventory only",
                                   "shared_tree": True}})
    return {"status": "inverted_question_full_boundary_bridge_shared_tree", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth,
        "max_states": max_states, "envelope": "Was it ... I saw?", "single_shared_derivation_tree": True,
        "character_residual_during_derivation": True, "edge_unit_filter_at_exposed_leaf": True,
        "full_two_character_boundary_bridge": True, "closure_requires_all_leaves_consumed": True,
        "independent_complete_reparse": True, "corpus_generation": False},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grammar_sha256": grammar.digest(), "material": "authored edge units plus typed beat bridge; no catalogue material"},
        "full_boundary_bridge_oracle": full_boundary_bridge(),
        "complete_recursive_control": control,
        "edge_unit_inventory": {"entries": [asdict(x) for x in MORPHOLOGY], "index_keys": len(UNIT_INDEX)},
        **result,
        "reader_facing_next_operator": "If this full bridge remains sparse, add another hand-audited typed object-gap verb ending in a bridge-compatible suffix and rerun independent closure checks.",
        "scope": "The bridge oracle establishes feasibility only; any exact output requires independent parse and blinded human evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"],
                      "exact_closures": len(result["exact_closures"]),
                      "admitted": len(result["mechanically_admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
