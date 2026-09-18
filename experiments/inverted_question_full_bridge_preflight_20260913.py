"""Preflighted full-boundary construction over one recursive grammar tree.

The previous bridge checked a verb suffix abstractly.  This branch first
simulates the whole local seam through the first lexical word.  Its only
known witness is the short ``Was it a bat I saw?`` oracle, which is retained
solely as an excluded negative control (never as candidate evidence).  The
same authored typed noun is then available inside the >=100-letter recursive
search, whose closures still require full-tree emission and independent
reparse.
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
SOURCE = ROOT / "experiments/inverted_question_full_bridge_edge_unit_20260913.py"
spec = importlib.util.spec_from_file_location("full_bridge_for_preflight_20260913", SOURCE)
FULL = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = FULL
spec.loader.exec_module(FULL)
BASE, QUESTION = FULL.BASE, FULL.QUESTION
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = FULL.MIN_LETTERS, FULL.MAX_LETTERS
FIXED_LABELS = FULL.FIXED_LABELS
BAT = FULL.EDGE.EdgeLexeme("bat", "noun", "object", units=("bat",), rank=7)
LEXEME_BY_FORM = dict(FULL.LEXEME_BY_FORM, bat=BAT)
MORPHOLOGY = FULL.MORPHOLOGY + (BAT,)


class PreflightGrammar(FULL.FullBridgeGrammar):
    """Full-bridge grammar with an authored object noun completing the probe."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        if lhs.name == "LEX_SLOT" and lhs.feature("category") == "noun" and lhs.feature("type") == "object":
            base = super().productions(lhs)
            return base + (BASE.Production("LEX_SLOT:bat", lhs,
                                           (BASE.sym("T", label=f"noun_{lhs.feature('role')}", form="bat"),)),)
        return super().productions(lhs)


def bridge_preflight(grammar: PreflightGrammar) -> dict[str, object]:
    """Simulate the complete seam through the first lexical word.

    The witness is deliberately below the search length band and explicitly
    excluded from all candidate collections.
    """
    witness = "was it a bat i saw"
    normalized = "".join(c for c in witness if "a" <= c <= "z")
    # Positions 4--6 are ``t`` from ``it`` plus ``a b`` from the first
    # variable lexical word; their counterparts are the complete reverse of
    # the terminal ``bat``.
    seam_pairs = [(normalized[i], normalized[-1 - i]) for i in range(4, 7)]
    tree = QUESTION.parse_tree(grammar, witness)
    exact = QUESTION.exact_audit(witness)
    return {"rendered": witness, "normalized": normalized, "excluded_negative_oracle": True,
            "reason": "known short palindrome used only to validate the bridge; never candidate evidence",
            "seam_pairs": seam_pairs, "left_bridge_stream": "tab",
            "right_terminal_reverse_stream": "tab", "full_multi_character_match": seam_pairs == [("t", "t"), ("a", "a"), ("b", "b")],
            "independent_parse": tree is not None, "exact_audit": exact,
            "central_exclusion": mechanical_admission_checks(witness, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)}


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
        unit = FULL.EDGE.active_unit(lexeme, leaf, side)
        if unit[0 if side == 1 else -1] != expected:
            return False
        # The preflighted branch uses the same edge-unit invariant, but the
        # bridge noun is deliberately added to this branch's local inventory.
        if not any(x.form == leaf.word for x in MORPHOLOGY):
            return False
    return True


def expand_edge(grammar: PreflightGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = set(content_words(state)); previous = BASE.leaf_map(state); result = []
    for candidate in BASE.expand(grammar, state, index):
        added = [x for x in candidate.leaves if x.identifier not in previous]
        if not lexical_unit_accepts(state, side, added):
            continue
        if any(x.word in used or x.word == x.word[::-1] for x in added if x.label not in FIXED_LABELS):
            continue
        result.append(candidate)
    return tuple(result)


def audit(grammar: PreflightGrammar, text: str, kind: str, trace: tuple[tuple[int, str], ...]) -> dict[str, object]:
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


def solver(grammar: PreflightGrammar, *, max_states: int = 100000) -> dict[str, object]:
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
                text = BASE.render(state); row = audit(grammar, text, "complete_preflighted_bridge_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]: admitted.setdefault(text, row)
                    elif len(rejected) < 100:
                        rejected.setdefault(text, row); stats["independent_reparse_rejections"] += 1
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
    grammar = PreflightGrammar(6); result = solver(grammar, max_states=max_states)
    control_state = FULL.EDGE.explicit_control(grammar); control_text = BASE.render(control_state)
    control = audit(grammar, control_text, "complete_preflighted_recursive_control", control_state.trace)
    content = [x.word for x in BASE.ordered_leaves(control_state) if x.label not in FIXED_LABELS]
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True,
                    "content_word_forms": content, "content_word_forms_unique": len(content) == len(set(content)),
                    "provenance": {"construction": "explicit six-level object-gap relative tree",
                                   "lexical_source": "hand-authored edge-unit inventory only", "shared_tree": True},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "inverted_question_full_bridge_preflight_shared_tree", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth,
        "max_states": max_states, "envelope": "Was it ... I saw?", "single_shared_derivation_tree": True,
        "character_residual_during_derivation": True, "full_terminal_to_prefix_preflight": True,
        "edge_unit_filter_at_exposed_leaf": True, "closure_requires_all_leaves_consumed": True,
        "independent_complete_reparse": True, "corpus_generation": False},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grammar_sha256": grammar.digest(), "material": "authored edge-unit inventory and excluded bat seam oracle; no catalogue construction material"},
        "full_boundary_preflight": bridge_preflight(grammar), "complete_recursive_control": control,
        "edge_unit_inventory": {"entries": [asdict(x) for x in MORPHOLOGY]}, **result,
        "reader_facing_next_operator": "Add a hand-audited typed lexical sequence matching the preflighted seam, then require complete-tree closure and independent reader evaluation.",
        "scope": "The short seam witness is excluded diagnostic material; it is not a candidate or readability evidence."}


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
