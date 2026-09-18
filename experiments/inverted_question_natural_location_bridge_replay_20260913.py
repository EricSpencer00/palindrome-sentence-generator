"""Natural typed location compound with replayed multi-slot bridge.

This replaces the rejected ``tempo hall`` construction.  ``temple hall`` is
an authored institutional-location compound: ``temple`` is the modifier and
``hall`` the location head.  The locative relative ``the help met at`` is
typed as a collective-person subject.  The executable shared-tree replay
matches nine pairs (``tatempleh``), crossing the third lexical slot ``hall``.
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
SOURCE = ROOT / "experiments/inverted_question_locative_bridge_second_slot_replay_20260913.py"
spec = importlib.util.spec_from_file_location("second_replay_for_natural_location_20260913", SOURCE)
SECOND = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = SECOND
spec.loader.exec_module(SECOND)
BASE, QUESTION = SECOND.BASE, SECOND.QUESTION
EDGE = SECOND.SECOND.LOCATIVE.EDGE
FIXED_LABELS = SECOND.SECOND.FIXED_LABELS
MIN_LETTERS, MAX_LETTERS = SECOND.SECOND.MIN_LETTERS, SECOND.SECOND.MAX_LETTERS
from llm_palindrome.admission import mechanical_admission_checks
TEMPLE = EDGE.EdgeLexeme("temple", "noun", "location_modifier", units=("tem", "ple"), rank=1)
HALL = EDGE.EdgeLexeme("hall", "noun", "location", units=("hall",), rank=2)
HELP = EDGE.EdgeLexeme("help", "noun", "person_group", units=("help",), rank=1)
MET = EDGE.EdgeLexeme("met", "verb", "location", units=("met",), rank=1)
MORPHOLOGY = (TEMPLE, HALL, HELP, MET) + tuple(x for x in SECOND.SECOND.MORPHOLOGY if x.form not in {"temple", "met"})
LEXEME_BY_FORM = {x.form: x for x in MORPHOLOGY}


class NaturalLocationGrammar(SECOND.SECOND.SecondSlotGrammar):
    """Feature grammar with licensed compound and collective-person option."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "DET_SLOT":
            form = "the" if f.get("number") == "plur" else "a"
            return (BASE.Production(f"DET_SLOT:{form}", lhs, (BASE.sym("T", label="det", form=form),)),)
        if lhs.name == "NP" and f.get("type") == "location" and int(f["depth"]) > 0:
            base = super().productions(lhs)
            d = int(f["depth"])
            return base + (BASE.Production(f"NP:{f['role']}:location:{d}:compound", lhs, (
                BASE.sym("DET_SLOT", number="sing", form_role=f["role"]),
                BASE.sym("LEX_SLOT", category="noun", type="location_modifier", role="compound_modifier", number="sing"),
                BASE.sym("LEX_SLOT", category="noun", type="location", role=f["role"], number="sing"),
                BASE.sym("REL", head_type="location", depth=str(d - 1)))),)
        if lhs.name == "NP" and f.get("type") == "person_group":
            d = int(f["depth"])
            slots = (BASE.sym("DET_SLOT", number="plur", form_role=f["role"]),
                     BASE.sym("LEX_SLOT", category="noun", type="person_group", role=f["role"], number="plur"))
            rows = [BASE.Production(f"NP:{f['role']}:person_group:{d}:plain", lhs, slots)]
            if d > 0:
                rows.append(BASE.Production(f"NP:{f['role']}:person_group:{d}:relative", lhs,
                                            slots + (BASE.sym("REL", head_type="person_group", depth=str(d - 1)),)))
            return tuple(rows)
        if lhs.name == "REL" and f["head_type"] == "location":
            d = int(f["depth"])
            group = BASE.Production(f"REL:location:{d}:locative-group", lhs, (
                BASE.sym("T", label="relative_that", form="that"),
                BASE.sym("NP", role="relative_agent", type="person_group", number="plur", depth=str(d)),
                BASE.sym("V", event="locative_gap", patient_type="location", subject_number="plur"),
                BASE.sym("T", label="relative_prep", form="at")))
            person = BASE.Production(f"REL:location:{d}:locative", lhs, (
                BASE.sym("T", label="relative_that", form="that"),
                BASE.sym("NP", role="relative_agent", type="person", number="sing", depth=str(d)),
                BASE.sym("V", event="locative_gap", patient_type="location", subject_number="sing"),
                BASE.sym("T", label="relative_prep", form="at")))
            return (group, person)
        if lhs.name == "REL" and f["head_type"] == "person_group":
            d = int(f["depth"])
            return (BASE.Production(f"REL:person_group:{d}", lhs, (
                BASE.sym("T", label="relative_that", form="that"),
                BASE.sym("NP", role="relative_agent", type="person", number="sing", depth=str(d)),
                BASE.sym("V", event="transitive_gap", patient_type="person_group", subject_number="sing"))),)
        if lhs.name == "V" and f.get("patient_type") == "location":
            return (BASE.Production("V:locative_met", lhs, (BASE.sym("LEX_SLOT", category="verb", patient_type="location", role="relative_verb"),)),)
        if lhs.name == "V" and f.get("patient_type") == "person_group":
            return (BASE.Production("V:group_met", lhs, (BASE.sym("LEX_SLOT", category="verb", patient_type="person_group", role="relative_verb"),)),)
        if lhs.name == "LEX_SLOT":
            if f.get("type") == "location_modifier":
                return (BASE.Production("LEX_SLOT:temple", lhs, (BASE.sym("T", label="noun_compound_modifier", form="temple"),)),)
            if f.get("type") == "location":
                base = (BASE.Production("LEX_SLOT:temple", lhs, (BASE.sym("T", label=f"noun_{f['role']}", form="temple"),)),)
                return base + (BASE.Production("LEX_SLOT:hall", lhs, (BASE.sym("T", label=f"noun_{f['role']}", form="hall"),)),)
            if f.get("type") == "person_group":
                return (BASE.Production("LEX_SLOT:help", lhs, (BASE.sym("T", label=f"noun_{f['role']}", form="help"),)),)
            if f.get("patient_type") == "location":
                return (BASE.Production("LEX_SLOT:met", lhs, (BASE.sym("T", label="relative_verb", form="met"),)),)
            if f.get("patient_type") == "person_group":
                return (BASE.Production("LEX_SLOT:met_group", lhs, (BASE.sym("T", label="relative_verb", form="met"),)),)
        return super().productions(lhs)


def feature_witness(tree: BASE.Tree | None) -> dict[str, object]:
    roles, relatives, agreement, valency = [], 0, True, True
    def visit(node: BASE.Tree) -> None:
        nonlocal relatives, agreement, valency
        if node.symbol.name == "NP": roles.append(dict(node.symbol.features))
        if node.symbol.name == "REL":
            relatives += 1
            if len(node.children) == 4:
                if node.children[0].terminal != "that" or node.children[2].symbol.name != "V" or node.children[3].terminal != "at": valency = False
                elif node.children[2].symbol.feature("patient_type") != node.symbol.feature("head_type"): valency = False
                elif node.children[1].symbol.feature("number") != node.children[2].symbol.feature("subject_number"): agreement = False
            elif len(node.children) == 3:
                if node.children[0].terminal != "that" or node.children[2].symbol.name != "V": valency = False
                elif node.children[2].symbol.feature("patient_type") != node.symbol.feature("head_type"): valency = False
            else: valency = False
        for child in node.children: visit(child)
    if tree: visit(tree)
    return {"semantic_roles": roles, "relative_count": relatives, "agreement_ok": agreement, "valency_ok": valency, "complete_tree": bool(tree and tree.symbol.name == "Q")}


def audit(grammar: NaturalLocationGrammar, text: str, kind: str, trace: tuple[tuple[int, str], ...]) -> dict[str, object]:
    exact, tree = QUESTION.exact_audit(text), QUESTION.parse_tree(grammar, text); witness = feature_witness(tree)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [k for k, v in central.items() if not v]
    if tree is None: codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]: codes.append("agreement_failure")
    if not witness["valency_ok"]: codes.append("valency_failure")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact, "independent_parse": tree is not None, "feature_witness": witness, "central_admission": central, "mechanically_admitted": not codes, "rejection_codes": codes, "shared_tree_trace": list(trace), "reader_status": "unreviewed; programmatic measures do not certify readability"}


def choose(grammar: NaturalLocationGrammar, state: BASE.State, index: int, ident: str) -> BASE.State:
    options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
    if len(options) != 1: raise AssertionError(f"nonunique production {ident}")
    return options[0]


def build_probe_tree(grammar: NaturalLocationGrammar) -> BASE.State:
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]
        if not unresolved: break
        i = unresolved[0]; s = nodes[state.frontier[i]].symbol; f = dict(s.features)
        if s.name == "Q": ident = "Q:was-it-location-i-saw"
        elif s.name == "NP": ident = "NP:complement:location:1:compound" if f.get("type") == "location" else f"NP:{f['role']}:person_group:0:plain"
        elif s.name == "DET_SLOT": ident = "DET_SLOT:the" if f.get("number") == "plur" else "DET_SLOT:a"
        elif s.name == "LEX_SLOT": ident = f"LEX_SLOT:{'temple' if f.get('role') == 'compound_modifier' else 'hall' if f.get('type') == 'location' else 'help' if f.get('type') == 'person_group' else 'met'}"
        elif s.name == "REL": ident = "REL:location:0:locative-group"
        elif s.name == "V": ident = "V:locative_met"
        else: raise AssertionError(s.name)
        state = choose(grammar, state, i, ident)
    return state


def replay_bridge(grammar: NaturalLocationGrammar) -> dict[str, object]:
    state = build_probe_tree(NaturalLocationGrammar(1)); text = BASE.render(state)
    sides = (1, 1, 1, 1, -1, -1, -1, -1) + tuple(x for _ in range(9) for x in (1, -1))
    emitted, events = state, []
    for step, side in enumerate(sides, 1):
        nodes, leaves = BASE.node_map(emitted), BASE.leaf_map(emitted); ref = next(r for r in (emitted.frontier if side == 1 else reversed(emitted.frontier)) if nodes[r].terminal); leaf = leaves[ref]
        char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]; emitted = BASE.emit(emitted, side)
        if emitted is None: raise AssertionError(step)
        if step >= 9: events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "residual_after": emitted.residual})
    pairs = [(events[i]["character"], events[i + 1]["character"]) for i in range(0, len(events), 2)]
    return {"rendered": text, "bridge_pairs": pairs, "bridge_stream": "".join(x[0] for x in pairs), "terminal_reverse_stream": "".join(x[1] for x in pairs), "emitter_events": events, "nine_pairs_replayed": len(events) == 18, "third_slot_event": events[-2]["word"] == "hall", "independent_parse": QUESTION.parse_tree(grammar, text) is not None, "exact_audit": QUESTION.exact_audit(text), "diagnostic_only": True}


def solver(grammar: NaturalLocationGrammar, *, max_states: int = 100000) -> dict[str, object]:
    root = BASE.Node(0, grammar.start()); initial = BASE.State((0,), (root,), (), "", 0, 0, ())
    queue, seen, exact_rows, admitted, rejected = deque([initial]), set(), {}, {}, {}; stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0, complete_trees=0, exact_closures=0, independent_reparse_rejections=0)
    while queue and stats["states"] < max_states:
        state = queue.pop(); stats["states"] += 1; key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.length)
        if key in seen: continue
        seen.add(key)
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS and state.residual == state.residual[::-1]:
                text = BASE.render(state); row = audit(grammar, text, "complete_natural_bridge_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]: admitted.setdefault(text, row)
                    elif len(rejected) < 100: rejected.setdefault(text, row); stats["independent_reparse_rejections"] += 1
            continue
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]; index = None
        if state.residual:
            side = -state.owner; edge = 0 if side == 1 else len(state.frontier) - 1
            if not nodes[state.frontier[edge]].terminal: index = edge
        elif unresolved:
            if not nodes[state.frontier[0]].terminal: index, side = 0, 1
            elif not nodes[state.frontier[-1]].terminal: index, side = len(state.frontier) - 1, -1
            else: index, side = unresolved[0], 1
        if index is not None:
            for nxt in expand_edge(grammar, state, index, side): queue.append(nxt); stats["expansions"] += 1
        left, right = bool(nodes[state.frontier[0]].terminal), bool(nodes[state.frontier[-1]].terminal); dirs = (-state.owner,) if state.residual else (1, -1)
        for side in dirs:
            if (side == 1 and not left) or (side == -1 and not right): continue
            nxt = BASE.emit(state, side)
            if nxt is not None: queue.append(nxt); stats["emissions"] += 1
            else: stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states, "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()), "independent_reparse_rejections": list(rejected.values())}


def content_words(state: BASE.State) -> tuple[str, ...]:
    return tuple(x.word for x in BASE.ordered_leaves(state) if x.label not in FIXED_LABELS)


def expand_edge(grammar: NaturalLocationGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = set(content_words(state)); previous = BASE.leaf_map(state); out = []
    for candidate in BASE.expand(grammar, state, index):
        added = [x for x in candidate.leaves if x.identifier not in previous]
        if state.residual and state.owner != side:
            expected = state.residual[0]
            for leaf in added:
                if leaf.label in FIXED_LABELS: continue
                lexeme = LEXEME_BY_FORM.get(leaf.word)
                if lexeme and EDGE.active_unit(lexeme, leaf, side)[0 if side == 1 else -1] != expected: break
            else:
                if not any(x.word in used or x.word == x.word[::-1] for x in added if x.label not in FIXED_LABELS): out.append(candidate)
        elif not any(x.word in used or x.word == x.word[::-1] for x in added if x.label not in FIXED_LABELS): out.append(candidate)
    return tuple(out)


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = NaturalLocationGrammar(6); result = solver(grammar, max_states=max_states); replay = replay_bridge(grammar)
    control_state = SECOND.SECOND.LOCATIVE.explicit_control(grammar); control_text = BASE.render(control_state); control = audit(grammar, control_text, "complete_natural_recursive_control", control_state.trace)
    content = [x.word for x in BASE.ordered_leaves(control_state) if x.label not in FIXED_LABELS]
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "content_word_forms": content, "content_word_forms_unique": len(content) == len(set(content)), "provenance": {"construction": "explicit six-level locative relative tree", "lexical_source": "hand-authored natural location inventory only", "shared_tree": True}, "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "inverted_question_natural_location_bridge_replay", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth, "max_states": max_states, "single_shared_derivation_tree": True, "character_residual_during_derivation": True, "programmatic_emitter_replay": True, "nine_pair_bridge": True, "natural_compound_semantic_license": "temple modifier + hall location head", "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False}, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "authored temple-hall/help locative bridge; no catalogue or known palindrome material"}, "replayed_boundary_preflight": replay, "complete_recursive_control": control, "edge_unit_inventory": {"entries": [asdict(x) for x in MORPHOLOGY]}, **result, "reader_facing_next_operator": "Only after this natural nine-pair replay, add another independently authored typed lexical span.", "scope": "The bridge replay is diagnostic only; only fresh >=100-letter exact closures enter reader evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
