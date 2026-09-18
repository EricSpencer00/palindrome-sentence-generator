"""Connected-tree search with typed edge-unit lexicalization.

This experiment is a separate construction from the residual-indexed word
inventory.  Each authored word is represented by hand-audited stem/affix (or
syllable-like) edge units.  A lexical slot is expanded only at the currently
live edge, and its candidate must expose a compatible unit there.  The
character residual is retained across unit and word boundaries; no second
tree or frozen half is constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "experiments/inverted_question_shared_tree_20260913.py"
spec = importlib.util.spec_from_file_location("edge_unit_question_tree_20260913", SOURCE)
QUESTION = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = QUESTION
spec.loader.exec_module(QUESTION)
BASE = QUESTION.BASE
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
FIXED_LABELS = {"fixed_was", "fixed_it", "fixed_i", "fixed_saw", "relative_that", "det"}


@dataclass(frozen=True)
class EdgeLexeme:
    form: str
    category: str
    type: str = ""
    patient_type: str = ""
    units: tuple[str, ...] = ()
    rank: int = 0

    @property
    def first_unit(self) -> str:
        return self.units[0]

    @property
    def last_unit(self) -> str:
        return self.units[-1]


# Units are authored morphological segments, never extracted phrase material.
MORPHOLOGY = (
    EdgeLexeme("portrait", "noun", "object", units=("por", "trait"), rank=1),
    EdgeLexeme("packet", "noun", "object", units=("pack", "et"), rank=2),
    EdgeLexeme("artifact", "noun", "object", units=("arti", "fact"), rank=3),
    EdgeLexeme("cabinet", "noun", "object", units=("cabin", "et"), rank=4),
    EdgeLexeme("helmet", "noun", "object", units=("helm", "et"), rank=5),
    EdgeLexeme("garment", "noun", "object", units=("gar", "ment"), rank=6),
    EdgeLexeme("teacher", "noun", "person", units=("teach", "er"), rank=1),
    EdgeLexeme("artist", "noun", "person", units=("art", "ist"), rank=2),
    EdgeLexeme("guard", "noun", "person", units=("guard",), rank=3),
    EdgeLexeme("writer", "noun", "person", units=("writ", "er"), rank=4),
    EdgeLexeme("singer", "noun", "person", units=("sing", "er"), rank=5),
    EdgeLexeme("farmer", "noun", "person", units=("farm", "er"), rank=6),
    EdgeLexeme("nurse", "noun", "person", units=("nurs", "e"), rank=7),
    EdgeLexeme("pilot", "noun", "person", units=("pil", "ot"), rank=8),
    EdgeLexeme("scholar", "noun", "person", units=("schol", "ar"), rank=9),
    EdgeLexeme("friend", "noun", "person", units=("friend",), rank=10),
    EdgeLexeme("quiet", "adjective", units=("qui", "et"), rank=1),
    EdgeLexeme("careful", "adjective", units=("care", "ful"), rank=2),
    EdgeLexeme("patient", "adjective", units=("pati", "ent"), rank=3),
    EdgeLexeme("brave", "adjective", units=("brav", "e"), rank=4),
    EdgeLexeme("young", "adjective", units=("young",), rank=5),
    EdgeLexeme("gentle", "adjective", units=("gent", "le"), rank=6),
    EdgeLexeme("steady", "adjective", units=("stead", "y"), rank=7),
    EdgeLexeme("honest", "adjective", units=("hon", "est"), rank=8),
    EdgeLexeme("alert", "adjective", units=("al", "ert"), rank=9),
    EdgeLexeme("warm", "adjective", units=("warm",), rank=10),
    EdgeLexeme("watched", "verb", patient_type="object", units=("watch", "ed"), rank=1),
    EdgeLexeme("painted", "verb", patient_type="object", units=("paint", "ed"), rank=2),
    EdgeLexeme("saved", "verb", patient_type="object", units=("sav", "ed"), rank=3),
    EdgeLexeme("praised", "verb", patient_type="object", units=("prais", "ed"), rank=4),
    # ``built`` supplies the otherwise missing right-edge ``t`` signature at
    # the outer object-gap seam; it remains an ordinary authored verb.
    EdgeLexeme("built", "verb", patient_type="object", units=("buil", "t"), rank=5),
    EdgeLexeme("greeted", "verb", patient_type="person", units=("greet", "ed"), rank=1),
    EdgeLexeme("trusted", "verb", patient_type="person", units=("trust", "ed"), rank=2),
    EdgeLexeme("followed", "verb", patient_type="person", units=("follow", "ed"), rank=3),
    EdgeLexeme("helped", "verb", patient_type="person", units=("help", "ed"), rank=4),
    EdgeLexeme("met", "verb", patient_type="person", units=("met",), rank=5),
    EdgeLexeme("joined", "verb", patient_type="person", units=("join", "ed"), rank=6),
)
NOUNS = tuple(x for x in MORPHOLOGY if x.category == "noun")
ADJECTIVES = tuple(x for x in MORPHOLOGY if x.category == "adjective")
VERBS = tuple(x for x in MORPHOLOGY if x.category == "verb")
LEXEME_BY_FORM = {x.form: x for x in MORPHOLOGY}
UNIT_INDEX: dict[tuple[str, int, str], tuple[EdgeLexeme, ...]] = {}
_unit_index = defaultdict(list)
for item in MORPHOLOGY:
    _unit_index[(item.category, 1, item.first_unit[0])].append(item)
    _unit_index[(item.category, -1, item.last_unit[-1])].append(item)
UNIT_INDEX = {key: tuple(value) for key, value in _unit_index.items()}


def boundary_feasibility(category: str, side: int, expected: str) -> tuple[str, ...]:
    """Return authored forms whose exposed edge unit can consume ``expected``."""
    return tuple(x.form for x in UNIT_INDEX.get((category, side, expected), ()))


class EdgeUnitGrammar(QUESTION.InvertedQuestionGrammar):
    """One recursive feature tree whose lexical slots expose authored words."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "NP":
            role, typ, number, depth = f["role"], f["type"], f["number"], int(f["depth"])
            slots = (BASE.sym("DET_SLOT", number=number, form_role=role),
                     BASE.sym("ADJ_SLOT", type=typ),
                     BASE.sym("LEX_SLOT", category="noun", type=typ, role=role, number=number))
            rows = [BASE.Production(f"NP:{role}:{typ}:{depth}:plain", lhs, (slots[0], slots[2])),
                    BASE.Production(f"NP:{role}:{typ}:{depth}:modified", lhs, slots)]
            if depth > 0:
                rel = BASE.sym("REL", head_type=typ, depth=str(depth - 1))
                rows += [BASE.Production(f"NP:{role}:{typ}:{depth}:plain-relative", lhs,
                                         (slots[0], slots[2], rel)),
                         BASE.Production(f"NP:{role}:{typ}:{depth}:modified-relative", lhs, slots + (rel,))]
            return tuple(rows)
        if lhs.name == "REL":
            depth, head_type = int(f["depth"]), f["head_type"]
            subject = BASE.sym("NP", role="relative_agent", type="person", number="sing", depth=str(depth))
            return (BASE.Production(f"REL:{head_type}:{depth}", lhs,
                                    (BASE.sym("T", label="relative_that", form="that"), subject,
                                     BASE.sym("V", event="transitive_gap", patient_type=head_type,
                                              subject_number="sing"))),)
        if lhs.name == "DET_SLOT":
            return (BASE.Production("DET_SLOT:a", lhs,
                                    (BASE.sym("T", label="det", form="a"),)),)
        if lhs.name == "ADJ_SLOT":
            return tuple(BASE.Production(f"ADJ_SLOT:{x.form}", lhs,
                                         (BASE.sym("T", label="adj", form=x.form),)) for x in ADJECTIVES)
        if lhs.name == "V":
            return (BASE.Production("V:edge_unit", lhs,
                                    (BASE.sym("LEX_SLOT", category="verb", patient_type=f["patient_type"],
                                              role="relative_verb"),)),)
        if lhs.name == "LEX_SLOT":
            category = f["category"]
            values = NOUNS if category == "noun" else VERBS
            return tuple(BASE.Production(f"LEX_SLOT:{x.form}", lhs,
                                         (BASE.sym("T", label=(f"noun_{f['role']}" if category == "noun" else "relative_verb"),
                                                   form=x.form),))
                         for x in values if (not f.get("type") or x.type == f.get("type"))
                         and (not f.get("patient_type") or x.patient_type == f.get("patient_type")))
        return super().productions(lhs)


def content_words(state: BASE.State) -> tuple[str, ...]:
    return tuple(leaf.word for leaf in BASE.ordered_leaves(state) if leaf.label not in FIXED_LABELS)


def active_unit(lexeme: EdgeLexeme, leaf: BASE.Leaf, side: int) -> str:
    """Return the stem/affix currently touched by the live edge."""
    offset = leaf.left if side == 1 else leaf.right
    units = lexeme.units if side == 1 else tuple(reversed(lexeme.units))
    remaining = offset
    for unit in units:
        if remaining < len(unit):
            return unit
        remaining -= len(unit)
    raise AssertionError("emission offset exceeded authored unit sequence")


def lexical_unit_accepts(state: BASE.State, side: int, added: list[BASE.Leaf]) -> bool:
    """Filter a newly exposed lexical slot by its live edge unit signature."""
    if not state.residual or state.owner == side:
        return True
    expected = state.residual[0]
    for leaf in added:
        if leaf.label in FIXED_LABELS:
            continue
        lexeme = LEXEME_BY_FORM.get(leaf.word)
        if lexeme is None:
            continue
        unit = active_unit(lexeme, leaf, side)
        if (lexeme.category, side, expected) not in UNIT_INDEX:
            return False
        if lexeme not in UNIT_INDEX[(lexeme.category, side, expected)] or unit[0 if side == 1 else -1] != expected:
            return False
    return True


def expand_edge(grammar: EdgeUnitGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = set(content_words(state)); previous = BASE.leaf_map(state); output = []
    for candidate in BASE.expand(grammar, state, index):
        added = [leaf for leaf in candidate.leaves if leaf.identifier not in previous]
        if not lexical_unit_accepts(state, side, added):
            continue
        if any(leaf.word in used or leaf.word == leaf.word[::-1]
               for leaf in added if leaf.label not in FIXED_LABELS):
            continue
        output.append(candidate)
    return tuple(output)


def audit(grammar: EdgeUnitGrammar, text: str, kind: str, trace: tuple[tuple[int, str], ...]) -> dict[str, object]:
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


def solver(grammar: EdgeUnitGrammar, *, max_states: int = 100000) -> dict[str, object]:
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
                text = BASE.render(state)
                row = audit(grammar, text, "complete_shared_tree_closure", state.trace)
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


def _choose(grammar: EdgeUnitGrammar, state: BASE.State, index: int, identifier: str) -> BASE.State:
    options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == identifier]
    if len(options) != 1: raise AssertionError(f"nonunique control production {identifier}")
    return options[0]


def explicit_control(grammar: EdgeUnitGrammar) -> BASE.State:
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    adjectives = iter(("quiet", "careful", "patient", "brave", "young", "gentle", "steady"))
    people = iter(("teacher", "guard", "writer", "singer", "farmer", "nurse"))
    verbs = iter(("greeted", "trusted", "followed", "helped", "met", "joined"))
    while True:
        nodes = BASE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved: break
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "Q": ident = "Q:was-it-interior-i-saw"
        elif symbol.name == "NP":
            d = int(f["depth"]); shape = "modified-relative" if d > 0 else "modified"
            ident = f"NP:{f['role']}:{f['type']}:{d}:{shape}"
        elif symbol.name == "DET_SLOT": ident = "DET_SLOT:a"
        elif symbol.name == "ADJ_SLOT": ident = f"ADJ_SLOT:{next(adjectives)}"
        elif symbol.name == "V": ident = "V:edge_unit"
        elif symbol.name == "LEX_SLOT":
            if f["category"] == "noun": form = "portrait" if f["type"] == "object" else next(people)
            else: form = "watched" if f["patient_type"] == "object" else next(verbs)
            ident = f"LEX_SLOT:{form}"
        elif symbol.name == "REL": ident = f"REL:{f['head_type']}:{f['depth']}"
        else: raise AssertionError(symbol.name)
        state = _choose(grammar, state, index, ident)
    if any(not BASE.node_map(state)[ref].terminal for ref in state.frontier):
        raise AssertionError("control tree is not fully expanded")
    return state


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = EdgeUnitGrammar(6)
    result = solver(grammar, max_states=max_states)
    control_state = explicit_control(grammar); control_text = BASE.render(control_state)
    control = audit(grammar, control_text, "complete_edge_unit_grammar_control", control_state.trace)
    content = [x.word for x in BASE.ordered_leaves(control_state) if x.label not in FIXED_LABELS]
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True,
                    "content_word_forms": content,
                    "content_word_forms_unique": len(content) == len(set(content)),
                    "unit_sequence": [LEXEME_BY_FORM[x].units for x in content],
                    "provenance": {"construction": "explicit six-level object-gap relative tree",
                                   "lexical_source": "hand-authored edge-unit inventory only",
                                   "shared_tree": True,
                                   "trace_sha256": sha256(json.dumps(list(control_state.trace), sort_keys=True).encode()).hexdigest()},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "inverted_question_edge_unit_shared_tree", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth,
        "max_states": max_states, "envelope": "Was it ... I saw?", "single_shared_derivation_tree": True,
        "character_residual_during_derivation": True, "edge_unit_filter_at_exposed_leaf": True,
        "residual_crosses_unit_and_word_boundaries": True, "closure_requires_all_leaves_consumed": True,
        "independent_complete_reparse": True, "corpus_generation": False},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grammar_sha256": grammar.digest(),
                       "material": "small hand-audited typed edge-unit inventory; no catalogue construction material"},
        "edge_unit_inventory": {"entries": [asdict(x) for x in MORPHOLOGY], "index_keys": len(UNIT_INDEX),
                                "outer_object_right_t_feasibility": boundary_feasibility("verb", -1, "t")},
        "complete_recursive_control": control, **result,
        "reader_facing_next_operator": "Add a new hand-audited typed stem/affix variant at an exposed edge unit, then independently reparse every closure.",
        "scope": "This bounded construction is not a readability certificate; any exact output requires blinded human evaluation."}


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
