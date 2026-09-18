"""Residual-indexed typed morphology over one inverted-question parse tree.

This is a distinct repair after endpoint starvation.  It keeps one connected
``Was it ... I saw?`` grammar tree, but exposes a small, hand-audited inventory
of ordinary lexical variants through role/type/number slots.  The inventory is
indexed by the first letter (left edge) and last letter (right edge); when a
character residual is live, only variants with the required side signature
are expanded.  Optional unmodified NPs remove the forced-adjective bottleneck
without introducing fragments. No corpus or catalogue supplies construction
material.
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
SOURCE = ROOT / "experiments/inverted_question_typed_transducer_20260913.py"
spec = importlib.util.spec_from_file_location("typed_transducer_20260913", SOURCE)
TRANSDUCER = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = TRANSDUCER
spec.loader.exec_module(TRANSDUCER)
BASE, QUESTION = TRANSDUCER.BASE, TRANSDUCER.QUESTION
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
FIXED_LABELS = {"fixed_was", "fixed_it", "fixed_i", "fixed_saw", "relative_that", "det"}


@dataclass(frozen=True)
class Morph:
    form: str
    category: str
    type: str = ""
    number: str = "sing"
    patient_type: str = ""
    rank: int = 0

    @property
    def first(self) -> str:
        return self.form[0]

    @property
    def last(self) -> str:
        return self.form[-1]


# A finite authored inventory, not a broad dictionary or borrowed source.
MORPHOLOGY = (
    Morph("portrait", "noun", "object", rank=1), Morph("packet", "noun", "object", rank=2),
    Morph("artifact", "noun", "object", rank=3), Morph("cabinet", "noun", "object", rank=4),
    Morph("helmet", "noun", "object", rank=5), Morph("garment", "noun", "object", rank=6),
    Morph("teacher", "noun", "person", rank=1), Morph("artist", "noun", "person", rank=2),
    Morph("guard", "noun", "person", rank=3), Morph("writer", "noun", "person", rank=4),
    Morph("singer", "noun", "person", rank=5), Morph("farmer", "noun", "person", rank=6),
    Morph("nurse", "noun", "person", rank=7), Morph("pilot", "noun", "person", rank=8),
    Morph("scholar", "noun", "person", rank=9), Morph("friend", "noun", "person", rank=10),
    Morph("quiet", "adjective", rank=1), Morph("careful", "adjective", rank=2),
    Morph("patient", "adjective", rank=3), Morph("brave", "adjective", rank=4),
    Morph("young", "adjective", rank=5), Morph("gentle", "adjective", rank=6),
    Morph("steady", "adjective", rank=7), Morph("honest", "adjective", rank=8),
    Morph("alert", "adjective", rank=9), Morph("warm", "adjective", rank=10),
    Morph("calm", "adjective", rank=11), Morph("clear", "adjective", rank=12),
    Morph("mild", "adjective", rank=13), Morph("kind", "adjective", rank=14),
    Morph("watched", "verb", patient_type="object", rank=1), Morph("painted", "verb", patient_type="object", rank=2),
    Morph("saved", "verb", patient_type="object", rank=3), Morph("praised", "verb", patient_type="object", rank=4),
    Morph("greeted", "verb", patient_type="person", rank=1), Morph("trusted", "verb", patient_type="person", rank=2),
    Morph("followed", "verb", patient_type="person", rank=3), Morph("helped", "verb", patient_type="person", rank=4),
    Morph("met", "verb", patient_type="person", rank=5), Morph("joined", "verb", patient_type="person", rank=6),
)
NOUNS = tuple(x for x in MORPHOLOGY if x.category == "noun")
ADJECTIVES = tuple(x for x in MORPHOLOGY if x.category == "adjective")
VERBS = tuple(x for x in MORPHOLOGY if x.category == "verb")

SIGNATURE_INDEX: dict[tuple[str, int, str], tuple[Morph, ...]] = {}
_index = defaultdict(list)
for item in MORPHOLOGY:
    _index[(item.category, 1, item.first)].append(item)
    _index[(item.category, -1, item.last)].append(item)
SIGNATURE_INDEX = {key: tuple(value) for key, value in _index.items()}


class ResidualIndexedGrammar(TRANSDUCER.TransducerGrammar):
    """Feature grammar whose lexical slots are backed by the indexed inventory."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "NP":
            role, typ, number, depth = f["role"], f["type"], f["number"], int(f["depth"])
            slots = (BASE.sym("DET_SLOT", number=number, form_role=role),
                     BASE.sym("ADJ_SLOT", type=typ), BASE.sym("N_SLOT", role=role, type=typ, number=number))
            rows = [BASE.Production(f"NP:{role}:{typ}:{depth}:plain", lhs,
                                    (slots[0], slots[2]))]
            rows.append(BASE.Production(f"NP:{role}:{typ}:{depth}:modified", lhs, slots))
            if depth > 0:
                rows.extend((BASE.Production(f"NP:{role}:{typ}:{depth}:plain-relative", lhs,
                                              (slots[0], slots[2], BASE.sym("REL", head_type=typ, depth=str(depth - 1)))),
                             BASE.Production(f"NP:{role}:{typ}:{depth}:modified-relative", lhs,
                                              slots + (BASE.sym("REL", head_type=typ, depth=str(depth - 1)),))))
            return tuple(rows)
        if lhs.name == "REL":
            depth, head_type = int(f["depth"]), f["head_type"]
            subject = BASE.sym("NP", role="relative_agent", type="person", number="sing", depth=str(depth))
            return (BASE.Production(f"REL:{head_type}:{depth}", lhs,
                                    (BASE.sym("T", label="relative_that", form="that"), subject,
                                     BASE.sym("V", event="transitive_gap", patient_type=head_type,
                                              subject_number="sing"))),)
        if lhs.name == "DET_SLOT":
            form = "a" if lhs.feature("number") == "sing" else "the"
            return (BASE.Production(f"DET_SLOT:{form}", lhs, (BASE.sym("T", label="det", form=form),)),)
        if lhs.name == "ADJ_SLOT":
            return tuple(BASE.Production(f"ADJ_SLOT:{item.form}", lhs,
                                         (BASE.sym("T", label="adj", form=item.form),)) for item in ADJECTIVES)
        if lhs.name == "N_SLOT":
            return tuple(BASE.Production(f"N_SLOT:{item.type}:{item.form}", lhs,
                                         (BASE.sym("T", label=f"noun_{lhs.feature('role')}", form=item.form),))
                         for item in NOUNS if item.type == lhs.feature("type") and item.number == lhs.feature("number"))
        if lhs.name == "V":
            return tuple(BASE.Production(f"V:{item.form}", lhs,
                                         (BASE.sym("T", label="relative_verb", form=item.form),))
                         for item in VERBS if item.patient_type == lhs.feature("patient_type"))
        return super().productions(lhs)


def content_words(state: BASE.State) -> frozenset[str]:
    return frozenset(leaf.word for leaf in state.leaves if leaf.label not in FIXED_LABELS)


def residual_index_accepts(state: BASE.State, side: int, added: list[BASE.Leaf]) -> bool:
    if not state.residual:
        return True
    expected = state.residual[0]
    if state.owner == side:
        return True
    for leaf in added:
        if leaf.label in FIXED_LABELS:
            continue
        category = "adjective" if leaf.label == "adj" else ("verb" if leaf.label == "relative_verb" else "noun" if leaf.label.startswith("noun_") else "")
        if category and not any(item.form == leaf.word for item in SIGNATURE_INDEX.get((category, side, expected), ())):
            return False
    return True


def expand_residual(grammar: ResidualIndexedGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = content_words(state); prior = BASE.leaf_map(state); out = []
    for candidate in BASE.expand(grammar, state, index):
        added = [leaf for leaf in candidate.leaves if leaf.identifier not in prior]
        if not residual_index_accepts(state, side, added):
            continue
        if any(leaf.word in used or leaf.word == leaf.word[::-1] for leaf in added if leaf.label not in FIXED_LABELS):
            continue
        out.append(candidate)
    return tuple(out)


def audit(grammar: ResidualIndexedGrammar, text: str, kind: str, trace: tuple[tuple[int, str], ...]) -> dict[str, object]:
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


def _choose_expansion(grammar: ResidualIndexedGrammar, state: BASE.State, index: int,
                      identifier: str) -> BASE.State:
    """Choose one authored production while retaining the ordinary tree trace."""
    options = BASE.expand(grammar, state, index)
    matches = [candidate for candidate in options if candidate.trace[-1][1] == identifier]
    if len(matches) != 1:
        raise AssertionError(f"control production not unique: {identifier!r}")
    return matches[0]


def explicit_control(grammar: ResidualIndexedGrammar) -> BASE.State:
    """Build a long, intact recursive control by explicit grammar productions.

    This is deliberately *not* a palindrome candidate: it is a positive
    grammar/reparse control showing that the recursive envelope can express
    distinct ordinary words.  Every lexical slot is still selected as a
    grammar leaf, so the rendered text has a complete connected derivation.
    """
    root = BASE.Node(0, grammar.start())
    state = BASE.State((0,), (root,), (), "", 0, 0, ())
    adjectives = iter(("quiet", "careful", "patient", "brave", "young", "gentle", "steady"))
    person_nouns = iter(("teacher", "guard", "writer", "singer", "farmer", "nurse"))
    person_verbs = iter(("greeted", "trusted", "followed", "helped", "met", "joined"))
    np_count = 0
    verb_count = 0
    while True:
        nodes = BASE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved:
            break
        index = unresolved[0]
        symbol = nodes[state.frontier[index]].symbol
        features = dict(symbol.features)
        if symbol.name == "Q":
            identifier = "Q:was-it-interior-i-saw"
        elif symbol.name == "NP":
            depth = int(features["depth"])
            identifier = (f"NP:{features['role']}:{features['type']}:{depth}:"
                          f"{'modified-relative' if depth > 0 else 'modified'}")
            np_count += 1
        elif symbol.name == "DET_SLOT":
            identifier = "DET_SLOT:a"
        elif symbol.name == "ADJ_SLOT":
            identifier = f"ADJ_SLOT:{next(adjectives)}"
        elif symbol.name == "N_SLOT":
            if features["type"] == "object":
                form = "portrait"
            else:
                form = next(person_nouns)
            identifier = f"N_SLOT:{features['type']}:{form}"
        elif symbol.name == "REL":
            identifier = f"REL:{features['head_type']}:{features['depth']}"
        elif symbol.name == "V":
            if features["patient_type"] == "object":
                form = "watched"
            else:
                form = next(person_verbs)
            verb_count += 1
            identifier = f"V:{form}"
        else:
            raise AssertionError(f"unexpected control nonterminal: {symbol.name}")
        state = _choose_expansion(grammar, state, index, identifier)
    # This control is a fully expanded grammar tree, but it is intentionally
    # not emitted through the palindrome tape: its ordinary prose is a
    # diagnostic grammar witness rather than a claimed closure.
    if any(not BASE.node_map(state)[ref].terminal for ref in state.frontier):
        raise AssertionError("explicit control did not fully expand")
    return state


def solver(grammar: ResidualIndexedGrammar, *, max_states: int = 100000) -> dict[str, object]:
    root = BASE.Node(0, grammar.start()); initial = BASE.State((0,), (root,), (), "", 0, 0, ())
    queue, seen, exact_rows, admitted, rejects = deque([initial]), set(), {}, {}, {}
    stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0, complete_trees=0, exact_closures=0, independent_reparse_rejections=0)
    while queue and stats["states"] < max_states:
        state = queue.pop(); stats["states"] += 1
        key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.length)
        if key in seen: continue
        seen.add(key)
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS and state.residual == state.residual[::-1]:
                text = BASE.render(state); row = audit(grammar, text, "complete_shared_tree_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]: admitted.setdefault(text, row)
                    elif len(rejects) < 100: rejects.setdefault(text, row); stats["independent_reparse_rejections"] += 1
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
            for next_state in expand_residual(grammar, state, index, side):
                queue.append(next_state); stats["expansions"] += 1
        left_leaf = bool(nodes[state.frontier[0]].terminal); right_leaf = bool(nodes[state.frontier[-1]].terminal)
        sides = (-state.owner,) if state.residual else (1, -1)
        for side in sides:
            if (side == 1 and not left_leaf) or (side == -1 and not right_leaf): continue
            next_state = BASE.emit(state, side)
            if next_state is not None: queue.append(next_state); stats["emissions"] += 1
            else: stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()),
            "independent_reparse_rejections": list(rejects.values())}


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = ResidualIndexedGrammar(6); result = solver(grammar, max_states=max_states)
    control_state = explicit_control(grammar)
    control_text = BASE.render(control_state)
    control = audit(grammar, control_text, "complete_recursive_grammar_control", control_state.trace)
    control_words = [leaf.word for leaf in BASE.ordered_leaves(control_state)
                     if leaf.label not in FIXED_LABELS]
    control["provenance"] = {
        "construction": "explicit six-level object-gap relative derivation",
        "lexical_source": "hand-authored MORPHOLOGY only",
        "shared_tree": True,
        "trace_sha256": sha256(json.dumps(list(control_state.trace), sort_keys=True).encode()).hexdigest(),
    }
    control["content_word_forms"] = control_words
    control["content_word_forms_unique"] = len(control_words) == len(set(control_words))
    control["grammar_tree_fully_expanded"] = True
    control["diagnostic_only"] = True
    control["reader_status"] = "grammar control only; not a palindrome candidate and not human readability evidence"
    return {"status": "residual_indexed_typed_morphology_shared_tree", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth,
        "max_states": max_states, "envelope": "Was it ... I saw?", "single_shared_derivation_tree": True,
        "character_residual_during_derivation": True, "lexical_filter_at_exposed_leaf": True,
        "residual_signature_index": True, "optional_unmodified_np": True,
        "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True,
        "catalogue_used_only_for_central_exclusion": True, "corpus_generation": False},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                       "material": "small hand-audited typed morphology inventory; no catalogue or corpus construction material"},
        "morphology": {"entries": [asdict(x) | {"first": x.first, "last": x.last} for x in MORPHOLOGY],
                       "index_keys": len(SIGNATURE_INDEX), "rank_semantics": "authored deterministic ordering only"},
        "complete_recursive_control": control,
        **result,
        "reader_facing_next_operator": "Add a morphology-preserving derivational variant at the same indexed leaf only if its typed independent reparse remains valid.",
        "scope": "This bounded constructive search is not a readability certificate; any exact output requires blinded human evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
