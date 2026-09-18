"""Typed lexical-substitution transducer over the inverted-question tree.

This is a new constructive operator over the shared ``Was it ... I saw?``
grammar.  Lexical alternatives are authored with semantic type, number, and
valency features.  The transducer expands a single parse-tree frontier and
filters a newly exposed lexical leaf against the current character residual;
it never supplies a second half or a catalogue phrase.  Content words already
used in the same tree are not reused.  Completed trees require a fresh full
reparse, exact two-pointer audit, and central admission before being reported.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QUESTION_PATH = ROOT / "experiments/inverted_question_shared_tree_20260913.py"
spec = importlib.util.spec_from_file_location("question_tree_20260913", QUESTION_PATH)
QUESTION = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = QUESTION
spec.loader.exec_module(QUESTION)
BASE = QUESTION.BASE
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
FIXED_WORDS = {"was", "it", "that", "a", "the", "i", "saw"}
FIXED_LABELS = {"fixed_was", "fixed_it", "fixed_i", "fixed_saw", "relative_that", "det"}


@dataclass(frozen=True)
class Lexeme:
    form: str
    type: str
    number: str = "sing"
    role: str = ""
    patient_type: str = ""
    rank: int = 0  # authored ordering, not a corpus frequency or readability score


LEXICON = (
    # Ordinary typed nouns, deliberately authored for this experiment.
    Lexeme("portrait", "object", role="noun", rank=1), Lexeme("packet", "object", role="noun", rank=2),
    Lexeme("artifact", "object", role="noun", rank=3), Lexeme("cabinet", "object", role="noun", rank=4),
    Lexeme("helmet", "object", role="noun", rank=5), Lexeme("garment", "object", role="noun", rank=6),
    Lexeme("teacher", "person", role="noun", rank=1), Lexeme("artist", "person", role="noun", rank=2),
    Lexeme("guard", "person", role="noun", rank=3), Lexeme("writer", "person", role="noun", rank=4),
    Lexeme("singer", "person", role="noun", rank=5), Lexeme("farmer", "person", role="noun", rank=6),
    Lexeme("nurse", "person", role="noun", rank=7), Lexeme("pilot", "person", role="noun", rank=8),
    Lexeme("scholar", "person", role="noun", rank=9), Lexeme("friend", "person", role="noun", rank=10),
    # These are surface modifiers, not generated phrase fragments.
    Lexeme("quiet", "adjective", role="adjective", rank=1), Lexeme("careful", "adjective", role="adjective", rank=2),
    Lexeme("patient", "adjective", role="adjective", rank=3), Lexeme("brave", "adjective", role="adjective", rank=4),
    Lexeme("young", "adjective", role="adjective", rank=5), Lexeme("gentle", "adjective", role="adjective", rank=6),
    Lexeme("steady", "adjective", role="adjective", rank=7), Lexeme("honest", "adjective", role="adjective", rank=8),
    Lexeme("alert", "adjective", role="adjective", rank=9), Lexeme("warm", "adjective", role="adjective", rank=10),
    # Object-gap relative verbs; the patient type is a feature, not a POS tag.
    Lexeme("watched", "verb", role="verb", patient_type="object", rank=1),
    Lexeme("painted", "verb", role="verb", patient_type="object", rank=2),
    Lexeme("saved", "verb", role="verb", patient_type="object", rank=3),
    Lexeme("praised", "verb", role="verb", patient_type="object", rank=4),
    Lexeme("greeted", "verb", role="verb", patient_type="person", rank=1),
    Lexeme("trusted", "verb", role="verb", patient_type="person", rank=2),
    Lexeme("followed", "verb", role="verb", patient_type="person", rank=3),
    Lexeme("helped", "verb", role="verb", patient_type="person", rank=4),
    Lexeme("met", "verb", role="verb", patient_type="person", rank=5),
)

NOUNS = tuple(x for x in LEXICON if x.role == "noun")
ADJECTIVES = tuple(x for x in LEXICON if x.role == "adjective")
VERBS = tuple(x for x in LEXICON if x.role == "verb")


class TransducerGrammar(QUESTION.InvertedQuestionGrammar):
    """The inverted-question feature grammar with an expanded typed lexicon."""

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "NP":
            role, typ, number, depth = f["role"], f["type"], f["number"], int(f["depth"])
            prefix = (BASE.sym("DET_SLOT", number=number, form_role=role),
                      BASE.sym("ADJ_SLOT", type=typ),
                      BASE.sym("N_SLOT", role=role, type=typ, number=number))
            rows = [BASE.Production(f"NP:{role}:{typ}:{depth}:base", lhs, prefix)]
            if depth > 0:
                rows.append(BASE.Production(f"NP:{role}:{typ}:{depth}:relative", lhs,
                                            prefix + (BASE.sym("REL", head_type=typ, depth=str(depth - 1)),)))
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
            return (BASE.Production(f"DET_SLOT:{form}", lhs,
                                    (BASE.sym("T", label="det", form=form),)),)
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


def boundary_compatible(state: BASE.State, side: int) -> bool:
    """Check a newly exposed lexical edge against the current residual."""
    if not state.residual:
        return True
    nodes, leaves = BASE.node_map(state), BASE.leaf_map(state)
    ref = state.frontier[0] if side == 1 else state.frontier[-1]
    if not nodes[ref].terminal or ref not in leaves:
        return True  # Expand through the nonterminal before exposing its lexeme.
    word = leaves[ref].word
    incoming = word[0] if side == 1 else word[-1]
    return state.owner == side or incoming == state.residual[0]


def expand_transducer(grammar: TransducerGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    """Expand one tree node, filtering lexical alternatives at the exposed edge."""
    used = content_words(state)
    options = []
    for candidate in BASE.expand(grammar, state, index):
        if not boundary_compatible(candidate, side):
            continue
        new_nodes = BASE.node_map(candidate)
        new_leaves = BASE.leaf_map(candidate)
        # A candidate lexical form is rejected at its grammar leaf if it is a
        # repeated content form or a self-palindromic word. Fixed function
        # words remain grammar material and are handled by central admission.
        added = [leaf for leaf in candidate.leaves if leaf.identifier not in BASE.leaf_map(state)]
        if any(leaf.word in used or leaf.word == leaf.word[::-1] for leaf in added if leaf.label not in FIXED_LABELS):
            continue
        options.append(candidate)
    return tuple(options)


def audit(grammar: TransducerGrammar, text: str, kind: str, trace: tuple[tuple[int, str], ...], reason: str | None = None) -> dict[str, object]:
    exact, tree = QUESTION.exact_audit(text), QUESTION.parse_tree(grammar, text)
    witness = QUESTION.feature_witness(tree)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if tree is None: codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]: codes.append("agreement_failure")
    if not witness["valency_ok"]: codes.append("valency_failure")
    if reason and reason not in codes: codes.append(reason)
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact,
            "independent_parse": tree is not None, "feature_witness": witness,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "shared_tree_trace": list(trace),
            "reader_status": "unreviewed; programmatic measures do not certify readability"}


def solver(grammar: TransducerGrammar, *, max_states: int = 100000, max_records: int = 100) -> dict[str, object]:
    root = BASE.Node(0, grammar.start())
    initial = BASE.State((0,), (root,), (), "", 0, 0, ())
    queue, seen, exact_rows, admitted, rejects = deque([initial]), set(), {}, {}, {}
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
                text = BASE.render(state); row = audit(grammar, text, "complete_shared_tree_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]: admitted.setdefault(text, row)
                    elif len(rejects) < max_records:
                        rejects.setdefault(text, row); stats["independent_reparse_rejections"] += 1
            continue
        nodes = BASE.node_map(state)
        unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        index = None
        if state.residual:
            # A live residual constrains the opposite edge. Keep all inner
            # lexical slots unassigned until that edge consumes its debt.
            side = -state.owner
            edge = 0 if side == 1 else len(state.frontier) - 1
            if not nodes[state.frontier[edge]].terminal:
                index = edge
        elif unresolved:
            if not nodes[state.frontier[0]].terminal:
                index, side = 0, 1
            elif not nodes[state.frontier[-1]].terminal:
                index, side = len(state.frontier) - 1, -1
            else:
                index, side = unresolved[0], 1
        if index is not None:
            for next_state in expand_transducer(grammar, state, index, side):
                queue.append(next_state); stats["expansions"] += 1
        left_leaf = bool(nodes[state.frontier[0]].terminal)
        right_leaf = bool(nodes[state.frontier[-1]].terminal)
        sides = (-state.owner,) if state.residual else (1, -1)
        for side in sides:
            if (side == 1 and not left_leaf) or (side == -1 and not right_leaf): continue
            next_state = BASE.emit(state, side)
            if next_state is not None:
                queue.append(next_state); stats["emissions"] += 1
            else:
                stats["residual_contradictions"] += 1
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()),
            "independent_reparse_rejections": list(rejects.values())}


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = TransducerGrammar(6); result = solver(grammar, max_states=max_states)
    return {"status": "inverted_question_typed_lexical_substitution_transducer", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth,
        "max_states": max_states, "envelope": "Was it ... I saw?", "single_shared_derivation_tree": True,
        "character_residual_during_derivation": True, "lexical_filter_at_exposed_leaf": True,
        "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True,
        "catalogue_used_only_for_central_exclusion": True, "corpus_generation": False},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grammar_sha256": grammar.digest(), "material": "authored typed semantic lexicon; no catalogue or corpus construction material"},
        "lexicon": {"entries": [asdict(x) for x in LEXICON], "rank_semantics": "authored deterministic ordering only"},
        **result,
        "reader_facing_next_operator": "If this filter yields no closure, add typed morphology-preserving lexical variants at the same exposed leaves; retain full-tree closure and independent reparse.",
        "scope": "This bounded constructive search is not a readability certificate; any exact output requires blinded human evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
