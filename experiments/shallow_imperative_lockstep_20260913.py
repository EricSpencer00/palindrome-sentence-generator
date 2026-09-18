"""Shallow semantic imperative trees with a deterministic character zipper.

Repair after the nested-question grammar failed: change the sentence family
to shallow, right-branching imperatives and eliminate duplicate left/right
emission schedules. One root owns all syntax. Words are selected only when
their W slot is the active outer leaf; neutral states always emit on the left,
and a live character debt forces the right (or opposite owning) edge. This
schedule is complete for each finite grammar yield: every palindrome has the
same outer character pairs regardless of which side emits first.

No catalogue examples, phrases, frozen halves, or generated text seed this
grammar. A hand-authored lexical inventory supplies individual word options.
The intact prose control tests the grammar, not English palindrome success.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "experiments/cfg_feature_palindrome_intersection_20260913.py"
SPEC = importlib.util.spec_from_file_location("shallow_lockstep_base", SOURCE)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)
from llm_palindrome.admission import mechanical_admission_checks, REPEATABLE_FUNCTION_WORDS

MIN_LETTERS, MAX_LETTERS = 100, 220
LEXICON = {
    "det": ("the", "a", "this", "that", "my", "your", "our", "each"),
    "person": ("teacher", "artist", "guard", "writer", "singer", "farmer", "nurse", "pilot",
               "scholar", "friend", "painter", "actor", "poet", "baker", "doctor", "leader",
               "reader", "drawer", "student", "dancer", "driver", "guide", "owner", "helper"),
    "object": ("drawing", "portrait", "letter", "book", "map", "basket", "parcel", "garment",
               "helmet", "medal", "prize", "ribbon", "scarf", "coin", "ticket", "flower",
               "card", "paper", "record", "report", "vase", "cup", "torch", "watch", "drawer"),
    "place": ("garden", "studio", "school", "market", "harbor", "office", "park", "hall",
              "theater", "tunnel", "forest", "valley", "street", "ward", "chamber", "shelter"),
    "adj_person": ("patient", "careful", "young", "quiet", "brave", "gentle", "steady", "honest",
                   "kind", "calm", "eager", "tired", "alert", "bold", "wise", "loyal"),
    "adj_object": ("rare", "small", "old", "new", "fine", "red", "blue", "golden", "bright",
                   "plain", "clean", "neat", "delicate", "valuable", "fragile", "detailed", "nice"),
    "adj_place": ("northern", "coastal", "quiet", "empty", "open", "nearby", "public",
                  "local", "narrow", "hidden", "distant", "central", "ancient", "shaded"),
    "past_person": ("helped", "watched", "greeted", "trusted", "followed", "met", "joined",
                    "guided", "taught", "praised", "welcomed", "saved", "admired", "heard"),
    "past_object": ("carried", "painted", "saved", "watched", "praised", "found", "made",
                    "read", "held", "sent", "bought", "sold", "restored", "studied"),
    "reward": ("reward", "thank", "honor", "surprise"),
    "guide": ("guide", "lead", "escort", "follow"),
    "carry": ("carry", "bring", "send", "deliver", "take", "return"),
    "person_action": ("help", "teach", "praise", "greet", "watch", "train", "draw", "name"),
    "object_action": ("draw", "paint", "read", "keep", "show", "save", "study", "restore"),
    "draw_at": ("draw", "paint", "sketch"),
    "and": ("and",), "who": ("who",), "with": ("with",), "through": ("through",), "to": ("to",), "in": ("in",),
}


def word(category: str) -> BASE.Symbol:
    return BASE.sym("W", category=category)


def np(kind: str, relative: bool = True) -> BASE.Symbol:
    return BASE.sym("NP", type=kind, relative=str(int(relative)))


class Grammar(BASE.FeatureGrammar):
    def start(self):
        return BASE.sym("S")

    def terminal(self, symbol):
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs):
        rows = []
        def add(label, *rhs):
            rows.append(BASE.Production(label, lhs, tuple(rhs)))
        if lhs.name == "S":
            add("sentence:single", BASE.sym("IMP"))
            add("sentence:coordinated", BASE.sym("IMP"), word("and"), BASE.sym("IMP"))
        elif lhs.name == "IMP":
            add("imperative:draw_at", word("draw_at"), np("object", False), word("in"), np("place", False))
            add("imperative:reward", word("reward"), np("person"), word("with"), np("object", False))
            add("imperative:guide", word("guide"), np("person"), word("through"), np("place", False))
            add("imperative:carry", word("carry"), np("object", False), word("to"), np("person"))
            add("imperative:person", word("person_action"), np("person"))
            add("imperative:object", word("object_action"), np("object", False))
        elif lhs.name == "NP":
            kind = lhs.feature("type")
            for modified in (False, True):
                prefix = (word("det"),) + ((word("adj_" + kind),) if modified else ()) + (word(kind),)
                add(f"np:{kind}:{int(modified)}:plain", *prefix)
                if kind == "person" and lhs.feature("relative") == "1":
                    add(f"np:{kind}:{int(modified)}:relative", *prefix, BASE.sym("REL"))
        elif lhs.name == "REL":
            for patient in ("person", "object"):
                add(f"relative:{patient}", word("who"), word("past_" + patient), np(patient, False))
        elif lhs.name == "W":
            category = lhs.feature("category")
            for form in LEXICON[category]:
                add(f"word:{category}:{form}", BASE.sym("T", form=form, label=category))
        return tuple(rows)


def grammar_checks(tree):
    """Inspect an independently parsed tree's roles and shallow relative depth."""
    if tree is None:
        return {"complete_tree": False, "valency_ok": False, "max_relative_depth": None}
    okay, maximum = True, 0
    def walk(node, depth):
        nonlocal okay, maximum
        if node.symbol.name == "REL":
            depth += 1
            okay &= (len(node.children) == 3 and node.children[0].symbol.feature("category") == "who"
                     and node.children[1].symbol.feature("category") == "past_" + node.children[2].symbol.feature("type"))
        maximum = max(maximum, depth)
        for child in node.children:
            walk(child, depth)
    walk(tree, 0)
    return {"complete_tree": tree.symbol.name == "S", "valency_ok": okay, "max_relative_depth": maximum}


def article_ok(text):
    words = text.split()
    return all(next_word[0] not in "aeiou" for current, next_word in zip(words, words[1:]) if current == "a")


def audit(grammar, text, kind, trace=()):
    tree = BASE.parse_tree(grammar, text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    witness = grammar_checks(tree)
    checks.update(independent_complete_reparse=tree is not None, article_agreement=article_ok(text),
                  typed_valency=witness["valency_ok"])
    return {"record_kind": kind, "rendered": text[:1].upper() + text[1:] + ".",
            "independent_exact_audit": BASE.exact_audit(text), "independent_parse": tree is not None,
            "grammar_witness": witness, "mechanical_checks": checks,
            "mechanically_admitted": all(checks.values()), "shared_tree_trace": list(trace),
            "rejection_codes": [key for key, value in checks.items() if not value],
            "provenance": "task-authored grammar and individual words; no catalogue or corpus generation material",
            "reader_status": "unreviewed; programmatic checks do not certify readability"}


def active_edge(state):
    """One canonical emission order removes exponential schedule duplicates."""
    side = -state.owner if state.residual else 1
    return side, 0 if side == 1 else len(state.frontier) - 1


def successors(grammar, state, stats):
    if not state.frontier:
        return ()
    side, edge = active_edge(state)
    node = BASE.node_map(state)[state.frontier[edge]]
    if node.terminal:
        candidate = BASE.emit(state, side)
        if candidate is None:
            stats["character_contradictions"] += 1
            return ()
        stats["characters_emitted"] += 1
        return (candidate,)
    previous = BASE.leaf_map(state)
    content = {leaf.word for leaf in state.leaves if leaf.word not in REPEATABLE_FUNCTION_WORDS}
    accepted = []
    for candidate in BASE.expand(grammar, state, edge):
        added = [leaf for leaf in candidate.leaves if leaf.identifier not in previous]
        # Structural productions never choose a word. Only the exposed W
        # slot can create a terminal, so unseen interior words stay free.
        if added:
            assert node.symbol.name == "W" and len(added) == 1
            form = added[0].word
            if form not in REPEATABLE_FUNCTION_WORDS and (form in content or form == form[::-1]):
                stats["repeated_content_pruned"] += 1
                continue
            if state.residual and (form[0] if side == 1 else form[-1]) != state.residual[0]:
                stats["lexical_endpoint_pruned"] += 1
                continue
        stats["expansions"] += 1
        accepted.append(candidate)
    return tuple(accepted)


def solve(grammar, max_states=100000):
    initial = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    stack = [initial]
    stats = Counter(states=0, expansions=0, characters_emitted=0, complete_trees=0)
    exact, admitted = {}, {}
    while stack and stats["states"] < max_states:
        state = stack.pop()
        stats["states"] += 1
        if state.length > MAX_LETTERS:
            stats["length_pruned"] += 1
            continue
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if state.residual != state.residual[::-1]:
                continue
            text = BASE.render(state)
            row = audit(grammar, text, "complete_shared_tree_closure", state.trace)
            if not row["independent_exact_audit"]["exact"]:
                raise AssertionError("character zipper admitted a non-palindrome")
            exact.setdefault(text, row)
            if row["mechanically_admitted"]:
                admitted.setdefault(text, row)
            continue
        # Reversal makes the first declared grammar/lexical choice run first.
        stack.extend(reversed(successors(grammar, state, stats)))
    return {"stats": dict(stats), "states_exhausted": not stack,
            "exact_closures": list(exact.values()), "mechanically_admitted_closures": list(admitted.values())}


CONTROL = ("reward the patient teacher who helped a young artist with a rare drawing and "
           "guide the careful scholar who watched a quiet painter through the northern garden")


def seam_oracle(grammar):
    text = "draw a nice portrait in a ward"
    normalized = "".join(text.split())
    paired = 0
    for left, right in zip(normalized, reversed(normalized)):
        if left != right:
            break
        paired += 1
    assert BASE.parse_tree(grammar, text) is not None and paired >= 7
    return {"diagnostic_only": True, "rendered": text.capitalize() + ".",
            "independent_parse": True, "outer_matching_character_pairs": paired,
            "matching_prefix": normalized[:paired], "matching_suffix": normalized[-paired:],
            "exact": BASE.exact_audit(text)["exact"],
            "purpose": "proves the root, article, and following word-start seam is satisfiable; does not claim a complete palindrome"}


def run(max_states=100000):
    grammar = Grammar()
    control = audit(grammar, CONTROL, "intact_prose_grammar_control")
    control["diagnostic_only"] = True
    control["reader_status"] = "grammar control only; not a palindrome candidate or human evidence"
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    return {"method": "shallow_typed_imperatives_with_canonical_character_zipper",
            "construction_change": "replaces deep object-gap questions with shallow subject relatives and complete imperative valency frames; one deterministic emission schedule per derivation",
            "config": {"max_states": max_states, "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                       "lexicalize_only_exposed_leaf": True, "single_shared_tree": True,
                       "every_emitted_character_checked_against_live_debt": True},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "grammar_sha256": grammar.digest(), "base_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
                           "lexicon": LEXICON, "catalogue_generation_material": False},
            "complete_grammar_control": control, "outer_seam_oracle": seam_oracle(grammar), **solve(grammar, max_states),
            "next_reader_facing_test": "For any admitted new long closure: randomized blinded intact/shuffled control package, grammaticality and coherent-meaning ratings with free paraphrase; no AI rating treated as human evidence.",
            "next_construction_if_no_candidate": "Replace the small word inventory with typed inflection families and test exact suffix feasibility before choosing a syntactic frame; retain this complete shallow tree and deterministic character order."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-states", default=100000, type=int)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing artifact")
    result = run(args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "stats": result["stats"],
                      "exact": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted_closures"])}))


if __name__ == "__main__":
    main()
