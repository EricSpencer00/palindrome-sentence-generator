"""Construct exact palindromes inside one lexicalized feature-grammar tree.

This replaces endpoint/menu assembly.  A state owns one connected sentence
derivation.  It expands its outer lexical leaves under the letter demand left
by the opposite edge, so a grammatical choice and a palindrome choice cannot
be made independently.  The finite grammar includes agreement, determiner,
valency, and light semantic selection constraints.  These constraints reject
bad syntax; they do *not* certify ordinary-reader readability.

The vocabulary and grammar are authored before this search and are deliberately
small.  The run is a falsifiable bounded construction attempt, not a claim that
the grammar's source controls are readable palindromes.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS = 100
MAX_LETTERS = 180
DEFAULT_STATE_CAP = 50_000
WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Lexeme:
    word: str
    category: str
    number: str | None = None
    semantic: str | None = None
    subject_types: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class Leaf:
    """A lexical leaf in a single connected grammar derivation.

    ``role`` binds a chosen noun/verb/modifier to constraints elsewhere in the
    same tree.  Literal leaves still appear in the tree and are expanded one at
    a time; they are not a separately authored endpoint string.
    """

    identifier: str
    category: str
    role: str | None = None
    literal: str | None = None


@dataclass(frozen=True)
class Grammar:
    identifier: str
    label: str
    leaves: tuple[Leaf, ...]
    tree: tuple[tuple[str, tuple[str, ...]], ...]


# These are common, ordinary words selected before any search result is read.
# Past-tense declaratives avoid hidden agreement; the present-tense material
# gives the independent parser an explicit subject--verb agreement constraint.
LEXICON = (
    Lexeme("set", "base_verb", semantic="recording", object_types=("record",)),
    Lexeme("draft", "base_verb", semantic="writing", object_types=("report",)),
    Lexeme("record", "noun", "singular", "record"),
    Lexeme("report", "noun", "singular", "report"),
    Lexeme("curator", "noun", "singular", "person"),
    Lexeme("archivist", "noun", "singular", "person"),
    Lexeme("researcher", "noun", "singular", "person"),
    Lexeme("editor", "noun", "singular", "person"),
    Lexeme("survey", "noun", "singular", "organization"),
    Lexeme("museum", "noun", "singular", "organization"),
    Lexeme("archive", "noun", "singular", "organization"),
    Lexeme("team", "noun", "singular", "organization"),
    Lexeme("manuscript", "noun", "singular", "artifact"),
    Lexeme("artifact", "noun", "singular", "artifact"),
    Lexeme("sketch", "noun", "singular", "artifact"),
    Lexeme("collection", "noun", "singular", "artifact"),
    Lexeme("trial", "noun", "singular", "event"),
    Lexeme("review", "noun", "singular", "event"),
    Lexeme("study", "noun", "singular", "event"),
    Lexeme("storm", "noun", "singular", "event"),
    Lexeme("operates", "sg_verb", "singular", "operation", ("organization",)),
    Lexeme("continues", "sg_verb", "singular", "operation", ("organization",)),
    Lexeme("monitors", "sg_verb", "singular", "observation", ("organization",), ("artifact",)),
    Lexeme("records", "sg_verb", "singular", "recording", ("person",), ("artifact", "record")),
    Lexeme("reviews", "sg_verb", "singular", "reviewing", ("person",), ("artifact", "record", "report")),
    Lexeme("catalogs", "sg_verb", "singular", "recording", ("person",), ("artifact",)),
    Lexeme("reviewed", "past_verb", semantic="reviewing", subject_types=("person",), object_types=("artifact", "record", "report")),
    Lexeme("cataloged", "past_verb", semantic="recording", subject_types=("person",), object_types=("artifact",)),
    Lexeme("repaired", "past_verb", semantic="repairing", subject_types=("person",), object_types=("artifact",)),
    Lexeme("recorded", "past_verb", semantic="recording", subject_types=("person",), object_types=("artifact", "record")),
    Lexeme("annual", "adjective", semantic="any"),
    Lexeme("careful", "adjective", semantic="any"),
    Lexeme("coastal", "adjective", semantic="organization"),
    Lexeme("damaged", "adjective", semantic="artifact"),
    Lexeme("detailed", "adjective", semantic="artifact"),
    Lexeme("northern", "adjective", semantic="organization"),
    Lexeme("patient", "adjective", semantic="person"),
    Lexeme("expert", "adjective", semantic="person"),
)
LEX_BY_CATEGORY: dict[str, tuple[Lexeme, ...]] = {}
for _item in LEXICON:
    LEX_BY_CATEGORY.setdefault(_item.category, ())
    LEX_BY_CATEGORY[_item.category] += (_item,)


def _leaves(*items: tuple[str, str, str | None] | tuple[str, str, str | None, str | None]) -> tuple[Leaf, ...]:
    out = []
    for item in items:
        identifier, category, role, *literal = item
        out.append(Leaf(identifier, category, role, literal[0] if literal else None))
    return tuple(out)


# Each grammar has a genuine root-to-leaf structure.  The linear leaf order is
# derived from that tree; it is not a left phrase stitched to a right phrase.
GRAMMARS = (
    Grammar(
        "imperative-record", "Imperative with a coordinated temporal clause",
        _leaves(
            ("matrix_verb", "base_verb", "matrix_verb"), ("matrix_det", "literal", None, "a"),
            ("matrix_object", "noun", "matrix_object"), ("for", "literal", None, "for"),
            ("event_det", "literal", None, "the"), ("event_adj", "adjective", "event_adj"),
            ("event", "noun", "event"), ("during", "literal", None, "during"),
            ("review_det", "literal", None, "the"), ("review_adj", "adjective", "review_adj"),
            ("review", "noun", "review"), ("near", "literal", None, "near"),
            ("org_det", "literal", None, "the"), ("org_adj", "adjective", "org_adj"),
            ("organization", "noun", "organization"), ("while", "literal", None, "while"),
            ("subject_det", "literal", None, "the"), ("subject_adj", "adjective", "subject_adj"),
            ("subject", "noun", "subject"), ("final_verb", "sg_verb", "final_verb"),
        ),
        (("S", ("ImperativeVP", "TemporalClause")),
         ("ImperativeVP", ("matrix_verb", "matrix_det", "matrix_object", "for", "event_det", "event_adj", "event", "during", "review_det", "review_adj", "review", "near", "org_det", "org_adj", "organization")),
         ("TemporalClause", ("while", "subject_det", "subject_adj", "subject", "final_verb"))),
    ),
    Grammar(
        "relative-declarative", "Declarative with an object relative clause",
        _leaves(
            ("subject_det", "literal", None, "an"), ("subject_adj", "adjective", "subject_adj"),
            ("subject", "noun", "subject"), ("who", "literal", None, "who"),
            ("relative_verb", "sg_verb", "relative_verb"), ("relative_det", "literal", None, "the"),
            ("relative_object_adj", "adjective", "relative_object_adj"), ("relative_object", "noun", "relative_object"),
            ("matrix_verb", "sg_verb", "matrix_verb"), ("matrix_det", "literal", None, "the"),
            ("matrix_object_adj", "adjective", "matrix_object_adj"), ("matrix_object", "noun", "matrix_object"),
            ("after", "literal", None, "after"), ("event_det", "literal", None, "the"),
            ("event_adj", "adjective", "event_adj"), ("event", "noun", "event"),
            ("while", "literal", None, "while"), ("org_det", "literal", None, "the"),
            ("org_adj", "adjective", "org_adj"), ("organization", "noun", "organization"),
            ("final_verb", "sg_verb", "final_verb"), ("final_det", "literal", None, "the"),
            ("final_object_adj", "adjective", "final_object_adj"), ("final_object", "noun", "final_object"),
        ),
        (("S", ("NP", "Relative", "MatrixVP", "TemporalClause")),
         ("NP", ("subject_det", "subject_adj", "subject")),
         ("Relative", ("who", "relative_verb", "relative_det", "relative_object_adj", "relative_object")),
         ("MatrixVP", ("matrix_verb", "matrix_det", "matrix_object_adj", "matrix_object", "after", "event_det", "event_adj", "event")),
         ("TemporalClause", ("while", "org_det", "org_adj", "organization", "final_verb", "final_det", "final_object_adj", "final_object"))),
    ),
    Grammar(
        "causal-declarative", "Causally connected two-clause declarative",
        _leaves(
            ("subject_det", "literal", None, "a"), ("subject_adj", "adjective", "subject_adj"),
            ("subject", "noun", "subject"), ("matrix_verb", "past_verb", "matrix_verb"),
            ("object_det", "literal", None, "the"), ("object_adj", "adjective", "object_adj"),
            ("object", "noun", "object"), ("because", "literal", None, "because"),
            ("cause_det", "literal", None, "an"), ("cause_adj", "adjective", "cause_adj"),
            ("cause_subject", "noun", "cause_subject"), ("cause_verb", "past_verb", "cause_verb"),
            ("cause_object_det", "literal", None, "the"), ("cause_object_adj", "adjective", "cause_object_adj"),
            ("cause_object", "noun", "cause_object"), ("after", "literal", None, "after"),
            ("event_det", "literal", None, "the"), ("event_adj", "adjective", "event_adj"),
            ("event", "noun", "event"), ("near", "literal", None, "near"),
            ("org_det", "literal", None, "the"), ("org_adj", "adjective", "org_adj"),
            ("organization", "noun", "organization"),
        ),
        (("S", ("MatrixClause", "CauseClause")),
         ("MatrixClause", ("subject_det", "subject_adj", "subject", "matrix_verb", "object_det", "object_adj", "object")),
         ("CauseClause", ("because", "cause_det", "cause_adj", "cause_subject", "cause_verb", "cause_object_det", "cause_object_adj", "cause_object", "after", "event_det", "event_adj", "event", "near", "org_det", "org_adj", "organization"))),
    ),
)


def _surface(words: Iterable[str]) -> str:
    sentence = " ".join(words)
    return sentence[:1].upper() + sentence[1:] + "."


# Canonical controls are selected by role, not by arbitrary lexicon order.  They
# are ordinary complete sentences used to prove that the parser recognizes the
# actual grammar independently of a search trace; they are never candidates.
CANONICAL = {
    "imperative-record": {
        "matrix_verb": "set", "matrix_object": "record", "event_adj": "annual", "event": "trial",
        "review_adj": "careful", "review": "review", "org_adj": "northern", "organization": "archive",
        "subject_adj": "coastal", "subject": "survey", "final_verb": "operates",
    },
    "relative-declarative": {
        "subject_adj": "expert", "subject": "archivist", "relative_verb": "catalogs",
        "relative_object_adj": "detailed", "relative_object": "manuscript", "matrix_verb": "reviews",
        "matrix_object_adj": "annual", "matrix_object": "report", "event_adj": "careful", "event": "study",
        "org_adj": "coastal", "organization": "museum", "final_verb": "monitors",
        "final_object_adj": "damaged", "final_object": "collection",
    },
    "causal-declarative": {
        "subject_adj": "patient", "subject": "curator", "matrix_verb": "recorded",
        "object_adj": "damaged", "object": "artifact", "cause_adj": "expert", "cause_subject": "archivist",
        "cause_verb": "reviewed", "cause_object_adj": "annual", "cause_object": "report",
        "event_adj": "careful", "event": "study", "org_adj": "coastal", "organization": "museum",
    },
}


def _choices(leaf: Leaf) -> tuple[Lexeme, ...]:
    if leaf.literal is not None:
        return (Lexeme(leaf.literal, "literal"),)
    return LEX_BY_CATEGORY[leaf.category]


def _canonical_choice(grammar: Grammar, leaf: Leaf, assigned: dict[str, Lexeme]) -> Lexeme:
    if leaf.literal is not None:
        return _choices(leaf)[0]
    if leaf.role and leaf.role in assigned:
        return assigned[leaf.role]
    requested = CANONICAL[grammar.identifier].get(leaf.role or "")
    if requested:
        return next(candidate for candidate in _choices(leaf) if candidate.word == requested)
    return _choices(leaf)[0]


def _assigned(values: tuple[tuple[str, Lexeme], ...]) -> dict[str, Lexeme]:
    return dict(values)


def _is_adjective_valid(adjective: Lexeme, noun: Lexeme) -> bool:
    return adjective.semantic in ("any", noun.semantic)


def _verb_valid(verb: Lexeme, subject: Lexeme, obj: Lexeme | None, *, requires_object: bool) -> bool:
    if verb.subject_types and subject.semantic not in verb.subject_types:
        return False
    if requires_object:
        return obj is not None and bool(verb.object_types) and obj.semantic in verb.object_types
    return obj is None and not verb.object_types


def valid_features(grammar: Grammar, values: tuple[tuple[str, Lexeme], ...], *, final: bool) -> bool:
    """Feature and selectional constraints shared by distant grammar leaves."""
    value = _assigned(values)
    noun_roles = ("subject", "cause_subject", "organization", "event", "review", "matrix_object",
                  "relative_object", "final_object", "object", "cause_object", "matrix_object")
    for role in noun_roles:
        item = value.get(role)
        if item is not None and item.category == "noun" and item.number != "singular":
            return False
    for adjective_role, noun_role in (("subject_adj", "subject"), ("cause_adj", "cause_subject"),
                                      ("org_adj", "organization"), ("event_adj", "event"),
                                      ("review_adj", "review"), ("object_adj", "object"),
                                      ("cause_object_adj", "cause_object"), ("relative_object_adj", "relative_object"),
                                      ("matrix_object_adj", "matrix_object"), ("final_object_adj", "final_object")):
        if adjective_role in value and noun_role in value and not _is_adjective_valid(value[adjective_role], value[noun_role]):
            return False
    # Every finite verb agrees with its real subject and satisfies valency.
    # The roles differ by tree shape; never treat an organizational temporal
    # clause as if it inherited the matrix subject.
    relationships = {
        "imperative-record": (("final_verb", "subject", None, False),),
        "relative-declarative": (("relative_verb", "subject", "relative_object"),
                                   ("matrix_verb", "subject", "matrix_object"),
                                   ("final_verb", "organization", "final_object")),
        "causal-declarative": (("matrix_verb", "subject", "object"),
                                ("cause_verb", "cause_subject", "cause_object")),
    }[grammar.identifier]
    relationships = [entry if len(entry) == 4 else (*entry, True) for entry in relationships]
    for verb_role, subject_role, object_role, requires_object in relationships:
        verb, subject = value.get(verb_role), value.get(subject_role)
        obj = value.get(object_role) if object_role else None
        if verb is not None and verb.category == "sg_verb" and subject is not None and subject.number != "singular":
            return False
        if verb is not None and subject is not None and (obj is not None or final):
            if not _verb_valid(verb, subject, obj, requires_object=requires_object):
                return False
    # Imperatives take a base form and their record-writing verbs require a record object.
    if grammar.identifier == "imperative-record":
        verb, obj = value.get("matrix_verb"), value.get("matrix_object")
        if verb is not None and verb.category != "base_verb":
            return False
        if verb is not None and obj is not None and not _verb_valid(verb, Lexeme("you", "pronoun", "singular", "person"), obj, requires_object=True):
            return False
    return True


def independent_parse(grammar: Grammar, text: str) -> dict[str, object]:
    """Freshly parse a surface; never inspect a saved search derivation."""
    if re.sub(r"[A-Za-z\s.]", "", text) or not text.endswith("."):
        return {"intact": False, "reason": "unsupported_rendering", "roles": {}}
    words = tuple(WORD.findall(text.casefold()))
    if len(words) != len(grammar.leaves):
        return {"intact": False, "reason": "leaf_count", "roles": {}}
    values: list[tuple[str, Lexeme]] = []
    for leaf, word in zip(grammar.leaves, words):
        candidates = [candidate for candidate in _choices(leaf) if candidate.word == word]
        if len(candidates) != 1:
            return {"intact": False, "reason": f"illegal_{leaf.identifier}", "roles": {}}
        if leaf.role:
            values.append((leaf.role, candidates[0]))
    intact = valid_features(grammar, tuple(values), final=True)
    return {"intact": intact, "reason": None if intact else "feature_or_valency", 
            "roles": {key: value.word for key, value in values},
            "tree": [[parent, list(children)] for parent, children in grammar.tree]}


def independent_exact(text: str) -> dict[str, object]:
    try:
        tape = normalize_letters(text)
    except ValueError:
        return {"exact": False, "letters": 0, "normalized": "", "first_mismatch": None}
    for index in range(len(tape) // 2):
        if tape[index] != tape[-index - 1]:
            return {"exact": False, "letters": len(tape), "normalized": tape,
                    "first_mismatch": [index, len(tape) - index - 1]}
    return {"exact": bool(tape), "letters": len(tape), "normalized": tape, "first_mismatch": None}


def _consume(debt: str, owner: int, incoming: str, side: int) -> tuple[str, int] | None:
    """Cancel an outside-in character chunk against the current residual.

    ``owner`` identifies the edge that supplied ``debt`` (zero iff it is
    empty); ``side`` identifies the edge that emitted ``incoming``.  Any
    remaining letters retain the owner of the chunk they came from.
    """
    if side not in (-1, 1) or (debt and side != -owner):
        raise ValueError("incoming characters must come from the opposite edge")
    if not debt:
        return incoming, side
    shared = min(len(debt), len(incoming))
    if debt[:shared] != incoming[:shared]:
        return None
    if len(debt) >= len(incoming):
        return debt[shared:], owner if len(debt) > shared else 0
    return incoming[shared:], side


def _next_side(debt: str, owner: int) -> int:
    """Return the edge that can cancel the current character residual."""
    return -owner if debt else 1


@dataclass(frozen=True)
class State:
    lo: int
    hi: int
    left: tuple[str, ...]
    right: tuple[str, ...]
    values: tuple[tuple[str, Lexeme], ...]
    debt: str
    owner: int
    trace: tuple[dict[str, object], ...]


def _state_surface(grammar: Grammar, state: State) -> str:
    """Complete any untouched leaves canonically for an auditable rejection."""
    assigned = _assigned(state.values)
    words = list(state.left)
    for leaf in grammar.leaves[state.lo:state.hi + 1]:
        words.append(_canonical_choice(grammar, leaf, assigned).word)
    words.extend(state.right)
    return _surface(words)


def _admission(grammar: Grammar, state: State, *, kind: str) -> dict[str, object]:
    rendered = _state_surface(grammar, state)
    parse = independent_parse(grammar, rendered)
    exact = independent_exact(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    checks["independent_exact_audit"] = exact["exact"]
    checks["independent_feature_grammar_parse"] = parse["intact"]
    return {"kind": kind, "rendered": rendered, "independent_exact_audit": exact,
            "independent_sentence_witness": parse, "current_central_admission": checks,
            "rejection_codes": [key for key, value in checks.items() if not value],
            "mechanically_admitted": bool(all(checks.values())),
            "reader_status": "No human reader evidence; this record makes no readability claim."}


def solve(grammar: Grammar, *, state_cap: int = DEFAULT_STATE_CAP,
          min_letters: int = MIN_LETTERS, max_letters: int = MAX_LETTERS) -> dict[str, object]:
    """Outside-in lexical search over a *single* feature-grammar derivation."""
    if state_cap < 1 or min_letters < 0 or max_letters < min_letters:
        raise ValueError("invalid search bounds")
    queue = deque([State(0, len(grammar.leaves) - 1, (), (), (), "", 1, ())])
    stats = Counter(states=0, lexical_expansions=0, feature_rejections=0,
                    letter_rejections=0, completed_derivations=0, exact_closures=0)
    completions, exact_rows = [], []
    exhausted = True
    while queue:
        if stats["states"] >= state_cap:
            exhausted = False
            break
        state = queue.popleft()
        stats["states"] += 1
        if state.lo > state.hi:
            stats["completed_derivations"] += 1
            record = _admission(grammar, state, kind="complete_derivation")
            record["residual"] = state.debt
            record["cancellation_trace"] = list(state.trace)
            if state.debt == state.debt[::-1] and min_letters <= record["independent_exact_audit"]["letters"] <= max_letters:
                if not record["independent_exact_audit"]["exact"]:
                    raise AssertionError("outside-in closure contradicted independent audit")
                stats["exact_closures"] += 1
                exact_rows.append(record)
            elif len(completions) < 80:
                completions.append(record)
            continue
        # ``owner`` is the side that supplied the still-unmatched letters.
        # The next lexical leaf must therefore be emitted from the opposite
        # edge.  Expanding the owner again would merely spell an entire
        # sentence from left to right and test it after the fact, which is not
        # a joint grammar--palindrome construction.
        side = _next_side(state.debt, state.owner)
        index = state.lo if side == 1 else state.hi
        leaf = grammar.leaves[index]
        for lexical in _choices(leaf):
            stats["lexical_expansions"] += 1
            values = state.values + ((leaf.role, lexical),) if leaf.role else state.values
            if not valid_features(grammar, values, final=False):
                stats["feature_rejections"] += 1
                continue
            incoming = normalize_letters(lexical.word)
            if side == -1:
                incoming = incoming[::-1]
            consumed = _consume(state.debt, state.owner, incoming, side)
            if consumed is None:
                stats["letter_rejections"] += 1
                continue
            debt, owner = consumed
            if len(state.left) + len(state.right) + 1 > len(grammar.leaves):
                raise AssertionError("a word was placed outside its one derivation")
            trace = state.trace + ({"leaf": leaf.identifier, "side": "left" if side == 1 else "right",
                                    "word": lexical.word, "incoming": incoming,
                                    "debt_before": state.debt, "debt_after": debt},)
            if side == 1:
                next_state = State(index + 1, state.hi, state.left + (lexical.word,), state.right,
                                   values, debt, owner, trace)
            else:
                next_state = State(state.lo, index - 1, state.left, (lexical.word,) + state.right,
                                   values, debt, owner, trace)
            queue.append(next_state)
    # A source control is parsed independently and makes an actual rendered
    # grammatical diagnostic available even when early letter mismatch prunes
    # every branch before a full derivation closes.
    source = State(0, -1, tuple(_canonical_choice(grammar, leaf, {}) .word for leaf in grammar.leaves), (), (), "", 1, ())
    source_control = _admission(grammar, source, kind="authored_nonpalindromic_grammar_control")
    for row in exact_rows:
        if not row["independent_sentence_witness"]["intact"]:
            raise AssertionError("an exact row must remain a connected grammar derivation")
    return {"grammar": grammar.identifier, "label": grammar.label, "exhausted": exhausted,
            "stats": dict(stats), "source_control": source_control, "complete_rejections": completions,
            "exact_closures": exact_rows,
            "mechanically_admitted": [row for row in exact_rows if row["mechanically_admitted"]]}


def run(*, state_cap: int = DEFAULT_STATE_CAP) -> dict[str, object]:
    runs = [solve(grammar, state_cap=state_cap) for grammar in GRAMMARS]
    exact = [row for result in runs for row in result["exact_closures"]]
    return {"status": "complete_feature_unified_grammar_construction_run",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                       "state_cap_per_grammar": state_cap, "fixed_tape": False,
                       "single_connected_derivation": True,
                       "exactness_enforced_after_each_lexical_leaf": True,
                       "human_readability_claimed": False},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "admission_sha256": sha256((ROOT / "llm_palindrome/admission.py").read_bytes()).hexdigest(),
                           "catalogue_sha256": sha256((ROOT / "data/known_palindromes.json").read_bytes()).hexdigest(),
                           "material": "Task-authored feature grammar and ordinary lexicon; known palindrome catalogue used only by the central exclusion gate."},
            "runs": runs, "exact_closures": exact,
            "mechanically_admitted": [row for row in exact if row["mechanically_admitted"]],
            "readable_survivors": [],
            "reader_facing_next_test": "Only a mechanically admitted, independently parsed complete sentence can enter a blinded reader package; no current row qualifies."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-cap", type=int, default=DEFAULT_STATE_CAP)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing artifact")
    result = run(state_cap=args.state_cap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": sum(row["stats"]["states"] for row in result["runs"]),
                      "exact": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
