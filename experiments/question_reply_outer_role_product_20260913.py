"""Productive question/reply grammar with a short functional endpoint.

This successor changes the outer topology rather than patching the previous
cause/result words.  A typed question about a group's state is answered by a
semantically licensed action, using a normal final object.  The question trie
is intersected with the reversed ``yes + reply`` trie before a complete
question/reply pair is selected.  Role alternatives are generated from the
relation inventory, not from a prebuilt palindrome or a sentence catalogue.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

MAX_STATES = 100_000


@dataclass(frozen=True)
class ReplyOption:
    verb: str
    object: str


@dataclass(frozen=True)
class QuestionState:
    predicate: str
    replies: tuple[ReplyOption, ...]


@dataclass(frozen=True)
class DiscourseFrame:
    relation: str
    subjects: tuple[str, ...]
    states: tuple[QuestionState, ...]


FRAME = DiscourseFrame(
    "group_presence_and_response",
    ("people", "patrons", "guests"),
    (
        QuestionState("present", (ReplyOption("attend", "opera"), ReplyOption("applaud", "opera"), ReplyOption("enter", "hall"))),
        QuestionState("nearby", (ReplyOption("enter", "arena"), ReplyOption("enter", "hall"))),
        QuestionState("ready", (ReplyOption("enter", "arena"), ReplyOption("attend", "opera"))),
    ),
)


@dataclass(frozen=True)
class Derivation:
    relation: str
    subject: str
    predicate: str
    reply_verb: str
    reply_object: str

    @property
    def question_words(self) -> tuple[str, ...]:
        return ("are", self.subject, self.predicate)

    @property
    def reply_words(self) -> tuple[str, ...]:
        return ("yes", "they", self.reply_verb, "the", self.reply_object)

    @property
    def question_tape(self) -> str:
        return normalize_letters(" ".join(self.question_words))

    @property
    def reply_suffix_tape(self) -> str:
        return normalize_letters(" ".join(self.reply_words))

    @property
    def reversed_reply_tape(self) -> str:
        return self.reply_suffix_tape[::-1]

    @property
    def rendered(self) -> str:
        return f"Are {self.subject} {self.predicate}? Yes, they {self.reply_verb} the {self.reply_object}."


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.endings: list[Derivation] = []


def add_to_trie(root: TrieNode, tape: str, derivation: Derivation) -> None:
    node = root
    for char in tape:
        node = node.children.setdefault(char, TrieNode())
    node.endings.append(derivation)


def productive_derivations() -> tuple[Derivation, ...]:
    return tuple(
        Derivation(FRAME.relation, subject, state.predicate, reply.verb, reply.object)
        for subject in FRAME.subjects
        for state in FRAME.states
        for reply in state.replies
    )


def compile_productive_grammars() -> tuple[TrieNode, TrieNode, list[dict[str, object]], list[dict[str, object]]]:
    left_root, right_root = TrieNode(), TrieNode()
    left_rows: list[dict[str, object]] = []
    right_rows: list[dict[str, object]] = []
    for derivation in productive_derivations():
        add_to_trie(left_root, derivation.question_tape, derivation)
        add_to_trie(right_root, derivation.reversed_reply_tape, derivation)
        left_rows.append({
            "relation": derivation.relation,
            "words": derivation.question_words,
            "rendered_clause": "Are " + " ".join(derivation.question_words[1:]) + "?",
            "tape": derivation.question_tape,
        })
        right_rows.append({
            "relation": derivation.relation,
            "words": derivation.reply_words,
            "rendered_clause": "Yes, " + " ".join(derivation.reply_words[1:]) + ".",
            "forward_suffix_tape": derivation.reply_suffix_tape,
            "reversed_suffix_tape": derivation.reversed_reply_tape,
        })
    return left_root, right_root, left_rows, right_rows


def independent_discourse_parse(derivation: Derivation, rendered: str) -> dict[str, object]:
    """Reparse the complete surface as one question/reply relation."""
    try:
        units = tuple(tokenize(rendered))
    except ValueError:
        return {"ok": False, "reason": "tokenization_failed"}
    if "yes" not in units:
        return {"ok": False, "reason": "missing_reply_operator"}
    pivot = units.index("yes")
    question, reply = units[:pivot], units[pivot:]
    question_shape = len(question) == 3 and question[0] == "are"
    reply_shape = len(reply) == 5 and reply[:2] == ("yes", "they") and reply[3] == "the"
    allowed = {
        state.predicate: {(option.verb, option.object) for option in state.replies}
        for state in FRAME.states
    }
    relation_ok = (
        question_shape
        and reply_shape
        and question[1] in FRAME.subjects
        and question[2] in allowed
        and (reply[2], reply[4]) in allowed[question[2]]
    )
    return {
        "ok": relation_ok,
        "relation": derivation.relation,
        "question_shape": question_shape,
        "reply_shape": reply_shape,
        "shared_question_reply_relation": relation_ok,
        "question_words": question,
        "reply_words": reply,
    }


def expansion_hard_rejection(rendered: str) -> list[str]:
    gate = mechanical_admission_checks(rendered, min_letters=30, max_letters=260)
    hard_names = (
        "not_word_order_symmetry",
        "no_repeated_nontrivial_unit",
        "no_self_palindromic_proper_multiword_span",
        "not_forbidden_catalogue_control",
        "not_catalogue_family_derivative",
        "not_forbidden_catalogue_endpoint_scaffold",
    )
    return [name for name in hard_names if not gate.get(name, False)]


def replay_character_ledger(row: dict[str, object]) -> bool:
    left = str(row["question_tape"])
    right = str(row["reversed_reply_tape"])
    ledger = row.get("character_ledger", [])
    if len(left) != len(right) or len(ledger) != len(left):
        return False
    return all(
        entry.get("position") == index
        and entry.get("left_character") == left[index - 1]
        and entry.get("right_reversed_character") == right[index - 1]
        and entry.get("matched") is True
        and left[index - 1] == right[index - 1]
        for index, entry in enumerate(ledger, start=1)
    )


def intersect_product_tries(left_root: TrieNode, right_root: TrieNode, *, state_limit: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    terminal_pairs: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    states = 0
    truncated = False
    deepest = {"prefix": "", "characters": 0, "left_terminal": False, "right_terminal": False}
    first_failure: dict[str, object] | None = None

    def visit(left: TrieNode, right: TrieNode, prefix: str) -> None:
        nonlocal states, truncated, deepest, first_failure
        if states >= state_limit:
            truncated = True
            return
        states += 1
        if len(prefix) > int(deepest["characters"]):
            deepest = {"prefix": prefix, "characters": len(prefix), "left_terminal": bool(left.endings), "right_terminal": bool(right.endings)}
        common = sorted(set(left.children) & set(right.children))
        if prefix and not common and first_failure is None:
            first_failure = {"prefix": prefix, "left_next_chars": sorted(left.children), "right_next_chars": sorted(right.children), "reason": "no_common_next_character"}
        if left.endings and right.endings:
            for left_derivation in left.endings:
                for right_derivation in right.endings:
                    if left_derivation.relation != right_derivation.relation:
                        continue
                    row = {
                        "relation": left_derivation.relation,
                        "question_words": left_derivation.question_words,
                        "reply_words": right_derivation.reply_words,
                        "question_tape": left_derivation.question_tape,
                        "reversed_reply_tape": right_derivation.reversed_reply_tape,
                        "joined_rendered": right_derivation.__class__(left_derivation.relation, left_derivation.subject, left_derivation.predicate, right_derivation.reply_verb, right_derivation.reply_object).rendered,
                        "character_ledger": [
                            {"position": index, "left_character": char, "right_reversed_character": char, "matched": True}
                            for index, char in enumerate(prefix, start=1)
                        ],
                    }
                    codes = expansion_hard_rejection(row["joined_rendered"])
                    if codes:
                        rejected.append({**row, "rejection_codes": codes})
                    else:
                        terminal_pairs.append(row)
        for char in common:
            visit(left.children[char], right.children[char], prefix + char)

    visit(left_root, right_root, "")
    return terminal_pairs, {
        "states_examined": states,
        "states_exhausted": not truncated,
        "search_truncated": truncated,
        "deepest_joint_state": deepest,
        "first_joint_failure": first_failure,
        "terminal_pairs_rejected_during_expansion": rejected,
    }


def audit_closure(row: dict[str, object]) -> dict[str, object]:
    rendered = str(row["joined_rendered"])
    tape = normalize_letters(rendered)
    gate = mechanical_admission_checks(rendered, min_letters=30, max_letters=260)
    derivation = Derivation(str(row["relation"]), str(row["question_words"][1]), str(row["question_words"][2]), str(row["reply_words"][2]), str(row["reply_words"][4]))
    parsed = independent_discourse_parse(derivation, rendered)
    codes = [key for key, value in gate.items() if not value]
    ledger_ok = replay_character_ledger(row)
    if not ledger_ok:
        codes.append("character_ledger_replay_failed")
    if not parsed["ok"]:
        codes.append("independent_single_relation_reparse_failed")
    return {**row, "letters": len(tape), "normalized": tape, "character_ledger_replayed": ledger_ok, "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()}, "independent_parse": parsed, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(*, state_limit: int = MAX_STATES) -> dict[str, object]:
    left_root, right_root, left_rows, right_rows = compile_productive_grammars()
    pairs, intersection = intersect_product_tries(left_root, right_root, state_limit=state_limit)
    records = [audit_closure(row) for row in pairs]
    exact = [row for row in records if row["independent_exact_audit"]["exact"]]
    parsed = [row for row in exact if row["independent_parse"]["ok"]]
    admitted = [row for row in parsed if row["mechanically_admitted"]]
    return {
        "status": "question_reply_outer_role_product",
        "operator": "productive typed question/reply grammar; intersect forward 'are' question with reversed 'yes + reply' stream before pairing",
        "config": {
            "state_limit": state_limit,
            "productive_typed_role_alternatives": True,
            "short_functional_question_operator": "are",
            "semantically_normal_final_reply_object": True,
            "shared_question_reply_relation": True,
            "grammar_product_intersection": True,
            "complete_pair_selected_after_intersection": True,
            "source_reverse_terminal_compatibility_constraint": True,
            "expansion_hard_gate": True,
            "independent_single_relation_reparse": True,
            "finite_inventory_exhausted_not_global_search_complete": True,
            "human_readability_required_after_admission": True,
            "corpus_or_catalogue_source": False,
        },
        "grammar_inventory": {
            "relation": FRAME.relation,
            "subjects": FRAME.subjects,
            "states": [
                {"predicate": state.predicate, "replies": [option.__dict__ for option in state.replies]}
                for state in FRAME.states
            ],
        },
        "derivation_count": len(productive_derivations()),
        "left_grammar_rows": left_rows,
        "right_grammar_rows": right_rows,
        "intersection": intersection,
        "states_exhausted": intersection["states_exhausted"],
        "completeness_statement": "The finite authored question/reply inventory was exhausted under the stated limit; this bounded run makes no claim of global grammar-search completeness.",
        "records": records,
        "exact_candidates": exact,
        "parsed_exact_candidates": parsed,
        "mechanically_admitted": admitted,
        "first_jointly_reachable_failure": intersection["first_joint_failure"] or {"reason": "no jointly reachable grammar state"},
        "next_repair_operator": "Expand the semantic outer-role inventory with another ordinary question/reply state whose full typed relation can cross the persisted frontier; do not insert a token solely to match a character.",
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "material": "task-authored question/reply relation and lexical alternatives; no corpus, catalogue, or prebuilt palindrome"},
        "reader_facing_test": {"status": "not triggered unless an exact parsed survivor exists", "required_when_triggered": "randomized blinded intact prose and word-shuffled controls"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-limit", type=int, default=MAX_STATES)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "derivations": result["derivation_count"], "states": result["intersection"]["states_examined"], "states_exhausted": result["states_exhausted"], "records": len(result["records"]), "exact": len(result["exact_candidates"]), "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
