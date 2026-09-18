"""Entailment-linked endpoint-flexible confirmation grammar.

This successor keeps endpoint selection separate from prose selection while
making the question/reply relation truth-conditionally explicit.  Each typed
plan supplies a proposition and a direct-evidence rule.  The endpoint
automaton selects an opening operator, subject boundary, and final object
jointly; only then are the plan's lexical alternatives compiled into forward
and reversed tries.  A complete pair must independently replay its evidence
rule from the parsed surface before central admission.
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
class ResponseOption:
    verb: str
    object: str


@dataclass(frozen=True)
class ConfirmationPlan:
    relation: str
    operator: str
    subjects: tuple[str, ...]
    question_frames: tuple[tuple[str, ...], ...]
    responses: tuple[ResponseOption, ...]
    proposition: str
    evidence_rule: str


# Every response is a direct evidence witness for the question, rather than a
# merely topical continuation.  The rules are replayed independently below.
PLANS = (
    ConfirmationPlan(
        "information_access",
        "can",
        ("analysts", "advisers"),
        (("access", "information"), ("reach", "sources")),
        (ResponseOption("consult", "almanac"),),
        "the agent can access an information source",
        "consulting_a_source_entails_information_access",
    ),
    ConfirmationPlan(
        "agent_presence",
        "are",
        ("mechanics", "musicians"),
        (("present",), ("nearby",)),
        (ResponseOption("use", "camera"), ResponseOption("operate", "camera")),
        "the agent is present",
        "an_agent_action_entails_presence",
    ),
    ConfirmationPlan(
        "instruction_following",
        "do",
        ("helpers", "historians"),
        (("follow", "instructions"), ("apply", "guidance")),
        (ResponseOption("apply", "method"), ResponseOption("use", "method")),
        "the agent follows an instruction",
        "applying_a_method_entails_instruction_following",
    ),
    ConfirmationPlan(
        "future_activity",
        "will",
        ("workers", "teams"),
        (("be", "active"), ("start", "work")),
        (ResponseOption("wash", "window"), ResponseOption("clean", "window")),
        "the agent will be active",
        "an_observed_task_entails_activity",
    ),
)


@dataclass(frozen=True)
class Derivation:
    relation: str
    operator: str
    subject: str
    question_frame: tuple[str, ...]
    response: ResponseOption
    proposition: str
    evidence_rule: str

    @property
    def question_words(self) -> tuple[str, ...]:
        return (self.operator, self.subject, *self.question_frame)

    @property
    def reply_words(self) -> tuple[str, ...]:
        return ("yes", "they", self.response.verb, "the", self.response.object)

    @property
    def question_tape(self) -> str:
        return normalize_letters(" ".join(self.question_words))

    @property
    def reply_tape(self) -> str:
        return normalize_letters(" ".join(self.reply_words))

    @property
    def reversed_reply_tape(self) -> str:
        return self.reply_tape[::-1]

    @property
    def rendered(self) -> str:
        return " ".join(self.question_words).capitalize() + "? Yes, " + " ".join(self.reply_words[1:]) + "."


@dataclass(frozen=True)
class EndpointPair:
    relation: str
    operator: str
    subject: str
    final_object: str
    shared_prefix: str


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.endings: list[Derivation] = []


def add_to_trie(root: TrieNode, tape: str, derivation: Derivation) -> None:
    node = root
    for char in tape:
        node = node.children.setdefault(char, TrieNode())
    node.endings.append(derivation)


def longest_common_prefix(left: str, right: str) -> str:
    index = 0
    while index < min(len(left), len(right)) and left[index] == right[index]:
        index += 1
    return left[:index]


def endpoint_automaton_inventory() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for plan in PLANS:
        for subject in plan.subjects:
            for response in plan.responses:
                left = normalize_letters(plan.operator + subject)
                right = normalize_letters(response.object)[::-1]
                shared = longest_common_prefix(left, right)
                rows.append({"relation": plan.relation, "operator": plan.operator, "subject": subject, "final_object": response.object, "shared_prefix": shared, "crosses_operator_subject_boundary": len(shared) >= len(normalize_letters(plan.operator)) + 1})
    return tuple(rows)


def endpoint_compatibility_automaton() -> tuple[EndpointPair, ...]:
    pairs: list[EndpointPair] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in endpoint_automaton_inventory():
        if not row["crosses_operator_subject_boundary"]:
            continue
        key = (str(row["relation"]), str(row["operator"]), str(row["subject"]), str(row["final_object"]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(EndpointPair(*key, str(row["shared_prefix"])))
    return tuple(pairs)


def productive_derivations() -> tuple[Derivation, ...]:
    compatible = {(pair.relation, pair.operator, pair.subject, pair.final_object) for pair in endpoint_compatibility_automaton()}
    rows: list[Derivation] = []
    for plan in PLANS:
        for subject in plan.subjects:
            for question_frame in plan.question_frames:
                for response in plan.responses:
                    if (plan.relation, plan.operator, subject, response.object) in compatible:
                        rows.append(Derivation(plan.relation, plan.operator, subject, question_frame, response, plan.proposition, plan.evidence_rule))
    return tuple(rows)


def compile_productive_grammars() -> tuple[TrieNode, TrieNode, list[dict[str, object]], list[dict[str, object]]]:
    left_root, right_root = TrieNode(), TrieNode()
    left_rows: list[dict[str, object]] = []
    right_rows: list[dict[str, object]] = []
    for derivation in productive_derivations():
        add_to_trie(left_root, derivation.question_tape, derivation)
        add_to_trie(right_root, derivation.reversed_reply_tape, derivation)
        left_rows.append({"relation": derivation.relation, "proposition": derivation.proposition, "evidence_rule": derivation.evidence_rule, "words": derivation.question_words, "rendered_clause": " ".join(derivation.question_words).capitalize() + "?", "tape": derivation.question_tape})
        right_rows.append({"relation": derivation.relation, "words": derivation.reply_words, "rendered_clause": "Yes, " + " ".join(derivation.reply_words[1:]) + ".", "forward_suffix_tape": derivation.reply_tape, "reversed_suffix_tape": derivation.reversed_reply_tape})
    return left_root, right_root, left_rows, right_rows


def independent_discourse_parse(derivation: Derivation, rendered: str) -> dict[str, object]:
    try:
        units = tuple(tokenize(rendered))
    except ValueError:
        return {"ok": False, "reason": "tokenization_failed"}
    if "yes" not in units:
        return {"ok": False, "reason": "missing_reply_operator"}
    pivot = units.index("yes")
    question, reply = units[:pivot], units[pivot:]
    plan = next((item for item in PLANS if item.relation == derivation.relation), None)
    question_shape = bool(plan) and question[0] == plan.operator and question[1] in plan.subjects
    reply_shape = len(reply) == 5 and reply[:2] == ("yes", "they") and reply[3] == "the"
    question_frame = tuple(question[2:])
    relation_ok = bool(plan) and question_shape and question_frame in plan.question_frames and reply_shape and ResponseOption(reply[2], reply[4]) in plan.responses
    return {"ok": relation_ok, "relation": derivation.relation, "question_shape": question_shape, "reply_shape": reply_shape, "shared_proposition": derivation.proposition, "question_frame": question_frame, "reply_words": reply}


def independent_entailment_verify(parsed: dict[str, object]) -> dict[str, object]:
    """Replay the evidence rule from surface fields, not a plan identifier."""
    if not parsed.get("ok"):
        return {"ok": False, "rule_replayed": False, "reason": "parse_failed"}
    frame = tuple(parsed["question_frame"])
    reply = tuple(parsed["reply_words"])
    rule = str(parsed["evidence_rule"]) if "evidence_rule" in parsed else ""
    if rule == "consulting_a_source_entails_information_access":
        ok = frame in (("access", "information"), ("reach", "sources")) and reply[2:5] == ("consult", "the", "almanac")
    elif rule == "an_agent_action_entails_presence":
        ok = frame in (("present",), ("nearby",)) and reply[2:5] in (("use", "the", "camera"), ("operate", "the", "camera"))
    elif rule == "applying_a_method_entails_instruction_following":
        ok = frame in (("follow", "instructions"), ("apply", "guidance")) and reply[2:5] in (("apply", "the", "method"), ("use", "the", "method"))
    elif rule == "an_observed_task_entails_activity":
        ok = frame in (("be", "active"), ("start", "work")) and reply[2:5] in (("wash", "the", "window"), ("clean", "the", "window"))
    else:
        ok = False
    return {"ok": ok, "rule_replayed": True, "rule": rule, "surface_witness": {"question_frame": frame, "reply_action_object": reply[2:5]}}


def expansion_hard_rejection(rendered: str) -> list[str]:
    gate = mechanical_admission_checks(rendered, min_letters=30, max_letters=260)
    hard_names = ("not_word_order_symmetry", "no_repeated_nontrivial_unit", "no_self_palindromic_proper_multiword_span", "not_forbidden_catalogue_control", "not_catalogue_family_derivative", "not_forbidden_catalogue_endpoint_scaffold")
    return [name for name in hard_names if not gate.get(name, False)]


def replay_character_ledger(row: dict[str, object]) -> bool:
    left, right, ledger = str(row["question_tape"]), str(row["reversed_reply_tape"]), row.get("character_ledger", [])
    if len(left) != len(right) or len(ledger) != len(left):
        return False
    return all(entry.get("position") == index and entry.get("left_character") == left[index - 1] and entry.get("right_reversed_character") == right[index - 1] and entry.get("matched") is True and left[index - 1] == right[index - 1] for index, entry in enumerate(ledger, start=1))


def intersect_product_tries(left_root: TrieNode, right_root: TrieNode, *, state_limit: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    terminal_pairs: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    states, truncated = 0, False
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
                    row = {"relation": left_derivation.relation, "proposition": left_derivation.proposition, "evidence_rule": left_derivation.evidence_rule, "question_words": left_derivation.question_words, "reply_words": right_derivation.reply_words, "question_tape": left_derivation.question_tape, "reversed_reply_tape": right_derivation.reversed_reply_tape, "joined_rendered": Derivation(left_derivation.relation, left_derivation.operator, left_derivation.subject, left_derivation.question_frame, right_derivation.response, left_derivation.proposition, left_derivation.evidence_rule).rendered, "character_ledger": [{"position": index, "left_character": char, "right_reversed_character": char, "matched": True} for index, char in enumerate(prefix, start=1)]}
                    codes = expansion_hard_rejection(row["joined_rendered"])
                    if codes:
                        rejected.append({**row, "rejection_codes": codes})
                    else:
                        terminal_pairs.append(row)
        for char in common:
            visit(left.children[char], right.children[char], prefix + char)

    visit(left_root, right_root, "")
    return terminal_pairs, {"states_examined": states, "states_exhausted": not truncated, "search_truncated": truncated, "deepest_joint_state": deepest, "first_joint_failure": first_failure, "terminal_pairs_rejected_during_expansion": rejected}


def audit_closure(row: dict[str, object]) -> dict[str, object]:
    rendered, tape = str(row["joined_rendered"]), normalize_letters(str(row["joined_rendered"]))
    gate = mechanical_admission_checks(rendered, min_letters=30, max_letters=260)
    question = tuple(row["question_words"])
    plan = next(plan for plan in PLANS if plan.relation == row["relation"])
    response = ResponseOption(str(row["reply_words"][2]), str(row["reply_words"][4]))
    predicate = tuple(question[2:])
    derivation = Derivation(plan.relation, plan.operator, str(question[1]), predicate, response, plan.proposition, plan.evidence_rule)
    parsed = independent_discourse_parse(derivation, rendered)
    parsed["evidence_rule"] = plan.evidence_rule
    entailment = independent_entailment_verify(parsed)
    codes = [key for key, value in gate.items() if not value]
    ledger_ok = replay_character_ledger(row)
    if not ledger_ok:
        codes.append("character_ledger_replay_failed")
    if not entailment["ok"]:
        codes.append("independent_entailment_replay_failed")
    return {**row, "letters": len(tape), "normalized": tape, "character_ledger_replayed": ledger_ok, "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()}, "independent_parse": parsed, "independent_entailment": entailment, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(*, state_limit: int = MAX_STATES) -> dict[str, object]:
    endpoints = endpoint_compatibility_automaton()
    left_root, right_root, left_rows, right_rows = compile_productive_grammars()
    pairs, intersection = intersect_product_tries(left_root, right_root, state_limit=state_limit)
    records = [audit_closure(row) for row in pairs]
    exact = [row for row in records if row["independent_exact_audit"]["exact"]]
    parsed = [row for row in exact if row["independent_parse"]["ok"] and row["independent_entailment"]["ok"]]
    admitted = [row for row in parsed if row["mechanically_admitted"]]
    return {"status": "entailment_linked_endpoint_confirmation", "operator": "semantic proposition/evidence precheck, endpoint-compatible operator/object automaton, then productive question/reply trie intersection", "config": {"state_limit": state_limit, "semantic_relation_selected_before_expansion": True, "machine_checkable_entailment_witness": True, "independent_entailment_replay": True, "endpoint_automaton_before_full_derivation": True, "operator_final_object_selected_jointly": True, "match_crosses_operator_subject_boundary": True, "productive_typed_role_alternatives": True, "grammar_product_intersection": True, "complete_pair_selected_after_intersection": True, "source_reverse_terminal_compatibility_constraint": True, "expansion_hard_gate": True, "independent_single_relation_reparse": True, "finite_inventory_exhausted_not_global_search_complete": True, "human_readability_required_after_admission": True, "corpus_or_catalogue_source": False}, "endpoint_automaton_inventory": list(endpoint_automaton_inventory()), "endpoint_compatibility_pairs": [pair.__dict__ for pair in endpoints], "grammar_inventory": [{"relation": plan.relation, "operator": plan.operator, "subjects": plan.subjects, "question_frames": plan.question_frames, "responses": [option.__dict__ for option in plan.responses], "proposition": plan.proposition, "evidence_rule": plan.evidence_rule} for plan in PLANS], "derivation_count": len(productive_derivations()), "left_grammar_rows": left_rows, "right_grammar_rows": right_rows, "intersection": intersection, "states_exhausted": intersection["states_exhausted"], "completeness_statement": "The finite entailment-linked endpoint inventory was exhausted under the stated limit; this bounded run makes no claim of global grammar-search completeness.", "records": records, "exact_candidates": exact, "parsed_exact_candidates": parsed, "mechanically_admitted": admitted, "first_jointly_reachable_failure": intersection["first_joint_failure"] or {"reason": "no jointly reachable grammar state"}, "next_repair_operator": "Add a new independently witnessed semantic relation with a natural endpoint pair that crosses the persisted frontier; do not patch a selected token or phrase.", "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "material": "task-authored entailment-linked question/reply plans; no corpus, catalogue, or prebuilt palindrome"}, "reader_facing_test": {"status": "not triggered unless an exact parsed survivor exists", "required_when_triggered": "randomized blinded intact prose and word-shuffled controls"}}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=MAX_STATES); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "endpoints": len(result["endpoint_compatibility_pairs"]), "derivations": result["derivation_count"], "states": result["intersection"]["states_examined"], "states_exhausted": result["states_exhausted"], "records": len(result["records"]), "exact": len(result["exact_candidates"]), "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__": main()
