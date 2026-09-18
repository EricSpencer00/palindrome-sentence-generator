"""Productive action/result grammar-product search.

This is a successor to the fixed-clause lexical probe.  A shared event frame
owns both clauses: the first describes an action and the second its ordinary
result, joined by ``so``.  Each typed role has lexical alternatives.  The
action grammar is compiled into a forward character trie and the result
grammar (including the connector) into a reversed trie.  Only a pair of
complete derivations whose terminal streams agree is materialized; no pair of
source sentences is selected in advance.

The central admission gate is also applied at expansion time, so a terminal
that is a word-boundary mirror, repeated unit, embedded self-palindrome, or
catalogue scaffold is rejected before it can become a candidate.  Surviving
closures are independently parsed as one action/result discourse relation and
remain human-unreviewed until a blinded reader test.
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
class EventFrame:
    relation: str
    action_agents: tuple[str, ...]
    action_verbs: tuple[str, ...]
    action_objects: tuple[str, ...]
    # Subject/verb/complement triples preserve agreement and valency inside
    # the result clause.  The complement prefix is e.g. ``the`` for a
    # transitive predicate or ``in the`` for an intransitive one.
    result_predicates: tuple[tuple[str, str, tuple[str, ...]], ...]
    result_places: tuple[str, ...]


# These frames are semantic constructions, not cross-products of unrelated
# nouns.  Every alternative stays in the same action/result relation.
EVENT_FRAMES = (
    EventFrame(
        "baking_aroma",
        ("baker", "chef"),
        ("bakes", "warms"),
        ("bread", "cake"),
        (("aroma", "fills", ("the",)), ("scent", "fills", ("the",)), ("aroma", "spreads", ("through", "the")), ("scent", "spreads", ("through", "the"))),
        ("area", "arena", "plaza"),
    ),
    EventFrame(
        "planting_growth",
        ("farmer", "gardener"),
        ("plants", "sows"),
        ("seeds", "beans"),
        (("garden", "grows", ("in", "the")), ("garden", "thrives", ("in", "the")), ("plants", "grow", ("in", "the")), ("plants", "thrive", ("in", "the"))),
        ("yard", "plot"),
    ),
    EventFrame(
        "painting_display",
        ("artist", "painter"),
        ("paints", "draws"),
        ("mural", "panel"),
        (("picture", "brightens", ("the",)), ("portrait", "brightens", ("the",)), ("picture", "hangs", ("in", "the")), ("portrait", "hangs", ("in", "the"))),
        ("gallery", "hall"),
    ),
)


@dataclass(frozen=True)
class Derivation:
    relation: str
    action_words: tuple[str, ...]
    result_words: tuple[str, ...]
    connector: str = "so"

    @property
    def action_tape(self) -> str:
        return normalize_letters(" ".join(self.action_words))

    @property
    def suffix_tape(self) -> str:
        return normalize_letters(" ".join((self.connector, *self.result_words)))

    @property
    def reversed_suffix_tape(self) -> str:
        return self.suffix_tape[::-1]

    @property
    def rendered(self) -> str:
        return " ".join(self.action_words).capitalize() + ", " + self.connector + " " + " ".join(self.result_words) + "."


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
    rows: list[Derivation] = []
    for frame in EVENT_FRAMES:
        for agent in frame.action_agents:
            determiner = "an" if agent[0] in "aeiou" else "a"
            for verb in frame.action_verbs:
                for obj in frame.action_objects:
                        for subject, result_verb, complement_prefix in frame.result_predicates:
                            for place in frame.result_places:
                                rows.append(Derivation(
                                    frame.relation,
                                    (determiner, agent, verb, obj),
                                    ("the", subject, result_verb, *complement_prefix, place),
                                ))
    return tuple(rows)


def compile_productive_grammars() -> tuple[TrieNode, TrieNode, list[dict[str, object]], list[dict[str, object]]]:
    left_root, right_root = TrieNode(), TrieNode()
    derivations = productive_derivations()
    left_rows: list[dict[str, object]] = []
    right_rows: list[dict[str, object]] = []
    for derivation in derivations:
        add_to_trie(left_root, derivation.action_tape, derivation)
        add_to_trie(right_root, derivation.reversed_suffix_tape, derivation)
        left_rows.append({
            "relation": derivation.relation,
            "words": derivation.action_words,
            "rendered_clause": " ".join(derivation.action_words).capitalize() + ".",
            "tape": derivation.action_tape,
        })
        right_rows.append({
            "relation": derivation.relation,
            "words": derivation.result_words,
            "connector": derivation.connector,
            "rendered_clause": derivation.connector + " " + " ".join(derivation.result_words) + ".",
            "forward_suffix_tape": derivation.suffix_tape,
            "reversed_suffix_tape": derivation.reversed_suffix_tape,
        })
    return left_root, right_root, left_rows, right_rows


def independent_discourse_parse(derivation: Derivation, rendered: str) -> dict[str, object]:
    """Reparse the joined text as one typed action/result relation."""
    try:
        units = tuple(tokenize(rendered))
    except ValueError:
        return {"ok": False, "reason": "tokenization_failed"}
    if "so" not in units:
        return {"ok": False, "reason": "missing_result_connector"}
    pivot = units.index("so")
    action = units[:pivot]
    result = units[pivot + 1:]
    expected_action = ("a", "an")
    action_shape = len(action) == 4 and action[0] in expected_action
    result_shape = len(result) >= 5 and result[0] == "the" and result[-1] in {place for frame in EVENT_FRAMES for place in frame.result_places}
    event = next((frame for frame in EVENT_FRAMES if frame.relation == derivation.relation), None)
    result_prefix = tuple(result[3:-1])
    relation_ok = bool(event) and action[1] in event.action_agents and action[2] in event.action_verbs and action[3] in event.action_objects and any((result[1], result[2], result_prefix) == predicate for predicate in event.result_predicates) and result[-1] in event.result_places
    return {
        "ok": action_shape and result_shape and relation_ok,
        "relation": derivation.relation,
        "action_shape": action_shape,
        "result_shape": result_shape,
        "shared_event_relation": relation_ok,
        "action_words": action,
        "result_words": result,
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
    """Independently replay every character match recorded at a terminal."""
    left = str(row["action_tape"])
    right = str(row["reversed_suffix_tape"])
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
        nonlocal states, deepest, first_failure, truncated
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
                        "action_words": left_derivation.action_words,
                        "result_words": right_derivation.result_words,
                        "connector": right_derivation.connector,
                        "action_tape": left_derivation.action_tape,
                        "reversed_suffix_tape": right_derivation.reversed_suffix_tape,
                        "joined_rendered": right_derivation.__class__(left_derivation.relation, left_derivation.action_words, right_derivation.result_words, right_derivation.connector).rendered,
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
    derivation = Derivation(str(row["relation"]), tuple(row["action_words"]), tuple(row["result_words"]), str(row["connector"]))
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
        "status": "coupled_event_so_grammar_product",
        "operator": "productive shared-event action/result grammar; intersect forward action with reversed 'so + result' terminal streams before pairing",
        "config": {
            "state_limit": state_limit,
            "productive_typed_role_alternatives": True,
            "shared_event_relation_in_both_clauses": True,
            "connector": "so",
            "grammar_product_intersection": True,
            "complete_pair_selected_after_intersection": True,
            "source_reverse_terminal_compatibility_constraint": True,
            "expansion_hard_gate": True,
            "independent_single_relation_reparse": True,
            "no_repeated_content_required": True,
            "human_readability_required_after_admission": True,
            "corpus_or_catalogue_source": False,
            "finite_inventory_exhausted_not_global_search_complete": True,
        },
        "grammar_inventory": [frame.__dict__ for frame in EVENT_FRAMES],
        "derivation_count": len(productive_derivations()),
        "left_grammar_rows": left_rows,
        "right_grammar_rows": right_rows,
        "intersection": intersection,
        "states_exhausted": intersection["states_exhausted"],
        "completeness_statement": "The finite authored inventory was exhausted under the stated limit; this bounded run makes no claim of global grammar-search completeness.",
        "records": records,
        "exact_candidates": exact,
        "parsed_exact_candidates": parsed,
        "mechanically_admitted": admitted,
        "first_jointly_reachable_failure": intersection["first_joint_failure"] or {"reason": "no jointly reachable grammar state"},
        "next_repair_operator": "Use the persisted incompatible frontier to add a new ordinary shared-event frame with compatible terminal streams; preserve productive role alternatives and the single-relation reparse.",
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "material": "task-authored coupled event frames; no corpus, catalogue, or prebuilt palindrome"},
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
    print(json.dumps({"out": str(args.out), "derivations": result["derivation_count"], "states": result["intersection"]["states_examined"], "records": len(result["records"]), "exact": len(result["exact_candidates"]), "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
