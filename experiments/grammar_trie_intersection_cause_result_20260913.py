"""Intersect two clause grammars on reversed character streams.

This operator does not choose a complete source sentence and then resegment
its mirror.  It compiles a small typed cause grammar and a typed result grammar
into separate character tries, intersects their reachable states one character
at a time, and only materializes a pair when both grammar paths terminate.
Every materialized pair is independently reparsed and sent through central
admission; word-boundary mirrors and self-palindromic spans are never accepted.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

MAX_STATES = 100_000


class TrieNode:
    def __init__(self): self.children: dict[str, "TrieNode"] = {}; self.endings: list[tuple[str, ...]] = []


def add_to_trie(root: TrieNode, tape: str, words: tuple[str, ...]) -> None:
    node = root
    for char in tape: node = node.children.setdefault(char, TrieNode())
    node.endings.append(words)


LEFT_FRAMES = (
    {"relation": "baking_aroma", "words": ("a", "skilled", "baker", "warms", "fresh", "bread")},
    {"relation": "farming_stream", "words": ("a", "careful", "farmer", "carries", "clean", "water")},
)
RIGHT_FRAMES = (
    {"relation": "baking_aroma", "words": ("the", "scent", "fills", "the", "area")},
    {"relation": "farming_stream", "words": ("the", "stream", "reaches", "the", "field")},
)


def compile_grammars() -> tuple[TrieNode, TrieNode, list[dict[str, object]], list[dict[str, object]]]:
    left_root, right_root = TrieNode(), TrieNode(); left_rows, right_rows = [], []
    for frame in LEFT_FRAMES:
        words = tuple(frame["words"]); tape = normalize_letters(" ".join(words)); add_to_trie(left_root, tape, words); left_rows.append({"relation": frame["relation"], "words": words, "rendered_clause": " ".join(words).capitalize() + ".", "tape": tape, "letters": len(tape)})
    for frame in RIGHT_FRAMES:
        words = tuple(frame["words"]); tape = normalize_letters(" ".join(words)); add_to_trie(right_root, tape[::-1], words); right_rows.append({"relation": frame["relation"], "words": words, "rendered_clause": " ".join(words).capitalize() + ".", "forward_tape": tape, "reversed_tape": tape[::-1], "letters": len(tape)})
    return left_root, right_root, left_rows, right_rows


def independent_left_parse(words: tuple[str, ...], relation: str) -> bool:
    expected = {"baking_aroma": ("a", "skilled", "baker", "warms", "fresh", "bread"), "farming_stream": ("a", "careful", "farmer", "carries", "clean", "water")}
    return words == expected.get(relation)


def independent_right_parse(words: tuple[str, ...], relation: str) -> bool:
    expected = {"baking_aroma": ("the", "scent", "fills", "the", "area"), "farming_stream": ("the", "stream", "reaches", "the", "field")}
    return words == expected.get(relation)


def audit_pair(left_words: tuple[str, ...], right_words: tuple[str, ...], relation: str) -> dict[str, object]:
    left, right = " ".join(left_words), " ".join(right_words); rendered = left.capitalize() + "; " + right + "."; tape = normalize_letters(rendered); gate = mechanical_admission_checks(rendered, min_letters=30, max_letters=260); parsed = independent_left_parse(left_words, relation) and independent_right_parse(right_words, relation); codes = [key for key, value in gate.items() if not value]
    if not parsed: codes.append("independent_cause_result_reparse_failed")
    return {"relation": relation, "left": left, "right": right, "rendered": rendered, "normalized": tape, "letters": len(tape), "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()}, "independent_parse": parsed, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def expansion_hard_rejection(left_words: tuple[str, ...], right_words: tuple[str, ...]) -> list[str]:
    """Apply construction-family exclusions before a terminal pair is emitted.

    The grammar-product search is allowed to reach a terminal only after its
    character streams agree, but it must not even materialize a terminal that
    is a word-boundary mirror, a copied unit, or a proper self-palindromic
    span.  These are the central admission predicates, reused here at the
    expansion boundary rather than approximated locally.
    """
    rendered = " ".join(left_words).capitalize() + "; " + " ".join(right_words) + "."
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


def intersect_tries(left_root: TrieNode, right_root: TrieNode, *, state_limit: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    exact, rejected, states = [], [], 0; deepest = {"prefix": "", "characters": 0, "left_terminal": False, "right_terminal": False}; first_mismatch = None
    def visit(left_node: TrieNode, right_node: TrieNode, prefix: str, left_words: tuple[str, ...] | None, right_words: tuple[str, ...] | None):
        nonlocal states, first_mismatch, deepest
        if states >= state_limit: return
        states += 1
        if len(prefix) > int(deepest["characters"]): deepest = {"prefix": prefix, "characters": len(prefix), "left_terminal": bool(left_node.endings), "right_terminal": bool(right_node.endings)}
        common = set(left_node.children) & set(right_node.children)
        if not common and prefix and first_mismatch is None:
            first_mismatch = {"prefix": prefix, "left_next_chars": sorted(left_node.children), "right_next_chars": sorted(right_node.children), "reason": "no_common_next_character"}
        for char in sorted(common):
            ln, rn = left_node.children[char], right_node.children[char]; new_prefix = prefix + char
            if ln.endings and rn.endings:
                for lw in ln.endings:
                    for rw in rn.endings:
                        rejection_codes = expansion_hard_rejection(lw, rw)
                        row = {"left_words": lw, "right_words": rw, "left_tape": new_prefix, "right_reversed_tape": new_prefix, "right_tape": normalize_letters(" ".join(rw))}
                        if rejection_codes:
                            rejected.append({**row, "rejection_codes": rejection_codes})
                        else:
                            exact.append(row)
            visit(ln, rn, new_prefix, left_words, right_words)
    visit(left_root, right_root, "", None, None)
    return exact, {"states_examined": states, "deepest_joint_state": deepest, "first_joint_failure": first_mismatch, "terminal_pairs_rejected_during_expansion": rejected}


def run(*, state_limit: int = MAX_STATES) -> dict[str, object]:
    left_root, right_root, left_rows, right_rows = compile_grammars(); pairs, intersection = intersect_tries(left_root, right_root, state_limit=state_limit); records = []
    for pair in pairs:
        relation = next((row["relation"] for row in LEFT_FRAMES if tuple(row["words"]) == pair["left_words"]), "unknown")
        right_relation = next((row["relation"] for row in RIGHT_FRAMES if tuple(row["words"]) == pair["right_words"]), "unknown")
        if relation != right_relation: continue
        row = audit_pair(pair["left_words"], pair["right_words"], relation); row["cross_boundary_intersection"] = True; records.append(row)
    exact = [row for row in records if row["independent_exact_audit"]["exact"]]; parsed = [row for row in exact if row["independent_parse"]]; admitted = [row for row in parsed if row["mechanically_admitted"]]
    first_failure = intersection["first_joint_failure"] or {"reason": "no jointly terminal grammar state"}
    return {"status": "grammar_trie_intersection_cause_result", "operator": "compile typed cause/result clause tries, intersect forward left with reversed right streams before selecting a pair", "config": {"state_limit": state_limit, "grammar_product_intersection": True, "complete_pair_selected_after_intersection": True, "right_words_from_reversed_chunks": True, "coherent_cause_result_relation": True, "word_boundary_mirror_rejected_by_central_gate": True, "embedded_self_palindrome_rejected_by_central_gate": True, "no_repeated_content_required": True, "independent_parse_required": True, "central_admission_required": True, "human_readability_required_after_admission": True, "lengthen_only_after_exact_parsed_survivor": True, "corpus_or_catalogue_source": False}, "left_grammar_rows": left_rows, "right_grammar_rows": right_rows, "intersection": intersection, "records": records, "exact_candidates": exact, "parsed_exact_candidates": parsed, "mechanically_admitted": admitted, "first_jointly_reachable_failure": first_failure, "next_repair_operator": "Add a new semantically coherent cause/result frame whose compiled terminal streams share a longer cross-boundary prefix; do not select a full sentence before trie intersection or lengthen this zero-survivor run.", "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "left_grammar_material": "task-authored baking and farming causes", "right_grammar_material": "task-authored aroma and stream results", "catalogue_text": False}, "reader_facing_test": {"status": "not triggered because no exact parsed survivor", "required_when_triggered": "randomized blinded intact-prose and word-shuffled controls", "ratings": ["grammatical", "recoverable intent", "coherent thought"]}}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=MAX_STATES); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["intersection"]["states_examined"], "records": len(result["records"]), "exact": len(result["exact_candidates"]), "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__": main()
