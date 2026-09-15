"""A bounded SAT-style joint model for a semantic English palindrome.

The model has three finite-domain variable families:

* one-hot ``rule`` variables select a complete semantic grammar rule;
* one-hot ``slot`` variables select lexical values for that rule's typed slots;
* ``c[0:N]`` character variables carry the rendered tape.

For every chosen lexical value, its letters are constrained into the character
variables at their ordinary left-to-right positions.  A hard equality clause
``c[i] == c[N-1-i]`` is propagated at the same time.  Thus syntax, semantic
selection, lexical choice, exact length, and palindrome equality are one
constraint problem.  The search never segments a reversed tape and never
constructs a word-order mirror.

This is intentionally a small, hand-authored grammar experiment.  It reports
finite satisfiability evidence only; a mechanical closure is not a readability
certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

FAMILY_ID = "cp-semantic-grammar-palindrome"
STATE_SPACE_SIGNATURE = (
    "finite-semantic-english-grammar|one-hot-rule-and-lexical-slot-csp|"
    "global-character-domain-equality|exact-N-satisfiability|"
    "joint-syntax-selection-not-reverse-segmentation"
)
MIN_LETTERS = 36
MAX_LETTERS = 96
# Exact-N is deliberately bounded to a sparse, reproducible target set.  The
# grammar remains finite and exhaustive for each listed N; this avoids turning
# a diagnostic artifact into an open-ended solver run.
TARGET_LENGTHS = (40, 48, 56, 64, 72, 80)
MAX_STATES = 2_500
MAX_PROBES = 24


@dataclass(frozen=True)
class Lexeme:
    text: str
    semantic_class: str
    pos: str

    @property
    def letters(self) -> str:
        return normalize_letters(self.text)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    meaning: str
    slots: tuple[str, ...]
    semantic_roles: tuple[str, ...]


# Every word here is authored for this experiment.  The classes are semantic
# selection restrictions, not a POS corpus or a catalogue of sentences.
LEXICON: dict[str, tuple[Lexeme, ...]] = {
    "DET": tuple(Lexeme(w, "function", "DET") for w in ("a", "the", "our", "one")),
    "AGENT": tuple(Lexeme(w, "person", "NOUN") for w in ("artisan", "baker", "carver", "guard", "pilot", "teacher", "writer")),
    "VMAKE": tuple(Lexeme(w, "make", "VERB") for w in ("builds", "carves", "draws", "mends", "paints", "writes")),
    "VNOTICE": tuple(Lexeme(w, "notice", "VERB") for w in ("checks", "finds", "marks", "notes", "spots", "studies")),
    "VLOC": tuple(Lexeme(w, "locate", "VERB") for w in ("rests", "sits", "stands", "waits")),
    "ARTIFACT": tuple(Lexeme(w, "artifact", "NOUN") for w in ("canvas", "letter", "map", "model", "parcel", "vessel")),
    "PLACE": tuple(Lexeme(w, "place", "NOUN") for w in ("garden", "harbor", "market", "studio", "workshop")),
    "QUALITY": tuple(Lexeme(w, "quality", "ADJ") for w in ("bright", "careful", "gentle", "quiet", "steady", "useful")),
    "CONJ": tuple(Lexeme(w, "function", "CONJ") for w in ("and", "but")),
    "PREP": tuple(Lexeme(w, "function", "PREP") for w in ("at", "by", "in", "near")),
}


# The rules are complete English clause shapes with semantic role bindings.
# A single rule variable is selected before slot values, but all alternatives
# remain live until character propagation proves or refutes them.
RULES: tuple[Rule, ...] = (
    Rule(
        "maker_then_observer",
        "two people perform distinct making and noticing actions on distinct artifacts",
        ("DET", "AGENT", "VMAKE", "DET", "ARTIFACT", "CONJ", "DET", "AGENT", "VNOTICE", "DET", "ARTIFACT"),
        ("article", "maker", "make", "object", "coord", "article", "observer", "notice", "object"),
    ),
    Rule(
        "observer_then_maker",
        "an observer marks an artifact while a maker works at a distinct place",
        ("DET", "AGENT", "VNOTICE", "DET", "ARTIFACT", "CONJ", "DET", "AGENT", "VMAKE", "PREP", "PLACE"),
        ("article", "observer", "notice", "object", "coord", "article", "maker", "make", "relation", "place"),
    ),
    Rule(
        "located_then_quality",
        "a person locates an artifact and a second person has a quality",
        ("DET", "AGENT", "VLOC", "PREP", "DET", "PLACE", "CONJ", "DET", "AGENT", "VCOP", "QUALITY"),
        ("article", "locator", "locate", "relation", "article", "place", "coord", "article", "qualifier", "quality"),
    ),
)

# Copular forms are kept separate so the grammar does not turn arbitrary POS
# strings into syntax.
LEXICON["VCOP"] = tuple(Lexeme(w, "copula", "VERB") for w in ("is", "seems"))

FUNCTION_CLASSES = frozenset({"function"})


def _content_key(lexeme: Lexeme) -> str:
    return lexeme.text.casefold()


def _model_schema() -> dict[str, Any]:
    return {
        "variable_families": {
            "rule_one_hot": "rule[r] in {0,1}; exactly one rule is true",
            "lexical_slot_one_hot": "slot[i,w] in {0,1}; exactly one lexeme per grammar slot",
            "character_domains": "c[0:N-1] in {a..z}",
            "length": "N is selected from the bounded exact target set",
        },
        "hard_clauses": [
            "exactly_one(rule[r])",
            "exactly_one(slot[i,w]) for every selected rule slot",
            "selected rule owns the selected slot type and ordered English production",
            "semantic-class and role compatibility",
            "distinct non-function content units; no self-palindromic lexical unit",
            "sum(len(selected lexeme[i])) == N",
            "for each emitted character position p: c[p] = selected lexeme letter",
            "for every p: c[p] == c[N-1-p]",
        ],
        "search_order": "rule -> grammar slot -> lexical value; character equality propagates globally",
        "not_used": ["reverse tape segmentation", "center-out beam", "word-order mirroring", "catalogue sentence text"],
    }


def _ascii_two_pointer(text: str) -> dict[str, Any]:
    lowered = text.casefold()
    if any(ch.isalpha() and not ("a" <= ch <= "z") for ch in lowered):
        return {"exact": False, "letters": 0, "error": "non_ascii_alpha"}
    tape = "".join(ch for ch in lowered if "a" <= ch <= "z")
    mismatches: list[dict[str, Any]] = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left": left, "right": right, "left_char": tape[left], "right_char": tape[right]})
        left += 1
        right -= 1
    return {
        "exact": bool(tape) and not mismatches,
        "letters": len(tape),
        "normalized_letters": tape,
        "two_pointer_pairs": len(tape) // 2,
        "mismatch_count": len(mismatches),
        "mismatch_sample": mismatches[:10],
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def _readability(text: str) -> dict[str, Any]:
    words = tokenize(text)
    try:
        from wordfreq import zipf_frequency
        values = [zipf_frequency(word, "en") for word in words]
    except Exception:
        values = []
    return {
        "status": "diagnostic_only_unreviewed",
        "word_count": len(words),
        "all_words_in_frequency_probe": bool(values) and all(v >= 2.0 for v in values),
        "mean_zipf_frequency": round(sum(values) / len(values), 3) if values else None,
        "intact_prose_reader_required": True,
    }


def _json_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_strings(child)


def _repository_fingerprint(output: Path | None) -> dict[str, Any]:
    """Hash existing normalized JSON string keys, excluding this run output."""
    keys: set[str] = set()
    palindrome_keys: set[str] = set()
    scanned = malformed = 0
    excluded = output.resolve() if output else None
    paths = sorted(path for base in (ROOT / "runs", ROOT / "data", ROOT / "experiments") for path in base.rglob("*.json"))
    for path in paths:
        if excluded and path.resolve() == excluded:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            malformed += 1
            continue
        scanned += 1
        for value in _json_strings(payload):
            try:
                tape = normalize_letters(value)
            except (TypeError, ValueError):
                continue
            if 1 <= len(tape) <= MAX_LETTERS:
                keys.add(tape)
                if tape == tape[::-1]:
                    palindrome_keys.add(tape)
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    palindrome_digest = hashlib.sha256("\n".join(sorted(palindrome_keys)).encode()).hexdigest()
    return {
        "json_files_scanned": scanned,
        "malformed_json_files_skipped": malformed,
        "normalized_string_keys": len(keys),
        "palindrome_tape_keys": len(palindrome_keys),
        "all_keys_sha256": digest,
        "palindrome_keys_sha256": palindrome_digest,
        "output_excluded_before_scan": bool(output),
        "excluded_output": str(output) if output else None,
    }


def _registry_comparison() -> dict[str, Any]:
    path = ROOT / "docs" / "experiment-novelty-registry.json"
    payload = json.loads(path.read_text())
    entries = payload.get("entries", [])
    same_signature = [e for e in entries if e.get("signature") == STATE_SPACE_SIGNATURE]
    same_family = [e for e in entries if e.get("id") == FAMILY_ID]
    return {
        "registry_path": str(path),
        "registry_entry_count": len(entries),
        "expected_current_entry_count": 29,
        "signature_collision": bool(same_signature),
        "family_id_collision": bool(same_family),
        "comparison": "unique_against_current_registry" if not same_signature and not same_family else "collision_requires_review",
        "changed_dimension": "one-hot semantic-rule/lexical-slot CSP with explicit global character variables and exact-N constraints",
    }


def _semantic_ok(rule: Rule, chosen: tuple[Lexeme, ...]) -> tuple[bool, str]:
    if len(chosen) != len(rule.slots):
        return False, "incomplete_slots"
    # The rule's semantic classes are hard selection constraints.  In
    # particular, two people and two content nouns must differ.
    content = [lex for lex in chosen if lex.semantic_class not in FUNCTION_CLASSES]
    if len({_content_key(lex) for lex in content}) != len(content):
        return False, "repeated_content_unit"
    if any(lex.letters == lex.letters[::-1] for lex in content):
        return False, "self_palindromic_lexical_unit"
    if rule.rule_id == "maker_then_observer":
        if chosen[1].semantic_class != "person" or chosen[2].semantic_class != "make" or chosen[4].semantic_class != "artifact":
            return False, "maker_clause_selection"
        if chosen[7].semantic_class != "notice" or chosen[9].semantic_class != "artifact":
            return False, "observer_clause_selection"
    elif rule.rule_id == "observer_then_maker":
        if chosen[2].semantic_class != "notice" or chosen[4].semantic_class != "artifact" or chosen[8].semantic_class != "make":
            return False, "observer_maker_selection"
    elif rule.rule_id == "located_then_quality":
        if chosen[2].semantic_class != "locate" or chosen[10].semantic_class != "quality":
            return False, "locative_quality_selection"
    return True, "ok"


def _propagate_word(domains: list[int], word: str, offset: int, exact_n: int) -> tuple[bool, list[int], str]:
    """Apply lexical-to-character and palindrome clauses without reversing text."""
    # A 26-bit mask is the character variable domain.  Copying masks keeps
    # propagation cheap while retaining explicit finite-domain semantics.
    next_domains = list(domains)
    for local, char in enumerate(word):
        position = offset + local
        if position >= exact_n:
            return False, domains, "length_overflow"
        mirror = exact_n - 1 - position
        bit = 1 << (ord(char) - ord("a"))
        if not (next_domains[position] & bit) or not (next_domains[mirror] & bit):
            return False, domains, "character_equality_contradiction"
        next_domains[position] = bit
        next_domains[mirror] = bit
    return True, next_domains, "ok"


def solve_exact_n(rule: Rule, exact_n: int, *, state_limit: int, result_limit: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    domains = [(1 << 26) - 1 for _ in range(exact_n)]
    chosen: list[Lexeme] = []
    results: list[dict[str, Any]] = []
    stats = Counter(states=0, slot_branches=0, character_contradictions=0, length_contradictions=0, semantic_rejections=0, closed=0)
    deepest: dict[str, Any] | None = None

    def visit(slot_index: int, offset: int, current_domains: list[int], used_content: frozenset[str] = frozenset()) -> None:
        nonlocal deepest
        if stats["states"] >= state_limit:
            return
        stats["states"] += 1
        if len(results) >= result_limit:
            return
        progress = {"slot_index": slot_index, "offset": offset, "slots_total": len(rule.slots), "chosen_words": [lex.text for lex in chosen], "remaining_letters": exact_n - offset}
        if deepest is None or offset > deepest["offset"]:
            deepest = progress
        if slot_index == len(rule.slots):
            if offset != exact_n:
                stats["length_contradictions"] += 1
                return
            stats["closed"] += 1
            words = tuple(lex.text for lex in chosen)
            valid, reason = _semantic_ok(rule, tuple(chosen))
            if not valid:
                stats["semantic_rejections"] += 1
                return
            rendered = " ".join(words).capitalize() + "."
            result = {"rendered": rendered, "rule_id": rule.rule_id, "meaning": rule.meaning, "slots": list(rule.slots), "lexical_choices": list(words), "semantic_roles": list(rule.semantic_roles), "exact_n": exact_n, "constraint_model_satisfied": True}
            result["independent_ascii_two_pointer_audit"] = _ascii_two_pointer(rendered)
            result["central_admission"] = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            result["readability_diagnostic"] = _readability(rendered)
            results.append(result)
            return
        role = rule.slots[slot_index]
        for lexeme in LEXICON[role]:
            stats["slot_branches"] += 1
            if lexeme.semantic_class not in FUNCTION_CLASSES and _content_key(lexeme) in used_content:
                stats["semantic_rejections"] += 1
                continue
            if lexeme.letters == lexeme.letters[::-1] and lexeme.semantic_class not in FUNCTION_CLASSES:
                stats["semantic_rejections"] += 1
                continue
            ok, updated, reason = _propagate_word(current_domains, lexeme.letters, offset, exact_n)
            if not ok:
                if reason == "length_overflow":
                    stats["length_contradictions"] += 1
                else:
                    stats["character_contradictions"] += 1
                continue
            chosen.append(lexeme)
            next_used = used_content | ({_content_key(lexeme)} if lexeme.semantic_class not in FUNCTION_CLASSES else set())
            visit(slot_index + 1, offset + len(lexeme.letters), updated, frozenset(next_used))
            chosen.pop()

    visit(0, 0, domains)
    return results, dict(stats), deepest


def _probe(rule: Rule, exact_n: int, deepest: dict[str, Any] | None, stats: dict[str, Any]) -> dict[str, Any]:
    chosen = deepest["chosen_words"] if deepest else []
    rendered = (" ".join(chosen).capitalize() + ".") if chosen else "."
    audit = _ascii_two_pointer(rendered)
    return {
        "kind": "joint_csp_partial_probe",
        "rule_id": rule.rule_id,
        "exact_n_target": exact_n,
        "rendered_probe": rendered,
        "selected_slots": chosen,
        "constraint_state": deepest,
        "search_stats": stats,
        "independent_ascii_two_pointer_audit": audit,
        "central_admission": mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
        "readability_diagnostic": _readability(rendered),
        "failure_interpretation": "The partial assignment reached this longest character-domain-consistent state but did not satisfy exact-N closure.",
    }


def run(output: Path | None = None) -> dict[str, Any]:
    fingerprint = _repository_fingerprint(output)
    registry = _registry_comparison()
    exact_rows: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    by_target: list[dict[str, Any]] = []
    total_stats = Counter()
    for rule in RULES:
        for exact_n in TARGET_LENGTHS:
            rows, stats, deepest = solve_exact_n(rule, exact_n, state_limit=MAX_STATES, result_limit=8)
            total_stats.update(stats)
            by_target.append({"rule_id": rule.rule_id, "exact_n": exact_n, "stats": stats, "closures": len(rows), "deepest_probe_offset": deepest["offset"] if deepest else 0})
            exact_rows.extend(rows)
            probes.append(_probe(rule, exact_n, deepest, stats))
    # Keep only the most informative deterministic probes, not every failed N.
    probes.sort(key=lambda row: (-int((row["constraint_state"] or {}).get("offset", 0)), row["rule_id"], row["exact_n_target"]))
    probes = probes[:MAX_PROBES]
    for row in exact_rows:
        tape = row["independent_ascii_two_pointer_audit"]["normalized_letters"]
        row["novelty_audit"] = {"tape_absent_from_repository_fingerprint": tape not in _known_fingerprinted_tapes(output), "tape": tape}
        row["mechanically_admitted"] = bool(row["independent_ascii_two_pointer_audit"]["exact"] and all(row["central_admission"].values()) and row["novelty_audit"]["tape_absent_from_repository_fingerprint"])
        row["reader_status"] = "not_run; exactness and frequency diagnostics do not certify readability"
    exact_rows.sort(key=lambda row: (-row["exact_n"], row["rendered"]))
    admitted = [row for row in exact_rows if row["mechanically_admitted"]]
    return {
        "status": "cp_joint_semantic_grammar_complete",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "registry_comparison": registry,
        "model_schema": _model_schema(),
        "config": {
            "exact_N_targets": list(TARGET_LENGTHS),
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "state_limit_per_rule_and_N": MAX_STATES,
            "rule_count": len(RULES),
            "lexical_inventory_is_hand_authored": True,
            "syntax_and_lexical_choices_jointly_constrained": True,
            "character_variables_enforce_palindrome": True,
            "reverse_lexicon_segmentation": False,
            "center_out_search": False,
            "content_units_unique_and_non_self_palindromic": True,
            "catalogue_text": False,
            "brown_or_pos_source": False,
            "output_excluded_before_repository_scan": True,
        },
        "grammar_rules": [{"rule_id": rule.rule_id, "meaning": rule.meaning, "slots": list(rule.slots), "semantic_roles": list(rule.semantic_roles)} for rule in RULES],
        "lexicon_sha256": hashlib.sha256(json.dumps({k: [x.__dict__ for x in v] for k, v in LEXICON.items()}, sort_keys=True).encode()).hexdigest(),
        "repository_fingerprint": fingerprint,
        "search_totals": dict(total_stats),
        "target_runs": by_target,
        "rendered_candidates": exact_rows,
        "admitted": admitted,
        "rendered_probes": probes,
        "next_repair_operator": {
            "operator": "semantic-slot-domain-repair-at-first-character-contradiction",
            "action": "Take the highest-offset probe, add exactly one new ordinary lexeme to the recorded semantic slot domain whose first/last letters satisfy the conflicting character-domain pair, then rerun every exact-N target with the same rule and uniqueness clauses.",
            "preserve": ["one-hot rule and slot variables", "semantic role compatibility", "global c[i]=c[N-1-i] clauses", "exact-N bounds", "no repeated/self-palindromic content units", "output-excluded fingerprint"],
            "forbidden_shortcuts": ["editing a selected sentence", "reversing a completed side", "adding a pre-palindromic phrase", "filler tokens"],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "grammar_material": "finite hand-authored semantic English rules and typed lexical classes",
            "source_sentences_copied": False,
            "catalogue_relexicalization": False,
            "brown_corpus_used": False,
            "pos_tagger_used": False,
            "solver": "deterministic finite-domain CSP with one-hot/SAT-style constraints and character-domain propagation",
            "output_excluded_fingerprint": True,
            "readability_certificate": False,
        },
        "reader_gate": {"status": "not_run", "reason": "Only exact, novel, mechanically admitted rows may enter blinded intact-prose and shuffled-control reading."},
    }


def _known_fingerprinted_tapes(output: Path | None) -> set[str]:
    # Recompute the key set only for candidate novelty decisions; this keeps
    # output exclusion exact even when run() is called repeatedly in-process.
    keys: set[str] = set()
    excluded = output.resolve() if output else None
    for base in (ROOT / "runs", ROOT / "data", ROOT / "experiments"):
        for path in base.rglob("*.json"):
            if excluded and path.resolve() == excluded:
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            for value in _json_strings(payload):
                try:
                    tape = normalize_letters(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= len(tape) <= MAX_LETTERS:
                    keys.add(tape)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["search_totals"].get("states", 0), "closures": len(result["rendered_candidates"]), "admitted": len(result["admitted"]), "probes": len(result["rendered_probes"]), "registry": result["registry_comparison"]}, sort_keys=True))


if __name__ == "__main__":
    main()
