"""Outside-in integer-span constraint propagation for a narrative palindrome.

This experiment uses a small dependency grammar whose lexical boundaries are
integer variables.  A target ``N`` is fixed, then each recursion step assigns
the leftmost and rightmost *dependency slots* together.  Their independently
chosen lexemes occupy integer spans and constrain the character domains at
mirrored positions.  The semantic state records which dependency roles have
been filled; it is not a reverse-tape decoder and never segments a reversed
string.

The grammar is a fresh two-clause narrative: a witness performs an observation
because a custodian performs a preparation.  Some object realizations are
multiword NPs, so whitespace boundaries are variable consequences of lexeme
choice.  The run is finite and diagnostic.  Exactness is independently
two-pointer audited, then checked against the shared mechanical admission
gate; no row is a readability claim.
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

FAMILY_ID = "semantic-dependency-outside-in"
STATE_SPACE_SIGNATURE = (
    "integer-span-boundaries|outside-in-arc-consistency|"
    "semantic-dependency-state|variable-length-lexeme-NPs|"
    "fresh-cause-preparation-narrative|paired-character-domain-constraints"
)
MIN_LETTERS = 39
MAX_LETTERS = 120
TARGET_LENGTHS = (39, 47, 55, 63, 71, 79, 87)
MAX_STATES = 18_000
MAX_PROBES = 20


@dataclass(frozen=True)
class Lexeme:
    text: str
    semantic_class: str
    content_units: tuple[str, ...]

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)

    @property
    def length(self) -> int:
        return len(self.tape)


@dataclass(frozen=True)
class DependencySlot:
    slot_id: str
    role: str
    semantic_node: str
    domain: str
    required_state: str


@dataclass(frozen=True)
class NarrativeGrammar:
    grammar_id: str
    meaning: str
    slots: tuple[DependencySlot, ...]
    connective_choices: tuple[str, ...]

    def render(self, chosen: tuple[Lexeme, ...]) -> str:
        # Slots are [det, witness, predicate, det, object, connective,
        # det, custodian, predicate, det, object].  Object lexemes may
        # themselves contain an adjective+noun NP.
        words = [lex.text for lex in chosen]
        return (
            f"{words[0].capitalize()} {words[1]} {words[2]} {words[3]} {words[4]} "
            f"{words[5]} {words[6]} {words[7]} {words[8]} {words[9]} {words[10]}."
        )


GRAMMAR = NarrativeGrammar(
    grammar_id="observation-because-preparation",
    meaning="a witness records an artifact because a custodian prepares a different artifact",
    slots=(
        DependencySlot("s0", "det", "observation_clause", "DET", "open_observation"),
        DependencySlot("s1", "nsubj", "witness", "WITNESS", "witness_bound"),
        DependencySlot("s2", "root_predicate", "observe", "OBSERVE", "observation_bound"),
        DependencySlot("s3", "obj_det", "observed_artifact", "DET", "object_det_bound"),
        DependencySlot("s4", "obj", "observed_artifact", "ARTIFACT", "observation_complete"),
        DependencySlot("s5", "causal_marker", "cause", "CAUSE", "cause_open"),
        DependencySlot("s6", "subj_det", "preparation_clause", "DET", "subj_det_bound"),
        DependencySlot("s7", "nsubj", "custodian", "CUSTODIAN", "custodian_bound"),
        DependencySlot("s8", "advcl_predicate", "prepare", "PREPARE", "preparation_bound"),
        DependencySlot("s9", "obj_det", "prepared_artifact", "DET", "prepared_det_bound"),
        DependencySlot("s10", "obj", "prepared_artifact", "ARTIFACT", "narrative_complete"),
    ),
    connective_choices=("because", "while", "after"),
)


def _lexemes(words: Iterable[str], cls: str) -> tuple[Lexeme, ...]:
    return tuple(Lexeme(word, cls, (normalize_letters(word),)) for word in words)


# These are authored semantic domains, not corpus spans or a palindrome
# catalogue.  Multiword artifact NPs make both lexeme lengths and word
# boundaries variables in the integer model.
LEXICON: dict[str, tuple[Lexeme, ...]] = {
    "DET": _lexemes(("a", "the", "our"), "function"),
    "WITNESS": _lexemes(("scout", "scribe", "ranger", "observer", "cartographer", "witness"), "person"),
    "CUSTODIAN": _lexemes(("keeper", "warden", "curator", "steward", "guardian", "porter"), "person"),
    "OBSERVE": _lexemes(("marks", "records", "checks", "studies", "sketches", "notes"), "observation"),
    "PREPARE": _lexemes(("packs", "mends", "folds", "seals", "sorts", "labels"), "preparation"),
    "ARTIFACT": tuple(
        Lexeme(text, "artifact", tuple(normalize_letters(part) for part in text.split()))
        for text in (
            "map", "chart", "letter", "lantern", "compass", "parcel", "camera", "field map",
            "route chart", "small lantern", "sealed letter", "travel log", "paper map",
        )
    ),
    "CAUSE": _lexemes(GRAMMAR.connective_choices, "function"),
}


FUNCTION_CLASSES = frozenset({"function"})


def _model_schema() -> dict[str, Any]:
    return {
        "integer_variables": {
            "N": "one selected exact target length",
            "start[i], end[i]": "integer span boundaries for dependency slot i",
            "lexeme[i]": "finite-domain index; span length is constrained by selected lexeme",
        },
        "semantic_state": "dependency-state set tracks witness/object/custodian/cause obligations",
        "hard_constraints": [
            "end[i] - start[i] == len(lexeme[i])",
            "start[i+1] == end[i] + one rendered word separator",
            "distinct content units across the two clauses",
            "semantic role/domain compatibility and causal grammar order",
            "character domain at p equals character domain at N-1-p",
            "all dependency obligations fulfilled at closure",
        ],
        "propagation": "outside-in arc consistency over paired integer spans and character domains",
        "not_used": [
            "reverse-tape segmentation or decoding",
            "one-hot global rule CSP",
            "clause-lattice DP",
            "character FST or transducer",
            "Brown/POS corpus search",
            "event, dialogue, scene, morphology, or catalogue scaffold",
        ],
    }


def _independent_audit(text: str) -> dict[str, Any]:
    lowered = text.casefold()
    if any(ch.isalpha() and not ("a" <= ch <= "z") for ch in lowered):
        return {"exact": False, "letters": 0, "error": "non_ascii_alpha"}
    tape = "".join(ch for ch in lowered if "a" <= ch <= "z")
    mismatches: list[dict[str, Any]] = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left_index": left, "right_index": right, "left_char": tape[left], "right_char": tape[right]})
        left += 1
        right -= 1
    return {
        "exact": bool(tape) and not mismatches,
        "two_pointer_exact": bool(tape) and not mismatches,
        "letters": len(tape),
        "normalized_letters": tape,
        "two_pointer_pairs_checked": len(tape) // 2,
        "mismatch_count": len(mismatches),
        "mismatch_sample": mismatches[:12],
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _fingerprint(output: Path | None) -> tuple[set[str], dict[str, Any]]:
    keys: set[str] = set()
    files = malformed = 0
    excluded = output.resolve() if output else None
    for base in (ROOT / "runs", ROOT / "data", ROOT / "experiments"):
        for path in sorted(base.rglob("*.json")):
            if excluded and path.resolve() == excluded:
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                malformed += 1
                continue
            files += 1
            for value in _strings(payload):
                try:
                    tape = normalize_letters(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= len(tape) <= MAX_LETTERS:
                    keys.add(tape)
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    return keys, {
        "json_files_scanned": files,
        "malformed_json_files_skipped": malformed,
        "normalized_string_keys": len(keys),
        "all_keys_sha256": digest,
        "output_excluded_before_scan": bool(output),
        "excluded_output": str(output) if output else None,
    }


def _registry_comparison() -> dict[str, Any]:
    path = ROOT / "docs" / "experiment-novelty-registry.json"
    entries = json.loads(path.read_text()).get("entries", [])
    signatures = [entry.get("signature") for entry in entries]
    families = [entry.get("id") for entry in entries]
    return {
        "registry_path": str(path),
        "registry_entry_count_at_generation": len(entries),
        "expected_baseline_entry_count": 32,
        "family_id_collision": FAMILY_ID in families,
        "signature_collision": STATE_SPACE_SIGNATURE in signatures,
        "comparison": "unique_against_32_entry_registry" if len(entries) == 32 and FAMILY_ID not in families and STATE_SPACE_SIGNATURE not in signatures else "registry_changed_or_collision_requires_review",
    }


def _mask(char: str) -> int:
    return 1 << (ord(char) - ord("a"))


def _assign_span(domains: list[int], start: int, word: str, N: int) -> tuple[bool, list[int], str]:
    """Intersect one lexical span with mirrored character domains."""
    updated = list(domains)
    if start < 0 or start + len(word) > N:
        return False, domains, "integer_span_out_of_bounds"
    for offset, char in enumerate(word):
        pos = start + offset
        mirror = N - 1 - pos
        bit = _mask(char)
        if not (updated[pos] & bit) or not (updated[mirror] & bit):
            return False, domains, "paired_character_domain_empty"
        updated[pos] = bit
        updated[mirror] = bit
    return True, updated, "ok"


def _semantic_transition(state: frozenset[str], slot: DependencySlot, lexeme: Lexeme, used: frozenset[str]) -> tuple[bool, frozenset[str], str]:
    """Apply dependency selection and return the next semantic state."""
    if slot.role in {"nsubj", "obj"} and slot.domain in {"WITNESS", "CUSTODIAN", "ARTIFACT"}:
        for unit in lexeme.content_units:
            if unit in used:
                return False, state, "repeated_content_unit"
            if unit == unit[::-1] and len(unit) > 1:
                return False, state, "self_palindromic_content_unit"
    next_state = set(state)
    next_state.add(slot.required_state)
    # Outside-in assignment reaches a dependent object before its head may be
    # assigned from the other side.  Therefore dependency prerequisites are
    # propagated as obligations and checked at closure, rather than imposing a
    # misleading left-to-right order on this bidirectional state.
    return True, frozenset(next_state), "ok"


def _slot_candidates(slot: DependencySlot) -> tuple[Lexeme, ...]:
    return LEXICON[slot.domain]


def solve_target(N: int, *, state_limit: int, result_limit: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    slots = GRAMMAR.slots
    min_len = [min(x.length for x in _slot_candidates(slot)) for slot in slots]
    max_len = [max(x.length for x in _slot_candidates(slot)) for slot in slots]
    stats = Counter(states=0, paired_span_branches=0, span_domain_rejections=0, semantic_rejections=0, length_rejections=0, closures=0, budget_exhausted=0)
    selected: list[Lexeme | None] = [None] * len(slots)
    probes: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    all_chars = [(1 << 26) - 1] * N

    def visit(lo: int, hi: int, left_pos: int, right_pos: int, domains: list[int], state: frozenset[str], used: frozenset[str]) -> None:
        nonlocal probes
        if stats["states"] >= state_limit or len(results) >= result_limit:
            stats["budget_exhausted"] = 1
            return
        stats["states"] += 1
        remaining = tuple(range(lo, hi + 1))
        min_remaining = sum(min_len[i] for i in remaining)
        max_remaining = sum(max_len[i] for i in remaining)
        if left_pos + min_remaining > right_pos or left_pos + max_remaining < right_pos:
            stats["length_rejections"] += 1
            return
        progress = {
            "left_slot": lo,
            "right_slot": hi,
            "left_integer_boundary": left_pos,
            "right_integer_boundary": right_pos,
            "semantic_state": sorted(state),
            "chosen_left_to_right": [item.text if item else None for item in selected],
            "unassigned_slot_count": max(0, hi - lo + 1),
        }
        if probes is None or (left_pos + (N - right_pos)) > probes["outer_letters_assigned"]:
            probes = {**progress, "outer_letters_assigned": left_pos + (N - right_pos)}
        if lo > hi:
            if left_pos != right_pos or state != frozenset(slot.required_state for slot in slots):
                stats["length_rejections"] += 1
                return
            stats["closures"] += 1
            chosen = tuple(item for item in selected if item is not None)
            rendered = GRAMMAR.render(chosen)
            result = {
                "rendered": rendered,
                "grammar_id": GRAMMAR.grammar_id,
                "meaning": GRAMMAR.meaning,
                "lexical_choices": [item.text for item in chosen],
                "integer_spans": [{"slot_id": slot.slot_id, "start": sum(len(chosen[j].tape) for j in range(i)), "end": sum(len(chosen[j].tape) for j in range(i + 1)), "length": chosen[i].length} for i, slot in enumerate(slots)],
                "semantic_state": sorted(state),
                "constraint_model_satisfied": True,
                "independent_ascii_two_pointer_audit": _independent_audit(rendered),
                "central_admission": mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
                "readability_diagnostic": _readability(rendered),
            }
            results.append(result)
            return

        left_slot, right_slot = slots[lo], slots[hi]
        left_options = _slot_candidates(left_slot)
        right_options = _slot_candidates(right_slot) if hi != lo else (None,)
        for left_lexeme in left_options:
            left_start = left_pos
            ok, after_left, reason = _assign_span(domains, left_start, left_lexeme.tape, N)
            if not ok:
                stats["span_domain_rejections"] += 1
                continue
            sem_ok, after_state, sem_reason = _semantic_transition(state, left_slot, left_lexeme, used)
            if not sem_ok:
                stats["semantic_rejections"] += 1
                continue
            next_used = used | frozenset(left_lexeme.content_units)
            selected[lo] = left_lexeme
            if hi == lo:
                # A central slot may be crossed by the mirror relation.  This
                # is a span assignment, not a prebuilt centre palindrome.
                visit(lo + 1, hi - 1, left_pos + left_lexeme.length, right_pos, after_left, after_state, next_used)
            else:
                for right_lexeme in right_options:
                    assert right_lexeme is not None
                    right_start = right_pos - right_lexeme.length
                    if left_pos + left_lexeme.length > right_start:
                        stats["span_domain_rejections"] += 1
                        continue
                    ok, after_right, right_reason = _assign_span(after_left, right_start, right_lexeme.tape, N)
                    if not ok:
                        stats["paired_span_branches"] += 1
                        stats["span_domain_rejections"] += 1
                        continue
                    sem_ok, right_state, sem_reason = _semantic_transition(after_state, right_slot, right_lexeme, next_used)
                    if not sem_ok:
                        stats["semantic_rejections"] += 1
                        continue
                    selected[hi] = right_lexeme
                    stats["paired_span_branches"] += 1
                    visit(lo + 1, hi - 1, left_pos + left_lexeme.length, right_pos - right_lexeme.length, after_right, right_state, next_used | frozenset(right_lexeme.content_units))
                    selected[hi] = None
            selected[lo] = None

    visit(0, len(slots) - 1, 0, N, all_chars, frozenset(), frozenset())
    return results, dict(stats), probes


def _readability(text: str) -> dict[str, Any]:
    words = tokenize(text)
    try:
        from wordfreq import zipf_frequency
        freqs = [zipf_frequency(word, "en") for word in words]
    except Exception:
        freqs = []
    return {
        "status": "diagnostic_only_unreviewed",
        "word_count": len(words),
        "all_words_frequency_ge_2": bool(freqs) and all(freq >= 2.0 for freq in freqs),
        "mean_zipf_frequency": round(sum(freqs) / len(freqs), 3) if freqs else None,
        "intact_prose_reader_required": True,
    }


def run(output: Path | None = None) -> dict[str, Any]:
    existing, fingerprint = _fingerprint(output)
    target_runs: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    totals = Counter()
    for target in TARGET_LENGTHS:
        rows, stats, probe = solve_target(target, state_limit=MAX_STATES, result_limit=8)
        totals.update(stats)
        target_runs.append({"target_letters": target, "stats": stats, "closures": len(rows), "deepest_outer_letters_assigned": (probe or {}).get("outer_letters_assigned", 0)})
        for row in rows:
            tape = row["independent_ascii_two_pointer_audit"]["normalized_letters"]
            row["novelty_audit"] = {"tape_absent_from_repository_fingerprint": tape not in existing, "tape": tape}
            row["mechanically_admitted"] = bool(row["independent_ascii_two_pointer_audit"]["exact"] and tape not in existing and all(row["central_admission"].values()))
            row["reader_status"] = "not_run; exactness and lexical diagnostics do not certify readability"
            candidate_rows.append(row)
        if probe:
            chosen = [word for word in probe["chosen_left_to_right"] if word]
            rendered_probe = GRAMMAR.render(tuple(Lexeme(word, "probe", (normalize_letters(word),)) for word in chosen)) if len(chosen) == len(GRAMMAR.slots) else " ".join(chosen) + "."
            probe_rows.append({
                "kind": "outside_in_dependency_partial_probe",
                "target_letters": target,
                "rendered_probe": rendered_probe,
                "constraint_state": probe,
                "independent_ascii_two_pointer_audit": _independent_audit(rendered_probe),
                "central_admission": mechanical_admission_checks(rendered_probe, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
                "readability_diagnostic": _readability(rendered_probe),
                "failure_interpretation": "Longest recorded outside-in span assignment before exact integer-span closure; unresolved slots remain semantic obligations.",
                "reader_status": "not_run",
            })
    for row in candidate_rows:
        row["novelty_audit"]["tape_absent_from_repository_fingerprint"] = row["novelty_audit"]["tape"] not in existing
    candidate_rows.sort(key=lambda row: (-row["independent_ascii_two_pointer_audit"]["letters"], row["rendered"]))
    probe_rows.sort(key=lambda row: (-row["constraint_state"]["outer_letters_assigned"], row["target_letters"]))
    return {
        "status": "semantic_dependency_outside_in_complete",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "registry_comparison": _registry_comparison(),
        "model_schema": _model_schema(),
        "config": {
            "target_lengths": list(TARGET_LENGTHS),
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "state_limit_per_target": MAX_STATES,
            "outside_in_pair_assignment": True,
            "integer_word_boundaries": True,
            "variable_length_multiword_lexemes": True,
            "semantic_dependency_state": True,
            "reverse_tape_segmentation": False,
            "catalogue_text_or_scaffold": False,
            "brown_pos_event_dialogue_scene_morphology_transducer_routes": False,
            "output_excluded_before_repository_scan": True,
            "bounded_state_budget_is_not_global_exhaustion": True,
        },
        "grammar": {
            "grammar_id": GRAMMAR.grammar_id,
            "meaning": GRAMMAR.meaning,
            "surface_skeleton": "DET WITNESS OBSERVE DET ARTIFACT CAUSE DET CUSTODIAN PREPARE DET ARTIFACT",
            "dependency_roles": [slot.__dict__ for slot in GRAMMAR.slots],
        },
        "lexicon_sha256": hashlib.sha256(json.dumps({key: [item.__dict__ for item in value] for key, value in LEXICON.items()}, sort_keys=True).encode()).hexdigest(),
        "repository_fingerprint": fingerprint,
        "search_totals": dict(totals),
        "target_runs": target_runs,
        "rendered_candidates": candidate_rows,
        "admitted": [row for row in candidate_rows if row["mechanically_admitted"]],
        "rendered_probes": probe_rows[:MAX_PROBES],
        "next_repair_operator": {
            "operator": "semantic-arc-domain-repair-at-deepest-integer-seam",
            "action": "At the highest-offset probe, add one independently authored ordinary lexeme to the recorded dependency slot domain; choose its length and boundary so the first contradicted mirrored character pair is restored, then rerun every target with the same semantic-state obligations.",
            "preserve": ["integer start/end span variables", "outside-in paired-domain propagation", "dependency-state closure", "distinct non-self-palindromic content", "output-excluded novelty fingerprint"],
            "forbidden_shortcuts": ["reversing a completed clause", "copying a catalogue palindrome", "preassembling a symmetric centre", "filler or punctuation characters"],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "grammar_material": "fresh hand-authored cause/preparation narrative dependency grammar",
            "lexical_material": "fresh hand-authored semantic domains, including variable-length artifact NPs",
            "source_sentences_copied": False,
            "catalogue_relexicalization": False,
            "corpus_or_pos_tagger_used": False,
            "solver": "bounded integer-span CSP with outside-in arc consistency and semantic dependency states",
            "output_excluded_fingerprint": True,
            "readability_certificate": False,
        },
        "reader_gate": {"status": "not_run", "reason": "Only exact, novel, mechanically admitted rows may enter blinded intact-prose and shuffled-control reading."},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "targets": len(result["target_runs"]), "closures": len(result["rendered_candidates"]), "admitted": len(result["admitted"]), "states": result["search_totals"].get("states", 0)}, sort_keys=True))


if __name__ == "__main__":
    main()
