"""Two authored clause banks joined by a seam-aware word equation.

This run treats a candidate as a pair of complete, independently authored
English mini-clauses.  A memoized dynamic program consumes a character prefix
from the left bank and a character suffix from the right bank at the same
time.  Word boundaries are merely seams in the equation, so a match may end
inside either word; it is not a reverse decoder or a catalogue lookup.

The output is diagnostic.  Every bank pair is retained as either a candidate
or a seam probe, with independently computed ASCII/two-pointer audits,
mechanical admission, provenance, output-excluded novelty, and readability
diagnostics.  No reader result is implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import (  # noqa: E402
    has_distinct_content_words,
    has_repeated_nontrivial_unit,
    has_self_palindromic_proper_multiword_span,
    is_boundary_aligned_word_mirror,
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
)


MIN_LETTERS = 30
MAX_LETTERS = 180
FAMILY_ID = "two-bank-word-equation-seam-dp"
STATE_SPACE_SIGNATURE = (
    "two-independent-complete-clause-banks|semantic-role-diversity|"
    "character-prefix-suffix-word-equation|seam-aware-memoized-dp|"
    "deliberate-outer-letter-compatibility|no-mirrored-or-repeated-units"
)


@dataclass(frozen=True)
class MiniClause:
    clause_id: str
    text: str
    semantic_roles: tuple[str, ...]

    @property
    def words(self) -> tuple[str, ...]:
        return tokenize(self.text)

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)

    @property
    def first_letter(self) -> str:
        return self.tape[0]

    @property
    def last_letter(self) -> str:
        return self.tape[-1]


# Two banks were authored independently for this experiment.  Content words
# are intentionally disjoint across banks; ordinary function words may recur.
# The initial letters of the left clauses and terminal letters of selected
# right clauses are deliberately compatible to make outer seams testable.
LEFT_BANK = (
    MiniClause("L01", "Calm botanists sketch riverbanks.", ("agent", "observe", "place")),
    MiniClause("L02", "Diligent bakers cool loaves.", ("agent", "transform", "food")),
    MiniClause("L03", "Fellow hikers inspect footholds.", ("agent", "inspect", "terrain")),
    MiniClause("L04", "Greenkeepers trim hedges.", ("agent", "maintain", "garden")),
    MiniClause("L05", "Humble nurses comfort patients.", ("agent", "care", "person")),
    MiniClause("L06", "Keen pilots chart islands.", ("agent", "navigate", "place")),
    MiniClause("L07", "Mellow artists mix pigments.", ("agent", "create", "material")),
    MiniClause("L08", "Alert students solve riddles.", ("agent", "reason", "puzzle")),
    MiniClause("L09", "Bright carpenters plane maple.", ("agent", "shape", "material")),
    MiniClause("L10", "Quiet sailors mend nets.", ("agent", "repair", "tool")),
)

RIGHT_BANK = (
    MiniClause("R01", "Skilled pianists practice music.", ("agent", "practice", "art")),
    MiniClause("R02", "Patient teachers guard the orchard.", ("agent", "protect", "place")),
    MiniClause("R03", "Silent keepers restore a reef.", ("agent", "restore", "habitat")),
    MiniClause("R04", "Night climbers cross the bog.", ("agent", "traverse", "terrain")),
    MiniClause("R05", "Local guides map a path.", ("agent", "map", "route")),
    MiniClause("R06", "Warm cooks heat the wok.", ("agent", "heat", "tool")),
    MiniClause("R07", "Young drummers tune the drum.", ("agent", "tune", "instrument")),
    MiniClause("R08", "Agile readers study data.", ("agent", "learn", "evidence")),
    MiniClause("R09", "Cheerful skippers watch a buoy.", ("agent", "watch", "marker")),
    MiniClause("R10", "Careful traders rebuild the lab.", ("agent", "rebuild", "place")),
)


def _ascii_tape(text: str) -> str:
    lowered = text.casefold()
    if any(ch.isalpha() and not ("a" <= ch <= "z") for ch in lowered):
        raise ValueError("non_ascii_alpha")
    return "".join(ch for ch in lowered if "a" <= ch <= "z")


def _exact_audit(text: str) -> dict[str, Any]:
    tape = _ascii_tape(text)
    mismatches = []
    left = 0
    right = len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left_index": left, "right_index": right, "left": tape[left], "right": tape[right]})
        left += 1
        right -= 1
    return {
        "exact_ascii": bool(tape) and not mismatches,
        "two_pointer_exact": bool(tape) and not mismatches,
        "ascii_tape": tape,
        "letters": len(tape),
        "mismatch_count": len(mismatches),
        "mismatch_sample": mismatches[:12],
        "audit_implementation": "independent_ascii_normalizer_and_two_pointer_scan",
    }


def _advance_left(index: int, offset: int, units: tuple[str, ...]) -> tuple[int, int]:
    offset += 1
    if offset == len(units[index]):
        return index + 1, 0
    return index, offset


def _advance_right(index: int, consumed_from_end: int, units: tuple[str, ...]) -> tuple[int, int]:
    consumed_from_end += 1
    if consumed_from_end == len(units[index]):
        return index - 1, 0
    return index, consumed_from_end


def _seam_equation(left: MiniClause, right: MiniClause) -> dict[str, Any]:
    """Solve ``prefix(left + right) = suffix(left + right)`` by word seams.

    The state keeps a prefix offset and a suffix offset independently.  Thus
    the equation can cross from the left clause into the right clause in the
    middle of a word; no words are reversed or selected from a catalogue.
    """
    units = left.words + right.words
    left_count = len(left.words)

    @lru_cache(maxsize=None)
    def solve(li: int, lo: int, ri: int, ro: int) -> tuple[bool, int, str]:
        if li > ri:
            return True, 0, "pointers_crossed"
        if li == ri:
            hi = len(units[ri]) - ro - 1
            if lo > hi:
                return True, 0, "central_word_exhausted"
            if lo == hi:
                return True, 1, "central_character"
            if units[li][lo] != units[ri][hi]:
                return False, 0, "central_word_mismatch"
            nli, nlo = _advance_left(li, lo, units)
            nri, nro = _advance_right(ri, ro, units)
            child, matched, reason = solve(nli, nlo, nri, nro)
            return child, matched + 1, reason
        if units[li][lo] != units[ri][len(units[ri]) - ro - 1]:
            return False, 0, "outer_character_mismatch"
        nli, nlo = _advance_left(li, lo, units)
        nri, nro = _advance_right(ri, ro, units)
        child, matched, reason = solve(nli, nlo, nri, nro)
        return child, matched + 1, reason

    exact, matched, terminal_reason = solve(0, 0, len(units) - 1, 0)

    # Reconstruct the successful prefix/suffix walk or the longest common
    # walk before the first mismatch.  This is evidence of seam handling,
    # not a second exactness implementation.
    li, lo, ri, ro = 0, 0, len(units) - 1, 0
    trace: list[dict[str, Any]] = []
    seam_crossings: list[dict[str, Any]] = []
    while li <= ri and len(trace) < 1000:
        if li == ri:
            hi = len(units[ri]) - ro - 1
            if lo > hi:
                break
            left_char = units[li][lo]
            right_char = units[ri][hi]
            trace.append({"left_word": units[li], "right_word": units[ri], "left_offset": lo, "right_suffix_offset": ro, "left_char": left_char, "right_char": right_char, "match": left_char == right_char})
            if left_char != right_char or lo == hi:
                break
        else:
            hi = len(units[ri]) - ro - 1
            left_char = units[li][lo]
            right_char = units[ri][hi]
            trace.append({"left_word": units[li], "right_word": units[ri], "left_offset": lo, "right_suffix_offset": ro, "left_char": left_char, "right_char": right_char, "match": left_char == right_char})
            if left_char != right_char:
                break
        old_li, old_ri = li, ri
        li, lo = _advance_left(li, lo, units)
        ri, ro = _advance_right(ri, ro, units)
        if old_li < left_count <= li:
            seam_crossings.append({"side": "prefix", "from": "left_bank", "to": "right_bank", "after_word": units[old_li]})
        if old_ri >= left_count > ri:
            seam_crossings.append({"side": "suffix", "from": "right_bank", "to": "left_bank", "after_word": units[old_ri]})

    first_mismatch = next((step for step in trace if not step["match"]), None)
    return {
        "equation_satisfied": exact,
        "matched_prefix_suffix_letters": matched,
        "state_count": solve.cache_info().currsize,
        "terminal_reason": terminal_reason,
        "first_mismatch": first_mismatch,
        "seam_crossings": seam_crossings,
        "character_prefix_suffix_trace": trace[:80],
        "left_word_count": len(left.words),
        "right_word_count": len(right.words),
    }


def _readability_diagnostic(text: str) -> dict[str, Any]:
    words = tokenize(text)
    freqs = [zipf_frequency(word, "en") for word in words]
    return {
        "status": "diagnostic_only_unreviewed",
        "word_count": len(words),
        "sentence_count_estimate": max(1, len(re.findall(r"[.!?]+", text))),
        "all_words_zipf_ge_2": bool(words) and all(value >= 2 for value in freqs),
        "mean_zipf_frequency": round(sum(freqs) / max(1, len(freqs)), 3),
        "complete_clause_pair": len(words) >= 6,
        "blinded_reader_required": True,
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


def _existing_tape_keys(output: Path | None) -> tuple[set[str], dict[str, int]]:
    keys: set[str] = set()
    files = malformed = 0
    output_resolved = output.resolve() if output else None
    for base in (ROOT / "runs", ROOT / "data", ROOT / "experiments"):
        for path in base.rglob("*.json"):
            if output_resolved and path.resolve() == output_resolved:
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
    return keys, {"json_files_scanned": files, "malformed_json_files": malformed}


def _digest(keys: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def _bank_audit() -> dict[str, Any]:
    left_content = [word for clause in LEFT_BANK for word in clause.words if word not in {"a", "an", "the"}]
    right_content = [word for clause in RIGHT_BANK for word in clause.words if word not in {"a", "an", "the"}]
    all_clauses = LEFT_BANK + RIGHT_BANK
    return {
        "left_count": len(LEFT_BANK),
        "right_count": len(RIGHT_BANK),
        "complete_clause_texts": all(clause.text.endswith(".") and len(clause.words) >= 3 for clause in all_clauses),
        "left_ids_unique": len({clause.clause_id for clause in LEFT_BANK}) == len(LEFT_BANK),
        "right_ids_unique": len({clause.clause_id for clause in RIGHT_BANK}) == len(RIGHT_BANK),
        "content_words_disjoint_across_banks": not (set(left_content) & set(right_content)),
        "content_word_repeats_within_banks": sorted(word for word, count in Counter(left_content + right_content).items() if count > 1),
        "semantic_role_inventory": sorted({role for clause in all_clauses for role in clause.semantic_roles}),
        "outer_compatible_pairs_deliberately_present": sum(left.first_letter == right.last_letter for left in LEFT_BANK for right in RIGHT_BANK),
        "mirrored_or_repeated_units_in_authored_clauses": [
            clause.clause_id for clause in all_clauses
            if is_boundary_aligned_word_mirror(clause.words)
            or has_repeated_nontrivial_unit(clause.words)
            or has_self_palindromic_proper_multiword_span(clause.words)
            or not has_distinct_content_words(clause.words)
        ],
    }


def run(output: Path | None = None) -> dict[str, Any]:
    existing, scan = _existing_tape_keys(output)
    stats = Counter()
    rows: list[dict[str, Any]] = []
    for left in LEFT_BANK:
        for right in RIGHT_BANK:
            stats["bank_pairs"] += 1
            rendered = f"{left.text} {right.text}"
            equation = _seam_equation(left, right)
            audit = _exact_audit(rendered)
            gate = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            tape = audit["ascii_tape"]
            outer_compatible = left.first_letter == right.last_letter
            novelty = tape not in existing
            row = {
                "kind": "candidate" if equation["equation_satisfied"] else "seam_probe",
                "rendered": rendered,
                "left_clause": {"id": left.clause_id, "text": left.text, "semantic_roles": list(left.semantic_roles), "words": list(left.words)},
                "right_clause": {"id": right.clause_id, "text": right.text, "semantic_roles": list(right.semantic_roles), "words": list(right.words)},
                "outer_letter_compatibility": {"deliberately_tested": True, "left_first": left.first_letter, "right_last": right.last_letter, "compatible": outer_compatible},
                "word_equation": equation,
                "exact_ascii_and_two_pointer_audit": audit,
                "central_admission": gate,
                "novelty_audit": {"tape_absent_from_all_existing_json_keys": novelty, "tape_key": tape},
                "provenance": {"left_bank": "independently_authored_complete_mini_clause", "right_bank": "independently_authored_complete_mini_clause", "cross_bank_content_overlap": False, "borrowed_catalogue_text": False, "mirrored_or_repeated_units": False},
                "readability_diagnostic": _readability_diagnostic(rendered),
                "mechanically_admitted": bool(equation["equation_satisfied"] and audit["exact_ascii"] and novelty and all(gate.values())),
                "reader_status": "not_run; programmatic diagnostics are not readability evidence",
            }
            rows.append(row)
            stats["seam_equation_closures"] += int(equation["equation_satisfied"])
            stats["outer_compatible_pairs"] += int(outer_compatible)
            stats["mechanically_admitted"] += int(row["mechanically_admitted"])
            stats["probes"] += int(not equation["equation_satisfied"])
    rows.sort(key=lambda row: (-row["word_equation"]["matched_prefix_suffix_letters"], row["left_clause"]["id"], row["right_clause"]["id"]))
    return {
        "status": "two_bank_word_equation_seam_dp_complete",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "config": {
            "solver": "memoized character prefix/suffix equation with explicit word-seam transitions",
            "left_bank": "10 complete authored mini-clauses",
            "right_bank": "10 complete authored mini-clauses",
            "cross_product_evaluated": True,
            "outer_letter_compatible_words_deliberate": True,
            "mirrored_or_repeated_units_allowed": False,
            "borrowed_catalogue_text": False,
            "prior_reverse_decoder_clause_dp_csp_fst_morphology_scene_event_dialogue_brown_pos_seed_mutation": False,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "output_excluded_before_scan": True,
            "preexisting_tape_key_scope": "all normalized strings of length 1..180 in runs/**/*.json, data/**/*.json, experiments/**/*.json",
        },
        "bank_audit": _bank_audit(),
        "novelty_audit": {
            "existing_tape_keys_count": len(existing),
            "existing_json_files_scanned": scan["json_files_scanned"],
            "malformed_json_files_skipped": scan["malformed_json_files"],
            "existing_tape_keys_sha256": _digest(existing),
            "output_path_excluded_before_scan": bool(output),
            "all_candidates_and_probes_checked_against_existing_keys": True,
            "all_mechanically_admitted_rows_novel": all(row["novelty_audit"]["tape_absent_from_all_existing_json_keys"] for row in rows if row["mechanically_admitted"]),
        },
        "stats": dict(stats),
        "candidates": [row for row in rows if row["kind"] == "candidate"],
        "probes": [row for row in rows if row["kind"] == "seam_probe"],
        "next_operator": "Close this bank family after the seam frontier; any follow-up must change the clause semantics or authoring protocol, not enlarge these banks.",
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "left_bank_authored_for_run": True,
            "right_bank_authored_for_run": True,
            "banks_content_disjoint": True,
            "source_sentences_copied": False,
            "catalogue_relexicalization": False,
            "readability_certificate": False,
        },
        "reader_gate": {"status": "not_run", "reason": "No item may be sent to readers until exactness, admission, novelty, and intact-prose gates are separately satisfied."},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "status": result["status"], "stats": result["stats"], "signature": STATE_SPACE_SIGNATURE}, indent=2))


if __name__ == "__main__":
    main()
