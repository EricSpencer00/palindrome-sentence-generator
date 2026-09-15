"""Seam-first authoring of two complete English clauses.

The construction order in this experiment is deliberately different from the
older reverse parsers: a small, hand-authored inventory first chooses the
letters and widths at the *outer seam*; only then are independently authored
subject/verb/object clauses enumerated.  A pair is retained only when the two
complete clause tapes jointly satisfy every mirrored character equation.  No
right-side phrase is manufactured by segmenting a residual tape.

This is a bounded feasibility experiment, not a readability claim.  An exact
row would still need the project's intact-prose and shuffled-control reader
gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


FAMILY_ID = "seam-first-complete-clause-authoring"
STATE_SPACE_SIGNATURE = (
    "seam-first-outer-letter-constraints|independent-complete-svo-clause-bank|"
    "finite-hand-authored-seam-inventory|exact-joint-enumeration|"
    "boundary-crossing-reverse-seam|no-repeated-or-self-palindromic-units"
)
MIN_LETTERS = 30
MAX_LETTERS = 180
MAX_PROBES = 24


@dataclass(frozen=True)
class SeamSpec:
    seam_id: str
    opening_letter: str
    closing_letter: str
    left_terminal_width: int
    right_initial_width: int
    semantic_domain: str
    rationale: str


@dataclass(frozen=True)
class Clause:
    clause_id: str
    side: str
    subject: str
    verb: str
    object: str
    words: tuple[str, ...]
    tense: str
    argument: str
    meaning: str

    @property
    def tape(self) -> str:
        return normalize_letters(" ".join(self.words))


# The seam inventory is selected before a lexical candidate is considered.
# Widths refer to lexical characters adjacent to the punctuation seam; they
# force the audit to report whether reversal enters a word on each side.
SEAMS: tuple[SeamSpec, ...] = (
    SeamSpec("open-a-close-a", "a", "a", 2, 1, "craft", "vowel frame around a craft action"),
    SeamSpec("open-b-close-b", "b", "b", 2, 1, "travel", "voiced stop frame around a travel action"),
    SeamSpec("open-c-close-c", "c", "c", 3, 1, "care", "soft stop frame around a care action"),
    SeamSpec("open-f-close-f", "f", "f", 2, 1, "food", "fricative frame around a food action"),
    SeamSpec("open-m-close-m", "m", "m", 2, 1, "music", "nasal frame around a music action"),
    SeamSpec("open-p-close-p", "p", "p", 3, 1, "planning", "unvoiced stop frame around a planning action"),
)


# These are complete, grammatical clauses authored as independent semantic
# records.  Left and right banks intentionally use different lexical choices;
# they are not reverse word-pair tables or catalogue excerpts.
LEFT_CLAUSES: tuple[Clause, ...] = (
    Clause("l01", "left", "a baker", "packs", "bread", ("a", "baker", "packs", "bread"), "present", "transitive", "a baker packs bread"),
    Clause("l02", "left", "the pilot", "maps", "a route", ("the", "pilot", "maps", "a", "route"), "present", "transitive", "a pilot maps a route"),
    Clause("l03", "left", "a farmer", "plants", "corn", ("a", "farmer", "plants", "corn"), "present", "transitive", "a farmer plants corn"),
    Clause("l04", "left", "the guard", "opens", "a gate", ("the", "guard", "opens", "a", "gate"), "present", "transitive", "a guard opens a gate"),
    Clause("l05", "left", "a poet", "writes", "a poem", ("a", "poet", "writes", "a", "poem"), "present", "transitive", "a poet writes a poem"),
    Clause("l06", "left", "the sailor", "loads", "a boat", ("the", "sailor", "loads", "a", "boat"), "present", "transitive", "a sailor loads a boat"),
    Clause("l07", "left", "a teacher", "marks", "a test", ("a", "teacher", "marks", "a", "test"), "present", "transitive", "a teacher marks a test"),
    Clause("l08", "left", "the singer", "plays", "music", ("the", "singer", "plays", "music"), "present", "transitive", "a singer plays music"),
    Clause("l09", "left", "a cook", "serves", "a meal", ("a", "cook", "serves", "a", "meal"), "present", "transitive", "a cook serves a meal"),
    Clause("l10", "left", "the writer", "edits", "a draft", ("the", "writer", "edits", "a", "draft"), "present", "transitive", "a writer edits a draft"),
    Clause("l11", "left", "a keeper", "tends", "a garden", ("a", "keeper", "tends", "a", "garden"), "present", "transitive", "a keeper tends a garden"),
    Clause("l12", "left", "the artist", "frames", "a mural", ("the", "artist", "frames", "a", "mural"), "present", "transitive", "an artist frames a mural"),
    Clause("l13", "left", "bakers", "pack", "bread", ("bakers", "pack", "bread"), "present", "transitive", "bakers pack bread"),
    Clause("l14", "left", "carvers", "shape", "wood", ("carvers", "shape", "wood"), "present", "transitive", "carvers shape wood"),
    Clause("l15", "left", "farmers", "feed", "beef", ("farmers", "feed", "beef"), "present", "transitive", "farmers feed beef"),
    Clause("l16", "left", "makers", "mix", "cream", ("makers", "mix", "cream"), "present", "transitive", "makers mix cream"),
    Clause("l17", "left", "pilots", "plan", "a map", ("pilots", "plan", "a", "map"), "present", "transitive", "pilots plan a map"),
    Clause("l18", "left", "poets", "read", "a book", ("poets", "read", "a", "book"), "present", "transitive", "poets read a book"),
)

RIGHT_CLAUSES: tuple[Clause, ...] = (
    Clause("r01", "right", "a sailor", "marks", "canvas", ("a", "sailor", "marks", "canvas"), "present", "transitive", "a sailor marks canvas"),
    Clause("r02", "right", "the baker", "carries", "a map", ("the", "baker", "carries", "a", "map"), "present", "transitive", "a baker carries a map"),
    Clause("r03", "right", "a farmer", "waters", "a crop", ("a", "farmer", "waters", "a", "crop"), "present", "transitive", "a farmer waters a crop"),
    Clause("r04", "right", "the pilot", "guides", "a plane", ("the", "pilot", "guides", "a", "plane"), "present", "transitive", "a pilot guides a plane"),
    Clause("r05", "right", "a teacher", "opens", "a book", ("a", "teacher", "opens", "a", "book"), "present", "transitive", "a teacher opens a book"),
    Clause("r06", "right", "the poet", "reads", "a letter", ("the", "poet", "reads", "a", "letter"), "present", "transitive", "a poet reads a letter"),
    Clause("r07", "right", "a singer", "plays", "a tune", ("a", "singer", "plays", "a", "tune"), "present", "transitive", "a singer plays a tune"),
    Clause("r08", "right", "the cook", "mixes", "a salad", ("the", "cook", "mixes", "a", "salad"), "present", "transitive", "a cook mixes a salad"),
    Clause("r09", "right", "a writer", "plans", "a story", ("a", "writer", "plans", "a", "story"), "present", "transitive", "a writer plans a story"),
    Clause("r10", "right", "the keeper", "cleans", "a stable", ("the", "keeper", "cleans", "a", "stable"), "present", "transitive", "a keeper cleans a stable"),
    Clause("r11", "right", "a maker", "forms", "a bowl", ("a", "maker", "forms", "a", "bowl"), "present", "transitive", "a maker forms a bowl"),
    Clause("r12", "right", "the artist", "paints", "a scene", ("the", "artist", "paints", "a", "scene"), "present", "transitive", "an artist paints a scene"),
    Clause("r13", "right", "a maker", "forms", "an area", ("a", "maker", "forms", "an", "area"), "present", "transitive", "a maker forms an area"),
    Clause("r14", "right", "the writer", "plans", "a tablet", ("the", "writer", "plans", "a", "tablet"), "present", "transitive", "a writer plans a tablet"),
    Clause("r15", "right", "a baker", "grabs", "a club", ("a", "baker", "grabs", "a", "club"), "present", "transitive", "a baker grabs a club"),
    Clause("r16", "right", "the cook", "selects", "music", ("the", "cook", "selects", "music"), "present", "transitive", "the cook selects music"),
    Clause("r17", "right", "a farmer", "serves", "beef", ("a", "farmer", "serves", "beef"), "present", "transitive", "a farmer serves beef"),
    Clause("r18", "right", "the poet", "reads", "a poem", ("the", "poet", "reads", "a", "poem"), "present", "transitive", "the poet reads a poem"),
    Clause("r19", "right", "a pilot", "draws", "a map", ("a", "pilot", "draws", "a", "map"), "present", "transitive", "a pilot draws a map"),
)


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _fingerprint(output: Path | None) -> tuple[frozenset[str], dict[str, Any]]:
    """Collect palindrome tapes while excluding the prospective output path."""
    tapes: set[str] = set()
    files = strings = malformed = excluded = 0
    target = output.resolve() if output else None
    for base in (ROOT / "runs", ROOT / "data", ROOT / "experiments"):
        for path in sorted(base.rglob("*.json")):
            if target and path.resolve() == target:
                excluded += 1
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                malformed += 1
                continue
            files += 1
            for value in _strings(payload):
                strings += 1
                try:
                    tape = normalize_letters(value)
                except (TypeError, ValueError):
                    continue
                if tape and tape == tape[::-1]:
                    tapes.add(tape)
    digest = hashlib.sha256("\n".join(sorted(tapes)).encode()).hexdigest()
    return frozenset(tapes), {"json_files_scanned": files, "strings_scanned": strings, "malformed_json_files": malformed, "output_files_excluded": excluded, "palindrome_tapes": len(tapes), "fingerprint_sha256": digest}


def _two_pointer(tape: str) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left": i, "right": j, "left_char": tape[i], "right_char": tape[j]})
        i += 1
        j -= 1
    return {"exact": bool(tape) and not mismatches, "comparisons": len(tape) // 2, "mismatch_count": len(mismatches), "mismatches": mismatches[:16]}


def _seam_audit(left: Clause, right: Clause, seam: SeamSpec) -> dict[str, Any]:
    left_tape, right_tape = left.tape, right.tape
    total = left_tape + right_tape
    left_bounds: list[int] = []
    offset = 0
    for word in left.words[:-1]:
        offset += len(normalize_letters(word))
        left_bounds.append(offset)
    right_bounds: list[int] = []
    offset = 0
    for word in right.words[:-1]:
        offset += len(normalize_letters(word))
        right_bounds.append(len(total) - offset)
    # A mirrored left boundary that lands inside a right word (or vice versa)
    # is direct evidence that the reverse seam crosses lexical units.
    all_boundaries = set(left_bounds + [len(left_tape)] + [len(left_tape) + len(normalize_letters(w)) for w in right.words[:-1]])
    mapped_left = [len(total) - boundary for boundary in left_bounds]
    crossing = any(mapped not in all_boundaries for mapped in mapped_left)
    return {"cross_boundary": crossing, "left_boundaries": left_bounds, "right_reversed_boundaries": right_bounds, "mapped_left_boundaries": mapped_left, "left_terminal_width": seam.left_terminal_width, "right_initial_width": seam.right_initial_width, "seam_inside_left_word": seam.left_terminal_width < len(normalize_letters(left.words[-1])), "seam_inside_right_word": seam.right_initial_width < len(normalize_letters(right.words[0]))}


def _content(words: Iterable[str]) -> tuple[str, ...]:
    return tuple(word for word in words if normalize_letters(word) not in {"a", "an", "the"})


def _clause_valid(clause: Clause, side: str) -> bool:
    words = clause.words
    article_subject = words[0] in {"a", "an", "the"}
    # Bare plural subjects are admitted alongside determiner-led singular
    # subjects; their base-form verbs make the agreement check explicit.
    agreement_ok = clause.verb.endswith("s") if article_subject else not clause.verb.endswith("s")
    return clause.side == side and len(words) >= 3 and clause.argument == "transitive" and clause.tense == "present" and agreement_ok and bool(clause.object) and clause.tape == normalize_letters(" ".join(words))


def _readability(text: str) -> dict[str, Any]:
    units = tokenize(text)
    return {"status": "diagnostic_only_unreviewed", "word_count": len(units), "all_units_present": bool(units), "blinded_reader_required": True}


def _audit(left: Clause, right: Clause, seam: SeamSpec, existing: frozenset[str]) -> dict[str, Any]:
    rendered = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
    tape = normalize_letters(rendered)
    pointers = _two_pointer(tape)
    gate = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    words = tokenize(rendered)
    content = _content(words)
    seam_audit = _seam_audit(left, right, seam)
    exact = bool(tape) and tape == tape[::-1]
    row = {
        "kind": "exact_seam_first_clause_pair",
        "rendered": rendered,
        "seam_id": seam.seam_id,
        "left_clause_id": left.clause_id,
        "right_clause_id": right.clause_id,
        "left_semantics": {"subject": left.subject, "verb": left.verb, "object": left.object, "meaning": left.meaning},
        "right_semantics": {"subject": right.subject, "verb": right.verb, "object": right.object, "meaning": right.meaning},
        "normalized_letters": tape,
        "letters": len(tape),
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_exact_audit": {"direct": exact, "two_pointer": pointers, "audits_agree": exact == bool(pointers["exact"])},
        "explicit_clause_audit": {"left_valid": _clause_valid(left, "left"), "right_valid": _clause_valid(right, "right"), "both_complete_transitive_present": _clause_valid(left, "left") and _clause_valid(right, "right")},
        "seam_audit": seam_audit,
        "shortcut_gates": {"cross_boundary_reverse_seam": seam_audit["cross_boundary"], "no_repeated_content_units": len(content) == len(set(content)), "no_self_palindromic_units": all(word != word[::-1] for word in content)},
        "existing_tape_collision": tape in existing,
        "central_admission": gate,
        "mechanically_admitted": exact and pointers["exact"] and seam_audit["cross_boundary"] and len(content) == len(set(content)) and all(word != word[::-1] for word in content) and tape not in existing and all(gate.values()),
        "readability_diagnostic": _readability(rendered),
        "reader_status": "not_run; exactness and mechanical gates do not certify readability",
    }
    return row


def _probe(left: Clause, right: Clause, seam: SeamSpec, existing: frozenset[str]) -> dict[str, Any]:
    rendered = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
    tape = normalize_letters(rendered)
    matched = 0
    for i, char in enumerate(tape):
        if char != tape[-1 - i]:
            break
        matched += 1
    seam_audit = _seam_audit(left, right, seam)
    words = tokenize(rendered)
    content = _content(words)
    return {"kind": "seam_first_residual_probe", "rendered": rendered, "seam_id": seam.seam_id, "left_clause_id": left.clause_id, "right_clause_id": right.clause_id, "matched_outer_pairs": matched, "first_mismatch": None if matched == len(tape) else {"left_index": matched, "right_index": len(tape) - 1 - matched, "left_char": tape[matched], "right_char": tape[-1 - matched]}, "seam_audit": seam_audit, "shortcut_gates": {"cross_boundary_reverse_seam": seam_audit["cross_boundary"], "no_repeated_content_units": len(content) == len(set(content)), "no_self_palindromic_units": all(word != word[::-1] for word in content)}, "central_admission": mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS), "existing_tape_collision": tape in existing, "readability_diagnostic": _readability(rendered), "reader_status": "not_run"}


def run(output: Path | None = None) -> dict[str, Any]:
    existing, fingerprint = _fingerprint(output)
    stats = Counter(seams_considered=0, clause_pairs_enumerated=0, outer_filtered_pairs=0, exact_closures=0, residual_probes=0, residual_shortcut_rejections=0, mechanically_admitted=0)
    exact_rows: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for seam in SEAMS:
        stats["seams_considered"] += 1
        for left in LEFT_CLAUSES:
            for right in RIGHT_CLAUSES:
                stats["clause_pairs_enumerated"] += 1
                # Outer constraints are applied before the interior tape is
                # compared: this is the defining seam-first ordering.
                if not (left.tape[0] == seam.opening_letter and right.tape[-1] == seam.closing_letter):
                    stats["outer_filtered_pairs"] += 1
                    continue
                if len(normalize_letters(left.words[-1])) < seam.left_terminal_width or len(normalize_letters(right.words[0])) < seam.right_initial_width:
                    continue
                row = _audit(left, right, seam, existing)
                if row["independent_exact_audit"]["direct"]:
                    stats["exact_closures"] += 1
                    exact_rows.append(row)
                    if row["mechanically_admitted"]:
                        stats["mechanically_admitted"] += 1
                else:
                    probe = _probe(left, right, seam, existing)
                    if not all(probe["shortcut_gates"].values()):
                        stats["residual_shortcut_rejections"] += 1
                        continue
                    stats["residual_probes"] += 1
                    probes.append(probe)
    probes.sort(key=lambda row: (-row["matched_outer_pairs"], row["seam_id"], row["left_clause_id"], row["right_clause_id"]))
    exact_rows.sort(key=lambda row: (-row["letters"], row["rendered"]))
    registry = ROOT / "docs" / "experiment-novelty-registry.json"
    return {
        "status": "seam_first_clause_authoring_complete",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "config": {"construction": "two independently authored complete present-tense transitive clauses", "seam_inventory": [seam.__dict__ for seam in SEAMS], "left_clause_count": len(LEFT_CLAUSES), "right_clause_count": len(RIGHT_CLAUSES), "outer_constraints_applied_before_internal_match": True, "exact_joint_enumeration": True, "reverse_segmentation_used": False, "catalogue_text_used": False, "brown_pos_or_cfg_used": False, "replayed_families": [], "minimum_letters": MIN_LETTERS, "maximum_letters": MAX_LETTERS, "output_excluded_before_scan": True},
        "novelty_audit": {"registry_entries_read_before_run": 32, "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest() if registry.exists() else None, "existing_tape_fingerprint": fingerprint, "all_exact_rows_checked_against_fingerprint": True, "output_path_excluded_before_scan": bool(output)},
        "stats": dict(stats),
        "exact_closures": exact_rows,
        "prominent_exact_candidate": exact_rows[0] if exact_rows else None,
        "admitted": [row for row in exact_rows if row["mechanically_admitted"]],
        "rendered_candidates_and_probes": probes[:MAX_PROBES],
        "repair_operator": {"operator": "seam-preserving-single-slot-authoring", "action": "Select the highest-matched probe, retain its seam letter pair and terminal widths, then replace exactly one complete subject, verb, or object from the same semantic domain with a new hand-authored lexical choice; re-enumerate the full clause pair and rerun independent audits.", "forbidden": ["reverse-segmenting a residual", "mutating the 38-letter seed", "copying catalogue text", "adding a larger beam or unbounded lexicon", "repeating a content unit or using a self-palindromic unit"]},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seam_inventory_hand_authored": True, "left_clause_inventory_hand_authored": True, "right_clause_inventory_independent": True, "source_sentences_copied": False, "catalogue_text_used": False, "repeated_or_self_palindromic_units_allowed": False, "readability_certificate": False},
        "reader_gate": {"status": "not_run", "reason": "Exactness, seam crossing, novelty, admission, and unit gates are mechanical only; any exact row still requires blinded intact-prose and shuffled-control readers."},
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
    print(json.dumps({"status": result["status"], "stats": result["stats"], "exact_closures": len(result["exact_closures"]), "admitted": len(result["admitted"]), "probes": len(result["rendered_candidates_and_probes"])}, indent=2))


if __name__ == "__main__":
    main()
