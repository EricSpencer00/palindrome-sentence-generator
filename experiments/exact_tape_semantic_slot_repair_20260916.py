"""Repair an existing exact tape through typed semantic slot substitutions.

The source is a mechanically checked exact tape emitted by an earlier
repository run.  This lane treats that tape as letters only, then applies a
typed subject/verb/object/adjunct map and optional boundary merges.  A
rendering is retained as a tape-preserving probe only when its independently
normalized letters equal the source tape.  The source tape is not in the
catalogue; no palindrome wrapper or mirrored clause is introduced.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT = "exact-tape-semantic-slot-repair-20260916"
SIGNATURE = (
    "repository-exact-tape-input|typed-semantic-slot-substitution|"
    "boundary-merge-preservation-constraint|ordinary-order-rendering|"
    "heldout-semantic-repair|independent-exact-admission-audit"
)
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
SOURCE_RUN = ROOT / "runs" / "benchmark-grammar-broad-lexical-sweep-20260916.json"
SOURCE_EXACT_TEXT = "An aide rips nine memos; Some men inspire Diana."
SOURCE_TAPE = normalize_letters(SOURCE_EXACT_TEXT)

# The exact tape's semantic slots are deliberately typed. Alternatives are
# held out from the source rendering and are never selected by a catalogue
# lookup. Identity entries provide a tape-preserving control; substitutions
# demonstrate why global semantic repair usually breaks the exact equation.
SLOTS = {
    "subject_agent": ("An aide", "A nurse", "The helper"),
    "subject_verb": ("rips", "writes", "sorts"),
    "object": ("nine memos", "some notes", "a short letter"),
    "adjunct_agent": ("Some men", "The clerks", "Two aides"),
    "adjunct_verb": ("inspire", "assist", "guide"),
    "adjunct_object": ("Diana", "the nurse", "the child"),
}
PUNCTUATION = {
    "semicolon": "; ",
    "comma": ", ",
    "period": ". ",
}
HELD_OUT = {
    "subject_agent": "The nurse",
    "subject_verb": "records",
    "object": "the notes",
    "adjunct_agent": "A clerk",
    "adjunct_verb": "helps",
    "adjunct_object": "the visitor",
}


def exact_slice(text: str) -> dict:
    value = normalize_letters(text)
    return {"algorithm": "normalized_tape_reverse_slice", "exact": bool(value) and value == value[::-1], "letters": len(value)}


def exact_two_pointer(text: str) -> dict:
    value = normalize_letters(text)
    mismatches = []
    left, right = 0, len(value) - 1
    while left < right:
        if value[left] != value[right]:
            mismatches.append({"offset": left, "right_offset": right, "left": value[left], "right": value[right]})
        left += 1
        right -= 1
    return {"algorithm": "independent_two_pointer", "exact": bool(value) and not mismatches, "letters": len(value), "mismatch_count": len(mismatches), "mismatches": mismatches[:10]}


def hash_audit(text: str) -> dict:
    value = normalize_letters(text)
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {"algorithm": "independent_sha256_tape_vs_reverse", "forward_sha256": forward, "reverse_sha256": reverse, "exact": bool(value) and forward == reverse}


def render(choice: tuple[int, int, int, int, int, int], punctuation: str) -> tuple[str, dict]:
    values = [SLOTS[name][choice[index]] for index, name in enumerate(SLOTS)]
    first = f"{values[0]} {values[1]} {values[2]}"
    second = f"{values[3]} {values[4]} {values[5]}"
    text = first + PUNCTUATION[punctuation] + second + "."
    return text, {"slot_values": dict(zip(SLOTS, values)), "punctuation": punctuation, "boundary_merge": "none"}


def boundary_merge_probe(choice: tuple[int, ...], punctuation: str) -> tuple[str, dict]:
    text, meta = render(choice, punctuation)
    # Merges are represented explicitly, but only whitespace/punctuation may
    # disappear without changing the tape. Lexical merges are therefore
    # tested and rejected by the same preservation check rather than invented.
    merged = text.replace(" ", "", 1)
    meta = dict(meta)
    meta["boundary_merge"] = "first_word_boundary_removed"
    return merged, meta


def independent_admission(text: str) -> dict:
    words = tuple(normalize_letters(word) for word in tokenize(text))
    return {"algorithm": "independent_complete_prose_and_slot_scan", "ascii_letters_only": all(not c.isalpha() or c.isascii() for c in text), "two_clause_shape": len(re.split(r"[;,.!?]+", text)) >= 2, "terminal_sentence_mark": text.endswith("."), "minimum_word_count": len(words) >= 7, "not_word_order_mirror": tuple(words) != tuple(reversed(words)), "no_catalogue_wrapper_marker": "marge" not in text.casefold() and "telegram" not in text.casefold()}


def readability(text: str) -> dict:
    words = [word.casefold() for word in tokenize(text)]
    try:
        from wordfreq import zipf_frequency
        mean_zipf = sum(zipf_frequency(word, "en") for word in words) / len(words)
    except Exception:
        mean_zipf = None
    return {"diagnostic_not_human_readability": True, "word_count": len(words), "mean_zipf_frequency": mean_zipf, "repeated_word_rate": 1 - len(set(words)) / len(words) if words else None, "reader_status": "not_run; source-derived probes require separate human review"}


def audit(text: str, choice: tuple[int, ...], punctuation: str, operation: str, rank: int = 0) -> dict:
    first, second, hashed = exact_slice(text), exact_two_pointer(text), hash_audit(text)
    central = mechanical_admission_checks(text, min_letters=30, max_letters=100)
    independent = independent_admission(text)
    preserved = normalize_letters(text) == SOURCE_TAPE
    row = {"rank": rank, "rendered": text, "operation": operation, "choice_indices": list(choice), "punctuation": punctuation, "source_tape": SOURCE_TAPE, "letters": len(normalize_letters(text)), "tape_preserved": preserved, "exact_check_1": first, "exact_check_2": second, "exact_check_hash": hashed, "independent_exact_agreement": first["exact"] == second["exact"] == hashed["exact"], "central_admission": central, "independent_admission": independent, "admission_agreement": central["not_word_order_symmetry"] == independent["not_word_order_mirror"], "mechanically_admitted": preserved and first["exact"] and second["exact"] and hashed["exact"] and all(central.values()) and all(independent.values()), "readability_evidence": readability(text), "provenance": {"source_run": str(SOURCE_RUN.relative_to(ROOT)), "source_exact_tape_only": True, "source_surface_reused": operation == "identity", "catalogue_text_used": False, "wrapper_used": False, "word_order_mirrored": False, "typed_semantic_slots": list(SLOTS)}}
    return row


def heldout_repair(choice: tuple[int, ...], punctuation: str, failed_row: dict) -> dict:
    offset = failed_row["exact_check_2"]["mismatches"][0]["offset"] if failed_row["exact_check_2"]["mismatches"] else 0
    slot_index = offset % len(SLOTS)
    slot_name = tuple(SLOTS)[slot_index]
    values = list(choice)
    replacement = HELD_OUT[slot_name]
    probe_values = dict(zip(SLOTS, [SLOTS[name][values[i]] for i, name in enumerate(SLOTS)]))
    probe_values[slot_name] = replacement
    probe = f"{probe_values['subject_agent']} {probe_values['subject_verb']} {probe_values['object']}{PUNCTUATION[punctuation]}{probe_values['adjunct_agent']} {probe_values['adjunct_verb']} {probe_values['adjunct_object']}."
    checked = exact_two_pointer(probe)
    return {"operator": "held-out one semantic slot substitution at first mismatch", "first_mismatch_offset": offset, "target_slot": slot_name, "changed_slot_count": 1, "held_out_value": replacement, "rendered_probe": probe, "probe_tape_preserved": normalize_letters(probe) == SOURCE_TAPE, "probe_exact": checked["exact"], "probe_mismatch_pairs": checked["mismatch_count"]}


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [entry["id"] for entry in entries if entry.get("id") != EXPERIMENT and entry.get("signature") == SIGNATURE]
    related = [entry["id"] for entry in entries if any(term in entry.get("signature", "") for term in ("semantic-slot", "tape", "lexicalization")) and entry.get("id") != EXPERIMENT]
    return {"entries_inspected": len(entries), "exact_signature_collisions_before_run": collisions, "related_families_for_manual_review": related[:24], "passed": not collisions, "state_space_distinction": "fixed repository exact tape with typed semantic substitutions and explicit tape-preserving boundary-merge constraint; no CFG resegmentation or wrapper"}


def load_source_evidence() -> dict:
    source_digest = hashlib.sha256(SOURCE_EXACT_TEXT.encode()).hexdigest()
    if SOURCE_RUN.exists():
        source_digest = hashlib.sha256(SOURCE_RUN.read_bytes()).hexdigest()
    return {"source_run": str(SOURCE_RUN.relative_to(ROOT)), "source_run_exists": SOURCE_RUN.exists(), "source_run_sha256": source_digest, "source_exact_text": SOURCE_EXACT_TEXT, "source_tape": SOURCE_TAPE, "source_exact_and_mechanically_checked": all(mechanical_admission_checks(SOURCE_EXACT_TEXT, min_letters=30, max_letters=100).values())}


def search() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_run']}")
    rows = []
    for choice in itertools.product(range(3), repeat=len(SLOTS)):
        # Source identity and punctuation-only controls are deliberately
        # excluded: this lane must test an altered semantic slot and a longer
        # resulting rendering, not count the inherited sentence as progress.
        if all(index == 0 for index in choice):
            continue
        for punctuation in PUNCTUATION:
            text, _ = render(choice, punctuation)
            if len(normalize_letters(text)) <= len(SOURCE_TAPE):
                continue
            operation = "typed_semantic_substitution"
            rows.append(audit(text, choice, punctuation, operation))
            merged, _ = boundary_merge_probe(choice, punctuation)
            if len(normalize_letters(merged)) > len(SOURCE_TAPE):
                rows.append(audit(merged, choice, punctuation, "boundary_merge_probe"))
    for row in rows:
        if not row["exact_check_1"]["exact"]:
            row["heldout_repair"] = heldout_repair(tuple(row["choice_indices"]), row["punctuation"], row)
            row["next_repair_operator"] = row["heldout_repair"]
        else:
            row["heldout_repair"] = {"operator": "none; exact source tape control"}
            row["next_repair_operator"] = row["heldout_repair"]
    rows.sort(key=lambda row: (not row["mechanically_admitted"], not row["tape_preserved"], not row["exact_check_1"]["exact"], row["exact_check_2"]["mismatch_count"], {"semicolon": 0, "period": 1, "comma": 2}[row["punctuation"]], row["operation"], row["rendered"]))
    for rank, row in enumerate(rows[:24], 1):
        row["rank"] = rank
    exact = [row for row in rows if row["exact_check_1"]["exact"] and row["tape_preserved"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete_exact_tape_semantic_slot_repair_search", "novelty_preflight": preflight, "source_evidence": load_source_evidence(), "typed_slot_count": len(SLOTS), "states_examined": len(rows), "tape_preserved_count": sum(row["tape_preserved"] for row in rows), "exact_count": len(exact), "mechanically_admitted_count": sum(row["mechanically_admitted"] for row in rows), "independent_exact_agreement_count": sum(row["independent_exact_agreement"] for row in rows), "admission_agreement_count": sum(row["admission_agreement"] for row in rows), "best_rendered_candidates": rows[:24], "failed_attempts": [row for row in rows if not row["exact_check_1"]["exact"]], "readability_evidence": {"status": "diagnostic_not_human_readability_result", "reader_eligible_count": 0, "method": "independent frequency, repetition, and prose-shape diagnostics; no human study run"}, "failure_evidence": {"all_nonexact_states_retain_heldout_repair": True, "next_repair_operator": "one held-out typed subject/verb/object/adjunct substitution at first mismatch, accepted only if source tape remains unchanged", "catalogue_or_wrapper_used": False}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "source_tape_origin": str(SOURCE_RUN.relative_to(ROOT)), "catalogue_or_corpus_import": False, "wrapper_used": False, "word_order_symmetry_used": False}}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = search()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("states_examined", "tape_preserved_count", "exact_count", "mechanically_admitted_count")}, indent=2))


if __name__ == "__main__":
    main()
