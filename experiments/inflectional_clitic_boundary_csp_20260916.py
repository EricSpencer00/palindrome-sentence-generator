"""Ordinary-order inflectional and clitic boundary CSP search.

This experiment starts from a fresh harbor scene and realizes it with a small
productive grammar.  Subject number and tense choose the verb ending; an
owner-number choice selects a possessive clitic (``crew's`` or ``crews'``) or
an ``of`` attachment; and a reported-complement choice changes the final
clause.  The boundary ledger records stem/ending/clitic positions before one
global character equation is scored.  It is not a morphology FST or an
affix-morpheme sampler, and no word order is reflected.
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

EXPERIMENT = "inflectional-clitic-boundary-csp-20260916"
SIGNATURE = (
    "ordinary-order-inflectional-clitic-grammar|agreement-tense-realization|"
    "stem-ending-clitic-boundary-csp|joint-complete-prose-rendering|"
    "heldout-boundary-inflection-repair|independent-exact-admission-audit"
)
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

SCENE = {
    "id": "harbor-pilot-dawn",
    "seed": "At sunrise, the harbor pilot checks the signal lamps, marks the tide in the crew's notebook, and alerts the waiting passengers that the ferry will depart.",
    "meaning": "At sunrise, a harbor pilot checks lights, records the tide in a crew notebook, and alerts waiting passengers that the ferry will leave.",
}

OBJECTS = {
    "lamps": ("the signal lamps", "the warning lights"),
    "book": ("notebook", "logbook"),
}
ATTACHMENTS = {
    "that_depart": "that the ferry will depart",
    "that_leave": "that the boat will leave",
}
PUNCTUATION = {
    "comma": (", ", ", ", "."),
    "dash": (" — ", ", ", "."),
}


def inflect(base: str, number: str, tense: str) -> tuple[str, str]:
    """Return a regular stem/ending realization with its boundary ledger."""
    if tense == "past":
        return base + "ed", "ed"
    if number == "singular":
        return base + "s", "s"
    return base, ""


def possessive(owner: str, mode: str, book: str) -> tuple[str, dict]:
    if mode == "singular_clitic":
        return f"the crew's {book}", {"owner_stem": "crew", "owner_ending": "", "clitic": "'s", "boundary_kind": "possessive_apostrophe_s"}
    if mode == "plural_clitic":
        return f"the crews' {book}", {"owner_stem": "crew", "owner_ending": "s", "clitic": "'", "boundary_kind": "plural_possessive_apostrophe"}
    return f"the {book} of the crew", {"owner_stem": "crew", "owner_ending": "", "clitic": "", "boundary_kind": "of_attachment"}


def render(number: str, tense: str, owner_mode: str, lamp_mode: int, book_mode: int, attachment: str, punctuation: str) -> tuple[str, dict]:
    subject = "harbor pilot" if number == "singular" else "harbor pilots"
    c1_verb, c1_ending = inflect("check", number, tense)
    c2_verb, c2_ending = inflect("mark", number, tense)
    c3_verb, c3_ending = inflect("alert", number, tense)
    book = OBJECTS["book"][book_mode]
    owner_phrase, owner_ledger = possessive("crew", owner_mode, book)
    joins = PUNCTUATION[punctuation]
    clauses = [
        f"At sunrise, the {subject} {c1_verb} {OBJECTS['lamps'][lamp_mode]}",
        f"{c2_verb} the tide in {owner_phrase}",
        f"and {c3_verb} the waiting passengers {ATTACHMENTS[attachment]}",
    ]
    text = clauses[0] + joins[0] + clauses[1] + joins[1] + clauses[2] + joins[2]
    state = {
        "number": number,
        "tense": tense,
        "owner_mode": owner_mode,
        "lamp_mode": lamp_mode,
        "book_mode": book_mode,
        "attachment": attachment,
        "punctuation": punctuation,
        "morphology": {
            "check": {"stem": "check", "ending": c1_ending, "surface": c1_verb},
            "mark": {"stem": "mark", "ending": c2_ending, "surface": c2_verb},
            "alert": {"stem": "alert", "ending": c3_ending, "surface": c3_verb},
        },
        "owner_boundary": owner_ledger,
    }
    return text, {"state": state, "clauses": clauses}


def tape(text: str) -> str:
    return normalize_letters(text)


def exact_slice(text: str) -> dict:
    value = tape(text)
    return {"algorithm": "normalized_tape_reverse_slice", "exact": bool(value) and value == value[::-1], "letters": len(value)}


def exact_two_pointer(text: str) -> dict:
    value = tape(text)
    mismatches = []
    left, right = 0, len(value) - 1
    while left < right:
        if value[left] != value[right]:
            mismatches.append({"offset": left, "right_offset": right, "left": value[left], "right": value[right]})
        left += 1
        right -= 1
    return {"algorithm": "independent_two_pointer", "exact": bool(value) and not mismatches, "letters": len(value), "mismatch_count": len(mismatches), "mismatches": mismatches[:10]}


def hash_audit(text: str) -> dict:
    value = tape(text)
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {"algorithm": "independent_sha256_forward_vs_reversed_tape", "forward_sha256": forward, "reverse_sha256": reverse, "exact": bool(value) and forward == reverse}


def boundary_csp(text: str, render_meta: dict) -> dict:
    value = tape(text)
    pairs = [(i, len(value) - 1 - i) for i in range(len(value) // 2)]
    matches = sum(value[i] == value[j] for i, j in pairs)
    mismatch = next((i for i, j in pairs if value[i] != value[j]), None)
    edge_records = []
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        word = match.group()
        if word.casefold() in {"checks", "check", "checked", "marks", "mark", "marked", "alerts", "alert", "alerted"}:
            surface = normalize_letters(word)
            edge_records.append({"surface": word, "normalized_start": len(tape(text[:match.start()])), "stem_ending_boundary": len(surface), "edge": render_meta["state"]["morphology"].get(word.casefold().rstrip("s ed"), {})})
        if "'" in word:
            edge_records.append({"surface": word, "normalized_start": len(tape(text[:match.start()])), "clitic_boundary": word.find("'"), "edge": render_meta["state"]["owner_boundary"]})
    return {"equation": "x[i] = x[N-1-i] on the complete ordinary-order rendering", "positions_checked": len(pairs), "matching_pairs": matches, "mismatch_pairs": len(pairs) - matches, "first_mismatch_offset": mismatch, "match_rate": matches / len(pairs) if pairs else 0.0, "joint_boundary_edges": edge_records, "boundary_choices_solved_jointly": True}


def independent_admission(text: str) -> dict:
    words = tuple(normalize_letters(word) for word in tokenize(text))
    function = {"a", "an", "the", "at", "in", "of", "and", "that", "will", "to", "the", "before", "after"}
    content = tuple(word for word in words if word not in function)
    return {"algorithm": "independent_complete_prose_shape_scan", "ascii_letters_only": all(not c.isalpha() or c.isascii() for c in text), "three_clause_segments": len(re.split(r"[,;:—]+", text)) >= 3, "terminal_period": text.endswith("."), "minimum_word_count": len(words) >= 15, "content_words_unique": len(content) == len(set(content)), "not_word_order_mirror": tuple(words) != tuple(reversed(words))}


def readability(text: str) -> dict:
    words = [word.casefold() for word in tokenize(text)]
    try:
        from wordfreq import zipf_frequency
        mean_zipf = sum(zipf_frequency(word, "en") for word in words) / len(words)
    except Exception:
        mean_zipf = None
    return {"diagnostic_not_human_readability": True, "word_count": len(words), "mean_zipf_frequency": mean_zipf, "repeated_word_rate": 1 - len(set(words)) / len(words) if words else None, "reader_status": "not_run; requires blinded human reader study"}


def repair_probe(params: tuple, text: str, equation: dict) -> dict:
    number, tense, owner_mode, lamp_mode, book_mode, attachment, punctuation = params
    offset = equation["first_mismatch_offset"] or 0
    # Held-out operator changes one morphological/boundary slot, selected by
    # the first mismatch region; all other semantic choices stay fixed.
    if offset < len(tape(text)) // 3:
        slot = "agreement_tense"
        replacement = ("plural" if number == "singular" else "singular", tense, owner_mode, lamp_mode, book_mode, attachment, punctuation)
    elif offset < 2 * len(tape(text)) // 3:
        slot = "clitic_boundary"
        replacement = (number, tense, "of_attachment" if owner_mode != "of_attachment" else "singular_clitic", lamp_mode, book_mode, attachment, punctuation)
    else:
        slot = "held_out_attachment"
        replacement = (number, tense, owner_mode, lamp_mode, book_mode, "that_leave" if attachment == "that_depart" else "that_depart", punctuation)
    probe, _ = render(*replacement)
    probe_exact = exact_two_pointer(probe)
    return {"operator": "held-out one-slot inflection/boundary repair at first mismatch", "target_slot": slot, "changed_slot_count": 1, "from": list(params), "to": list(replacement), "rendered_probe": probe, "probe_exact": probe_exact["exact"], "probe_mismatch_pairs": probe_exact["mismatch_count"]}


def audit(params: tuple, rank: int = 0) -> dict:
    text, meta = render(*params)
    first, second, hashed = exact_slice(text), exact_two_pointer(text), hash_audit(text)
    equation = boundary_csp(text, meta)
    central = mechanical_admission_checks(text, min_letters=95, max_letters=180)
    independent = independent_admission(text)
    row = {"rank": rank, "rendered": text, "letters": len(tape(text)), "grammar_state": meta["state"], "boundary_csp": equation, "exact_check_1": first, "exact_check_2": second, "exact_check_hash": hashed, "independent_exact_agreement": first["exact"] == second["exact"] == hashed["exact"], "central_admission": central, "independent_admission": independent, "admission_agreement": central["not_word_order_symmetry"] == independent["not_word_order_mirror"], "mechanically_admitted": first["exact"] and second["exact"] and hashed["exact"] and all(central.values()) and all(independent.values()), "readability_evidence": readability(text), "provenance": {"source": "human-authored harbor scene", "catalogue_text_used": False, "pre_existing_palindrome_wrapped": False, "word_order_mirrored": False, "repeated_palindromic_unit_used": False, "productive_inflectional_grammar": True}}
    row["heldout_repair"] = repair_probe(params, text, equation)
    row["next_repair_operator"] = "none; exact closure" if first["exact"] else f"apply one held-out {row['heldout_repair']['target_slot']} change at first mismatch offset {equation['first_mismatch_offset']}; rerun all stem/ending/clitic boundary equations"
    return row


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [entry["id"] for entry in entries if entry.get("id") != EXPERIMENT and entry.get("signature") == SIGNATURE]
    related = [entry["id"] for entry in entries if any(term in entry.get("signature", "") for term in ("morphology", "inflection", "clitic", "affix")) and entry.get("id") != EXPERIMENT]
    return {"entries_inspected": len(entries), "exact_signature_collisions_before_run": collisions, "related_families_for_manual_review": related[:24], "passed": not collisions, "state_space_distinction": "productive agreement/tense forms and possessive clitic boundaries are selected in ordinary word order, then jointly audited as complete prose; no FST or morpheme sampler"}


def search() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_run']}")
    states = itertools.product(("singular", "plural"), ("present", "past"), ("singular_clitic", "plural_clitic", "of_attachment"), range(2), range(2), tuple(ATTACHMENTS), tuple(PUNCTUATION))
    rows = [audit(params) for params in states]
    rows.sort(key=lambda row: (-row["boundary_csp"]["match_rate"], row["exact_check_2"]["mismatch_count"], {"comma": 0, "dash": 1}[row["grammar_state"]["punctuation"]], row["rendered"]))
    for rank, row in enumerate(rows[:24], 1):
        row["rank"] = rank
    exact = [row for row in rows if row["exact_check_1"]["exact"] and row["exact_check_2"]["exact"] and row["exact_check_hash"]["exact"]]
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete_inflectional_clitic_boundary_csp_search", "novelty_preflight": preflight, "human_seed": SCENE, "state_dimensions": {"agreement": 2, "tense": 2, "clitic_boundary_modes": 3, "lamp_realizations": 2, "book_realizations": 2, "reported_attachment": 2, "punctuation": len(PUNCTUATION)}, "states_examined": len(rows), "exact_count": len(exact), "mechanically_admitted_count": len(admitted), "independent_exact_agreement_count": sum(row["independent_exact_agreement"] for row in rows), "admission_agreement_count": sum(row["admission_agreement"] for row in rows), "best_rendered_candidates": rows[:24], "failed_attempts": [row for row in rows if not row["exact_check_1"]["exact"]], "readability_evidence": {"status": "diagnostic_not_human_readability_result", "reader_eligible_count": 0, "method": "independent word-frequency, repetition, and complete-prose shape diagnostics; no human readers run"}, "failure_evidence": {"all_nonexact_states_retained": True, "next_repair_operator": "one held-out agreement, tense, clitic-boundary, or attachment change at the first mismatch", "exact_closure_found": bool(exact)}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "catalogue_or_corpus_import": False, "pre_existing_palindrome_wrapped": False, "word_order_symmetry_used": False}}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = search()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("states_examined", "exact_count", "mechanically_admitted_count")}, indent=2))


if __name__ == "__main__":
    main()
