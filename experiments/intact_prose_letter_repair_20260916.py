"""Bounded letter-level repair of intact, ordinary multi-clause prose.

This route starts with complete sentences and applies a small edit script while
keeping clause order and ordinary syntax intact.  It is deliberately not a
word-order mirror: lexical substitutions, inflection changes, and whole
adjunct-clause rewrites are proposed from a held-out frontier, then scored by
the first mirrored character mismatch.  Every attempted rendering is kept so
that a failed repair is evidence rather than a silently discarded search.
"""
from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "intact-prose-letter-repair-20260916"
SIGNATURE = (
    "intact-multi-clause-prose-seeds|letter-level-edit-script|"
    "lexical-substitution-inflection-clause-edit|first-mismatch-frontier|"
    "nonmirror-word-order|no-self-palindromic-units|independent-two-audit"
)
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

# These are complete sentences written for this experiment.  Each has a main
# clause plus an ordinary adjunct/coordinated clause; no sentence is itself a
# palindrome or copied from the project's catalogue.
SEEDS = [
    {
        "id": "dawn-maps",
        "text": "At dawn, the curator catalogued three fragile maps while the rain tapped the roof.",
        "clauses": ["At dawn, the curator catalogued three fragile maps", "while the rain tapped the roof"],
        "meaning": "a curator records fragile maps during rainy morning work",
    },
    {
        "id": "carpenter-boards",
        "text": "Before lunch, the patient carpenter measured the narrow boards and marked each corner.",
        "clauses": ["Before lunch, the patient carpenter measured the narrow boards", "and marked each corner"],
        "meaning": "a carpenter measures boards and marks their corners before lunch",
    },
    {
        "id": "singer-costumes",
        "text": "After rehearsal, a careful singer folded the borrowed costumes because the stage was cold.",
        "clauses": ["After rehearsal, a careful singer folded the borrowed costumes", "because the stage was cold"],
        "meaning": "a singer folds borrowed costumes after rehearsal in a cold theatre",
    },
]

# All replacements preserve the seed's local syntactic frame.  The inflection
# table is explicit so a present/past change cannot masquerade as a synonym.
LEXICAL = {
    "curator": ("keeper", "editor"),
    "fragile": ("delicate", "brittle"),
    "maps": ("charts", "plans"),
    "rain": ("mist", "wind"),
    "roof": ("awning", "porch"),
    "patient": ("steady", "calm"),
    "narrow": ("slender", "thin"),
    "boards": ("planks", "panels"),
    "corner": ("edge", "joint"),
    "careful": ("alert", "quiet"),
    "borrowed": ("rented", "shared"),
    "costumes": ("outfits", "clothes"),
    "stage": ("platform", "theatre"),
    "cold": ("chilly", "dim"),
}
INFLECTIONS = {
    "catalogued": ("catalogs", "records"),
    "tapped": ("taps", "drummed"),
    "measured": ("measures", "checked"),
    "marked": ("marks", "labeled"),
    "folded": ("folds", "packed"),
    "was": ("is",),
}
CLAUSE_EDITS = {
    "while the rain tapped the roof": (
        "while the mist covered the porch",
        "while the wind shook the awning",
    ),
    "and marked each corner": (
        "and labeled each edge",
        "and checked each joint",
    ),
    "because the stage was cold": (
        "because the platform felt chilly",
        "because the theatre seemed dim",
    ),
}


def direct_tape(text: str) -> str:
    return normalize_letters(text)


def exact_check_direct(tape: str) -> dict:
    """Independent check one: direct normalized-tape reversal."""
    return {
        "algorithm": "normalized_tape_equals_slice_reverse",
        "exact": bool(tape) and tape == tape[::-1],
        "letters": len(tape),
    }


def exact_check_two_pointer(tape: str) -> dict:
    """Independent check two: pairwise pointers, with no slicing."""
    mismatches = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left": left, "right": right, "left_char": tape[left], "right_char": tape[right]})
        left += 1
        right -= 1
    return {
        "algorithm": "two_pointer_pairwise_comparison",
        "exact": bool(tape) and not mismatches,
        "letters": len(tape),
        "mismatches": mismatches[:8],
        "mismatch_count": len(mismatches),
    }


def first_mismatch(tape: str) -> dict | None:
    for index in range(len(tape) // 2):
        if tape[index] != tape[-index - 1]:
            return {
                "offset": index,
                "left": tape[index],
                "mirrored_right": tape[-index - 1],
                "distance": abs(ord(tape[index]) - ord(tape[-index - 1])),
            }
    if len(tape) % 2 == 0 and tape:
        return None
    return None


def edit_variants(seed: dict) -> list[dict]:
    """Enumerate identity, one-edit, and constrained two-edit scripts."""
    text = seed["text"]
    variants = [{"text": text, "operations": [], "edit_budget": 0}]
    one_edit: list[dict] = []

    for old, replacements in LEXICAL.items():
        if not re.search(rf"\b{re.escape(old)}\b", text, flags=re.I):
            continue
        for new in replacements:
            edited = re.sub(rf"\b{re.escape(old)}\b", new, text, count=1, flags=re.I)
            one_edit.append({"text": edited, "operations": [{"kind": "lexical_substitution", "from": old, "to": new}], "edit_budget": 1})

    for old, replacements in INFLECTIONS.items():
        if not re.search(rf"\b{re.escape(old)}\b", text, flags=re.I):
            continue
        for new in replacements:
            edited = re.sub(rf"\b{re.escape(old)}\b", new, text, count=1, flags=re.I)
            one_edit.append({"text": edited, "operations": [{"kind": "inflection", "from": old, "to": new}], "edit_budget": 1})

    for old, replacements in CLAUSE_EDITS.items():
        if old not in text:
            continue
        for new in replacements:
            edited = text.replace(old, new, 1)
            one_edit.append({"text": edited, "operations": [{"kind": "clause_edit", "from": old, "to": new}], "edit_budget": 1})

    variants.extend(one_edit)
    # A two-edit script is the largest allowed repair.  Do not make a second
    # edit to the same token/phrase: this keeps the frontier interpretable.
    for left, right in combinations(one_edit, 2):
        kinds = {(op["kind"], op["from"]) for op in left["operations"] + right["operations"]}
        if len(kinds) != 2:
            continue
        candidate = right["text"]
        # Reapply the left operation to the right rendering only when its old
        # form remains.  Otherwise the pair would be an accidental no-op.
        op = left["operations"][0]
        if not re.search(rf"\b{re.escape(op['from'])}\b", candidate, flags=re.I) and op["kind"] != "clause_edit":
            continue
        if op["kind"] == "clause_edit" and op["from"] not in candidate:
            continue
        candidate = (re.sub(rf"\b{re.escape(op['from'])}\b", op["to"], candidate, count=1, flags=re.I)
                     if op["kind"] != "clause_edit" else candidate.replace(op["from"], op["to"], 1))
        variants.append({"text": candidate, "operations": left["operations"] + right["operations"], "edit_budget": 2})

    dedup: dict[str, dict] = {}
    for variant in variants:
        dedup.setdefault(variant["text"], variant)
    return list(dedup.values())


def readability_diagnostics(text: str, tape: str) -> dict:
    words = [word.casefold() for word in tokenize(text)]
    try:
        from wordfreq import zipf_frequency
        mean_zipf = sum(zipf_frequency(word, "en") for word in words) / len(words) if words else None
    except Exception:
        mean_zipf = None
    return {
        "diagnostic_only_not_readability": True,
        "word_count": len(words),
        "punctuation_segments": len([part for part in re.split(r"[.!?]+", text) if re.search(r"[A-Za-z]", part)]),
        "mean_zipf_frequency": mean_zipf,
        "repeated_word_rate": (1 - len(set(words)) / len(words)) if words else None,
        "first_mismatch": first_mismatch(tape),
    }


def audit_candidate(seed: dict, variant: dict, baseline_mismatch: int) -> dict:
    text = variant["text"]
    tape = direct_tape(text)
    words = tokenize(text)
    direct = exact_check_direct(tape)
    pointers = exact_check_two_pointer(tape)
    mechanical = mechanical_admission_checks(text, min_letters=30, max_letters=1000)
    mismatch_count = pointers["mismatch_count"]
    improves_frontier = mismatch_count < baseline_mismatch or (
        mismatch_count == baseline_mismatch
        and (first_mismatch(tape) or {}).get("offset", -1) > 0
    )
    clause_count = len(seed["clauses"])
    complete = text.endswith(".") and len(words) >= 8 and text.count(",") >= 1
    return {
        "seed_id": seed["id"],
        "rendered": text,
        "operations": variant["operations"],
        "edit_budget": variant["edit_budget"],
        "provenance": {
            "seed_source": "task-authored intact grammatical prose",
            "seed_id": seed["id"],
            "meaning": seed["meaning"],
            "catalogue_text_used": False,
            "word_order_mirrored": False,
            "clause_count_preserved": clause_count == len(re.findall(r"\b(?:while|and|because)\b", text)) + 1,
        },
        "lengths": {"letters": len(tape), "words": len(words), "characters_including_space": len(text)},
        "exact_check_1": direct,
        "exact_check_2": pointers,
        "independent_exact_agreement": direct["exact"] == pointers["exact"],
        "first_mismatch_constraint": {
            "baseline_mismatch_count": baseline_mismatch,
            "candidate_mismatch_count": mismatch_count,
            "improves_or_ties_frontier": improves_frontier,
            "first_mismatch": first_mismatch(tape),
        },
        "mechanical_admission": mechanical,
        "no_word_order_symmetry": mechanical["not_word_order_symmetry"],
        "no_self_palindromic_units": mechanical["no_self_palindromic_word"] and mechanical["no_self_palindromic_proper_multiword_span"],
        "no_repeated_units": mechanical["no_repeated_nontrivial_unit"] and mechanical["distinct_words"],
        "complete_ordinary_prose_shape": complete,
        "readability_diagnostics": readability_diagnostics(text, tape),
        "mechanically_admitted": all(mechanical.values()) and direct["exact"] and pointers["exact"],
        "reader_eligible": False,
        "failure_evidence": (
            ["not_exact", "not_mechanically_admitted", "not_reader_eligible"]
            if not (direct["exact"] and all(mechanical.values()))
            else ["reader_gate_not_run"]
        ),
        "next_repair": "Expand the held-out lexical alternative at the first mirrored mismatch, then rerun the two-edit budget without changing clause order.",
    }


def novelty_preflight() -> dict:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("entries", [])
    prior = [entry for entry in entries if entry.get("id") != EXPERIMENT]
    collisions = [entry["id"] for entry in prior if entry.get("signature") == SIGNATURE]
    return {
        "registry": str(REGISTRY.relative_to(ROOT)),
        "entries_inspected": len(entries),
        "exact_signature_collisions_before_run": collisions,
        "passed": not collisions,
        "interpretation": "The edit-script state is retained only as a distinct family; near lexical-overlap flags still require human review.",
    }


def main() -> None:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_run']}")
    all_rows = []
    per_seed = []
    for seed in SEEDS:
        base_tape = direct_tape(seed["text"])
        base_audit = exact_check_two_pointer(base_tape)
        variants = edit_variants(seed)
        rows = [audit_candidate(seed, variant, base_audit["mismatch_count"]) for variant in variants]
        # The constraint is live in selection: retain identity, the best
        # one-edit candidate of each operation type, and the best two-edit
        # candidate.  All attempted rows remain under `attempted` below.
        selected = [row for row in rows if row["edit_budget"] == 0]
        for kind in ("lexical_substitution", "inflection", "clause_edit"):
            options = [row for row in rows if row["edit_budget"] == 1 and row["operations"][0]["kind"] == kind]
            if options:
                selected.append(min(options, key=lambda row: (row["exact_check_2"]["mismatch_count"], -len(row["rendered"]))))
        two = [row for row in rows if row["edit_budget"] == 2]
        if two:
            selected.append(min(two, key=lambda row: (row["exact_check_2"]["mismatch_count"], -len(row["rendered"]))))
        all_rows.extend(rows)
        per_seed.append({"seed": seed, "baseline": audit_candidate(seed, variants[0], base_audit["mismatch_count"]), "attempted": rows, "selected_frontier": selected})

    payload = {
        "experiment": EXPERIMENT,
        "signature": SIGNATURE,
        "operator": "bounded letter-level edit script: lexical substitution, explicit inflection, or complete adjunct-clause rewrite; first mirrored mismatch ranks the frontier",
        "novelty_preflight": preflight,
        "seed_count": len(SEEDS),
        "attempted_count": len(all_rows),
        "selected_frontier_count": sum(len(item["selected_frontier"]) for item in per_seed),
        "exact_count": sum(row["exact_check_1"]["exact"] for row in all_rows),
        "independent_exact_agreement_count": sum(row["independent_exact_agreement"] for row in all_rows),
        "mechanically_admitted_count": sum(row["mechanically_admitted"] for row in all_rows),
        "reader_eligible_count": 0,
        "seeds": SEEDS,
        "rendered_candidates": all_rows,
        # Keep the conventional append-only key as well, so the shared
        # readability auditor can consume this artifact without special cases.
        "rendered_probes": all_rows,
        "per_seed": per_seed,
        "failure_evidence": {
            "all_attempts_retained": True,
            "exact_failures": [row["rendered"] for row in all_rows if not row["exact_check_1"]["exact"]],
            "next_repair_operator": "Use a third edit only at the recorded first mismatch: substitute the held-out lemma or inflection in that clause, preserving ordinary clause order and rejecting any repeated or self-palindromic unit.",
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
            "seed_material": "three task-authored complete multi-clause sentences",
            "catalogue_or_corpus_import": False,
            "word_order_symmetry_used": False,
            "self_palindromic_units_used": False,
            "human_readability_status": "unreviewed; diagnostics are not a readability certificate",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("attempted_count", "selected_frontier_count", "exact_count", "mechanically_admitted_count")}))


if __name__ == "__main__":
    main()
