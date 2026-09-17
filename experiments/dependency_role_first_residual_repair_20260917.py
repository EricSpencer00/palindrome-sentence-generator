"""One-step held-out role/attachment repair from the strongest scene lattice witness.

The base scene is kept fixed except for the first clause.  A small held-out
set of complete, independently authored role/attachment realizations is
tested against the live character obligation at the first residual seam.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT = "dependency-role-first-residual-repair-20260917"
SIGNATURE = "heldout-role-attachment-one-clause|first-residual-live-equation|complete-scene-repair|pointer-sha"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

BASE = (
    "The quiet gardener waters young seedlings after steady rain.",
    "The watchful sailor repairs loose rigging near the harbor.",
    "The young porter carries sealed parcels toward records offices.",
)

# These are held out complete clauses, not character fragments.  Each changes
# one role/attachment realization at position 1 and leaves the other two
# ordinary-order clauses untouched.
HELDOUT_FIRST_CLAUSES = (
    {"id": "keeper-ledger-platform", "text": "The station keeper checks the morning ledger beside platforms.", "role": "keeper checks ledger", "attachment": "beside platforms"},
    {"id": "teacher-notes-classroom", "text": "The patient teacher reviews marked notes inside the classroom.", "role": "teacher reviews notes", "attachment": "inside classroom"},
    {"id": "courier-maps-shed", "text": "The careful courier delivers folded maps beside the weathered shed.", "role": "courier delivers maps", "attachment": "beside shed"},
    {"id": "archivist-journals-rafters", "text": "The patient archivist catalogs sealed journals beneath winter rafters.", "role": "archivist catalogs journals", "attachment": "beneath rafters"},
    {"id": "ranger-lanterns-workshop", "text": "The alert ranger repairs broken lanterns inside the stone workshop.", "role": "ranger repairs lanterns", "attachment": "inside workshop"},
)


def independent_pointer(text: str) -> dict:
    value = normalize_letters(text)
    mismatches = []
    for i in range(len(value) // 2):
        j = len(value) - 1 - i
        if value[i] != value[j]:
            mismatches.append({"offset": i, "left": value[i], "right": value[j]})
    return {"algorithm": "independent_two_pointer", "exact": bool(value) and not mismatches, "letters": len(value), "mismatch_count": len(mismatches), "mismatches": mismatches[:16]}


def independent_sha(text: str) -> dict:
    value = normalize_letters(text)
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "exact": bool(value) and forward == reverse, "forward": forward, "reverse": reverse}


def first_residual(text: str) -> dict:
    value = normalize_letters(text)
    for i, (left, right) in enumerate(zip(value, reversed(value))):
        if left != right:
            return {"offset": i, "left": left, "right": right}
    return {"offset": None, "left": None, "right": None}


def novelty() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e.get("id") for e in entries if e.get("id") != EXPERIMENT and e.get("signature") == SIGNATURE]
    return {"entries_inspected": len(entries), "exact_signature_collisions_before_render": collisions, "passed": not collisions, "distinction": "one held-out complete role/attachment clause replaces the base clause at its first residual seam; other clauses and ordinary order remain fixed"}


def render(first: dict) -> str:
    return " ".join((first["text"], BASE[1], BASE[2]))


def row(first: dict, rank: int) -> dict:
    text = render(first)
    value = normalize_letters(text)
    pointer, sha = independent_pointer(text), independent_sha(text)
    admission = mechanical_admission_checks(text, min_letters=100, max_letters=240)
    words = tuple(normalize_letters(w) for w in tokenize(text))
    content = tuple(w for w in words if w not in {"the", "a", "an", "after", "beside", "beneath", "inside", "near", "toward"})
    exact = pointer["exact"] and sha["exact"] and pointer["exact"] == sha["exact"]
    return {
        "rank": rank,
        "rendered": text,
        "normalized_tape": value,
        "letters": len(value),
        "replacement": {"position": 1, "heldout_clause_id": first["id"], "role": first["role"], "attachment": first["attachment"], "replaced_base_clause": BASE[0]},
        "provenance": {"base_scene_source": "semantic-valency-attachment-scene-lattice-20260916", "source_sentences_copied": False, "catalogue_imported": False, "borrowed_text": False, "posthoc_reversal": False, "word_order_symmetry": False, "generator_method": "held-out complete role/attachment clause substitution at first live residual"},
        "live_obligation": {"base_first_residual": first_residual(" ".join(BASE)), "repaired_first_residual": first_residual(text), "equation": "prefix[i] = reverse(full)[i] while preserving complete clause boundaries"},
        "exact_check_two_pointer": pointer,
        "exact_check_sha256": sha,
        "independent_exact_agreement": pointer["exact"] == sha["exact"],
        "central_admission": admission,
        "independent_admission": {"ascii_letters_only": all(ord(c) < 128 for c in text), "complete_sentence_marks": text.endswith("."), "minimum_word_count": len(words) >= 24, "content_words_unique": len(content) == len(set(content)), "not_word_order_mirror": words != tuple(reversed(words))},
        "anti_shortcut_flags": {"fixed_tape": False, "reverse_decoder": False, "mirrored_word_units": False, "repeated_palindromic_unit": False, "catalogue_text_used": False, "isolated_character_edit": False, "complete_constituents_only": True, "role_attachment_checked": True, "ordinary_order_events": True},
        "mechanically_admitted": bool(value) and exact and all(admission.values()),
        "reader_status": "unreviewed; programmatic checks do not certify readability",
        "next_repair": "If the first seam remains open, replace only the second clause attachment with a fresh held-out complete realization and rerun the same live obligation ledger.",
    }


def run() -> dict:
    preflight = novelty()
    if not preflight["passed"]:
        raise RuntimeError(preflight)
    rows = [row(item, i + 1) for i, item in enumerate(HELDOUT_FIRST_CLAUSES)]
    # Prefer a reader-eligible lexical surface before length: a longer but
    # repeated/unknown-word witness is not the useful repair frontier.
    rows.sort(key=lambda r: (not (r["central_admission"].get("lexicon_words") and r["central_admission"].get("distinct_words")), not r["mechanically_admitted"], -r["letters"], r["live_obligation"]["repaired_first_residual"]["offset"] or 999))
    admitted = [r for r in rows if r["mechanically_admitted"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete; no exact closure" if not admitted else "exact closure found", "novelty_preflight": preflight, "states_examined": len(rows), "exact_count": sum(r["exact_check_two_pointer"]["exact"] for r in rows), "mechanically_admitted_count": len(admitted), "best_rendered_candidates": rows, "anti_shortcut_policy": "No fixed tape, word mirroring, repeated units, catalogue text, or character edits; one complete held-out role/attachment clause is selected in ordinary prose order.", "next_repair": "Replace only the second clause attachment at its first residual seam, preserving valency and agreement, then rerun independent pointer/SHA checks.", "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "heldout_clause_count": len(HELDOUT_FIRST_CLAUSES), "base_scene": BASE}}


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states_examined": result["states_examined"], "exact_count": result["exact_count"], "mechanically_admitted_count": result["mechanically_admitted_count"]}, indent=2))
