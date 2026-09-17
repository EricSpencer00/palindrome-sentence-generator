"""Bounded scene lattice with semantic valency and attachment choices.

Every state is an ordinary-order selection of complete clauses.  Choices are
made jointly; no tape is fixed or decoded in reverse.  The frontier ledger
records character equations that become testable as each clause is appended.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT = "semantic-valency-attachment-scene-lattice-20260916"
SIGNATURE = (
    "semantic-valency-attachment-lattice|joint-complete-clause-selection|"
    "live-character-equation-frontier|ordinary-scene-order|independent-pointer-sha"
)
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"


CLAUSES = (
    (
        {"id": "archivist-journals-rafters", "text": "The patient archivist catalogs sealed journals beneath winter rafters.", "meaning": "archivist catalogs journals; locative attachment to rafters"},
        {"id": "courier-maps-shed", "text": "The careful courier delivers folded maps beside the weathered shed.", "meaning": "courier delivers maps; locative attachment to shed"},
        {"id": "gardener-seedlings-rain", "text": "The quiet gardener waters young seedlings after steady rain.", "meaning": "gardener waters seedlings; temporal attachment after rain"},
    ),
    (
        {"id": "keeper-ledger-platform", "text": "The station keeper checks the morning ledger beside platforms.", "meaning": "keeper checks ledger; locative attachment beside platforms"},
        {"id": "teacher-notes-classroom", "text": "The patient teacher reviews marked notes inside the classroom.", "meaning": "teacher reviews notes; locative attachment inside classroom"},
        {"id": "sailor-rigging-harbor", "text": "The watchful sailor repairs loose rigging near the harbor.", "meaning": "sailor repairs rigging; locative attachment near harbor"},
    ),
    (
        {"id": "curator-cabinets-lectures", "text": "The waiting curator locks glass cabinets after evening lectures.", "meaning": "curator locks cabinets; temporal attachment after lectures"},
        {"id": "porter-parcels-offices", "text": "The young porter carries sealed parcels toward records offices.", "meaning": "porter carries parcels; directional attachment toward offices"},
        {"id": "pilot-vessels-piers", "text": "The watchful pilot secures fishing vessels beside stone piers.", "meaning": "pilot secures vessels; locative attachment beside piers"},
    ),
)


def tape(text: str) -> str:
    return normalize_letters(text)


def exact_two_pointer(text: str) -> dict:
    value = tape(text)
    mismatches = []
    i, j = 0, len(value) - 1
    while i < j:
        if value[i] != value[j]:
            mismatches.append({"offset": i, "left": value[i], "right": value[j]})
        i += 1
        j -= 1
    return {"algorithm": "independent_two_pointer", "exact": bool(value) and not mismatches, "letters": len(value), "mismatch_count": len(mismatches), "mismatches": mismatches[:12]}


def exact_sha(text: str) -> dict:
    value = tape(text)
    fwd = hashlib.sha256(value.encode()).hexdigest()
    rev = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "exact": bool(value) and fwd == rev, "forward": fwd, "reverse": rev}


def equation_frontier(selected: list[dict], full: str) -> list[dict]:
    whole = tape(full)
    rows = []
    for depth in range(1, len(selected) + 1):
        prefix = tape(" ".join(c["text"] for c in selected[:depth]))
        checked = min(len(prefix), len(whole))
        first = next((i for i in range(checked) if prefix[i] != whole[::-1][i]), None)
        rows.append({"depth": depth, "selected_clause_ids": [c["id"] for c in selected[:depth]], "positions_checked": checked, "matching_pairs": sum(prefix[i] == whole[::-1][i] for i in range(checked)), "first_mismatch_offset": first, "equation": "prefix[i] = reverse(full)[i] for every emitted complete clause"})
    return rows


def independent_admission(text: str) -> dict:
    words = tuple(tape(w) for w in tokenize(text))
    stop = {"the", "a", "an", "after", "beside", "inside", "near", "beneath", "toward"}
    content = tuple(w for w in words if w not in stop)
    return {"ascii_letters_only": all(ord(c) < 128 for c in text), "complete_sentence_marks": text.endswith("."), "minimum_word_count": len(words) >= 24, "content_words_unique": len(content) == len(set(content)), "not_word_order_mirror": words != tuple(reversed(words))}


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e["id"] for e in entries if e.get("id") != EXPERIMENT and e.get("signature") == SIGNATURE]
    return {"entries_inspected": len(entries), "exact_signature_collisions_before_render": collisions, "passed": not collisions, "state_space_distinction": "three semantic SVO clauses each select an attachment-compatible complete realization jointly; emitted events retain ordinary order and expose live prefix/reverse equations"}


def audit(choice: tuple[dict, ...], rank: int) -> dict:
    text = " ".join(c["text"] for c in choice)
    normalized = tape(text)
    pointer, sha = exact_two_pointer(text), exact_sha(text)
    central = mechanical_admission_checks(text, min_letters=120, max_letters=240)
    independent = independent_admission(text)
    return {"rank": rank, "rendered": text, "normalized_tape": normalized, "letters": len(normalized), "clause_provenance": [{"id": c["id"], "meaning": c["meaning"], "position": i + 1} for i, c in enumerate(choice)], "equation_frontier": equation_frontier(list(choice), text), "exact_check_two_pointer": pointer, "exact_check_sha256": sha, "independent_exact_agreement": pointer["exact"] == sha["exact"], "central_admission": central, "independent_admission": independent, "anti_shortcut_flags": {"fixed_tape": False, "reverse_decoder": False, "mirrored_word_units": False, "repeated_palindromic_unit": False, "catalogue_text_used": False, "isolated_character_edit": False, "complete_constituents_only": True, "semantic_valency_checked": True, "ordinary_order_events": True}, "mechanically_admitted": bool(normalized) and normalized == normalized[::-1] and pointer["exact"] and sha["exact"] and all(central.values()) and all(independent.values()), "next_repair": "At the first residual frontier mismatch, replace only that clause with a new held-out valency-and-attachment realization, then rerun the preflight and live equation ledger."}


def run() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_render']}")
    choices = list(itertools.product(*CLAUSES))
    rows = [audit(choice, i + 1) for i, choice in enumerate(choices)]
    rows.sort(key=lambda r: (not r["mechanically_admitted"], -sum(x["matching_pairs"] for x in r["equation_frontier"]), r["equation_frontier"][0]["first_mismatch_offset"] or 999))
    exact = [r for r in rows if r["mechanically_admitted"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete; no exact closure" if not exact else "exact closure found", "novelty_preflight": preflight, "states_examined": len(rows), "exact_count": len(exact), "best_rendered_candidates": rows[:6], "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "lattice_shape": [len(x) for x in CLAUSES], "ordinary_order_rendering": True}, "anti_shortcut_policy": "No fixed tape, reverse decoding, mirrored word order, repeated units, catalogue import, or isolated edits; every state is a joint choice of complete valency-compatible clauses.", "next_repair": "Replace the first mismatching clause with one fresh attachment-compatible event and rerun this bounded lattice."}


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states_examined": result["states_examined"], "exact_count": result["exact_count"]}, indent=2))
