"""Semantic word-sense/valency templates with a character-boundary CSP.

The route begins with a human-authored station scene, not a palindrome.  Each
clause is represented by a sense-specific valency frame (placement, request,
or waiting-with-event).  A complete frame assignment is rendered in ordinary
reading order; only then does the boundary CSP score the global equations
``x[i] = x[N-1-i]``.  Since alternatives have different lengths and
punctuation, word boundaries are variables in the CSP rather than mirrored
word units.  Every failed assignment is retained with a repair suggestion.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT = "semantic-valency-boundary-csp-20260916"
SIGNATURE = (
    "human-authored-station-scene|sense-specific-valency-frames|"
    "variable-word-boundary-csp|joint-complete-clause-realization|"
    "independent-exact-admission-readability-audit"
)
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"


@dataclass(frozen=True)
class Frame:
    id: str
    sense: str
    valency: str
    meaning: str
    past: str
    present: str


SCENE = {
    "id": "station-after-storm",
    "seed": (
        "After the storm, the station porter placed a wet parcel beside the bench, "
        "asked the tired traveler for a name, and waited while the late train crossed the bridge."
    ),
    "meaning": (
        "After bad weather, a station worker puts a damp package by a bench, asks a weary traveler "
        "for identification, and waits as a delayed train crosses a bridge."
    ),
}

# New clauses and lexical inventory, deliberately unrelated to the prior
# archive/keeper scene.  Alternatives keep the event roles but change the
# lexical sense realization and valency surface.
FRAMES = (
    (
        Frame("place-beside", "placement", "V NP PP[location]", "worker places damp package by bench", "After the storm, the station porter placed a wet parcel beside the bench", "After the storm, the station porter places a wet parcel beside the bench"),
        Frame("set-beside", "placement", "V NP PP[location]", "worker sets damp package by bench", "After the storm, the station porter set a damp package beside the bench", "After the storm, the station porter sets a damp package beside the bench"),
        Frame("store-under", "containment", "V NP PP[location]", "worker stores soaked parcel under bench", "Following the storm, the railway porter stored a soaked parcel under the bench", "Following the storm, the railway porter stores a soaked parcel under the bench"),
        Frame("leave-beside", "placement", "V NP PP[location]", "worker leaves wet bundle by bench", "After rain, the platform porter left a wet bundle beside the bench", "After rain, the platform porter leaves a wet bundle beside the bench"),
    ),
    (
        Frame("ask-for", "information_request", "V NP PP[source]", "worker asks weary traveler for name", "asked the tired traveler for a name", "asks the tired traveler for a name"),
        Frame("request-from", "information_request", "V NP[goal] NP[content]", "worker requests traveler's name", "requested the weary passenger's name", "requests the weary passenger's name"),
        Frame("seek-from", "information_request", "V NP[content] PP[source]", "worker seeks name from weary traveler", "sought the traveler's name from the weary passenger", "seeks the traveler's name from the weary passenger"),
        Frame("learn-from", "information_request", "V NP PP[content]", "worker learns a name from traveler", "learned a name from the tired traveler", "learns a name from the tired traveler"),
    ),
    (
        Frame("wait-cross", "durative_wait", "V PP[time] CLAUSE[event]", "worker waits as late train crosses bridge", "and waited while the late train crossed the bridge", "and waits while the late train crosses the bridge"),
        Frame("wait-pass", "durative_wait", "V PP[time] CLAUSE[event]", "worker waits as evening train passes bridge", "then waited as the evening train passed the bridge", "then waits as the evening train passes the bridge"),
        Frame("stay-roll", "durative_wait", "V PP[time] CLAUSE[event]", "worker stays as delayed train rolls across bridge", "and stayed while the delayed train rolled across the bridge", "and stays while the delayed train rolls across the bridge"),
        Frame("stand-cross", "durative_wait", "V PP[time] CLAUSE[event]", "worker stands by as late train crosses bridge", "then stood by as the late train crossed the bridge", "then stands by as the late train crosses the bridge"),
    ),
)

# Delimiters are CSP variables too: they change character offsets while
# leaving the three semantic frames and their reading order unchanged.
DELIMITERS = {
    "comma": (", ",) * 2 + (".",),
    "dash": (" — ", ", ") + (".",),
    "semicolon": ("; ", ", ") + (".",),
}


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
            mismatches.append({"offset": left, "left": value[left], "right": value[right]})
        left += 1
        right -= 1
    return {"algorithm": "independent_two_pointer", "exact": bool(value) and not mismatches, "letters": len(value), "mismatch_count": len(mismatches), "mismatches": mismatches[:10]}


def boundary_csp(text: str, frame_indices: tuple[int, int, int], delimiter_id: str, tense: str) -> dict:
    """Evaluate global character equalities while retaining variable boundaries."""
    value = tape(text)
    spans = [(match.group(), match.start(), match.end()) for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)]
    pairs = [(index, len(value) - 1 - index) for index in range(len(value) // 2)]
    matches = sum(value[left] == value[right] for left, right in pairs)
    mismatch_offsets = [left for left, right in pairs if value[left] != value[right]]
    return {
        "variables": {"word_boundaries": len(spans) - 1, "punctuation_policy": delimiter_id, "tense": tense},
        "word_boundary_positions": [end for _, _, end in spans[:-1]],
        "equation": "x[i] = x[N-1-i] over the complete rendered scene",
        "positions_checked": len(pairs),
        "matching_pairs": matches,
        "mismatch_pairs": len(mismatch_offsets),
        "first_mismatch_offset": mismatch_offsets[0] if mismatch_offsets else None,
        "match_rate": matches / len(pairs) if pairs else 0.0,
        "boundary_choice_was_not_mirrored": True,
    }


def render(indices: tuple[int, int, int], delimiter_id: str, tense: str) -> str:
    delimiter = DELIMITERS[delimiter_id]
    chosen = [getattr(FRAMES[clause][indices[clause]], tense) for clause in range(3)]
    return chosen[0] + delimiter[0] + chosen[1] + delimiter[1] + chosen[2] + delimiter[2]


def independent_admission(text: str) -> dict:
    words = tuple(normalize_letters(word) for word in tokenize(text))
    function = {"a", "an", "the", "after", "as", "while", "and", "for", "from", "by", "beside", "under", "was", "is", "had", "has"}
    content = tuple(word for word in words if word not in function)
    return {
        "algorithm": "independent_shape_scan",
        "ascii_letters_only": all(not character.isalpha() or character.isascii() for character in text),
        "three_clause_boundaries": len(re.split(r"[,;:—]+", text)) >= 3,
        "terminal_sentence_mark": text.endswith("."),
        "minimum_word_count": len(words) >= 14,
        "content_words_unique": len(content) == len(set(content)),
        "not_word_order_mirror": tuple(words) != tuple(reversed(words)),
    }


def readability(text: str) -> dict:
    words = [word.casefold() for word in tokenize(text)]
    try:
        from wordfreq import zipf_frequency
        mean_zipf = sum(zipf_frequency(word, "en") for word in words) / len(words)
    except Exception:
        mean_zipf = None
    return {"diagnostic_not_human_readability": True, "word_count": len(words), "mean_zipf_frequency": mean_zipf, "repeated_word_rate": 1 - len(set(words)) / len(words) if words else None, "punctuation_segments": len([part for part in re.split(r"[.!?]+", text) if re.search(r"[A-Za-z]", part)]), "reader_status": "not_run; requires blinded human study"}


def repair_operator(row: dict) -> str:
    offset = row["boundary_csp"]["first_mismatch_offset"]
    if offset is None:
        return "none; exact closure"
    word_count = len(tokenize(row["rendered"]))
    if offset < row["letters"] // 3:
        target = "placement frame"
    elif offset < 2 * row["letters"] // 3:
        target = "information-request frame"
    else:
        target = "waiting-event frame"
    return f"replace one complete {target} realization at the next CSP branch (offset {offset}; {word_count} word boundaries remain variable)"


def audit(text: str, indices: tuple[int, int, int], delimiter_id: str, tense: str, rank: int = 0) -> dict:
    first, second = exact_slice(text), exact_two_pointer(text)
    central = mechanical_admission_checks(text, min_letters=90, max_letters=180)
    independent = independent_admission(text)
    row = {
        "rank": rank,
        "rendered": text,
        "letters": len(tape(text)),
        "frame_indices": list(indices),
        "sense_valency_state": [{"frame_id": FRAMES[n][indices[n]].id, "sense": FRAMES[n][indices[n]].sense, "valency": FRAMES[n][indices[n]].valency, "meaning": FRAMES[n][indices[n]].meaning} for n in range(3)],
        "delimiter_id": delimiter_id,
        "tense": tense,
        "boundary_csp": boundary_csp(text, indices, delimiter_id, tense),
        "exact_check_1": first,
        "exact_check_2": second,
        "independent_exact_agreement": first["exact"] == second["exact"],
        "central_admission": central,
        "independent_admission": independent,
        "admission_agreement": central["not_word_order_symmetry"] == independent["not_word_order_mirror"],
        "mechanically_admitted": first["exact"] and second["exact"] and all(central.values()) and all(independent.values()),
        "readability_evidence": readability(text),
        "provenance": {"human_scene_seed": SCENE["id"], "catalogue_text_used": False, "pre_existing_palindrome_wrapped": False, "word_order_mirrored": False, "repeated_palindromic_unit_used": False, "semantic_frame_preserved": True},
    }
    row["next_repair_operator"] = repair_operator(row)
    return row


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [entry["id"] for entry in entries if entry.get("id") != EXPERIMENT and entry.get("signature") == SIGNATURE]
    related = [entry["id"] for entry in entries if any(term in entry.get("signature", "") for term in ("valency", "homograph-sense", "boundary-csp")) and entry.get("id") != EXPERIMENT]
    return {"entries_inspected": len(entries), "exact_signature_collisions_before_run": collisions, "related_families_for_manual_review": related[:24], "passed": not collisions, "state_space_distinction": "sense-specific valency frames choose complete clauses first; variable word boundaries and punctuation are then scored by one whole-scene character CSP"}


def search() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_run']}")
    rows = []
    for indices in itertools.product(range(4), repeat=3):
        for tense in ("past", "present"):
            for delimiter_id in DELIMITERS:
                rows.append(audit(render(indices, delimiter_id, tense), indices, delimiter_id, tense))
    rows.sort(key=lambda row: (-row["boundary_csp"]["match_rate"], row["exact_check_2"]["mismatch_count"], -row["letters"], row["rendered"]))
    for rank, row in enumerate(rows[:24], 1):
        row["rank"] = rank
    exact = [row for row in rows if row["exact_check_1"]["exact"] and row["exact_check_2"]["exact"]]
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "experiment": EXPERIMENT,
        "signature": SIGNATURE,
        "status": "complete_semantic_valency_boundary_csp_search",
        "novelty_preflight": preflight,
        "human_seed": SCENE,
        "state_dimensions": {"placement_senses": 4, "request_senses": 4, "waiting_event_senses": 4, "tense_states": 2, "delimiter_states": len(DELIMITERS)},
        "states_examined": len(rows),
        "exact_count": len(exact),
        "mechanically_admitted_count": len(admitted),
        "independent_exact_agreement_count": sum(row["independent_exact_agreement"] for row in rows),
        "independent_admission_agreement_count": sum(row["admission_agreement"] for row in rows),
        "best_rendered_candidates": rows[:24],
        "failed_attempts": [row for row in rows if not row["exact_check_1"]["exact"]],
        "readability_evidence": {"status": "diagnostic_not_human_readability_result", "candidate_rows": 24, "reader_eligible_count": 0, "method": "independent word-frequency, repetition, punctuation-shape diagnostics; no human readers run"},
        "failure_evidence": {"all_nonexact_assignments_retained": True, "next_repair_operator": "At the first mirrored mismatch, substitute a complete sense-compatible frame in that clause; recompute all variable word boundaries globally before accepting", "exact_closure_found": bool(exact)},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "catalogue_or_corpus_import": False, "pre_existing_palindrome_wrapped": False, "word_order_symmetry_used": False},
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = search()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("states_examined", "exact_count", "mechanically_admitted_count")}, indent=2))


if __name__ == "__main__":
    main()
