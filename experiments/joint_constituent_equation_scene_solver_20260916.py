"""Bounded bilateral scene search with complete constituent equations.

Each side is authored as ordinary English from complete S-V-O-adjunct
constituents.  The solver chooses one constituent at a time on both sides and
updates a prefix/suffix character equation before rendering the whole scene.
It never reverse-renders, copies a tape, or treats a word as a mirrored unit.
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

EXPERIMENT = "joint-constituent-equation-scene-solver-20260916"
SIGNATURE = (
    "bilateral-complete-constituent-scene|paired-svo-adjunct-selection|"
    "ordinary-right-scene-order|constituent-frontier-equation|"
    "independent-pointer-sha-audit"
)
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"


@dataclass(frozen=True)
class Constituent:
    id: str
    subject: str
    verb: str
    object: str
    adjunct: str
    meaning: str

    @property
    def text(self) -> str:
        return f"{self.subject} {self.verb} {self.object} {self.adjunct}."


LEFT = (
    Constituent("clerk-manifest", "The harbor clerk", "records", "cargo manifests", "before sunrise", "clerk records cargo manifests before sunrise"),
    Constituent("keeper-ledger", "The station keeper", "checks", "the morning ledger", "beside platforms", "keeper checks morning ledger beside platforms"),
    Constituent("guide-gallery", "The museum guide", "opens", "the west gallery", "after visitors arrive", "guide opens west gallery after visitors arrive"),
)
RIGHT = (
    Constituent("pilot-vessel", "The watchful pilot", "secures", "fishing vessels", "beside stone piers", "pilot secures fishing vessels beside stone piers"),
    Constituent("curator-cabinet", "The waiting curator", "locks", "glass cabinets", "after evening lectures", "curator locks glass cabinets after evening lectures"),
    Constituent("porter-parcel", "The young porter", "carries", "sealed parcels", "toward records offices", "porter carries sealed parcels toward records offices"),
)


def tape(text: str) -> str:
    return normalize_letters(text)


def exact_two_pointer(text: str) -> dict:
    value = tape(text)
    mismatches = []
    left, right = 0, len(value) - 1
    while left < right:
        if value[left] != value[right]:
            mismatches.append({"offset": left, "left": value[left], "right": value[right]})
        left += 1
        right -= 1
    return {"algorithm": "independent_two_pointer", "exact": bool(value) and not mismatches, "letters": len(value), "mismatch_count": len(mismatches), "mismatches": mismatches[:12]}


def exact_sha(text: str) -> dict:
    value = tape(text)
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "exact": bool(value) and forward == reverse, "forward": forward, "reverse": reverse}


def equation_frontier(left_parts: list[str], right_parts: list[str]) -> dict:
    """Carry x[i] = x[N-1-i] for complete constituent additions."""
    left = tape(" ".join(left_parts))
    right = tape(" ".join(right_parts))
    obligation = right[::-1]
    checked = min(len(left), len(obligation))
    matches = sum(left[i] == obligation[i] for i in range(checked))
    first = next((i for i in range(checked) if left[i] != obligation[i]), None)
    return {
        "left_letters": len(left), "right_letters": len(right),
        "positions_checked": checked, "matching_pairs": matches,
        "mismatch_pairs": checked - matches, "first_mismatch_offset": first,
        "left_prefix": left[-24:], "right_reverse_obligation": obligation[:24],
        "equation": "x[i] = x[N-1-i] across complete constituent yields",
    }


def render(left_indices: tuple[int, int, int], right_indices: tuple[int, int, int]) -> str:
    left = [LEFT[i].text for i in left_indices]
    right = [RIGHT[i].text for i in right_indices]
    return " ".join(left + right)


def independent_admission(text: str) -> dict:
    words = tuple(normalize_letters(w) for w in tokenize(text))
    content = tuple(w for w in words if w not in {"the", "a", "an", "before", "beside", "after", "toward", "the"})
    return {
        "ascii_letters_only": all(not c.isalpha() or c.isascii() for c in text),
        "complete_sentence_marks": text.endswith("."),
        "minimum_word_count": len(words) >= 24,
        "content_words_unique": len(content) == len(set(content)),
        "not_word_order_mirror": words != tuple(reversed(words)),
    }


def audit(left_indices: tuple[int, int, int], right_indices: tuple[int, int, int], rank: int) -> dict:
    text = render(left_indices, right_indices)
    left_parts = [LEFT[i].text for i in left_indices]
    right_parts = [RIGHT[i].text for i in right_indices]
    direct = tape(text)
    pointer = exact_two_pointer(text)
    sha = exact_sha(text)
    central = mechanical_admission_checks(text, min_letters=90, max_letters=220)
    independent = independent_admission(text)
    trace = []
    for depth in range(2):
        trace.append({
            "depth": depth + 1,
            "left_constituent": LEFT[left_indices[depth]].id,
            "right_constituent": RIGHT[right_indices[1 - depth]].id,
            "left_meaning": LEFT[left_indices[depth]].meaning,
            "right_meaning": RIGHT[right_indices[1 - depth]].meaning,
            "frontier": equation_frontier(left_parts[: depth + 1], right_parts[2 - depth:]),
        })
    return {
        "rank": rank, "rendered": text, "normalized_tape": direct,
        "letters": len(direct), "left_indices": list(left_indices), "right_indices": list(right_indices),
        "constituent_provenance": [{"side": "left", "id": LEFT[i].id, "meaning": LEFT[i].meaning} for i in left_indices]
        + [{"side": "right", "id": RIGHT[i].id, "meaning": RIGHT[i].meaning} for i in right_indices],
        "equation_frontier": trace,
        "exact_check_direct": {"algorithm": "normalized_tape_reverse_slice", "exact": bool(direct) and direct == direct[::-1]},
        "exact_check_two_pointer": pointer,
        "exact_check_sha256": sha,
        "independent_exact_agreement": (bool(direct) and direct == direct[::-1]) == pointer["exact"] == sha["exact"],
        "central_admission": central, "independent_admission": independent,
        "anti_shortcut_flags": {"fixed_tape": False, "reverse_decoder": False, "mirrored_word_units": False, "repeated_palindromic_unit": False, "catalogue_text_used": False, "complete_constituents_only": True},
        "mechanically_admitted": bool(direct) and direct == direct[::-1] and pointer["exact"] and sha["exact"] and all(central.values()) and all(independent.values()),
        "next_repair": "At the first mirrored mismatch, replace the paired complete constituent on both sides with a new role-compatible realization, then recompute all later frontiers; do not edit an isolated character.",
    }


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e["id"] for e in entries if e.get("id") != EXPERIMENT and e.get("signature") == SIGNATURE]
    related = [e["id"] for e in entries if any(k in e.get("signature", "") for k in ("complete-constituent", "constituent-frontier", "paired-svo")) and e.get("id") != EXPERIMENT]
    return {"entries_inspected": len(entries), "exact_signature_collisions_before_render": collisions, "related_families_for_review": related[:20], "passed": not collisions, "state_space_distinction": "independently authored complete SVO-adjunct constituents are selected jointly while a reversible prefix/suffix character equation is carried at constituent boundaries; both sides remain normal-order prose"}


def search() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_render']}")
    rows = [audit(left, right, 0) for left in itertools.permutations(range(3), 2) for right in itertools.permutations(range(3), 2)]
    rows.sort(key=lambda r: (not all(v for k, v in r["central_admission"].items() if k != "exact_letter_palindrome") or not all(r["independent_admission"].values()), -sum(t["frontier"]["matching_pairs"] for t in r["equation_frontier"]), r["equation_frontier"][0]["frontier"]["mismatch_pairs"], -r["letters"]))
    for rank, row in enumerate(rows[:9], 1): row["rank"] = rank
    exact = [r for r in rows if r["exact_check_direct"]["exact"]]
    return {
        "experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete; no exact closure" if not exact else "exact closure found",
        "novelty_preflight": preflight, "states_examined": len(rows), "exact_count": len(exact),
        "best_rendered_candidates": rows[:9], "failed_attempts": len(rows) - len(exact),
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "left_bank": [x.id for x in LEFT], "right_bank": [x.id for x in RIGHT], "ordinary_order_rendering": True},
        "anti_shortcut_policy": "No fixed tape, reverse decoding, mirrored units, catalogue import, or isolated-character edits; every state is a joint choice of complete grammatical constituents.",
        "next_repair": "Use the best scene's first mismatch to author one held-out role-compatible constituent pair, preserving ordinary word order and recomputing the complete equation ledger.",
    }


def main() -> None:
    if OUT.exists(): raise SystemExit(f"output already exists: {OUT}")
    OUT.write_text(json.dumps(search(), indent=2) + "\n")
    print(json.dumps({k: json.loads(OUT.read_text())[k] for k in ("states_examined", "exact_count", "failed_attempts")}, indent=2))


if __name__ == "__main__": main()
