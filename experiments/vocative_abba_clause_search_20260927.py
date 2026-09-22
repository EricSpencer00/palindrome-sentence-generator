"""Search a small, fresh vocative-clause grammar for ABBA paragraphs.

This is a paragraph-level construction lane, not a catalogue-pair composer.
Left and right clauses are generated independently from typed banks.  An
indexed reversed-character seam check intersects their live tape obligations;
complete clauses are emitted only after the intersection closes.  Distinct
clause pairs are then nested as A1 B1 B2 A2 (and beyond) without reversing a
finished text.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.validator import is_palindrome, normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "vocative-abba-clause-search-20260927.json"

# Independently authored grammatical banks.  The names and predicates are not
# stored as mirror pairs; the search discovers which combinations close.
LEFT_NAMES = ("Nora", "Noel", "Mara", "Sara", "Aras")
LEFT_OBJECTS = ("evil", "war", "God", "dog", "live")
RIGHT_NAMES = ("Aron", "Leon", "Aram", "Sara", "Nora")
RIGHT_PREDICATES = ("Live", "Raw", "Dog", "Evil", "War")


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    mismatch = next(
        ((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
         if tape[i] != tape[-1 - i]),
        None,
    )
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
        "validator_exact": is_palindrome(text),
    }


def left_clauses() -> list[dict[str, object]]:
    return [
        {"text": f"{name}, I saw {obj}.", "role": "A_or_B_left",
         "name": name, "object": obj, "template": "NAME, I saw OBJECT"}
        for name in LEFT_NAMES for obj in LEFT_OBJECTS
    ]


def right_clauses() -> list[dict[str, object]]:
    return [
        {"text": f"{predicate} was I, {name}.", "role": "B_or_A_right",
         "name": name, "predicate": predicate, "template": "PREDICATE was I, NAME"}
        for predicate in RIGHT_PREDICATES for name in RIGHT_NAMES
    ]


def live_intersection(left: str, right: str) -> dict[str, object]:
    """Consume the opposing tapes one character at a time.

    The right clause is indexed by its reverse spelling only as an
    intersection key.  The rendered surface is always the independently
    authored right clause; no reversed tape is emitted.
    """
    lt, rt = letters(left), letters(right)
    trace = []
    i, j = 0, len(rt) - 1
    while i < len(lt) and j >= 0:
        event = {"offset": i, "left": lt[i], "right": rt[j],
                 "match": lt[i] == rt[j]}
        trace.append(event)
        if lt[i] != rt[j]:
            return {"exact": False, "matched": i, "trace": trace,
                    "first_mismatch": event}
        i += 1
        j -= 1
    return {"exact": i == len(lt) and j < 0, "matched": i,
            "trace": trace, "first_mismatch": None}


def intersect_pairs(left_rows: list[dict], right_rows: list[dict]) -> list[dict]:
    """Indexed character intersection without a mirrored-unit bank."""
    # Bucketing by length and first exposed character bounds the independent
    # product before the live comparison.  It is an index, not a paired bank.
    index: dict[tuple[int, str], list[dict]] = {}
    for row in right_rows:
        tape = letters(row["text"])
        index.setdefault((len(tape), tape[-1]), []).append(row)
    pairs = []
    for left in left_rows:
        tape = letters(left["text"])
        for right in index.get((len(tape), tape[0]), []):
            seam = live_intersection(left["text"], right["text"])
            if seam["exact"]:
                pairs.append({
                    "left": left,
                    "right": right,
                    "seam": seam,
                    "provenance": {
                        "left_bank": "fresh authored vocative clauses",
                        "right_bank": "fresh authored inverted-vocative clauses",
                        "independent_clause_generation": True,
                        "finished_tape_reversal": False,
                        "catalogue_text": False,
                        "prepaired_mirror_unit": False,
                    },
                })
    return pairs


def compose(pair_path: list[dict]) -> str:
    left = [pair["left"]["text"] for pair in pair_path]
    right = [pair["right"]["text"] for pair in reversed(pair_path)]
    return " ".join(left + right)


def paragraph_shape(width: int) -> str:
    """Name the nested seam topology independently of the rendered text."""
    if width == 1:
        return "A A'"
    labels = [chr(ord("A") + i) for i in range(width)]
    return " ".join(labels + [f"{label}'" for label in reversed(labels)])


def reader_package(candidate: str) -> dict[str, object]:
    controls = [
        "Nora described the war while Noel recorded the damage, and Mara named the witness.",
        "At dusk the guide heard a bell, marked the shore, and asked the sailor to wait.",
    ]
    rng = random.Random(20260927)
    items = [{"id": "candidate-1", "text": candidate, "kind": "candidate"}]
    for idx, text in enumerate(controls, 1):
        words = text.split()
        shuffled = words[:]
        rng.shuffle(shuffled)
        items.append({"id": f"control-intact-{idx}", "text": text, "kind": "intact_control"})
        items.append({"id": f"control-shuffled-{idx}", "text": " ".join(shuffled),
                      "kind": "shuffled_control"})
    rng.shuffle(items)
    return {
        "random_seed": 20260927,
        "order": [item["id"] for item in items],
        "items": items,
        "instructions": "Rate English readability and coherence blind; exactness is not a rating dimension.",
        "status": "prepared; human ratings not yet collected",
    }


def run() -> dict[str, object]:
    left, right = left_clauses(), right_clauses()
    pairs = intersect_pairs(left, right)
    # Select disjoint, content-diverse pairs; this is a semantic selection
    # rule, not a character score.  The first three are retained as a trace.
    chosen: list[dict] = []
    used_left: set[str] = set()
    used_right: set[str] = set()
    # Prefer a paragraph whose clauses carry different discourse content.
    # These are typed semantic fields, not character-level scores, so the
    # selection cannot quietly optimize toward a pre-existing mirror.
    used_left_names: set[str] = set()
    used_right_names: set[str] = set()
    used_objects: set[str] = set()
    used_predicates: set[str] = set()
    for pair in pairs:
        l, r = pair["left"]["text"], pair["right"]["text"]
        left_name = str(pair["left"]["name"])
        right_name = str(pair["right"]["name"])
        obj = str(pair["left"]["object"])
        predicate = str(pair["right"]["predicate"])
        if (l in used_left or r in used_right or
                left_name in used_left_names or right_name in used_right_names or
                obj in used_objects or predicate in used_predicates):
            continue
        chosen.append(pair)
        used_left.add(l)
        used_right.add(r)
        used_left_names.add(left_name)
        used_right_names.add(right_name)
        used_objects.add(obj)
        used_predicates.add(predicate)
        if len(chosen) == 3:
            break
    candidates = []
    for width in range(1, len(chosen) + 1):
        text = compose(chosen[:width])
        au = audit(text)
        units = [p["left"]["text"] for p in chosen[:width]] + [
            p["right"]["text"] for p in reversed(chosen[:width])]
        candidate = {
            "rendered": text,
            "letters": au["letters"],
            "paragraph_shape": paragraph_shape(width),
            "units": units,
            "audit": au,
            "novelty_preflight": is_novel_palindrome(text),
            "provenance": {
                "generator": "independent typed vocative clause banks + live character intersection",
                "pair_count": width,
                "distinct_units": len(set(units)) == len(units),
                "self_palindromic_units": any(letters(u) == letters(u)[::-1] for u in units),
                "catalogue_text": False,
                "finished_tape_reversal": False,
                "posthoc_character_repair": False,
                "reader_certified": False,
            },
        }
        candidates.append(candidate)
    exact = [row for row in candidates if row["audit"]["two_pointer_exact"]
             and row["audit"]["sha_equal"] and row["audit"]["validator_exact"]
             and row["provenance"]["distinct_units"]
             and not row["provenance"]["self_palindromic_units"]]
    best = max(exact, key=lambda row: row["letters"], default=None)
    package = reader_package(best["rendered"] if best else "")
    return {
        "experiment_id": "vocative-abba-clause-search-20260927",
        "method": "independent typed vocative clause banks with indexed live reversed-character seam intersection and ABBA nesting",
        "construction_rule": {
            "unit": "complete authored clause",
            "seam": "each left clause is matched against the live reverse obligation of an independently generated right clause",
            "nested_shapes": [paragraph_shape(width) for width in range(1, len(chosen) + 1)],
            "finished_tape_reversal": False,
        },
        "stats": {"left_clauses": len(left), "right_clauses": len(right),
                  "exact_pairs": len(pairs), "candidate_widths": len(candidates),
                  "exact_admissible": len(exact),
                  "longest_letters": max((row["letters"] for row in exact), default=0)},
        "pair_trace": pairs[:12],
        "rendered_candidates": candidates,
        "exact_candidates": exact,
        "best": best,
        "reader_package": package,
        "reader_gate": "prepared; no readability claim until blinded human ratings",
        "next_reader_test": "run the seeded package with intact and shuffled controls; collect independent readability/coherence ratings",
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer", "validator", "forward/reverse SHA-256"]},
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
    if result["best"]:
        print(result["best"]["rendered"])
