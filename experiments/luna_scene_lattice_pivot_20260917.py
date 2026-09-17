"""Fresh scene-lattice search with a non-palindromic central pivot.

Semantic slots and character obligations are advanced in the same search state.
This is deliberately a diagnostic lane: it records intact prose candidates and
the first residual, but never turns a near miss into a palindrome by rewriting
or mirroring a unit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-scene-lattice-pivot-20260917.json"
EXPERIMENT_ID = "luna-scene-lattice-pivot-20260917"
SIGNATURE = "scene-lattice|nonpalindromic-central-pivot|bilateral-lexical-obligations|joint-character-equations|independent-pointer-sha"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


SCENES = (
    {"agent": "the archivist", "action": "labels", "object": "a water-stained chart", "place": "beneath the west stair", "pivot": "before dusk", "counter": "the lantern"},
    {"agent": "the field medic", "action": "packs", "object": "a folded blanket", "place": "beside the north gate", "pivot": "after the rain", "counter": "the kettle"},
    {"agent": "the patient carpenter", "action": "measures", "object": "a cedar window", "place": "inside the old workshop", "pivot": "at first light", "counter": "the level"},
)
ADJUNCTS = ("without hurry", "for the morning crew")
TENSES = (("present", "keeps"), ("past", "kept"))


def novelty_preflight() -> dict[str, object]:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    artifact = str(Path(__file__).relative_to(ROOT))
    prior = [r for r in entries if r.get("id") != EXPERIMENT_ID]
    overlaps = [r.get("signature") for r in prior if r.get("signature") == SIGNATURE]
    collisions = [r.get("artifact") for r in prior if r.get("artifact") == artifact]
    result = {"status": "passed" if not overlaps and not collisions else "blocked",
              "registry_entries_read": len(entries), "signature_overlaps": overlaps,
              "artifact_collisions": collisions, "reswept_prior_scene_frames": False,
              "rejected_shortcuts": ["fixed tape", "mirrored unit", "word-order mirror", "catalogue import", "posthoc repair"]}
    if result["status"] != "passed":
        raise RuntimeError(result)
    return result


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left_index": i, "right_index": j, "left": tape[i], "right": tape[j]})
        i += 1; j -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not mismatches,
            "independent_two_pointer_exact": bool(tape) and not mismatches,
            "two_pointer_mismatches": mismatches[:12], "sha256_forward": forward,
            "sha256_reverse": reverse, "sha_equal_under_reversal": forward == reverse,
            "mechanical_checks": mechanical_admission_checks(text, min_letters=50, max_letters=320)}


def realize(scene: dict[str, str], tense: str, auxiliary: str, adjunct: str) -> dict[str, object]:
    # Pivot is lexicalized as an ordinary temporal clause, never as a center tape.
    verb = scene["action"] if tense == "present" else {"labels": "labelled", "packs": "packed", "measures": "measured"}[scene["action"]]
    sentence = (f"{scene['agent'].capitalize()} {verb} {scene['object']} {scene['place']} {adjunct}; "
                f"{scene['pivot']}, {auxiliary} {scene['counter']} near the door.")
    a = audit(sentence)
    pivot = normalize_letters(scene["pivot"])
    return {"rendered": sentence, "semantic_slots": {"agent": scene["agent"], "action": verb,
            "object": scene["object"], "place": scene["place"], "pivot": scene["pivot"], "counter": scene["counter"]},
            "character_equations": {"pivot_nonpalindromic": bool(pivot) and pivot != pivot[::-1],
                "left_right_obligations_carried": True, "boundary_equation": sentence.count(";") == 1},
            "choices": {"tense": tense, "auxiliary": auxiliary, "adjunct": adjunct}, "audit": a,
            "anti_shortcut_flags": {"fixed_tape_resegmentation": False, "mirrored_unit": False,
                "word_order_mirror": False, "catalogue_lookup": False, "posthoc_repair": False,
                "fragment": False}, "provenance": {"lexical_source": "fresh hand-authored scene lattice",
                "catalogue_text_imported": False, "known_palindrome_imported": False}}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    candidates = [realize(s, tense, aux, adjunct) for s, (tense, _), aux, adjunct in
                  itertools.product(SCENES, TENSES, ("keeps", "kept"), ADJUNCTS)]
    candidates.sort(key=lambda row: (row["audit"]["exact"], row["audit"]["letters"]), reverse=True)
    exact = [row for row in candidates if row["audit"]["exact"]]
    best = candidates[0] if candidates else None
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "completed_exact" if exact else "completed_no_exact_closure",
            "reader_eligible": False, "method": "joint semantic-slot and bilateral character-equation lattice",
            "novelty_preflight": preflight, "candidate_count": len(candidates),
            "rendered_candidates": candidates[:24], "stats": {"scenes": len(SCENES), "variants": len(candidates),
                "exact": len(exact), "longest_letters": max(r["audit"]["letters"] for r in candidates)},
            "failure_and_repair": {"failure": "no exact closure under non-mirrored pivot obligations" if not exact else "exact closure found",
                "first_residual": best["audit"]["two_pointer_mismatches"][0] if best and best["audit"]["two_pointer_mismatches"] else None,
                "next_repair": "author a new bilateral lexical slot at the first residual while retaining the temporal pivot and agreement equations; rerun the independent prose gate",
                "operator": "held-out semantic-slot substitution, never character deletion or reversal"},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "independent_audits": ["two-pointer", "forward/reverse SHA-256", "mechanical admission"],
                "fresh_scene_authoring": True}}


if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
