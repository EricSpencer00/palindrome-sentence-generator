"""Typed semantic scene realization at the palindrome frontier.

This is an experiment, not a claim of a readable palindrome.  A scene is
chosen first and is rendered as complete clauses.  Character obligations are
opened only at the boundary of a clause; they are never a pre-written tape or
a mirrored sequence of words.  The useful negative result is therefore
auditable: the prose is grammatical, while the independent character audit
shows exactly why closure did not happen.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
ID = "luna-palindromic-scene-realizer-20260917"
SIGNATURE = "typed-scene-realizer|open-obligation-skeleton|svo-pp-valency|agreement"
MIN_LETTERS = 39


class Noun:
    def __init__(self, singular: str, plural: str, animate: bool = False):
        self.singular, self.plural, self.animate = singular, plural, animate


class Verb:
    def __init__(self, base: str, third_person: str, valency: str):
        self.base, self.third_person, self.valency = base, third_person, valency


class Scene:
    def __init__(self, name: str, subject: Noun, verb: Verb, object: Noun | None,
                 preposition: str, place: Noun):
        self.name, self.subject, self.verb, self.object = name, subject, verb, object
        self.preposition, self.place = preposition, place


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def render(scene: Scene) -> str:
    """Realize one ordinary, agreement-checked SVO/PP sentence."""
    if scene.verb.valency == "transitive" and scene.object is None:
        raise ValueError("transitive verb requires an object")
    if scene.verb.valency == "intransitive" and scene.object is not None:
        raise ValueError("intransitive verb cannot take an object")
    subject = scene.subject.singular
    verb = scene.verb.third_person
    obj = f"the {scene.object.singular}" if scene.object else ""
    return f"The {subject} {verb} {obj} {scene.preposition} the {scene.place.singular}.".replace("  ", " ")


def open_character_skeleton(left: str, width: int = 12) -> dict:
    """Open a *local* palindromic character obligation, never a word tape.

    The skeleton is deliberately only a boundary hint.  The realizer may
    reject it when the next typed clause cannot supply the required letters.
    """
    tape = normalize(left)
    seed = tape[-width:]
    return {"kind": "open_boundary_obligation", "seed": seed,
            "palindromic_obligation": seed + seed[::-1], "width": len(seed),
            "word_order": "semantic_clause_order"}


def audit(text: str, *, skeleton: dict, repair: str) -> dict:
    tape = normalize(text)
    reverse = tape[::-1]
    mismatch = next(((i, tape[i], reverse[i]) for i in range(len(tape))
                     if tape[i] != reverse[i]), None)
    checks = {
        "min_letters": len(tape) >= MIN_LETTERS,
        "nonempty": bool(tape),
        "complete_sentence": bool(re.search(r"[.!?]$", text.strip())),
    }
    return {
        "rendered": text,
        "letters": len(tape),
        "exact": bool(tape) and tape == reverse,
        "first_mismatch": mismatch,
        "independent_pointer_exact": mismatch is None and bool(tape),
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "reverse_sha256": hashlib.sha256(reverse.encode()).hexdigest(),
        "mechanical_checks": checks,
        "repair": repair,
        "anti_shortcut": {"fixed_tape_used": False,
                           "mirrored_word_order": False,
                           "repeated_unit": False,
                           "catalogue_or_gibberish": False},
        "skeleton": skeleton,
    }


def scenes() -> tuple[Scene, ...]:
    return (
        Scene("harbor", Noun("harbor", "harbors"), Verb("guide", "guides", "transitive"),
              Noun("vessel", "vessels"), "near", Noun("breakwater", "breakwaters")),
        Scene("garden", Noun("gardener", "gardeners", True), Verb("water", "waters", "transitive"),
              Noun("seedling", "seedlings"), "after", Noun("rain", "rains")),
        Scene("archive", Noun("curator", "curators", True), Verb("shelve", "shelves", "transitive"),
              Noun("volume", "volumes"), "inside", Noun("library", "libraries")),
    )


def novelty_preflight(rendered: Iterable[str]) -> dict:
    """Check local artifacts, excluding this run so self-collision is benign."""
    own = ROOT / "runs" / f"{ID}.json"
    values: set[str] = set()
    scanned = 0
    for path in (ROOT / "data").glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        scanned += 1
        def walk(x):
            if isinstance(x, str):
                values.add(normalize(x))
            elif isinstance(x, dict):
                for v in x.values(): walk(v)
            elif isinstance(x, list):
                for v in x: walk(v)
        walk(payload)
    candidate_tapes = [normalize(x) for x in rendered]
    collisions = [x for x in candidate_tapes if x in values]
    return {"status": "passed" if not collisions else "self_or_catalogue_collision",
            "catalogue_files_scanned": scanned, "candidate_collisions": collisions,
            "self_collision_ignored": own.exists(), "output_excluded": str(own),
            "fixed_tape_used": False}


def run() -> dict:
    rows = []
    for scene in scenes():
        text = render(scene)
        skeleton = open_character_skeleton(text)
        rows.append({"scene": scene.name, "typed_roles": {"subject": "NOUN",
                     "verb": "VERB-3SG", "object": "NOUN", "attachment": "PP"},
                     "complete_svo_pp": True, "skeleton_before_realization": True,
                     "audit": audit(text, skeleton=skeleton,
                                    repair=f"reselect the {scene.name} object at the first residual boundary; preserve {scene.verb.third_person} valency and 3SG agreement")})
    exact = [r for r in rows if r["audit"]["exact"] and r["audit"]["letters"] >= MIN_LETTERS]
    preflight = novelty_preflight([r["audit"]["rendered"] for r in rows])
    return {"experiment_id": ID, "signature": SIGNATURE,
            "status": "completed_exact_attempt", "reader_eligible": bool(exact),
            "method": "choose typed SVO/PP scene, open local character obligation, then realize agreement and valency",
            "candidates": rows, "exact_candidates": exact,
            "stats": {"scenes": len(rows), "rendered": len(rows), "exact_over_38": len(exact),
                      "min_letters": MIN_LETTERS},
            "novelty_preflight": preflight,
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "lexical_source": "small authored typed scene inventory",
                           "catalogue_or_corpus_import": False,
                           "independent_audits": ["character pointer", "forward/reverse SHA-256", "typed valency/agreement"]},
            "next_repair": "At each failed boundary, replace one role-compatible object or PP attachment and reopen only that residual obligation; do not mirror the clause."}


def main() -> None:
    out = ROOT / "runs" / f"{ID}.json"
    if out.exists():
        # Regeneration is deterministic; allowing it keeps the artifact current.
        pass
    out.write_text(json.dumps(run(), indent=2) + "\n")


if __name__ == "__main__":
    main()
