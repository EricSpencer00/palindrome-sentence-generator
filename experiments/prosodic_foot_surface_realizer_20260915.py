"""Prosody-constrained surface realization for exact palindrome search.

This route changes the construction state rather than merely changing a beam:
each complete clause carries a CMU pronunciation, syllable count, lexical
stress contour, and a phrase-boundary obligation while its character yield is
indexed for an independent reverse join.  Stress is a construction filter and
ranking signal only; it cannot certify that a rendered candidate is readable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from nltk.corpus import cmudict

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
ID = "prosodic-foot-surface-realizer"
SIGNATURE = (
    "prosodic-foot-lattice|pronouncing-dictionary-stress-constraints|"
    "metrical-clause-realization|paired-character-yield-equation|"
    "typed-event-roles|stress-frontier-repair"
)
SEED_CONTENT = frozenset("aide rips nine memos some men inspire diana".split())
MIN_LETTERS = 39
MIN_CLAUSE_LETTERS = 16


@dataclass(frozen=True)
class Lexeme:
    text: str
    role: str
    number: str
    syllables: int
    stress: str


@dataclass(frozen=True)
class Clause:
    words: tuple[Lexeme, ...]
    frame: str
    stress: str
    syllables: int
    boundary: str

    @property
    def text(self) -> str:
        return " ".join(item.text for item in self.words)

    @property
    def tape(self) -> str:
        return "".join(item.text for item in self.words)

    @property
    def content(self) -> frozenset[str]:
        return frozenset(item.text for item in self.words if item.role in {"agent", "verb", "object", "adj"})


def normalize(text: str) -> str:
    return "".join(char.lower() for char in text if "a" <= char.lower() <= "z")


def independent_audit(text: str) -> dict:
    tape = normalize(text)
    return {
        "tape": tape,
        "letters": len(tape),
        "exact": bool(tape) and tape == tape[::-1],
        "two_pointer": bool(tape) and all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
    }


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text())


def novelty_preflight() -> dict:
    data = load_registry()
    rows = data.get("entries", [])
    excluded = data.get("excluded", [])
    self_rows = [row for row in rows if row.get("id") == ID]
    other_rows = [row for row in rows if row.get("id") != ID]
    if any(row.get("signature") == SIGNATURE for row in other_rows + excluded):
        raise RuntimeError("prosodic signature collides with another registered or excluded route")
    if any(row.get("id") == ID or row.get("signature") == SIGNATURE for row in excluded):
        raise RuntimeError("prosodic route is explicitly excluded")
    return {
        "registry_entries_read_before_run": len(rows),
        "excluded_routes_read_before_run": len(excluded),
        "signature_overlap": [],
        "self_entry_present": [ID] if self_rows else [],
        "replay_of_registered_family": bool(self_rows),
        "repair_of_registered_family": bool(self_rows),
    }


def _pronunciations() -> dict[str, tuple[int, str]]:
    """Choose a deterministic CMU pronunciation and retain lexical stress."""
    table: dict[str, tuple[int, str]] = {}
    for word, variants in cmudict.dict().items():
        clean = word.casefold()
        if not re.fullmatch(r"[a-z]+", clean) or not variants:
            continue
        chosen = next((variant for variant in variants if any(phone[-1] in "012" for phone in variant)), None)
        if chosen is None:
            continue
        stress = "".join(phone[-1] for phone in chosen if phone[-1] in "012")
        table.setdefault(clean, (len(stress), stress))
    return table


def lexical_inventory() -> dict[str, tuple[Lexeme, ...]]:
    pron = _pronunciations()
    # Fresh event vocabulary, deliberately separate from the seed and from
    # the borrowed catalogue family.  Reverse-compatible words are included
    # only as ordinary lexical options, never as preassembled pairs.
    banks = {
        "agent": ("artist baker captain child clerk dancer doctor farmer guard poet sailor teacher worker writer pilot singer parent", "agent"),
        "verb": ("admires carries cleans cooks draws finds follows gathers guides hears helps keeps learns opens paints repairs saves sees sends visits writes", "verb"),
        "object": ("anchor basket bridge candle garden harbor letter mirror parcel picture river signal stone story tower window idea camera button lamps boats", "object"),
        "adj": ("ancient blue calm clever distant gentle green hidden kind little narrow patient quiet silver young", "adj"),
    }
    # A small held-out bank contains inflected and semordnilap-compatible
    # forms that can improve the character frontier without importing a
    # sentence or a catalogue line.
    banks["agent"] = (banks["agent"][0] + " drawer liam", "agent")
    banks["verb"] = (banks["verb"][0] + " delivers flows lives stops", "verb")
    banks["object"] = (banks["object"][0] + " desserts evil mail reward trap wolf", "object")

    out: dict[str, list[Lexeme]] = defaultdict(list)
    for role, (words, role_name) in banks.items():
        for word in words.split():
            if word in SEED_CONTENT or word == word[::-1] or word not in pron:
                continue
            syllables, stress = pron[word]
            number = "plural" if word.endswith("s") and word not in {"desserts", "lives"} else "singular"
            out[role].append(Lexeme(word, role_name, number, syllables, stress))
    return {role: tuple(sorted(items, key=lambda x: (x.text, x.stress))) for role, items in out.items()}


DETERMINERS = (
    Lexeme("a", "det", "singular", 1, "0"),
    Lexeme("an", "det", "singular", 1, "0"),
    Lexeme("the", "det", "any", 1, "0"),
    Lexeme("some", "det", "plural", 1, "1"),
    Lexeme("one", "det", "singular", 1, "1"),
    Lexeme("our", "det", "plural", 1, "1"),
)
FRAMES = (
    ("det_agent_verb_det_object", ("det", "agent", "verb", "det", "object")),
    ("det_adj_agent_verb_det_object", ("det", "adj", "agent", "verb", "det", "object")),
    ("det_agent_verb_det_adj_object", ("det", "agent", "verb", "det", "adj", "object")),
)


def _agreement(words: tuple[Lexeme, ...]) -> bool:
    agents = [x for x in words if x.role == "agent"]
    verbs = [x for x in words if x.role == "verb"]
    if not agents or not verbs:
        return False
    # The hand-authored verb bank is mixed; retain only ordinary singular
    # third-person/plural-compatible pairings where the surface makes sense.
    agent = agents[0]
    verb = verbs[0].text
    if agent.number == "plural" and verb.endswith("s"):
        return False
    if agent.number == "singular" and verb in {"admires", "carries", "cleans", "cooks", "draws", "finds", "follows", "gathers", "guides", "hears", "helps", "keeps", "learns", "opens", "paints", "repairs", "saves", "sees", "sends", "visits", "writes", "delivers", "flows", "lives", "stops"}:
        return True
    return True


def _article_agreement(words: tuple[Lexeme, ...]) -> bool:
    """Reject visibly malformed ``a``/``an`` surfaces before indexing."""
    for index, word in enumerate(words[:-1]):
        if word.role != "det":
            continue
        following = words[index + 1].text
        if word.text == "a" and following[:1].lower() in "aeiou":
            return False
        if word.text == "an" and following[:1].lower() not in "aeiou":
            return False
    return True


def _prosody(words: tuple[Lexeme, ...]) -> tuple[str, int, str] | None:
    stress = "".join(item.stress for item in words)
    syllables = sum(item.syllables for item in words)
    if not 4 <= syllables <= 13 or len(stress) < 4:
        return None
    strong = stress.count("1") + stress.count("2")
    if not 0.18 <= strong / len(stress) <= 0.72:
        return None
    # A clause must contain at least two changes of metrical foot and end on a
    # lexical boundary whose final stress is not an unstressed run of three.
    transitions = sum(a != b for a, b in zip(stress, stress[1:]))
    if transitions < 2 or stress.endswith("000"):
        return None
    boundary = "strong" if stress[-1] in "12" else "weak"
    return stress, syllables, boundary


def clauses(inventory: dict[str, tuple[Lexeme, ...]], limit: int) -> list[Clause]:
    pools = {**inventory, "det": DETERMINERS}
    output: dict[str, Clause] = {}
    for frame_name, roles in FRAMES:
        for combo in product(*(pools[role] for role in roles)):
            words = tuple(combo)
            if not _agreement(words) or not _article_agreement(words):
                continue
            profile = _prosody(words)
            if profile is None:
                continue
            clause = Clause(words, frame_name, *profile)
            if MIN_CLAUSE_LETTERS <= len(clause.tape) <= 72:
                output.setdefault(clause.tape, clause)
            if len(output) >= limit:
                break
        if len(output) >= limit:
            break
    return list(output.values())


def _row(left: Clause, right: Clause, preflight: dict, operation: str) -> dict:
    rendered = f"{left.text.capitalize()}; {right.text}."
    tape = normalize(rendered)
    checks = independent_audit(rendered)
    return {
        "rendered": rendered,
        "letters": len(tape),
        "normalized_letters": tape,
        "left_frame": left.frame,
        "right_frame": right.frame,
        "left_stress": left.stress,
        "right_stress": right.stress,
        "left_syllables": left.syllables,
        "right_syllables": right.syllables,
        "phrase_boundaries": {"left": left.boundary, "right": right.boundary},
        "left_words": [x.text for x in left.words],
        "right_words": [x.text for x in right.words],
        "operation": operation,
        "independent_exact_audit": checks,
        "content_disjoint": not (left.content & right.content),
        "word_order_shortcut": tuple(x.text[::-1] for x in reversed(left.words)) == tuple(x.text for x in right.words),
        "mechanically_admitted": checks["exact"] and len(tape) >= MIN_LETTERS and not (left.content & right.content),
        "reader_status": "not_run; prosodic and character diagnostics do not certify readability",
        "novelty_audit": preflight,
    }


def run(limit: int = 80_000) -> dict:
    preflight = novelty_preflight()
    inventory = lexical_inventory()
    left = clauses(inventory, limit)
    right = clauses(inventory, limit)
    by_tape: dict[str, list[Clause]] = defaultdict(list)
    for clause in right:
        by_tape[clause.tape].append(clause)

    exact_rows: list[dict] = []
    near: list[dict] = []
    checked = 0
    prosody_pairs = 0
    for clause in left:
        reverse_tape = clause.tape[::-1]
        matches = by_tape.get(reverse_tape, [])
        prosody_pairs += len(matches)
        for other in matches:
            checked += 1
            if clause.content & other.content:
                continue
            if tuple(x.text[::-1] for x in reversed(clause.words)) == tuple(x.text for x in other.words):
                continue
            exact_rows.append(_row(clause, other, preflight, "paired-character-yield-equation"))

    # Retain a bounded, concrete frontier even when no exact join exists.  The
    # frontier is sorted by common outer characters, then by prosodic balance;
    # it is never promoted to a candidate.
    right_by_initial: dict[str, list[Clause]] = defaultdict(list)
    for clause in right:
        right_by_initial[clause.tape[:1]].append(clause)
    for clause in left[:500]:
        target = clause.tape[::-1]
        best = None
        for other in right_by_initial.get(target[:1], right[:32]):
            matched = 0
            for a, b in zip(target, other.tape):
                if a != b:
                    break
                matched += 1
            score = (matched, -abs(clause.syllables - other.syllables))
            if best is None or score > best[0]:
                best = (score, other)
        if best is not None:
            _, other = best
            rendered_probe = f"{clause.text.capitalize()}; {other.text}."
            near.append({
                "rendered": rendered_probe,
                "letters": len(normalize(rendered_probe)),
                "independent_exact_audit": independent_audit(rendered_probe),
                "left": clause.text,
                "right": other.text,
                "matched_outer_letters": best[0][0],
                "left_syllables": clause.syllables,
                "right_syllables": other.syllables,
                "stress_pair": [clause.stress, other.stress],
            })
    near.sort(key=lambda row: (row["matched_outer_letters"], -abs(row["left_syllables"] - row["right_syllables"])), reverse=True)

    return {
        "status": "complete_prosodic_foot_surface_realizer",
        "family_id": ID,
        "state_space_signature": SIGNATURE,
        "seed": "fresh event frames; seed content excluded",
        "config": {
            "frames": [name for name, _ in FRAMES],
            "pronunciation_source": "NLTK CMUdict first deterministic stress-valid pronunciation",
            "stress_constraints": "4--13 syllables, >=2 stress transitions, bounded strong-stress ratio",
            "minimum_letters": MIN_LETTERS,
            "character_join": "independent reverse tape index after prosodic filtering",
            "no_catalogue_sentence_import": True,
        },
        "novelty_audit": preflight,
        "repair_context": {
            "prior_artifact": "runs/prosodic-foot-surface-realizer-20260915.json",
            "prior_failure": "The first run applied the full-candidate minimum to each half and indexed zero clauses.",
            "repair": "Lowered the per-clause bound to 16 letters, added article-agreement filtering, and bucketed the near-miss frontier by outer character.",
        },
        "inventory_counts": {role: len(words) for role, words in inventory.items()},
        "stats": {
            "left_realizations": len(left),
            "right_realizations": len(right),
            "reverse_index_pairs_checked": checked,
            "prosody_compatible_pairs": prosody_pairs,
            "exact_candidates": len(exact_rows),
            "reader_eligible": 0,
        },
        "exact_candidates": exact_rows,
        "rendered_candidates_and_probes": near[:40],
        "next_repair_operator": {
            "name": "stress_frontier_repair",
            "action": "At the best surviving outer-character frontier, replace one lexical realization with a held-out same-role lexeme of matching syllable/stress profile or move one legal phrase boundary; retain semantic frame and re-audit exactness independently.",
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_text_copied": False,
            "fresh_event_frames": True,
        },
        "reader_gate": "No row is reader evidence; any exact row requires intact-prose versus shuffled-control blinded readers.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=80_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run(args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
