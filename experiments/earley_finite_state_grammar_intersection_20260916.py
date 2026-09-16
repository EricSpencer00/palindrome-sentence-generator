"""Character-synchronous Earley/finite-state grammar intersection.

The two sides are independent finite-state clause grammars.  A product search
advances one character on the forward parse and one character on the reverse
parse at the same time; word-boundary transitions are epsilon transitions.
Agreement and event roles are carried by parser state, so an exact closure is
already a pair of grammatical derivations rather than a tape decoded after
the fact.

This is an experimental search artifact, not a readability certificate.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs/earley-finite-state-grammar-intersection-20260916.json"
EXPERIMENT_ID = "earley-finite-state-grammar-intersection-20260916"
SIGNATURE = (
    "earley-finite-state-grammar-product|forward-reverse-character-lockstep|"
    "epsilon-word-boundaries|agreement-in-parser-state|event-role-clause-semantics|"
    "independent-exact-admission-audit"
)
MIN_LETTERS = 39
MAX_LETTERS = 180

import sys

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


@dataclass(frozen=True)
class Lexeme:
    word: str
    kind: str
    number: str | None = None
    lemma: str | None = None


@dataclass(frozen=True)
class Slot:
    kind: str
    role: str


@dataclass(frozen=True)
class Frame:
    name: str
    slots: tuple[Slot, ...]
    required_roles: tuple[str, ...]


@dataclass(frozen=True)
class Semantics:
    """The feature/meaning register carried by one parser branch."""

    subject_number: str | None = None
    object_number: str | None = None
    locative_number: str | None = None
    predicate: str | None = None
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductState:
    """An Earley-like dotted state for both parses.

    ``left_pos`` consumes a forward word; ``right_pos`` consumes the reverse
    spelling of a normal right-side word.  The semantic registers are part of
    the key and therefore cannot be repaired by a later decoder.
    """

    left_frame: str
    right_frame: str
    left_slot: int
    right_slot: int
    left_word: str | None
    right_word: str | None
    left_pos: int
    right_pos: int
    left_words: tuple[str, ...]
    right_words_reversed: tuple[str, ...]
    left_semantics: Semantics
    right_semantics: Semantics
    matched_letters: int


FUNCTION_WORDS = frozenset(
    "a an the some one each every this that these those many my our their in on"
    " at by for with near beside beyond is are was were can will and or but".split()
)

# The lexicon is deliberately small, fixed, and authored for this experiment.
# Singular/plural inflection is explicit rather than inferred after rendering.
LEXICON: dict[str, tuple[Lexeme, ...]] = {
    "subject_det": tuple(Lexeme(w, "det", n) for w, n in (
        ("a", "sg"), ("the", "sg"), ("one", "sg"), ("each", "sg"),
        ("this", "sg"), ("my", "sg"), ("some", "pl"), ("these", "pl"),
        ("many", "pl"), ("our", "pl"), ("their", "pl"),
    )),
    "object_det": tuple(Lexeme(w, "det", n) for w, n in (
        ("a", "sg"), ("the", "sg"), ("one", "sg"), ("each", "sg"),
        ("some", "pl"), ("these", "pl"), ("many", "pl"), ("our", "pl"),
    )),
    "loc_det": tuple(Lexeme(w, "det", n) for w, n in (
        ("the", "sg"), ("some", "pl"), ("our", "pl"),
    )),
    "subject_noun": tuple(Lexeme(w, "noun", n) for w, n in (
        ("nerd", "sg"), ("baker", "sg"), ("captain", "sg"), ("doctor", "sg"),
        ("farmer", "sg"), ("friend", "sg"), ("gardener", "sg"), ("harbor", "sg"),
        ("lantern", "sg"), ("nurse", "sg"), ("poet", "sg"), ("sailor", "sg"),
        ("teacher", "sg"), ("writer", "sg"), ("artists", "pl"), ("bakers", "pl"),
        ("captains", "pl"), ("doctors", "pl"), ("farmers", "pl"), ("friends", "pl"),
        ("gardeners", "pl"), ("lanterns", "pl"), ("nurses", "pl"), ("poets", "pl"),
        ("sailors", "pl"), ("teachers", "pl"), ("writers", "pl"),
    )),
    "object_noun": tuple(Lexeme(w, "noun", n) for w, n in (
        ("candle", "sg"), ("canvas", "sg"), ("letter", "sg"),
        ("message", "sg"), ("method", "sg"), ("notice", "sg"), ("parcel", "sg"),
        ("report", "sg"), ("story", "sg"), ("ticket", "sg"), ("village", "sg"),
        ("answers", "pl"), ("candles", "pl"), ("canvases", "pl"), ("letters", "pl"),
        ("messages", "pl"), ("methods", "pl"), ("notices", "pl"), ("parcels", "pl"),
        ("reports", "pl"), ("stories", "pl"), ("tickets", "pl"), ("villages", "pl"),
    )),
    "loc_noun": tuple(Lexeme(w, "noun", n) for w, n in (
        ("garden", "sg"), ("harbor", "sg"), ("market", "sg"), ("station", "sg"),
        ("arena", "sg"), ("theater", "sg"), ("village", "sg"), ("gardens", "pl"), ("harbors", "pl"),
        ("markets", "pl"), ("stations", "pl"), ("theaters", "pl"), ("villages", "pl"),
    )),
    "verb": tuple(Lexeme(w, "verb", n, lemma) for w, n, lemma in (
        ("carries", "sg", "carry"), ("draws", "sg", "draw"), ("guides", "sg", "guide"),
        ("helps", "sg", "help"), ("marks", "sg", "mark"), ("opens", "sg", "open"),
        ("reads", "sg", "read"), ("records", "sg", "record"), ("sends", "sg", "send"),
        ("shows", "sg", "show"), ("writes", "sg", "write"),
        ("carry", "pl", "carry"), ("draw", "pl", "draw"), ("guide", "pl", "guide"),
        ("help", "pl", "help"), ("mark", "pl", "mark"), ("open", "pl", "open"),
        ("read", "pl", "read"), ("record", "pl", "record"), ("send", "pl", "send"),
        ("show", "pl", "show"), ("write", "pl", "write"),
    )),
    "aux": tuple(Lexeme(w, "aux", n, w) for w, n in (
        ("is", "sg"), ("was", "sg"), ("can", "sg"), ("will", "sg"),
        ("are", "pl"), ("were", "pl"), ("can", "pl"), ("will", "pl"),
    )),
    "adjective": tuple(Lexeme(w, "adj") for w in "bright careful gentle quiet ready steady vivid patient useful".split()),
    "prep": tuple(Lexeme(w, "prep") for w in "in on at by near beside beyond with".split()),
}

FRAMES: dict[str, Frame] = {
    "event": Frame(
        "event",
        (Slot("subject_det", "agent_determiner"), Slot("subject_noun", "agent"),
         Slot("verb", "event"), Slot("object_det", "patient_determiner"),
         Slot("object_noun", "patient"), Slot("prep", "location_preposition"),
         Slot("loc_det", "setting_determiner"), Slot("loc_noun", "setting")),
        ("agent", "event", "patient", "setting"),
    ),
    "state": Frame(
        "state",
        (Slot("subject_det", "experiencer_determiner"), Slot("subject_noun", "experiencer"),
         Slot("aux", "state"), Slot("adjective", "property"), Slot("prep", "location_preposition"),
         Slot("loc_det", "setting_determiner"), Slot("loc_noun", "setting")),
        ("experiencer", "state", "property", "setting"),
    ),
}


def novelty_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("entries", [])
    prior = [row for row in entries if row.get("id") != EXPERIMENT_ID]
    overlaps = sorted({row.get("signature") for row in prior if row.get("signature") == SIGNATURE})
    artifact_collisions = [row.get("artifact") for row in prior if row.get("artifact") == str(Path(__file__).relative_to(ROOT))]
    result = {
        "status": "passed" if not overlaps and not artifact_collisions else "blocked",
        "registry_entries_before_run": len(entries),
        "signature": SIGNATURE,
        "signature_overlaps": overlaps,
        "artifact_collisions": artifact_collisions,
        "manual_review_required": False,
        "excluded_routes": ["reverse-tape decoder", "word-order mirror", "post-hoc agreement filter"],
    }
    if result["status"] != "passed":
        raise RuntimeError(f"novelty preflight blocked: {result}")
    return result


def _choices(slot: Slot, sem: Semantics) -> tuple[Lexeme, ...]:
    choices = LEXICON[slot.kind]
    if slot.kind in {"subject_det", "object_det", "loc_det"}:
        return choices
    if slot.kind in {"subject_noun", "object_noun", "loc_noun"}:
        expected = {
            "subject_noun": sem.subject_number,
            "object_noun": sem.object_number,
            "loc_noun": sem.locative_number,
        }[slot.kind]
        return tuple(item for item in choices if expected is None or item.number == expected)
    if slot.kind in {"verb", "aux"}:
        return tuple(item for item in choices if sem.subject_number is None or item.number == sem.subject_number)
    return choices


def _apply(slot: Slot, item: Lexeme, sem: Semantics) -> Semantics | None:
    """Apply one lexical transition, rejecting feature/role violations now."""
    if slot.kind == "subject_det":
        if sem.subject_number is not None and sem.subject_number != item.number:
            return None
        return Semantics(sem.subject_number or item.number, sem.object_number, sem.locative_number, sem.predicate, sem.roles)
    if slot.kind == "subject_noun":
        if sem.subject_number is not None and sem.subject_number != item.number:
            return None
        return Semantics(sem.subject_number or item.number, sem.object_number, sem.locative_number, sem.predicate,
                         sem.roles + (f"{slot.role}:{item.word}",))
    if slot.kind == "verb":
        if sem.subject_number is not None and sem.subject_number != item.number:
            return None
        return Semantics(sem.subject_number or item.number, sem.object_number, sem.locative_number, item.lemma,
                         sem.roles + (f"{slot.role}:{item.lemma}",))
    if slot.kind == "aux":
        if sem.subject_number is not None and sem.subject_number != item.number:
            return None
        return Semantics(sem.subject_number or item.number, sem.object_number, sem.locative_number, item.lemma,
                         sem.roles + (f"{slot.role}:{item.lemma}",))
    if slot.kind == "object_det":
        if sem.object_number is not None and sem.object_number != item.number:
            return None
        return Semantics(sem.subject_number, sem.object_number or item.number, sem.locative_number, sem.predicate, sem.roles)
    if slot.kind == "object_noun":
        if sem.object_number is not None and sem.object_number != item.number:
            return None
        return Semantics(sem.subject_number, sem.object_number or item.number, sem.locative_number, sem.predicate,
                         sem.roles + (f"{slot.role}:{item.word}",))
    if slot.kind == "loc_det":
        if sem.locative_number is not None and sem.locative_number != item.number:
            return None
        return Semantics(sem.subject_number, sem.object_number, sem.locative_number or item.number, sem.predicate, sem.roles)
    if slot.kind == "loc_noun":
        if sem.locative_number is not None and sem.locative_number != item.number:
            return None
        return Semantics(sem.subject_number, sem.object_number, sem.locative_number or item.number, sem.predicate,
                         sem.roles + (f"{slot.role}:{item.word}",))
    return Semantics(sem.subject_number, sem.object_number, sem.locative_number, sem.predicate,
                     sem.roles + (f"{slot.role}:{item.word}",))


def _complete(frame: Frame, sem: Semantics) -> bool:
    roles = {role.split(":", 1)[0] for role in sem.roles}
    return all(role in roles for role in frame.required_roles) and sem.predicate is not None


def _epsilon(state: ProductState, left: Frame, right: Frame) -> ProductState:
    """Close completed lexical states without consuming a character."""
    current = state
    while current.left_word is not None and current.left_pos >= len(current.left_word):
        current = ProductState(
            current.left_frame, current.right_frame, current.left_slot + 1, current.right_slot,
            None, current.right_word, 0, current.right_pos, current.left_words + (current.left_word,),
            current.right_words_reversed, current.left_semantics, current.right_semantics,
            current.matched_letters,
        )
    while current.right_word is not None and current.right_pos >= len(current.right_word):
        current = ProductState(
            current.left_frame, current.right_frame, current.left_slot, current.right_slot - 1,
            current.left_word, None, current.left_pos, 0, current.left_words,
            current.right_words_reversed + (current.right_word,), current.left_semantics,
            current.right_semantics, current.matched_letters,
        )
    return current


def _start_state(left: Frame, right: Frame) -> ProductState:
    return ProductState(left.name, right.name, 0, len(right.slots) - 1, None, None, 0, 0, (), (), Semantics(), Semantics(), 0)


def _render(left_words: tuple[str, ...], right_words_reversed: tuple[str, ...]) -> str:
    return " ".join(left_words).capitalize() + "; " + " ".join(reversed(right_words_reversed)) + "."


def independent_audit(text: str) -> dict[str, object]:
    """Audit without using the search trace or its character counter."""
    tape = normalize_letters(text)
    ascii_tape = "".join(ch.lower() for ch in text if "a" <= ch.lower() <= "z")
    two_pointer = bool(ascii_tape) and all(ascii_tape[i] == ascii_tape[-1 - i] for i in range(len(ascii_tape) // 2))
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_tape": tape,
        "independent_ascii_tape": ascii_tape,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer_exact": two_pointer,
        "independent_sha256": hashlib.sha256(ascii_tape.encode()).hexdigest(),
        "mechanical_checks": checks,
        "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
    }


def _expand(state: ProductState, left: Frame, right: Frame, stats: Counter) -> Iterable[ProductState]:
    state = _epsilon(state, left, right)
    if state.left_slot >= len(left.slots) or state.right_slot < 0:
        return ()
    left_slot = left.slots[state.left_slot]
    right_slot = right.slots[state.right_slot]
    # Entering a lexical state is an Earley scan transition.  Once entered,
    # only one next-character pair is emitted, never a completed tape decode.
    left_lexemes = ((Lexeme(state.left_word, left_slot.kind),) if state.left_word is not None else _choices(left_slot, state.left_semantics))
    right_lexemes = ((Lexeme(state.right_word, right_slot.kind),) if state.right_word is not None else _choices(right_slot, state.right_semantics))
    by_char: dict[str, list[Lexeme]] = defaultdict(list)
    for item in right_lexemes:
        by_char[item.word[::-1][state.right_pos]].append(item)
    emitted: list[ProductState] = []
    for litem in left_lexemes:
        lsem = state.left_semantics if state.left_word is not None else _apply(left_slot, litem, state.left_semantics)
        if lsem is None:
            stats["agreement_pruned"] += 1
            continue
        lchar = litem.word[state.left_pos]
        for ritem in by_char.get(lchar, ()):
            rsem = state.right_semantics if state.right_word is not None else _apply(right_slot, ritem, state.right_semantics)
            if rsem is None:
                stats["agreement_pruned"] += 1
                continue
            stats["character_matches"] += 1
            emitted.append(ProductState(
                state.left_frame, state.right_frame, state.left_slot, state.right_slot,
                litem.word, ritem.word, state.left_pos + 1, state.right_pos + 1,
                state.left_words, state.right_words_reversed, lsem, rsem,
                state.matched_letters + 1,
            ))
    stats["character_mismatch_pruned"] += sum(len(left_lexemes) for _ in [0]) - len(emitted)
    return tuple(emitted)


def search_pair(left: Frame, right: Frame, *, max_states: int = 160_000) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter]:
    queue = deque([_start_state(left, right)])
    seen: set[ProductState] = set()
    closures: list[dict[str, object]] = []
    dead: list[dict[str, object]] = []
    stats = Counter()
    while queue and len(seen) < max_states:
        raw = queue.popleft()
        state = _epsilon(raw, left, right)
        if state in seen:
            continue
        seen.add(state)
        stats["states"] += 1
        if state.left_slot >= len(left.slots) and state.right_slot < 0:
            if _complete(left, state.left_semantics) and _complete(right, state.right_semantics):
                rendered = _render(state.left_words, state.right_words_reversed)
                audit = independent_audit(rendered)
                closures.append({
                    "rendered": rendered,
                    "left_words": list(state.left_words),
                    "right_words": list(reversed(state.right_words_reversed)),
                    "left_semantics": state.left_semantics.__dict__,
                    "right_semantics": state.right_semantics.__dict__,
                    "letters_consumed_lockstep": state.matched_letters,
                    "audit": audit,
                    "reader_eligible": False,
                })
                stats["complete_grammar_closures"] += 1
            continue
        children = _expand(state, left, right, stats)
        if not children and len(dead) < 240:
            dead.append({
                "left_frame": left.name, "right_frame": right.name,
                "left_slot": state.left_slot, "right_slot": state.right_slot,
                "left_words": list(state.left_words),
                "right_words_reversed": list(state.right_words_reversed),
                "left_semantics": state.left_semantics.__dict__,
                "right_semantics": state.right_semantics.__dict__,
                "matched_letters": state.matched_letters,
                "reason": "no-character-compatible-Earley-scan-transition",
            })
        queue.extend(children)
    stats["node_budget_hit"] += int(bool(queue) and len(seen) >= max_states)
    return closures, dead, stats


def enumerate_clauses(frame: Frame, cap: int = 360) -> list[dict[str, object]]:
    """Enumerate complete grammar derivations for rendered near-miss probes."""
    out: list[dict[str, object]] = []

    def visit(index: int, words: tuple[str, ...], sem: Semantics) -> None:
        if len(out) >= cap:
            return
        if index == len(frame.slots):
            if _complete(frame, sem):
                out.append({"words": words, "semantics": sem, "text": " ".join(words)})
            return
        slot = frame.slots[index]
        for item in _choices(slot, sem):
            next_sem = _apply(slot, item, sem)
            if next_sem is not None:
                visit(index + 1, words + (item.word,), next_sem)

    visit(0, (), Semantics())
    return out


def _matched_prefix(left: str, right: str) -> int:
    a, b = normalize_letters(left), normalize_letters(right)[::-1]
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def run(*, max_states: int = 160_000) -> dict[str, object]:
    preflight = novelty_preflight()
    stats = Counter()
    closures: list[dict[str, object]] = []
    dead: list[dict[str, object]] = []
    frame_runs: list[dict[str, object]] = []
    for left in FRAMES.values():
        for right in FRAMES.values():
            found, misses, local = search_pair(left, right, max_states=max_states)
            closures.extend(found)
            dead.extend(misses)
            stats.update(local)
            frame_runs.append({"left_frame": left.name, "right_frame": right.name, **dict(local), "closures": len(found)})

    # These are independently derived complete clauses, rendered to preserve
    # the near-miss surface and give the next repair a concrete mismatch.
    banks = {name: enumerate_clauses(frame) for name, frame in FRAMES.items()}
    probes: list[dict[str, object]] = []
    for lname, left_bank in banks.items():
        for rname, right_bank in banks.items():
            pairs = []
            for left in left_bank[:160]:
                for right in right_bank[:160]:
                    text = left["text"].capitalize() + "; " + right["text"] + "."
                    if len(normalize_letters(text)) >= MIN_LETTERS:
                        pairs.append((
                            _matched_prefix(left["text"], right["text"]),
                            text, left, right,
                        ))
            for matched, text, left, right in sorted(pairs, key=lambda row: (-row[0], row[1]))[:3]:
                audit = independent_audit(text)
                probes.append({
                    "rendered": text, "left_frame": lname, "right_frame": rname,
                    "matched_outer_letters_before_mismatch": matched,
                    "left_semantics": left["semantics"].__dict__,
                    "right_semantics": right["semantics"].__dict__,
                    "audit": audit, "reader_eligible": False,
                    "provenance": "two independent finite-state clause derivations; probe is not an exact closure",
                })
    probes.sort(key=lambda row: (-row["matched_outer_letters_before_mismatch"], -row["audit"]["letters"], row["rendered"]))
    unique: dict[str, dict[str, object]] = {}
    for row in probes:
        unique.setdefault(row["audit"]["normalized_tape"], row)
    probes = list(unique.values())[:24]
    exact = [row for row in closures if row["audit"]["exact"]]
    admitted = [row for row in closures if row["audit"]["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_character_synchronous_earley_intersection",
        "method": "A finite Earley-style product scans one forward and one reverse lexical character per transition, with epsilon word-boundary closure. Agreement features and clause semantic roles are registers in each parser state.",
        "novelty_preflight": preflight,
        "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_states_per_frame_pair": max_states, "forward_frames": list(FRAMES), "reverse_frames": list(FRAMES), "catalogue_text_imported": False, "post_hoc_decoder": False},
        "stats": {**dict(stats), "frame_pairs": len(frame_runs), "rendered_candidates": len(exact), "exact_count": len(exact), "mechanically_admitted": len(admitted), "rendered_probes": len(probes), "reader_eligible": 0},
        "frame_runs": frame_runs,
        "rendered_candidates": exact,
        "rendered_probes": probes,
        "dead_frontier": dead[:240],
        "independent_audit": [row["audit"] for row in exact],
        "reader_gate": "No row is readability evidence; exact mechanically admitted rows require randomized blinded intact/shuffled reading.",
        "next_repair_operator": "Take the deepest dead Earley state at the first character mismatch and add one held-out lexical transition licensed by the same event-role and agreement registers; rerun the full lockstep product without reverse-segmenting a completed tape.",
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexicon": "authored finite lexical banks in this script", "source_sentences_copied": False, "known_catalogue_excluded": True, "independent_audits": ["normalize_letters tape reversal", "independent ASCII two-pointer", "mechanical_admission_checks"]},
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUT}")
    result = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": result["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
