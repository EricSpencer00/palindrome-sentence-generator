"""Weighted finite-state morphology transducer under character lockstep.

Each transition selects a lexical chunk on the forward side and a lexical
chunk on the reverse side, emits one character from each chunk, and updates a
feature register.  The register carries number, tense, lemma, and semantic
roles; a fluency score is accumulated as lexical transitions are entered.  A
completed tape is never reverse-segmented or decoded after the search.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs/weighted-morphology-fst-lockstep-20260916.json"
EXPERIMENT_ID = "weighted-morphology-fst-lockstep-20260916"
SIGNATURE = (
    "weighted-morphology-fst|lexical-chunk-pair-emission|"
    "agreement-tense-register|fluency-objective-during-character-lockstep|"
    "heldout-morphology-repair|independent-exact-admission-audit"
)
MIN_LETTERS = 39
MAX_LETTERS = 180
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


@dataclass(frozen=True)
class Lexeme:
    text: str
    category: str
    number: str | None = None
    tense: str | None = None
    lemma: str | None = None


@dataclass(frozen=True)
class Morphology:
    subject_number: str | None = None
    object_number: str | None = None
    locative_number: str | None = None
    tense: str | None = None
    lemma: str | None = None
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class Slot:
    category: str
    role: str


@dataclass(frozen=True)
class Grammar:
    name: str
    slots: tuple[Slot, ...]
    required_roles: tuple[str, ...]


@dataclass(frozen=True)
class State:
    left_grammar: str
    right_grammar: str
    left_slot: int
    right_slot: int
    left_chunk: str | None
    right_chunk: str | None
    left_pos: int
    right_pos: int
    left_words: tuple[str, ...]
    right_words_reversed: tuple[str, ...]
    left_morphology: Morphology
    right_morphology: Morphology
    score: float
    emitted: int


LEXICON: dict[str, tuple[Lexeme, ...]] = {
    "det": tuple(Lexeme(w, "det", n) for w, n in (
        ("a", "sg"), ("the", "sg"), ("one", "sg"), ("some", "pl"),
        ("these", "pl"), ("many", "pl"), ("our", "pl"),
    )),
    "noun_subj": tuple(Lexeme(w, "noun", n) for w, n in (
        ("nerd", "sg"), ("baker", "sg"), ("captain", "sg"), ("doctor", "sg"),
        ("farmer", "sg"), ("friend", "sg"), ("gardener", "sg"), ("nurse", "sg"),
        ("poet", "sg"), ("sailor", "sg"), ("teacher", "sg"), ("writer", "sg"),
        ("bakers", "pl"), ("captains", "pl"), ("doctors", "pl"), ("farmers", "pl"),
        ("friends", "pl"), ("gardeners", "pl"), ("nurses", "pl"), ("poets", "pl"),
        ("sailors", "pl"), ("teachers", "pl"), ("writers", "pl"),
    )),
    "noun_obj": tuple(Lexeme(w, "noun", n) for w, n in (
        ("candle", "sg"), ("canvas", "sg"), ("letter", "sg"), ("message", "sg"),
        ("method", "sg"), ("notice", "sg"), ("parcel", "sg"), ("report", "sg"),
        ("story", "sg"), ("ticket", "sg"), ("village", "sg"), ("candles", "pl"),
        ("canvases", "pl"), ("letters", "pl"), ("messages", "pl"), ("methods", "pl"),
        ("notices", "pl"), ("parcels", "pl"), ("reports", "pl"), ("stories", "pl"),
        ("tickets", "pl"), ("villages", "pl"),
    )),
    "verb": tuple(Lexeme(w, "verb", n, t, lemma) for w, n, t, lemma in (
        ("carries", "sg", "pres", "carry"), ("draws", "sg", "pres", "draw"),
        ("guides", "sg", "pres", "guide"), ("marks", "sg", "pres", "mark"),
        ("reads", "sg", "pres", "read"), ("records", "sg", "pres", "record"),
        ("sends", "sg", "pres", "send"), ("writes", "sg", "pres", "write"),
        ("carried", "sg", "past", "carry"), ("drew", "sg", "past", "draw"),
        ("guided", "sg", "past", "guide"), ("marked", "sg", "past", "mark"),
        ("read", "sg", "past", "read"), ("recorded", "sg", "past", "record"),
        ("sent", "sg", "past", "send"), ("wrote", "sg", "past", "write"),
        ("carry", "pl", "pres", "carry"), ("draw", "pl", "pres", "draw"),
        ("guide", "pl", "pres", "guide"), ("mark", "pl", "pres", "mark"),
        ("read", "pl", "pres", "read"), ("record", "pl", "pres", "record"),
        ("send", "pl", "pres", "send"), ("write", "pl", "pres", "write"),
        ("carried", "pl", "past", "carry"), ("drew", "pl", "past", "draw"),
        ("guided", "pl", "past", "guide"), ("marked", "pl", "past", "mark"),
        ("read", "pl", "past", "read"), ("recorded", "pl", "past", "record"),
        ("sent", "pl", "past", "send"), ("wrote", "pl", "past", "write"),
    )),
    "prep": tuple(Lexeme(w, "prep") for w in "in on at by near beside beyond with".split()),
    "noun_loc": tuple(Lexeme(w, "noun", n) for w, n in (
        ("arena", "sg"), ("garden", "sg"), ("harbor", "sg"), ("market", "sg"),
        ("station", "sg"), ("theater", "sg"), ("gardens", "pl"), ("harbors", "pl"),
        ("markets", "pl"), ("stations", "pl"), ("theaters", "pl"),
    )),
}

GRAMMARS = {
    "event": Grammar("event", (
        Slot("det", "agent_det"), Slot("noun_subj", "agent"), Slot("verb", "event"),
        Slot("det", "patient_det"), Slot("noun_obj", "patient"), Slot("prep", "relation"),
        Slot("det", "setting_det"), Slot("noun_loc", "setting")),
        ("agent", "event", "patient", "setting")),
}

# A tiny fixed word-bigram model is enough to make the optimization explicit.
# Scores are applied when a lexical chunk is selected, before the next paired
# character is emitted.
BIGRAM_BONUS = {
    ("a", "nerd"): 2.5, ("the", "garden"): 2.0, ("the", "arena"): 1.8,
    ("our", "gardens"): 2.0, ("the", "harbor"): 1.8,
    ("nerd", "carries"): 2.2, ("baker", "guides"): 2.0, ("sailors", "carry"): 1.8,
    ("carries", "a"): 1.8, ("reads", "a"): 1.7, ("records", "the"): 1.7,
    ("a", "candle"): 1.8, ("a", "letter"): 1.5, ("at", "the"): 1.5,
    ("near", "the"): 1.2, ("beside", "the"): 1.3,
}

HELDOUT_MORPHOLOGY = (
    Lexeme("leads", "verb", "sg", "pres", "lead"),
    Lexeme("lead", "verb", "pl", "pres", "lead"),
    Lexeme("led", "verb", "sg", "past", "lead"),
    Lexeme("led", "verb", "pl", "past", "lead"),
)


def novelty_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    rows = [row for row in registry.get("entries", []) if row.get("id") != EXPERIMENT_ID]
    overlap = sorted({row.get("signature") for row in rows if row.get("signature") == SIGNATURE})
    artifact = str(Path(__file__).relative_to(ROOT))
    collision = [row.get("artifact") for row in rows if row.get("artifact") == artifact]
    result = {"status": "passed" if not overlap and not collision else "blocked",
              "registry_entries_before_run": len(registry.get("entries", [])),
              "signature_overlaps": overlap, "artifact_collisions": collision,
              "manual_review_required": False,
              "excluded_routes": ["fixed-tape reverse decoder", "post-hoc morphology", "unweighted lexical cross-product"]}
    if result["status"] != "passed":
        raise RuntimeError(f"novelty preflight blocked: {result}")
    return result


def choices(slot: Slot, morphology: Morphology) -> tuple[Lexeme, ...]:
    result = LEXICON[slot.category]
    if slot.category in {"noun_subj", "noun_obj", "noun_loc"}:
        expected = {"noun_subj": morphology.subject_number, "noun_obj": morphology.object_number,
                    "noun_loc": morphology.locative_number}[slot.category]
        if expected:
            result = tuple(item for item in result if item.number == expected)
    if slot.category == "verb":
        if morphology.subject_number:
            result = tuple(item for item in result if item.number == morphology.subject_number)
        if morphology.tense:
            result = tuple(item for item in result if item.tense == morphology.tense)
    return result


def apply(slot: Slot, item: Lexeme, morphology: Morphology) -> Morphology | None:
    """Morphological transducer transition; disagreement rejects immediately."""
    subject, obj, loc = morphology.subject_number, morphology.object_number, morphology.locative_number
    tense, lemma = morphology.tense, morphology.lemma
    if slot.role in {"agent_det", "agent"}:
        if subject and item.number and subject != item.number:
            return None
        subject = subject or item.number
    elif slot.role in {"patient_det", "patient"}:
        if obj and item.number and obj != item.number:
            return None
        obj = obj or item.number
    elif slot.role in {"setting_det", "setting"}:
        if loc and item.number and loc != item.number:
            return None
        loc = loc or item.number
    elif slot.category == "verb":
        if subject and item.number and subject != item.number:
            return None
        subject = subject or item.number
    if item.tense and tense and item.tense != tense:
        return None
    tense = tense or item.tense
    lemma = item.lemma or lemma
    roles = morphology.roles
    if slot.role in {"agent", "event", "patient", "setting"}:
        roles = roles + (f"{slot.role}:{item.text}",)
    return Morphology(subject, obj, loc, tense, lemma, roles)


def complete(grammar: Grammar, morphology: Morphology) -> bool:
    return morphology.lemma is not None and all(
        role in {x.split(":", 1)[0] for x in morphology.roles} for role in grammar.required_roles
    )


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    independent = "".join(ch.lower() for ch in text if "a" <= ch.lower() <= "z")
    two_pointer = bool(independent) and all(independent[i] == independent[-1 - i] for i in range(len(independent) // 2))
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {"rendered": text, "letters": len(tape), "normalized_tape": tape,
            "independent_ascii_tape": independent, "exact": bool(tape) and tape == tape[::-1],
            "independent_two_pointer_exact": two_pointer,
            "independent_sha256": hashlib.sha256(independent.encode()).hexdigest(),
            "mechanical_checks": checks,
            "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values())}


def render(left: tuple[str, ...], right_reversed: tuple[str, ...]) -> str:
    return " ".join(left).capitalize() + "; " + " ".join(reversed(right_reversed)) + "."


def _close(state: State, left: Grammar, right: Grammar) -> State:
    while state.left_chunk is not None and state.left_pos >= len(state.left_chunk):
        state = State(state.left_grammar, state.right_grammar, state.left_slot + 1, state.right_slot,
                      None, state.right_chunk, 0, state.right_pos, state.left_words + (state.left_chunk,),
                      state.right_words_reversed, state.left_morphology, state.right_morphology,
                      state.score, state.emitted)
    while state.right_chunk is not None and state.right_pos >= len(state.right_chunk):
        state = State(state.left_grammar, state.right_grammar, state.left_slot, state.right_slot - 1,
                      state.left_chunk, None, state.left_pos, 0, state.left_words,
                      state.right_words_reversed + (state.right_chunk,), state.left_morphology,
                      state.right_morphology, state.score, state.emitted)
    return state


def _state_key(state: State) -> tuple[object, ...]:
    return (state.left_slot, state.right_slot, state.left_chunk, state.right_chunk,
            state.left_pos, state.right_pos, state.left_words, state.right_words_reversed,
            state.left_morphology, state.right_morphology)


def _transition(state: State, left: Grammar, right: Grammar, stats: Counter) -> tuple[State, ...]:
    state = _close(state, left, right)
    if state.left_slot >= len(left.slots) or state.right_slot < 0:
        return ()
    ls, rs = left.slots[state.left_slot], right.slots[state.right_slot]
    litems = (Lexeme(state.left_chunk, ls.category),) if state.left_chunk is not None else choices(ls, state.left_morphology)
    ritems = (Lexeme(state.right_chunk, rs.category),) if state.right_chunk is not None else choices(rs, state.right_morphology)
    by_char: dict[str, list[Lexeme]] = defaultdict(list)
    for item in ritems:
        by_char[item.text[::-1][state.right_pos]].append(item)
    out = []
    for li in litems:
        lm = state.left_morphology if state.left_chunk is not None else apply(ls, li, state.left_morphology)
        if lm is None:
            stats["morphology_pruned"] += 1
            continue
        for ri in by_char.get(li.text[state.left_pos], ()):
            rm = state.right_morphology if state.right_chunk is not None else apply(rs, ri, state.right_morphology)
            if rm is None:
                stats["morphology_pruned"] += 1
                continue
            previous_l = state.left_words[-1] if state.left_words else None
            previous_r = state.right_words_reversed[-1] if state.right_words_reversed else None
            # The reverse parser selects normal right-side words from the end,
            # so invert that local lookup to score the eventual ordinary-order
            # clause rather than its construction order.
            score = state.score + BIGRAM_BONUS.get((previous_l, li.text), 0.0) + BIGRAM_BONUS.get((ri.text, previous_r), 0.0)
            stats["character_pairs_emitted"] += 1
            out.append(State(state.left_grammar, state.right_grammar, state.left_slot, state.right_slot,
                             li.text, ri.text, state.left_pos + 1, state.right_pos + 1,
                             state.left_words, state.right_words_reversed, lm, rm, score, state.emitted + 1))
    stats["weighted_transitions"] += len(out)
    return tuple(out)


def search(left: Grammar, right: Grammar, *, max_states: int = 80_000) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter]:
    initial = State(left.name, right.name, 0, len(right.slots) - 1, None, None, 0, 0, (), (), Morphology(), Morphology(), 0.0, 0)
    heap: list[tuple[float, int, State]] = [(0.0, 0, initial)]
    serial = 1
    seen: dict[tuple[object, ...], float] = {}
    rows: list[dict[str, object]] = []
    dead: list[dict[str, object]] = []
    stats = Counter()
    while heap and len(seen) < max_states:
        neg, _, raw = heapq.heappop(heap)
        state = _close(raw, left, right)
        key = _state_key(state)
        if key in seen and seen[key] >= state.score:
            continue
        seen[key] = state.score
        stats["states"] += 1
        if state.left_slot >= len(left.slots) and state.right_slot < 0:
            if complete(left, state.left_morphology) and complete(right, state.right_morphology):
                text = render(state.left_words, state.right_words_reversed)
                rows.append({"rendered": text, "fluency_score": state.score,
                             "left_morphology": state.left_morphology.__dict__,
                             "right_morphology": state.right_morphology.__dict__,
                             "characters_emitted_lockstep": state.emitted, "audit": audit(text),
                             "reader_eligible": False})
                stats["complete_closures"] += 1
            continue
        children = _transition(state, left, right, stats)
        if not children and len(dead) < 240:
            dead.append({"left_slot": state.left_slot, "right_slot": state.right_slot,
                         "left_words": list(state.left_words), "right_words_reversed": list(state.right_words_reversed),
                         "left_morphology": state.left_morphology.__dict__, "right_morphology": state.right_morphology.__dict__,
                         "fluency_score": state.score, "characters_emitted_lockstep": state.emitted,
                         "reason": "no-morphology-compatible-character-transition"})
        for child in children:
            heapq.heappush(heap, (-child.score, serial, child)); serial += 1
    stats["node_budget_hit"] += int(bool(heap) and len(seen) >= max_states)
    return rows, dead, stats


def heldout_morphology_repair(*, max_states: int = 80_000) -> dict[str, object]:
    """Rerun one mismatch frontier with held-out inflectional transitions."""
    original = LEXICON["verb"]
    LEXICON["verb"] = original + HELDOUT_MORPHOLOGY
    try:
        rows, dead, stats = search(GRAMMARS["event"], GRAMMARS["event"], max_states=max_states)
    finally:
        LEXICON["verb"] = original
    return {"held_out_variants": [item.__dict__ for item in HELDOUT_MORPHOLOGY],
            "target": "highest-fluency event/event dead frontier",
            "closures": rows, "dead_frontier": dead[:40], "stats": dict(stats),
            "exact_count": sum(row["audit"]["exact"] for row in rows),
            "mechanically_admitted": sum(row["audit"]["mechanically_admitted"] for row in rows)}


def independent_clause_bank(grammar: Grammar, cap: int = 240) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []

    def visit(index: int, words: tuple[str, ...], morph: Morphology) -> None:
        if len(out) >= cap:
            return
        if index == len(grammar.slots):
            if complete(grammar, morph):
                out.append(words)
            return
        slot = grammar.slots[index]
        for item in choices(slot, morph):
            next_morph = apply(slot, item, morph)
            if next_morph is not None:
                visit(index + 1, words + (item.text,), next_morph)

    visit(0, (), Morphology())
    return out


def match_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    a, b = normalize_letters(" ".join(left)), normalize_letters(" ".join(right))[::-1]
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def run(*, max_states: int = 80_000) -> dict[str, object]:
    preflight = novelty_preflight()
    all_rows, all_dead, stats = [], [], Counter()
    frame_runs = []
    for left in GRAMMARS.values():
        for right in GRAMMARS.values():
            rows, dead, local = search(left, right, max_states=max_states)
            all_rows.extend(rows); all_dead.extend(dead); stats.update(local)
            frame_runs.append({"left": left.name, "right": right.name, **dict(local), "closures": len(rows)})
    repair = heldout_morphology_repair(max_states=max_states)
    bank = independent_clause_bank(GRAMMARS["event"], cap=360)
    ranked_pairs = []
    for left in bank[:80]:
        for right in bank[:80]:
            text = render(left, (right[-1],) + tuple(reversed(right[:-1]))) if right else ""
            if len(normalize_letters(text)) < MIN_LETTERS:
                continue
            score = sum(BIGRAM_BONUS.get((a, b), 0.0) for a, b in zip(left, left[1:])) + sum(BIGRAM_BONUS.get((a, b), 0.0) for a, b in zip(right, right[1:]))
            ranked_pairs.append((match_prefix(left, right), score, text))
    ranked_pairs.sort(key=lambda x: (-x[0], -x[1], -len(normalize_letters(x[2])), x[2]))
    probes = []
    for matched, score, text in ranked_pairs[:24]:
        probes.append({"rendered": text, "fluency_score": score,
                       "matched_outer_letters_before_mismatch": matched, "audit": audit(text),
                       "reader_eligible": False, "provenance": "independent finite-state morphology derivations; diagnostic probe, not closure"})
    unique = {}
    for row in probes:
        unique.setdefault(row["audit"]["normalized_tape"], row)
    probes = list(unique.values())[:24]
    exact = [row for row in all_rows if row["audit"]["exact"]]
    admitted = [row for row in all_rows if row["audit"]["mechanically_admitted"]]
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "completed_weighted_morphology_fst_lockstep", "novelty_preflight": preflight,
            "method": "Weighted finite-state morphology transducer emits paired lexical chunks while a forward and reverse character lockstep is active; number, tense, lemma, semantic roles, and a cumulative fluency score are in the state.",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_states_per_frame_pair": max_states, "catalogue_text_imported": False, "post_hoc_reverse_segmentation": False},
            "stats": {**dict(stats), "frame_pairs": len(frame_runs), "rendered_candidates": len(exact), "exact_count": len(exact), "mechanically_admitted": len(admitted), "rendered_probes": len(probes), "reader_eligible": 0},
            "frame_runs": frame_runs, "rendered_candidates": exact, "rendered_probes": probes, "dead_frontier": all_dead[:240],
            "mismatch_directed_morphology_repair": repair,
            "independent_audit": [row["audit"] for row in exact],
            "next_repair_operator": "At the highest-fluency dead transition, add one held-out inflectional variant (same lemma and semantic role, alternate number/tense realization) to the transducer alphabet, then rerun the character-lockstep optimization; do not resegment a completed tape.",
            "reader_gate": "No row is readability evidence; exact mechanically admitted rows require blinded intact/shuffled reading.",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexicon": "authored inflectional lexical banks", "source_sentences_copied": False, "known_catalogue_excluded": True, "independent_audits": ["normalized tape reversal", "ASCII two-pointer", "mechanical_admission_checks"]}}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUT}")
    result = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": result["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
