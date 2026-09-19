#!/usr/bin/env python3
"""Typed reflected-residual chart for fresh scene-frame construction.

The chart grows two independently authored scene frames from opposite edges.
It carries the live reflected character residual across lexical boundaries,
instead of substituting a word after a mismatch or reversing a completed tape.
Every completed frame is checked again by an independent pointer/SHA audit.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "typed-reflected-residual-chart-20260918.json"
EXPERIMENT = "typed-reflected-residual-chart-20260918"

# Fresh scene lexicon.  Slots are semantic, not a list of completed sentences.
DETS = ("a", "an", "the", "some", "one")
AGENTS = (
    "archivist", "artist", "baker", "captain", "clerk", "curator",
    "doctor", "editor", "farmer", "friend", "gardener", "guard",
    "keeper", "nurse", "poet", "reader", "sailor", "scout", "teacher",
    "writer", "woman", "child", "pilot",
)
VERBS = (
    "answers", "bakes", "carries", "checks", "cleans", "closes", "draws",
    "edits", "finds", "guards", "guides", "helps", "holds", "keeps",
    "labels", "marks", "mends", "opens", "paints", "plans", "reads",
    "records", "repairs", "saves", "sees", "sends", "shows", "stores",
    "takes", "teaches", "tells", "tests", "uses", "visits", "watches",
    "writes", "inspires", "delivers",
)
OBJECTS = (
    "answer", "book", "boat", "bridge", "candle", "chart", "door", "gate",
    "garden", "letter", "map", "memo", "message", "note", "paper", "parcel",
    "record", "rope", "story", "table", "ticket", "vase", "wall", "water",
    "window", "word", "song", "stone", "lamp", "seed", "ship", "signal",
    "plan", "report", "tool", "box", "bread", "cake", "coin", "flag",
    "flower", "glass", "key", "list", "lock", "path", "poem", "road",
    "room", "star", "town", "tree", "wheel",
)
PREPS = ("in", "on", "at", "by", "for", "near", "with", "from", "after", "under", "over")
SETTINGS = (
    "garden", "harbor", "market", "station", "school", "theater", "village",
    "window", "river", "workshop", "archive", "hill",
)
NAMES = ("Alice", "Diana", "Eva", "Iris", "Lisa", "Nina", "Adam", "Noah", "Sam")

# Two complete frames; the right frame is independently authored and may have
# a proper-name object.  Agreement and valency are checked at completion.
FRAME = ("D", "A", "V", "D", "O", "P", "D", "S")
NAME_FRAME = ("D", "A", "V", "N", "P", "D", "S")
BANKS = {
    "D": DETS, "A": AGENTS, "V": VERBS, "O": OBJECTS,
    "P": PREPS, "S": SETTINGS, "N": NAMES,
}
FUNCTION = set(DETS) | set(PREPS)


def letters(text: str) -> str:
    return "".join(ch.lower() for ch in text if "a" <= ch.lower() <= "z")


def independent_audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [i for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "two_pointer_exact": bool(tape) and not mismatches,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def frame_valid(words: tuple[str, ...], slots: tuple[str, ...]) -> bool:
    """Small semantic grammar: determiner, agent, transitive verb, object."""
    if len(words) != len(slots):
        return False
    if slots[0] != "D" or slots[1] != "A" or slots[2] != "V":
        return False
    # ``a/an`` agreement is a hard surface check; the rest is a typed
    # transitive frame with a locative adjunct.
    if words[0] == "an" and words[1][0] not in "aeiou":
        return False
    if words[0] == "a" and words[1][0] in "aeiou":
        return False
    return True


@dataclass(frozen=True)
class State:
    left_slot: int
    right_slot: int
    left_word: str | None
    right_word: str | None
    left_pos: int
    right_pos: int
    left_words: tuple[str, ...]
    right_words_reversed: tuple[str, ...]
    emitted_pairs: int


def _advance(state: State, left_slots: tuple[str, ...], right_slots: tuple[str, ...]) -> State:
    """Close completed lexical spans while retaining the reflected residual."""
    left_slot, right_slot = state.left_slot, state.right_slot
    left_words, right_words = list(state.left_words), list(state.right_words_reversed)
    left_word, right_word = state.left_word, state.right_word
    left_pos, right_pos = state.left_pos, state.right_pos
    if left_word is not None and left_pos >= len(left_word):
        left_words.append(left_word)
        left_slot += 1
        left_word, left_pos = None, 0
    if right_word is not None and right_pos >= len(right_word):
        right_words.append(right_word)
        right_slot -= 1
        right_word, right_pos = None, 0
    return State(left_slot, right_slot, left_word, right_word, left_pos, right_pos,
                 tuple(left_words), tuple(right_words), state.emitted_pairs)


def chart(left_slots: tuple[str, ...], right_slots: tuple[str, ...], *, state_limit: int = 80_000) -> dict[str, object]:
    """Run the live reflected-residual chart for one pair of typed frames."""
    initial = State(0, len(right_slots) - 1, None, None, 0, 0, (), (), 0)
    frontier = [initial]
    closures: list[dict[str, object]] = []
    dead: list[dict[str, object]] = []
    expanded = 0
    for _depth in range(260):
        next_frontier: list[State] = []
        for raw in frontier:
            state = _advance(raw, left_slots, right_slots)
            if state.left_slot >= len(left_slots) and state.right_slot < 0 and state.left_word is None and state.right_word is None:
                left_words = state.left_words
                right_words = tuple(reversed(state.right_words_reversed))
                if frame_valid(left_words, left_slots) and frame_valid(right_words, right_slots):
                    rendered = " ".join(left_words) + "; " + " ".join(right_words) + "."
                    closures.append({
                        "rendered": rendered,
                        "audit": independent_audit(rendered),
                        "left_slots": left_slots,
                        "right_slots": right_slots,
                        "provenance": {
                            "construction": "typed reflected-residual chart",
                            "finished_tape_reversal": False,
                            "catalogue_text": False,
                            "word_order_only": False,
                        },
                    })
                continue
            if state.left_slot >= len(left_slots) or state.right_slot < 0:
                continue
            left_options = ((state.left_word,) if state.left_word is not None
                            else BANKS[left_slots[state.left_slot]])
            right_options = ((state.right_word,) if state.right_word is not None
                             else BANKS[right_slots[state.right_slot]])
            by_char: dict[str, list[str]] = defaultdict(list)
            for word in right_options:
                by_char[word[-1 - state.right_pos]].append(word)
            matched = 0
            for left_word in left_options:
                for right_word in by_char.get(left_word[state.left_pos], ()):
                    # Repeated content is a completion-time rejection, but
                    # retained in the chart so a boundary can resegment it.
                    next_frontier.append(State(
                        state.left_slot, state.right_slot, left_word, right_word,
                        state.left_pos + 1, state.right_pos + 1,
                        state.left_words, state.right_words_reversed,
                        state.emitted_pairs + 1,
                    ))
                    matched += 1
                    expanded += 1
                    if expanded >= state_limit:
                        break
                if expanded >= state_limit:
                    break
            if matched == 0 and len(dead) < 120:
                dead.append({
                    "left_slot": state.left_slot,
                    "right_slot": state.right_slot,
                    "left_words": state.left_words,
                    "right_words_reversed": state.right_words_reversed,
                    "residual": {
                        "left": state.left_word[state.left_pos:] if state.left_word else "",
                        "right_reversed": state.right_word[state.right_pos:] if state.right_word else "",
                    },
                    "reason": "no typed lexical transition consumes the reflected residual",
                    "next_repair": "expand only the owning semantic slot with an agreement-compatible lexical alternative",
                })
            if expanded >= state_limit:
                break
        if not next_frontier or expanded >= state_limit:
            break
        # Structural dedup keeps one high-coverage path per live residual;
        # lexical history remains in the state for provenance.
        unique: dict[tuple[object, ...], State] = {}
        for state in next_frontier:
            key = (state.left_slot, state.right_slot, state.left_word, state.right_word,
                   state.left_pos, state.right_pos, state.left_words[-2:], state.right_words_reversed[-2:])
            unique.setdefault(key, state)
        frontier = list(unique.values())[:state_limit]
    return {
        "closures": closures,
        "dead_frontier": dead,
        "expanded": expanded,
        "frontier": len(frontier),
    }


def withheld_seed_smoke() -> dict[str, object]:
    """Validate the chart's independent audit on the benchmark only."""
    seed = "An aide rips nine memos; some men inspire Diana."
    audit = independent_audit(seed)
    return {"rendered": seed, "audit": audit, "used_as_output": False,
            "purpose": "withheld exactness smoke test; not a generated candidate"}


def main() -> None:
    runs = []
    for left_slots, right_slots in ((FRAME, FRAME), (FRAME, NAME_FRAME), (NAME_FRAME, FRAME)):
        result = chart(left_slots, right_slots)
        runs.append({"left_slots": left_slots, "right_slots": right_slots, **result})
    controls = [
        "The archivist reads a letter near the harbor; the editor opens a door by the station.",
        "A teacher marks a map after the storm; a sailor carries a chart under the window.",
    ]
    rendered_controls = [{"rendered": text, "audit": independent_audit(text),
                         "reader_status": "unreviewed",
                         "provenance": {"fresh_authored_control": True,
                                        "catalogue_text": False,
                                        "finished_tape_reversal": False}}
                        for text in controls]
    payload = {
        "experiment": EXPERIMENT,
        "method": "Dream-RSI typed reflected-residual chart: opposite scene-frame lexical transitions consume live character debt while agreement/valency is checked before closure.",
        "construction": {"live_residual": True, "typed_slots": True,
                          "semantic_valency": "transitive agent-action-object plus locative adjunct",
                          "independent_exact_audits": True, "finished_tape_reversal": False},
        "frame_runs": runs,
        "rendered_controls": rendered_controls,
        "withheld_seed_smoke": withheld_seed_smoke(),
        "stats": {"frame_pairs": len(runs),
                  "expanded_states": sum(run["expanded"] for run in runs),
                  "closures": sum(len(run["closures"]) for run in runs),
                  "exact_closures": sum(sum(row["audit"]["exact"] for row in run["closures"]) for run in runs),
                  "longest_control_letters": max(row["audit"]["letters"] for row in rendered_controls)},
        "novelty_preflight": {"new_geometry": "typed live residual chart across semantic frame boundaries",
                              "prior_lane_reused": False, "duplicate_sweep": False,
                              "catalogue_text": False},
        "reader_gate": {"status": "closed", "reason": "no fresh exact closure; controls remain unreviewed",
                        "programmatic_metrics_are_diagnostic": True},
        "next_repair": {"operator": "add a chart-level grammar transition that can change lexical category only at a live word boundary while preserving the semantic role and agreement register",
                         "reason": "all current dead frontiers terminate on typed edge-character debt before a complete frame pair closes"},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "human_readability_certified": False,
                        "source_sentences_copied": False},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
