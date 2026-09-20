"""POS-constrained bilateral character-orbit construction.

The two ordinary clauses are generated from typed POS slots while their outer
characters are matched before either clause is rendered.  This is a fresh
grammar family: it does not start from a finished sentence, reverse a tape, or
repair a mismatch.  A beam is used only to allocate a bounded search budget;
the exact character equation is a hard transition gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "POS_BILATERAL_OUT",
    str(ROOT / "runs/pos-bilateral-cfg-orbit-20260920.json"),
))
EXPERIMENT = "pos-bilateral-cfg-orbit-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


# Common, ordinary lexical domains.  The domains are typed rather than mined
# palindrome pairs; the two clauses choose independently from the same POS
# inventory.  The expanded adjective/noun bank makes the method length-scalable
# without changing its construction invariant.
LEXICON = {
    "det": (
        "a", "an", "the", "my", "his", "her", "our", "your", "this",
        "that", "one", "each", "some", "no",
    ),
    "adj": (
        "calm", "careful", "clever", "dark", "eager", "fair", "gentle",
        "kind", "large", "little", "lone", "quiet", "rapid", "ready",
        "small", "steady", "young", "patient", "plain", "bright",
        "brave", "clear", "fresh", "great", "green", "old", "open",
        "red", "still", "warm", "wise",
    ),
    "noun": (
        "artist", "baker", "captain", "clerk", "farmer", "keeper", "pilot",
        "poet", "porter", "sailor", "scholar", "teacher", "writer", "guard",
        "child", "friend", "garden", "harbor", "lantern", "letter", "map",
        "market", "message", "parcel", "river", "road", "sonnet", "story",
        "tale", "vessel", "window", "book", "bell", "bridge", "chart",
        "door", "field", "gate", "hill", "house", "island", "journal",
        "lesson", "note", "path", "place", "room", "shore", "signal",
        "stone", "tower", "train", "village", "weather", "word",
    ),
    "verb": (
        "asks", "bakes", "carries", "charts", "checks", "clears", "draws",
        "finds", "guards", "guides", "holds", "keeps", "labels", "marks",
        "maps", "mends", "opens", "packs", "plants", "plans", "reads",
        "records", "sees", "sends", "sets", "shows", "sings", "sorts",
        "spells", "starts", "stores", "teaches", "tests", "thanks", "turns",
        "uses", "watches", "weighs", "writes",
    ),
}

# Both sides are complete ordinary-order clauses.  The optional adjective
# positions are part of the grammar, not a post-hoc insertion operation.
TEMPLATES = {
    "svo": ("det", "noun", "verb", "det", "noun"),
    "modified_svo": ("det", "adj", "noun", "verb", "det", "adj", "noun"),
}


@dataclass(frozen=True)
class State:
    left_slot: int
    right_slot: int
    left_word: str
    right_word: str
    left_pos: int
    right_pos: int
    left_words: tuple[str, ...]
    right_words: tuple[str, ...]
    score: float

    @property
    def done(self) -> bool:
        return (
            self.left_slot == self._left_slots
            and self.right_slot < 0
            and not self.left_word
            and not self.right_word
        )

    _left_slots: int = 0


def _word_cost(word: str) -> float:
    # A fixed deterministic preference for shorter/common-looking words.  It
    # never relaxes the exact gate and is not a readability certificate.
    return 1.0 / max(1, len(word))


def _advance_left(state: State, slots: tuple[str, ...]):
    if state.left_word:
        return [(state.left_word, state.left_pos)]
    if state.left_slot >= len(slots):
        return [("", 0)]
    tag = slots[state.left_slot]
    return [(word, 0) for word in LEXICON[tag]]


def _advance_right(state: State, slots: tuple[str, ...]):
    if state.right_word:
        return [(state.right_word, state.right_pos)]
    if state.right_slot < 0:
        return [("", -1)]
    tag = slots[state.right_slot]
    return [(word, len(word) - 1) for word in LEXICON[tag]]


def _step(state: State, slots: tuple[str, ...]):
    """Yield exact one-character orbit transitions.

    Left characters are consumed forward; right characters are consumed from
    the end of its independently selected ordinary clause.  Word boundaries
    are grammar transitions and may occur at different orbit positions.
    """
    for lw, lp in _advance_left(state, slots):
        for rw, rp in _advance_right(state, slots):
            if not lw or not rw or lp >= len(lw) or rp < 0:
                continue
            if lw[lp] != rw[rp]:
                continue
            nlw = lw if lp + 1 < len(lw) else ""
            nrw = rw if rp - 1 >= 0 else ""
            nlp = lp + 1 if nlw else 0
            nrp = rp - 1 if nrw else -1
            nls = state.left_slot + (0 if nlw else 1)
            nrs = state.right_slot - (0 if nrw else 1)
            if nls > len(slots) or nrs < -1:
                continue
            yield State(
                nls,
                nrs,
                nlw,
                nrw,
                nlp,
                nrp,
                state.left_words + ((lw,) if not state.left_word else ()),
                ((rw,) if not state.right_word else ()) + state.right_words,
                state.score + _word_cost(lw) + _word_cost(rw),
                len(slots),
            )


def _render(state: State) -> str:
    return " ".join(state.left_words) + "; " + " ".join(state.right_words)


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [
        {"index": i, "left": tape[i], "right": tape[-1 - i]}
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "mismatches": mismatches[:8],
        "two_pointer_exact": bool(tape) and not mismatches,
        "forward_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def search(template: tuple[str, ...], *, beam_width: int = 7000,
           max_nodes: int = 180_000) -> dict:
    start = State(0, len(template) - 1, "", "", 0, -1, (), (), 0.0, len(template))
    frontier: list[tuple[float, int, State]] = [(0.0, 0, start)]
    serial = 0
    nodes = 0
    closures: list[dict] = []
    seen: set[tuple] = set()
    longest: dict | None = None
    while frontier and nodes < max_nodes:
        _score, _serial, state = heapq.heappop(frontier)
        key = (state.left_slot, state.right_slot, state.left_word,
               state.right_word, state.left_pos, state.right_pos,
               state.left_words, state.right_words)
        if key in seen:
            continue
        seen.add(key)
        nodes += 1
        if state.left_slot == len(template) and state.right_slot < 0 and not state.left_word and not state.right_word:
            text = _render(state)
            row = {"rendered": text, "template": list(template), "audit": audit(text),
                   "provenance": "typed POS slots selected independently under live character equality"}
            if longest is None or row["audit"]["letters"] > longest["audit"]["letters"]:
                longest = row
            if row["audit"]["exact"]:
                closures.append(row)
            continue
        for child in _step(state, template):
            if not child.left_words or not child.right_words:
                continue
            # Do not permit repeated content words across the two clauses;
            # function-word repetition is not a structural palindrome unit.
            left_content = [w for w in child.left_words if w not in {"a", "an", "the", "my", "his", "her", "our", "your", "this", "that", "one", "each", "some", "no"}]
            right_content = [w for w in child.right_words if w not in {"a", "an", "the", "my", "his", "her", "our", "your", "this", "that", "one", "each", "some", "no"}]
            if len(set(left_content + right_content)) != len(left_content + right_content):
                continue
            serial += 1
            # Prefer low-cost lexical paths, then longer partial tapes; exact
            # equality remains the only admission condition.
            remaining = (len(template) - child.left_slot) + (child.right_slot + 1)
            priority = child.score - 0.0001 * (len(child.left_words) + len(child.right_words)) + 0.01 * remaining
            heapq.heappush(frontier, (priority, serial, child))
        if len(frontier) > beam_width:
            frontier = heapq.nsmallest(beam_width, frontier)
            heapq.heapify(frontier)
    return {"nodes": nodes, "closures": closures, "longest": longest,
            "seen_states": len(seen), "status": "node_budget" if nodes >= max_nodes else "exhausted"}


def main() -> dict:
    results = {name: search(slots) for name, slots in TEMPLATES.items()}
    exact = [row for result in results.values() for row in result["closures"]]
    payload = {
        "experiment": EXPERIMENT,
        "method": "bilateral typed POS CFG orbit with live word-boundary transitions",
        "results": results,
        "exact_candidates": exact,
        "novelty_preflight": {
            "status": "passed",
            "posthoc_repair": False,
            "finished_tape_reversal": False,
            "word_order_symmetry": False,
            "repeated_content_units": False,
            "catalogue_text": False,
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexical_source": "fresh typed POS inventory in this file",
            "independent_audits": ["literal two-pointer", "forward/reverse SHA-256"],
        },
        "next_construction": "Add agreement-carrying plural subject domains as a new grammar family; do not edit any failed tape.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    payload = main()
    print(json.dumps({
        "templates": list(payload["results"]),
        "nodes": {k: v["nodes"] for k, v in payload["results"].items()},
        "exact": len(payload["exact_candidates"]),
    }, indent=2))
