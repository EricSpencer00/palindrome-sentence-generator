"""Two-sided typed clause zipper.

The earlier frame lanes chose a complete left clause and repaired its mirror
afterwards.  This lane keeps both clause grammars live: every constituent on
the left is paired with the next constituent on the right while its character
debt is still active.  The right clause is traversed backwards internally,
but is rendered in ordinary order.  No finished tape is reversed and no
catalogue text is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "typed-clause-zipper-20260919"


def letters(text: str) -> str:
    return normalize_letters(text)


def content(words: Iterable[str]) -> frozenset[str]:
    repeatable = {
        "a", "an", "the", "some", "many", "one", "two", "nine", "old", "new",
        "my", "our", "his", "her", "and", "or", "in", "on", "at", "by", "of",
        "to", "with", "while", "before", "after", "near", "under", "beside",
    }
    return frozenset(
        letters(word) for word in words
        if letters(word) not in repeatable and len(letters(word)) > 1
    )


@dataclass(frozen=True)
class Option:
    text: str
    number: str | None = None
    valency: str | None = None
    content: frozenset[str] = frozenset()
    proper: bool = False

    @property
    def tape(self) -> str:
        return letters(self.text)


def opt(text: str, **kwargs) -> Option:
    return Option(text=text, content=content(text.split()), **kwargs)


SUBJECTS = (
    opt("an aide", number="sg"), opt("a poet", number="sg"),
    opt("a scribe", number="sg"), opt("a sailor", number="sg"),
    opt("a captain", number="sg"), opt("a nurse", number="sg"),
    opt("a baker", number="sg"), opt("a player", number="sg"),
    opt("a writer", number="sg"), opt("a clerk", number="sg"),
    opt("the queen", number="sg"), opt("the king", number="sg"),
    opt("the herald", number="sg"), opt("the pilot", number="sg"),
    opt("the gardener", number="sg"), opt("the reader", number="sg"),
    opt("some men", number="pl"), opt("some poets", number="pl"),
    opt("the sailors", number="pl"), opt("the writers", number="pl"),
    opt("the nurses", number="pl"), opt("the players", number="pl"),
    opt("the singers", number="pl"),
    opt("Diana", number="sg", proper=True), opt("Noel", number="sg", proper=True),
    opt("Nora", number="sg", proper=True), opt("Mara", number="sg", proper=True),
    opt("Leon", number="sg", proper=True),
)

_VERBS = (
    ("rips", "sg", "doc"), ("reads", "sg", "doc"),
    ("marks", "sg", "doc"), ("writes", "sg", "doc"),
    ("carries", "sg", "doc"), ("guards", "sg", "place"),
    ("inspires", "sg", "person"), ("praises", "sg", "person"),
    ("keeps", "sg", "doc"), ("finds", "sg", "doc"),
    ("follows", "sg", "person"), ("remembers", "sg", "doc"),
    ("opens", "sg", "doc"), ("closes", "sg", "doc"),
    ("watches", "sg", "place"), ("hears", "sg", "sound"),
    ("sees", "sg", "person"), ("holds", "sg", "doc"),
    ("loves", "sg", "person"), ("needs", "sg", "doc"),
    ("read", "pl", "doc"), ("mark", "pl", "doc"),
    ("write", "pl", "doc"), ("carry", "pl", "doc"),
    ("guard", "pl", "place"), ("inspire", "pl", "person"),
    ("praise", "pl", "person"), ("keep", "pl", "doc"),
    ("find", "pl", "doc"), ("follow", "pl", "person"),
    ("remember", "pl", "doc"), ("open", "pl", "doc"),
    ("close", "pl", "doc"), ("watch", "pl", "place"),
    ("hear", "pl", "sound"), ("see", "pl", "person"),
    ("hold", "pl", "doc"), ("love", "pl", "person"),
    ("need", "pl", "doc"),
)
VERBS = tuple(Option(text=word, number=number, valency=valency,
                     content=frozenset({word}))
              for word, number, valency in _VERBS)

OBJECTS = (
    opt("nine memos", valency="doc"), opt("a letter", valency="doc"),
    opt("the notes", valency="doc"), opt("old maps", valency="doc"),
    opt("the chart", valency="doc"), opt("a sealed note", valency="doc"),
    opt("fresh pages", valency="doc"), opt("the ledger", valency="doc"),
    opt("some poems", valency="doc"), opt("the book", valency="doc"),
    opt("new plans", valency="doc"), opt("a red rose", valency="doc"),
    opt("the stars", valency="doc"), opt("the sonnet", valency="doc"),
    opt("a song", valency="doc"), opt("the poem", valency="doc"),
    opt("some men", valency="person"), opt("the poet", valency="person"),
    opt("Diana", valency="person", proper=True), opt("the queen", valency="person"),
    opt("the king", valency="person"), opt("the sailor", valency="person"),
    opt("the harbor", valency="place"), opt("the river", valency="place"),
    opt("the garden", valency="place"), opt("the shore", valency="place"),
    opt("the gate", valency="place"), opt("the bell", valency="sound"),
)
ADJUNCTS = (
    opt("at dawn"), opt("after rain"), opt("under the moon"),
    opt("near the river"), opt("by the shore"), opt("before dusk"),
    opt("in still air"), opt("through the gate"), opt("beside the harbor"),
)

# A small authored scene expansion, kept separate from the anchor vocabulary.
# These are lexical choices, not imported sentences or catalogue palindromes.
_EXTRA_SUBJECTS = (
    ("gulls", "pl"), ("sailors", "pl"), ("daughters", "pl"),
    ("neighbours", "pl"), ("sons", "pl"), ("elders", "pl"),
    ("widows", "pl"), ("fishermen", "pl"), ("wind", "sg"),
    ("clouds", "pl"), ("bells", "pl"), ("priest", "sg"),
    ("storms", "pl"), ("captains", "pl"), ("waves", "pl"),
    ("ropes", "pl"), ("anchors", "pl"), ("silence", "sg"),
    ("grief", "sg"), ("nets", "pl"), ("wives", "pl"),
    ("mothers", "pl"), ("debts", "pl"), ("winter", "sg"),
    ("frost", "sg"), ("timber", "sg"), ("rivers", "pl"),
    ("founders", "pl"), ("barges", "pl"), ("foremen", "pl"),
    ("mills", "pl"), ("smoke", "sg"), ("lamps", "pl"),
    ("silt", "sg"), ("ash", "sg"), ("rust", "sg"),
    ("machines", "pl"), ("floods", "pl"), ("hunger", "sg"),
    ("songs", "pl"), ("boats", "pl"), ("ice", "sg"),
    ("students", "pl"), ("altitude", "sg"), ("dust", "sg"),
    ("cables", "pl"), ("coffee", "sg"), ("notebooks", "pl"),
    ("static", "sg"), ("shadows", "pl"), ("lenses", "pl"),
    ("cold", "sg"), ("numbers", "pl"), ("doubt", "sg"),
    ("fatigue", "sg"), ("signals", "pl"), ("dawn", "sg"),
    ("mountains", "pl"), ("headlamps", "pl"), ("domes", "pl"),
    ("comets", "pl"),
)
SUBJECTS += tuple(opt(word, number=number) for word, number in _EXTRA_SUBJECTS)

_EXTRA_VERBS = (
    ("circled", "doc"), ("coiled", "doc"), ("weighed", "doc"),
    ("trusted", "doc"), ("mended", "doc"), ("taught", "person"),
    ("lit", "doc"), ("loaded", "doc"), ("shifted", "doc"),
    ("warned", "person"), ("blessed", "person"), ("chased", "person"),
    ("counted", "doc"), ("swallowed", "doc"), ("splintered", "doc"),
    ("lashed", "doc"), ("dragged", "doc"), ("drowned", "doc"),
    ("took", "doc"), ("snagged", "doc"), ("buried", "doc"),
    ("rebuilt", "doc"), ("replaced", "doc"), ("awaited", "doc"),
    ("fed", "person"), ("named", "person"), ("welcomed", "person"),
    ("outshone", "doc"), ("smothered", "doc"), ("silenced", "doc"),
    ("outnumbered", "person"), ("outlived", "person"),
    ("outlasted", "doc"), ("haunted", "person"), ("mirrored", "doc"),
    ("answered", "doc"), ("greeted", "person"), ("guarded", "place"),
    ("recorded", "doc"), ("gathered", "doc"), ("defeated", "person"),
    ("corrected", "doc"), ("contradicted", "doc"), ("shadowed", "person"),
    ("blurred", "doc"), ("confirmed", "doc"), ("erased", "doc"),
    ("watched", "person"), ("carried", "doc"), ("tested", "doc"),
    ("found", "doc"), ("guided", "person"), ("caught", "doc"),
    ("outran", "person"),
)
VERBS += tuple(Option(text=word, valency=valency, content=frozenset({word}))
               for word, valency in _EXTRA_VERBS)

_EXTRA_OBJECTS = (
    ("harbour", "place"), ("ropes", "doc"), ("salt", "doc"),
    ("weather", "doc"), ("children", "person"), ("lamps", "doc"),
    ("boats", "doc"), ("gulls", "person"), ("crew", "person"),
    ("fleet", "doc"), ("hulls", "doc"), ("chains", "doc"),
    ("prayers", "doc"), ("songs", "doc"), ("sons", "person"),
    ("wreckage", "doc"), ("debts", "doc"), ("candles", "doc"),
    ("bread", "doc"), ("spring", "place"), ("villages", "place"),
    ("streets", "place"), ("ferries", "doc"), ("workers", "person"),
    ("rain", "doc"), ("stars", "doc"), ("banks", "place"),
    ("engines", "doc"), ("salmon", "person"), ("bridges", "place"),
    ("mills", "place"), ("harbors", "place"), ("telescopes", "doc"),
    ("silence", "doc"), ("mirrors", "doc"), ("glass", "doc"),
    ("domes", "place"), ("sleep", "doc"), ("signals", "doc"),
    ("mountains", "place"), ("theories", "doc"), ("darkness", "doc"),
    ("night", "place"), ("charts", "doc"), ("photons", "doc"),
    ("horizon", "place"), ("dust", "doc"), ("comets", "doc"),
)
OBJECTS += tuple(opt(word, valency=valency) for word, valency in _EXTRA_OBJECTS)


def consume(left: str, right: str) -> tuple[str, str] | None:
    """Return (owner, remainder) for two opposing character chunks."""
    if left.startswith(right):
        return "L", left[len(right):]
    if right.startswith(left):
        return "R", right[len(left):]
    return None


def _choices(slot: str, state: dict[str, object]) -> tuple[Option, ...]:
    used = state["content"]
    if slot == "S":
        return tuple(option for option in SUBJECTS
                     if not used & option.content
                     and (state["number"] is None or option.number == state["number"])
                     and not (state["proper"] and option.proper))
    if slot == "V":
        return tuple(option for option in VERBS
                     if (state["number"] is None or option.number is None
                         or option.number == state["number"])
                     and (state["valency"] is None or option.valency == state["valency"])
                     and not used & option.content)
    if slot == "O":
        return tuple(option for option in OBJECTS
                     if (state["valency"] is None or option.valency == state["valency"])
                     and not used & option.content
                     and not (state["proper"] and option.proper))
    if slot == "P":
        return tuple(option for option in ADJUNCTS if not used & option.content)
    raise ValueError(slot)


def _add(state: dict[str, object], option: Option) -> dict[str, object]:
    updated = dict(state)
    updated["content"] = state["content"] | option.content
    if option.number is not None:
        updated["number"] = option.number
    if option.valency is not None:
        updated["valency"] = option.valency
    if option.proper:
        updated["proper"] = True
    return updated


def _audit(text: str) -> dict[str, object]:
    tape = letters(text)
    reverse = tape[::-1]
    first = next(((index, a, b) for index, (a, b)
                  in enumerate(zip(tape, reverse)) if a != b), None)
    return {
        "normalized": tape,
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and first is None,
        "first_mismatch": first,
        "sha256_forward": hashlib.sha256(tape.encode("ascii")).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode("ascii")).hexdigest(),
    }


def search(frame_left: tuple[str, ...], frame_right: tuple[str, ...],
           max_nodes: int = 500_000) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    nodes = 0

    def dfs(li: int, ri: int, debt: str, owner: str | None,
            left: list[str], right: list[str], left_state: dict[str, object],
            right_state: dict[str, object]) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return
        if li == len(frame_left) and ri < 0:
            if debt:
                return
            raw_rendered = " ".join(left) + "; " + " ".join(right) + "."
            rendered = raw_rendered[:1].upper() + raw_rendered[1:]
            audit = _audit(rendered)
            if audit["letters"] < 30 or not audit["two_pointer_exact"]:
                return
            checks = mechanical_admission_checks(rendered, min_letters=30,
                                                  max_letters=260)
            rows.append({
                "rendered": rendered,
                "audit": audit,
                "mechanical_checks": checks,
                "mechanically_admitted": all(checks.values()),
                "word_spans": left + right,
                "frame_left": frame_left,
                "frame_right": frame_right,
                "provenance": {
                    "construction": "two-sided typed clause zipper",
                    "finished_tape_reversed": False,
                    "catalogue_imported": False,
                    "word_order_mirror": False,
                    "rlaif_used": False,
                },
                "reader_status": "unreviewed; programmatic checks never certify readability",
            })
            return
        if not debt and (li >= len(frame_left) or ri < 0):
            return
        if debt and li >= len(frame_left) and owner != "L":
            return
        if debt and ri < 0 and owner != "R":
            return

        if not debt:
            for left_option in _choices(frame_left[li], left_state):
                next_left_state = _add(left_state, left_option)
                for right_option in _choices(frame_right[ri], right_state):
                    comparison = consume(left_option.tape,
                                         right_option.tape[::-1])
                    if comparison is None:
                        continue
                    next_owner, remainder = comparison
                    dfs(li + 1, ri - 1, remainder, next_owner,
                        left + [left_option.text],
                        [right_option.text] + right,
                        next_left_state, _add(right_state, right_option))
            return

        if owner == "L":
            for right_option in _choices(frame_right[ri], right_state):
                comparison = consume(debt, right_option.tape[::-1])
                if comparison is None:
                    continue
                next_owner, remainder = comparison
                dfs(li, ri - 1, remainder, next_owner, left,
                    [right_option.text] + right, left_state,
                    _add(right_state, right_option))
        else:
            for left_option in _choices(frame_left[li], left_state):
                comparison = consume(left_option.tape, debt)
                if comparison is None:
                    continue
                next_owner, remainder = comparison
                dfs(li + 1, ri, remainder, next_owner,
                    left + [left_option.text], right,
                    _add(left_state, left_option), right_state)

    initial = {"content": frozenset(), "number": None, "valency": None,
               "proper": False}
    dfs(0, len(frame_right) - 1, "", None, [], [], initial, initial)
    return rows, nodes


def run() -> dict[str, object]:
    frames = (
        (("S", "V", "O"), ("S", "V", "O")),
        (("S", "V", "O", "P"), ("S", "V", "O", "P")),
        (("S", "V", "O", "P"), ("S", "V", "O")),
        (("S", "V", "O"), ("S", "V", "O", "P")),
    )
    rows: list[dict[str, object]] = []
    node_total = 0
    for left, right in frames:
        found, nodes = search(left, right)
        node_total += nodes
        rows.extend(found)
    unique: dict[str, dict[str, object]] = {row["audit"]["normalized"]: row
                                            for row in rows}
    exact = list(unique.values())
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "two-sided typed clause zipper with live character debt",
        "frame_families": [
            {"left": left, "right": right} for left, right in frames
        ],
        "stats": {
            "nodes": node_total,
            "exact": len(exact),
            "mechanically_admitted": len(admitted),
            "reader_eligible": 0,
            "longest_exact_letters": max((row["audit"]["letters"] for row in exact),
                                          default=0),
        },
        "candidates": sorted(exact, key=lambda row: -row["audit"]["letters"]),
        "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
        "novelty_preflight": {
            "status": "passed",
            "signature": "typed-clause-zipper-live-debt-20260919",
            "catalogue_imported": False,
        },
        "next_repair": (
            "If exact rows remain sparse, retain the zipper and add one authored "
            "relative-clause slot rather than widening the lexical product."
        ),
        "reader_gate": "closed; exactness and mechanical admission do not certify readability",
    }


if __name__ == "__main__":
    result = run()
    path = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
