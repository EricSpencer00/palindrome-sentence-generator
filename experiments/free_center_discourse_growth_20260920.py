"""Free-center discourse growth with independently selected phrase chunks."""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "free-center-discourse-growth-20260920.json"
EXPERIMENT_ID = "free-center-discourse-growth-20260920"
SIGNATURE = "free-internal-center|center-out-discourse-growth|independent-phrase-chunks|complete-surface-parse"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    bad = [(i, len(tape) - i - 1) for i in range(len(tape) // 2)
           if tape[i] != tape[-i - 1]]
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and not bad,
            "first_mismatch": bad[0] if bad else None,
            "sha256_forward": fwd, "sha256_reverse": rev,
            "sha_equal": fwd == rev}


def boundary_compatible(left: str, right: str) -> bool:
    """Check only the currently exposed center-adjacent overlap."""
    n = min(len(left), len(right))
    return not n or left[-n:] == right[:n][::-1]


@dataclass(frozen=True)
class Chunk:
    role: str
    text: str
    referent: str
    valency: str


def c(role: str, referent: str, valency: str, *texts: str) -> tuple[Chunk, ...]:
    return tuple(Chunk(role, text, referent, valency) for text in texts)


def slots() -> tuple[tuple[Chunk, ...], ...]:
    # One complete contemporary discourse template.  The pivot is selected
    # freely from connector/event/object positions, then phrase chunks grow
    # outward.  The sides are not pre-made sentence pairs.
    return (
        c("subject", "speaker", "agent", "Mara", "Noah", "the reader", "the keeper"),
        c("verb", "event", "finite", "reads", "opens", "keeps", "guards"),
        c("object", "theme", "patient", "the letter", "the gate", "the book", "a message"),
        c("connector", "relation", "discourse", "and", "while", "because", "but"),
        c("subject", "agent2", "agent", "the sailor", "the guard", "the poet", "the queen"),
        c("verb", "event2", "finite", "marks", "opens", "reads", "keeps"),
        c("object", "theme2", "patient", "the seal", "the door", "the song", "the vow"),
    )


def complete_surface(text: str) -> bool:
    """Small final grammar gate for generated controls/candidates."""
    words = text.split()
    return len(words) >= 3 and any(w in words for w in ("and", "while", "because", "but"))


def controls() -> list[dict[str, object]]:
    texts = [
        "Mara reads the letter and the sailor marks the seal.",
        "Noah opens the gate while the guard keeps the vow.",
        "The reader keeps the book because the poet reads the song.",
        "The keeper guards a message but the queen opens the door.",
        "Mara opens the door and the sailor reads the letter.",
        "Noah keeps the vow while the poet marks the seal.",
        "The reader reads the book because the guard opens the gate.",
        "The keeper keeps the letter but the queen guards the door.",
        "Mara guards the gate and the poet reads the vow.",
        "Noah reads a message while the sailor opens the seal.",
        "The reader opens the book because the queen marks the song.",
        "The keeper reads the letter but the guard keeps the door.",
        "Mara keeps the book and the sailor guards the vow.",
        "Noah marks the seal while the poet opens the gate.",
        "The reader guards the door because the queen reads the song.",
        "The keeper opens the message but the guard marks the letter.",
        "Mara reads the book and the queen keeps the seal.",
        "Noah guards the letter while the sailor reads the vow.",
        "The reader marks the gate because the poet opens the door.",
        "The keeper reads the song but the guard guards the book.",
    ]
    return [{"rendered": t, "audit": audit(t),
             "reader_status": "complete contemporary prose control; not exact"} for t in texts]


def run(*, state_limit: int = 100_000) -> dict[str, object]:
    grammar = slots()
    states = pruned = advances = center_pruned = 0
    candidates: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []

    def grow(pivot: int, center: Chunk, depth: int, left_side: str, right_side: str,
             chosen_left: tuple[Chunk, ...], chosen_right: tuple[Chunk, ...]) -> None:
        nonlocal states, pruned, advances, center_pruned
        if states >= state_limit:
            return
        lo = pivot - depth
        hi = pivot + depth
        if lo < 0 and hi >= len(grammar):
            rendered = " ".join(x.text for x in chosen_left + (center,) + chosen_right)
            if not complete_surface(rendered):
                center_pruned += 1
                return
            checked = audit(rendered)
            if checked["exact"]:
                candidates.append({"rendered": rendered, "audit": checked,
                    "provenance": {"construction": "free-internal-center discourse growth",
                        "pivot_role": grammar[pivot][0].role, "left_roles": [x.role for x in chosen_left],
                        "right_roles": [x.role for x in chosen_right], "phrase_chunks": True,
                        "finished_tape_reversal": False, "post_hoc_repair": False,
                        "catalogue_text": False, "aligned_token_mirror": False},
                    "reader_status": "unreviewed; exactness does not certify readability"})
            return
        if lo < 0 or hi >= len(grammar):
            center_pruned += 1
            return
        for left_chunk in grammar[lo]:
            for right_chunk in grammar[hi]:
                states += 1
                new_left = letters(left_chunk.text) + left_side
                new_right = right_side + letters(right_chunk.text)
                if not boundary_compatible(new_left, new_right):
                    pruned += 1
                    if len(witnesses) < 24:
                        z = left_chunk.text + " " + center.text + " " + right_chunk.text
                        witnesses.append({"rendered": z, "depth": depth,
                                          "audit": audit(z), "reader_status": "diagnostic center-growth witness"})
                    continue
                advances += 1
                grow(pivot, center, depth + 1, new_left, new_right,
                     (left_chunk,) + chosen_left, chosen_right + (right_chunk,))

    # Free center choices are semantic positions, not a fixed seam template.
    # For multi-option pivots, each center phrase is selected before growth.
    for pivot in range(1, len(grammar) - 1):
        for center in grammar[pivot]:
            grow(pivot, center, 1, "", "", (), ())
            if states >= state_limit:
                break
        if states >= state_limit:
            break

    candidates.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    result = {"experiment": EXPERIMENT_ID, "method": "free internal center synchronous discourse growth",
              "complete_prose_controls": controls(), "candidates": candidates, "witnesses": witnesses,
              "stats": {"pivot_positions": len(grammar) - 2, "states": states, "pruned": pruned,
                        "center_pruned": center_pruned, "chart_advances": advances, "exact": len(candidates)},
              "provenance": {"novelty_signature": SIGNATURE,
                  "novelty_preflight": "registry search found no exact signature; this lane grows a single discourse from a free internal pivot rather than pairing complete clauses",
                  "free_internal_center": True, "independent_phrase_chunks": True,
                  "complete_surface_parse_gate": True, "independent_pointer_sha_audit": True,
                  "finished_tape_reversal": False, "post_hoc_repair": False,
                  "catalogue_text": False, "aligned_token_mirror": False,
                  "next_construction": "allow a two-word pivot connective and attach an optional relative clause to one discourse referent",
                  "reader_next_test": "blind the 20 controls against word-shuffled versions before any exact output is promoted"}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
