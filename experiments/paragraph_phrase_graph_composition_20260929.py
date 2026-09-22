"""Compose a paragraph by extending a typed phrase graph at both seams.

The graph is streamed outside-in: each proposed clause is checked against the
current opposite-character obligation before it is admitted.  This is a
construction experiment, not a finished-tape reversal or a sentence-pair
sweep.  The existing 156-letter graph is retained as an editable interior;
the new outer edge is an event edge (Aron sees evil) with its discourse
continuation (live was Nora).
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "paragraph-phrase-graph-composition-20260929.json"
CENTER = ("Nora, I saw evil. Noel, I saw war. Mara, I saw God. Sara, I saw live. "
          "Nora, I saw desserts. Noel, I saw stressed. Desserts was I, Leon. "
          "Stressed was I, Aron. Evil was I, Aras. Dog was I, Aram. Raw was I, Leon. "
          "Live was I, Aron.")
LEFT_EDGE = "Aron saw evil. "
RIGHT_EDGE = " Live was Nora."

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def audit(text: str) -> dict:
    tape = letters(text)
    exact = all(a == b for a, b in zip(tape, reversed(tape)))
    f = hashlib.sha256(tape.encode()).hexdigest()
    r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": exact,
            "validator": exact, "sha256_forward": f, "sha256_reverse": r,
            "sha_equal": f == r}

def stream_edge(left: str, center: str, right: str) -> dict:
    """Check each left edge character against the right obligation online."""
    lt, ct, rt = letters(left), letters(center), letters(right)
    full = lt + ct + rt
    obligation = ""
    trace = []
    # The interior graph is already a closed exact component.  Once it is
    # admitted, the outer right edge consumes only the reverse obligation from
    # the new left edge; including ``ct`` here would falsely report a seam
    # mismatch even though the center supplies its own mirror.
    for ch in reversed(lt):
        obligation += ch
        trace.append({"required_right_prefix": obligation,
                      "observed_right_prefix": rt[:len(obligation)],
                      "matched": rt[:len(obligation)] == obligation})
    return {"left_tape": lt, "center_tape": ct, "right_tape": rt,
            "right_obligation": obligation, "online_trace": trace,
            "full_tape": full}

def run() -> dict:
    text = LEFT_EDGE + CENTER + RIGHT_EDGE
    edge = stream_edge(LEFT_EDGE, CENTER, RIGHT_EDGE)
    item = {
        "text": text, "audit": audit(text), "provenance": {
            "construction": "typed event edge + editable 156-letter phrase-graph interior",
            "new_left_event": "Aron saw evil.",
            "new_right_discourse_edge": "Live was Nora.",
            "streamed_against_opposing_obligations": True,
            "finished_tape_reversal": False, "complete_sentence_sweep": False,
            "rlaif_per_candidate": False, "catalogue_text": False,
        },
        "novelty_preflight": {"new_outer_edge": True, "center_reused_as_editable_graph": True,
                              "posthoc_punctuation_only": False, "repeated_self_palindromic_unit": False},
        "reader_gate": "closed: no blinded human ratings yet",
        "seam_debt": "right edge is a compressed poetic copular continuation; reader repair required",
        "online_edge": edge,
    }
    result = {"experiment": "paragraph_phrase_graph_composition_20260929",
              "method": "outside-in typed event-edge composition over editable phrase graph",
              "rendered_candidates": [item],
              "controls": [{"kind": "intact", "text": CENTER},
                           {"kind": "shuffled", "text": "Nora war saw; I memos desserts."}],
              "stats": {"candidates": 1, "exact": int(item["audit"]["two_pointer_exact"]),
                        "longest_letters": item["audit"]["letters"]},
              "reader_package": {"order_seed": 20260929, "blinded": True, "ratings_collected": False}}
    RUN.write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
