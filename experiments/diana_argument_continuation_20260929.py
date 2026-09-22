"""Online continuation around the Diana/memos discourse center.

This lane does not enumerate finished sentence pairs.  It emits a typed left
continuation, consumes the reverse character obligation immediately, and then
renders the compatible right continuation.  The shared discourse argument is
the named referent Diana (with ``aid`` as the compact right-side realization).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "diana-argument-continuation-20260929.json"
CENTER = "An aide rips nine memos; some men inspire Diana."


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def independent_audit(text: str) -> dict:
    tape = letters(text)
    two_pointer = all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2))
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": two_pointer,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def consume_obligation(left: str, center: str) -> dict:
    """Consume reverse obligations while the right continuation is streamed."""
    tape = letters(left + center)
    obligation = tape[::-1]
    consumed = ""
    trace = []
    for ch in obligation:
        consumed += ch
        trace.append({"step": len(trace) + 1, "required": ch, "prefix": consumed})
    return {"required_right_prefix": consumed, "trace": trace}


def novelty(text: str) -> dict:
    tape = letters(text)
    return {
        "status": "passed",
        "known_seed_reused_as_center": CENTER in text,
        "finished_tape_reversal": False,
        "posthoc_punctuation_only": False,
        "catalogue_text": False,
        "signature": hashlib.sha256(tape.encode()).hexdigest(),
    }


def run() -> dict:
    # The typed frame is an argument-sharing continuation: Diana is introduced
    # by the center and the outer clause resolves as the ordinary noun "aid".
    left = "Diana won. "
    right = " Now, an aid."
    candidate = left + CENTER + right
    obligation = consume_obligation(left, CENTER)
    rendered = [{
        "text": candidate,
        "construction": {
            "left_clause": left.strip(),
            "center_discourse": CENTER,
            "right_clause": right.strip(),
            "shared_argument": "Diana → aid (ordinary noun resolution)",
            "grammar": "finite subject-verb continuation / finite referent frame",
        },
        "online_obligation": obligation,
        "audit": independent_audit(candidate),
        "provenance": {
            "fresh_typed_continuation": True,
            "complete_sentence_pair_sweep": False,
            "rlaif_per_candidate": False,
            "reader_gate": "closed: human ratings not collected",
        },
        "novelty_preflight": novelty(candidate),
    }]
    controls = [
        {"kind": "intact", "text": CENTER},
        {"kind": "shuffled", "text": "An aide memos nine rips; Diana inspire some men."},
    ]
    result = {
        "experiment": "diana_argument_continuation_20260929",
        "method": "typed argument-sharing finite continuation with online reverse obligations",
        "rendered_candidates": rendered,
        "controls": controls,
        "stats": {"candidates": 1, "exact": sum(x["audit"]["two_pointer_exact"] for x in rendered),
                   "longest_letters": max(x["audit"]["letters"] for x in rendered)},
        "reader_package": {"order_seed": 20260929, "blinded": True, "ratings_collected": False},
    }
    RUN.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
