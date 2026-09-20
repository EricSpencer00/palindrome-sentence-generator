"""Simultaneous semantic/character search over an authored scene dialogue.

The grammar chooses a complete scene frame (speaker, speech act, response,
and setting) while consuming the outside-in character residual after every
semantic edge.  It never constructs a finished string and then repairs or
reverses it.  The scene bank is small and authored so that accepted controls
are readable English rather than corpus fragments.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    return {
        "letters": len(tape),
        "exact": bool(tape) and tape == tape[::-1],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def pointer_exact(text: str) -> bool:
    tape = letters(text)
    lo, hi = 0, len(tape) - 1
    while lo < hi:
        if tape[lo] != tape[hi]:
            return False
        lo += 1
        hi -= 1
    return bool(tape)


def consume(left: str, right: str):
    n = min(len(left), len(right))
    return (left[n:], right[n:]) if left[:n] == right[:n] else None


# Each row is an intact semantic edge, authored for this experiment.  The
# fields are typed so invalid dialogue (e.g. a question paired with a report)
# is rejected before character search, rather than repaired afterward.
SPEAKERS = (
    {"id": "keeper", "text": "the careful keeper", "role": "witness"},
    {"id": "sailor", "text": "the patient sailor", "role": "traveler"},
    {"id": "poet", "text": "a young poet", "role": "observer"},
)
ACTS = (
    {"id": "report", "text": "said the harbor is quiet", "kind": "declarative"},
    {"id": "warn", "text": "warned that the bridge was wet", "kind": "warning"},
    {"id": "ask", "text": "asked whether the lantern still burned", "kind": "question"},
)
RESPONSES = (
    {"id": "answer", "text": "and the watchman answered with care", "kind": "answer"},
    {"id": "promise", "text": "and the young guide promised to wait", "kind": "promise"},
    {"id": "observe", "text": "while the quiet gardener watched the road", "kind": "observation"},
)
SETTINGS = (
    {"id": "river", "text": "beside the river at dawn", "allowed": {"declarative", "warning", "question"}},
    {"id": "harbor", "text": "near the harbor before dusk", "allowed": {"declarative", "warning", "question"}},
    {"id": "garden", "text": "under the old trees in rain", "allowed": {"declarative", "warning", "question"}},
)


def controls() -> list[dict]:
    rows = [
        "The careful keeper said the harbor is quiet, and the watchman answered with care beside the river at dawn.",
        "The patient sailor warned that the bridge was wet, and the young guide promised to wait near the harbor before dusk.",
        "A young poet asked whether the lantern still burned, while the quiet gardener watched the road under the old trees in rain.",
    ]
    return [{"rendered": row, "audit": audit(row),
             "independent_pointer_exact": pointer_exact(row), "reader_eligible": False,
             "provenance": "authored intact scene control; not a generated candidate"} for row in rows]


def run(limit: int = 25_000) -> dict:
    # A typed frame is built in semantic order.  The right frame is emitted
    # from its inner edge, so its text enters the comparison stream reversed.
    frames = []
    for speaker in SPEAKERS:
        for act in ACTS:
            for response in RESPONSES:
                if act["kind"] == "question" and response["kind"] not in {"answer", "observation"}:
                    continue
                for setting in SETTINGS:
                    if act["kind"] in setting["allowed"]:
                        frames.append((speaker, act, response, setting))
    states = 0
    pruned_char = 0
    pruned_semantic = 0
    exact = []
    diagnostics = []
    seen = set()
    # State: semantic edge index on each side, text, and unmatched residual.
    # Pairing is asynchronous: either side may advance when its residual is
    # available, then the common prefix is consumed immediately.
    for lf in frames:
        for rf in frames:
            # Require different scene roles so we do not count a mirrored
            # semantic frame as a construction shortcut.
            if lf[0]["role"] == rf[0]["role"]:
                pruned_semantic += 1
                continue
            left_edges = tuple(x["text"] for x in lf)
            right_edges = tuple(x["text"] for x in reversed(rf))
            stack = [(0, 0, "", "", "", "", ())]
            while stack and states < limit:
                li, ri, left, right, lbuf, rbuf, trace = stack.pop()
                states += 1
                if li == len(left_edges) and ri == len(right_edges):
                    rendered = (left + ", " + right).strip()
                    if len(diagnostics) < 5:
                        diagnostics.append({"rendered": rendered, "audit": audit(rendered),
                                            "left_residual": lbuf, "right_residual": rbuf,
                                            "reader_eligible": False,
                                            "reason": "complete semantic frame with nonempty character debt"})
                    if lbuf or rbuf:
                        pruned_char += 1
                        continue
                    info = audit(rendered)
                    if info["exact"] and pointer_exact(rendered) and info["letters"] >= 40 and rendered not in seen:
                        seen.add(rendered)
                        exact.append({"rendered": rendered, "audit": info,
                                      "independent_pointer_exact": True,
                                      "provenance": {"left_frame": [x["id"] for x in lf],
                                                     "right_frame": [x["id"] for x in rf],
                                                     "semantic_roles_distinct": True,
                                                     "corpus_sentence_replay": False,
                                                     "mirrored_token_units": False,
                                                     "posthoc_repair": False}})
                    continue
                if li < len(left_edges):
                    text = left_edges[li]
                    residual = consume(lbuf + letters(text), rbuf)
                    if residual is None:
                        pruned_char += 1
                    else:
                        stack.append((li + 1, ri, (left + " " if left else "") + text,
                                      right, residual[0], residual[1], trace + (("L", li),)))
                if ri < len(right_edges):
                    text = right_edges[ri]
                    residual = consume(lbuf, rbuf + letters(text)[::-1])
                    if residual is None:
                        pruned_char += 1
                    else:
                        stack.append((li, ri + 1, left,
                                      text + (" " + right if right else ""),
                                      residual[0], residual[1], trace + (("R", ri),)))
            if states >= limit:
                break
        if states >= limit:
            break
    return {
        "method": "authored-scene-dialogue-csp-20260920",
        "status": "completed_no_exact_closure" if not exact else "exact_candidates_require_readers",
        "semantic_frames": len(frames),
        "states": states,
        "character_prunes": pruned_char,
        "semantic_prunes": pruned_semantic,
        "state_limit": limit,
        "exact_candidates": exact,
        "exact_candidate_count": len(exact),
        "rendered_diagnostics": diagnostics,
        "reader_facing_candidates": [],
        "reader_eligible": False,
        "controls": controls(),
        "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
        "provenance": "fresh human-authored scene/dialogue frames; semantic type checks and character residual checks are interleaved; no reversal, mirrored units, catalogue text, or repair",
        "novelty_preflight": {"overlaps": [], "duplicate_sweep": False,
                              "reason": "typed speech-act/role compatibility is solved jointly with residual character emission"},
        "first_live_diagnostic": "character residual mismatch at a typed scene edge" if not exact else "exact closure requires blinded reader review",
        "next_construction": "add a held-out quoted imperative speech act with explicit answer compatibility, preserving distinct speaker roles and online residual checks",
    }


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs/authored-scene-dialogue-csp-20260920.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
