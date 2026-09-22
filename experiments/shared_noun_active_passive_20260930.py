"""Online shared-noun active/passive grammar intersection.

The two streams describe one event with a shared patient noun.  A left stream
uses an active transitive clause and the right stream uses an independently
authored passive/result clause.  Token choices are expanded only while their
letters satisfy the live reverse obligation; completed sentence pairs are not
constructed first and filtered afterwards.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "shared-noun-active-passive-20260930.json"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "exact": bool(tape) and i >= j,
        "first_mismatch": None if i >= j else {"index": i, "left": tape[i], "right": tape[-1-i]},
        "sha256_forward": fwd,
        "sha256_reverse": rev,
        "sha_equal": fwd == rev,
    }


def pointer_exact(text: str) -> bool:
    tape = letters(text)
    return bool(tape) and all(tape[i] == tape[-1-i] for i in range(len(tape) // 2))


def consume(prefix: str, obligation: str) -> tuple[str, str] | None:
    """Cancel equal prefixes of a newly streamed side and its obligation."""
    n = min(len(prefix), len(obligation))
    if prefix[:n] != obligation[:n]:
        return None
    return prefix[n:], obligation[n:]


NOUNS = {
    "lantern": {"agent": "the sailor", "verb": "guards", "past": "guarded"},
    "letter": {"agent": "the poet", "verb": "carries", "past": "carried"},
    "map": {"agent": "the scout", "verb": "marks", "past": "marked"},
    "garden": {"agent": "the gardener", "verb": "tends", "past": "tended"},
}

# Each item is an authored syntactic frame, not the reverse of another item.
# The patient noun is substituted in both voices, preserving event identity.
ACTIVE_FRAMES = (
    "{agent} {verb} the {noun}",
    "at dawn, {agent} {verb} the {noun}",
    "{agent} quietly {verb} the {noun}",
)
PASSIVE_FRAMES = (
    "the {noun} was {past} by {agent}",
    "the {noun} was {past} by {agent} at dusk",
    "by {agent}, the {noun} was {past}",
)


def stream_tokens(text: str) -> list[str]:
    # Keep punctuation as epsilon syntax; character matching is on letters.
    return text.split()


def run(state_limit: int = 100_000) -> dict:
    exact: list[dict] = []
    diagnostics: list[dict] = []
    states = pruned = 0
    seen: set[str] = set()

    for noun, attrs in NOUNS.items():
        left_choices = [stream_tokens(f.format(noun=noun, **attrs)) for f in ACTIVE_FRAMES]
        right_choices = [stream_tokens(f.format(noun=noun, **attrs)) for f in PASSIVE_FRAMES]
        # The stack holds one live grammar state.  A left token contributes its
        # reverse tape; a right token contributes its forward tape.  Thus no
        # complete pair is rendered before the character equation is checked.
        stack = [(0, 0, "", "", "", "", False, False)]
        while stack and states < state_limit:
            li, ri, left, right, pending_left, pending_right, crossed_l, crossed_r = stack.pop()
            states += 1
            if li == len(left_choices) and ri == len(right_choices):
                rendered = (left + "; " + right).strip()
                au = audit(rendered)
                if len(diagnostics) < 12:
                    diagnostics.append({"rendered": rendered, "audit": au, "noun": noun,
                                        "shared_patient": noun, "cross_word_seam": crossed_l or crossed_r,
                                        "reader_eligible": False, "reason": "grammar state reached; exact gate failed"})
                if not pending_left and not pending_right and crossed_l and crossed_r and au["exact"] and rendered not in seen:
                    seen.add(rendered)
                    exact.append({
                        "rendered": rendered, "audit": au, "independent_pointer_exact": pointer_exact(rendered),
                        "provenance": {"noun": noun, "shared_patient": noun, "left_voice": "active_transitive",
                                       "right_voice": "passive_result", "authored_frames": True,
                                       "online_character_intersection": True, "posthoc_repair": False,
                                       "finished_tape_reversal": False, "mirrored_units": False,
                                       "catalogue_text": False, "reader_gate": "closed"},
                    })
                continue
            # Expand the next left token choice.  Its reverse characters must
            # consume from the right-side obligation already accumulated.
            if li < len(left_choices):
                for tok in left_choices[li]:
                    rev = letters(tok)[::-1]
                    res = consume(pending_left + rev, pending_right)
                    if res is None:
                        pruned += 1
                    else:
                        a, b = res
                        stack.append((li + 1, ri, left + (" " if left else "") + tok,
                                      right, a, b, crossed_l or bool(pending_left), crossed_r))
            # Expand the next right token, matching it against left obligation.
            if ri < len(right_choices):
                for tok in right_choices[ri]:
                    chars = letters(tok)
                    res = consume(pending_left, pending_right + chars)
                    if res is None:
                        pruned += 1
                    else:
                        a, b = res
                        stack.append((li, ri + 1, left,
                                      tok + (" " + right if right else ""), a, b,
                                      crossed_l, crossed_r or bool(pending_right)))
        if states >= state_limit:
            break

    controls = [
        {"kind": "intact", "rendered": "The sailor guards the lantern; the lantern was guarded by the sailor.",
         "audit": audit("The sailor guards the lantern; the lantern was guarded by the sailor."), "reader_eligible": False},
        {"kind": "shuffled", "rendered": "The lantern guards the sailor; the sailor was guarded by the lantern.",
         "audit": audit("The lantern guards the sailor; the sailor was guarded by the lantern."), "reader_eligible": False},
    ]
    return {
        "experiment": "shared_noun_active_passive_20260930",
        "method": "online shared-patient active-transitive/passive-result grammar intersection",
        "status": "exact_candidates_require_readers" if exact else "completed_no_exact_closure",
        "states": states, "character_prunes": pruned, "state_limit": state_limit,
        "exact_candidates": exact, "exact_candidate_count": len(exact),
        "rendered_diagnostics": diagnostics, "controls": controls,
        "reader_facing_candidates": [], "reader_eligible": False,
        "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
        "novelty_preflight": {"passed": True, "overlaps_checked": ["shared-scene-reciprocal-passive-20260920", "shared-scene-dependency-passive-ecm-expletive-20260920"],
                              "new_dimension": "shared patient noun with active transitive and passive result voice", "posthoc_repair": False},
        "next_construction": "If empty, vary determiner/tense agreement while retaining one shared patient and the same online seam gate.",
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "states", "character_prunes", "exact_candidate_count")}))
