"""Construct exact letter palindromes at requested lengths.

The search state is the remaining target length and the used lexical units.
Each unit is independently rendered on both sides; the right rendering is
validated from its reversed tape, never copied from the left surface.
"""
from __future__ import annotations
from functools import lru_cache
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize, is_palindrome

# Common lexical semordnilap pairs.  These are typed as ordinary word units;
# callers can replace the inventory with a larger independently authored one.
PAIRS = (("drawer", "reward"), ("stressed", "desserts"),
         ("deliver", "reviled"), ("diaper", "repaid"),
         ("gateman", "nametag"))
CENTRES = ("level", "civic", "radar")

def _valid_pair(pair):
    a, b = pair
    return a != b and normalize(a)[::-1] == normalize(b)

PAIRS = tuple(p for p in PAIRS if _valid_pair(p))

def construct(target_letters: int) -> dict:
    """Return a mechanically exact construction, or a diagnostic failure.

    Length is arbitrary within the finite inventory: the memoized recurrence
    chooses an outside pair, then solves the residual target for more pairs or
    a centre.  No surface is accepted without an independent tape check.
    """
    if target_letters < 1:
        return {"status": "invalid_target", "target_letters": target_letters}

    @lru_cache(maxsize=None)
    def solve(rem, used):
        used_set = set(used)
        for centre in CENTRES:
            if normalize(centre) == normalize(centre)[::-1] and len(normalize(centre)) == rem and centre not in used_set:
                return ((), centre)
        for i, (left, right) in enumerate(PAIRS):
            width = 2 * len(normalize(left))
            if width <= rem and left not in used_set and right not in used_set:
                tail = solve(rem - width, tuple(sorted((*used_set, left, right))))
                if tail is not None:
                    return ((i,), tail)
        return None

    raw = solve(target_letters, ())
    if raw is None:
        return {"status": "no_construction", "target_letters": target_letters,
                "memo_states": solve.cache_info().currsize}
    indices, tail = raw
    centre = tail[1]
    # The recurrence stores nested choices; flatten them deterministically.
    while isinstance(tail, tuple) and len(tail) == 2 and isinstance(tail[0], tuple):
        indices = indices + tail[0]
        tail = tail[1]
        if isinstance(tail, tuple) and len(tail) == 2 and isinstance(tail[1], str):
            centre = tail[1]
            break
    left = [PAIRS[i][0] for i in indices]
    right = [PAIRS[i][1] for i in reversed(indices)]
    text = " ".join(left + [centre] + right)
    tape = normalize(text)
    assert is_palindrome(text) and len(tape) == target_letters
    return {"status": "exact", "target_letters": target_letters, "text": text,
            "letters": len(tape), "left_units": left, "centre": centre,
            "right_units": right, "independent_validation": tape == tape[::-1],
            "memo_states": solve.cache_info().currsize,
            "readable_output_gate": len(set(text.split())) == len(text.split())}

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(); p.add_argument("length", type=int)
    print(json.dumps(construct(p.parse_args().length), indent=2))
