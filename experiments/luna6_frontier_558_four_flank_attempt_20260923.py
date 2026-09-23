"""One four-flank event replacement using the 558-letter repair frontier."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome

PARENT_PATH = ROOT / "runs/incumbent-550-typed-center-product-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-frontier-558-four-flank-attempt-20260923.json"
PARENT_ID = "typed-center-25"
PARENT_SHA = "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"
LEFT_SPAN = (244, 261)
RIGHT_SPAN = (297, 314)
LEFT_NEW = "Nadia delivered the ledger before dawn."
RIGHT_NEW = "The flood rose, and Tara warned Aidan."


def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isascii() and c.isalpha())


def raw_cut(text: str, offset: int) -> int:
    seen = 0
    for i, c in enumerate(text):
        if c.isascii() and c.isalpha():
            if seen == offset:
                return i
            seen += 1
    if seen == offset:
        return len(text)
    raise ValueError(offset)


def outside_in(s: str) -> dict:
    tape = letters(s)
    for i in range(len(tape) // 2):
        j = len(tape) - i - 1
        if tape[i] != tape[j]:
            return {"letters": len(tape), "exact": False,
                    "first_mismatch": {"left": i, "right": j,
                                       "left_char": tape[i], "right_char": tape[j]}}
    return {"letters": len(tape), "exact": True, "first_mismatch": None}


def tokens(s: str) -> list[str]:
    return [letters(x) for x in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", s)]


def main() -> None:
    source = json.loads(PARENT_PATH.read_text())
    row = next(r for r in source["rows"] if r["id"] == PARENT_ID)
    parent = row["rendered"]
    p = letters(parent)
    assert len(p) == 558
    assert hashlib.sha256(p.encode()).hexdigest() == PARENT_SHA
    assert outside_in(parent)["exact"] and is_palindrome(parent)
    assert RIGHT_SPAN == (len(p) - LEFT_SPAN[1], len(p) - LEFT_SPAN[0])
    assert p[slice(*LEFT_SPAN)] == "nadiadeliversmaps"
    assert p[slice(*RIGHT_SPAN)] == "spamsreviledaidan"

    # Novelty preflight: exact geometry and authored phrases in tracked history.
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    query = subprocess.run(
        ["git", "grep", "-n", "-F", "-e", "[244,261)", "-e", "[297,314)",
         head, "--", "runs", "experiments", "docs", "data"],
        cwd=ROOT, check=False, capture_output=True, text=True)
    phrase = subprocess.run(
        ["git", "grep", "-n", "-F", "-i", "-e", LEFT_NEW, "-e", RIGHT_NEW,
         head, "--", "runs", "experiments", "docs", "data"],
        cwd=ROOT, check=False, capture_output=True, text=True)
    geometry_hits = query.stdout.splitlines() if query.returncode == 0 else []
    phrase_hits = phrase.stdout.splitlines() if phrase.returncode == 0 else []

    l0, l1 = (raw_cut(parent, x) for x in LEFT_SPAN)
    r0, r1 = (raw_cut(parent, x) for x in RIGHT_SPAN)
    candidate = parent[:l0] + LEFT_NEW + " " + parent[l1:r0] + RIGHT_NEW + " " + parent[r1:]
    child = letters(candidate)
    left, right = letters(LEFT_NEW), letters(RIGHT_NEW)
    reverse_right = right[::-1]
    cursor = 0
    while cursor < min(len(left), len(reverse_right)) and left[cursor] == reverse_right[cursor]:
        cursor += 1
    exact = outside_in(candidate)
    project_exact = bool(is_palindrome(candidate))
    fwd = hashlib.sha256(child.encode()).hexdigest()
    rev = hashlib.sha256(child[::-1].encode()).hexdigest()
    lt, rt = tokens(LEFT_NEW), tokens(RIGHT_NEW)
    reverse_pairs = sorted({(a, b) for a in lt for b in rt if len(a) > 1 and a == b[::-1]})
    self_palindromes = sorted({w for w in lt + rt if len(w) > 1 and w == w[::-1]})
    repeated_new = sorted({w for w in lt + rt if (lt + rt).count(w) > 1})

    result = {
        "experiment_id": "luna6-frontier-558-four-flank-attempt-20260923",
        "status": "rejected_live_residual_and_not_reader_eligible",
        "frontier_choice": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)), "row": PARENT_ID,
            "letters": len(p), "normalized_sha256": PARENT_SHA,
            "lexical_owner_advantage": "center clause has the typed live residual `a|rat`: `a` leaves owner R with residual `rat`, then the noun closes it at a staggered a+rat/Tara boundary; this gives a concrete four-flank attachment site.",
            "rendered_parent": parent,
            "other_frontier_comparison": {
                "artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json",
                "row": "depth39-longest-f1g1h1r", "letters": 556,
                "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14",
                "reason_not_selected": "its strongest lexical clue is the inherited depth-39 `i|ts` closure, already used to produce this row, whereas 558 exposes the typed `rat` noun-owner state in a different center grammar.",
            },
        },
        "novelty_preflight": {
            "revision": head,
            "operator": "replace one complete finite-event pair on the 558 row, expanding outward from the existing typed center owner while preserving the exact central event and both outer reflection shells",
            "geometry": {"left_parent_span": list(LEFT_SPAN), "right_parent_span": list(RIGHT_SPAN)},
            "geometry_hits": geometry_hits,
            "authored_phrase_hits": phrase_hits,
            "status": "coordinate_and_phrase_clean" if not geometry_hits and not phrase_hits else "collision_review_required",
            "not_the_568_outer_wrapper": True,
            "not_the_active_560_[131,159)/[401,429)_edit": True,
        },
        "edit": {
            "left_old": p[slice(*LEFT_SPAN)], "left_new_surface": LEFT_NEW,
            "right_old": p[slice(*RIGHT_SPAN)], "right_new_surface": RIGHT_NEW,
            "left_new_letters": len(left), "right_new_letters": len(right),
            "predicted_growth": len(left) + len(right) - (LEFT_SPAN[1]-LEFT_SPAN[0]) - (RIGHT_SPAN[1]-RIGHT_SPAN[0]),
            "actual_growth": len(child) - len(p),
            "four_flanks": {
                "left_before": parent[max(0, l0-45):l0],
                "left_after": parent[l1:min(len(parent), l1+45)],
                "right_before": parent[max(0, r0-45):r0],
                "right_after": parent[r1:min(len(parent), r1+45)],
                "joined_candidate": ["Noel, was I stressed? Nadia delivered the ledger before dawn.",
                                      "Leon, Aidan stops a rat. Tara spots Nadia, Noel.",
                                      "The flood rose, and Tara warned Aidan. Desserts I saw, Leon."],
                "assessment": "Each authored sentence is grammatical and the local discourse concerns a ledger and flood warning; the full inherited tape remains rough and these joins do not repair its middle.",
            },
        },
        "live_residual": {
            "left_surface": LEFT_NEW, "left_tape": left,
            "right_surface": RIGHT_NEW, "right_reverse_obligation": reverse_right,
            "matched_prefix_letters": cursor, "first_mismatch_cursor": cursor,
            "left_emits": left[cursor:cursor+1], "right_requires": reverse_right[cursor:cursor+1],
            "left_remaining": len(left)-cursor, "right_remaining": len(right)-cursor,
            "closed": left == reverse_right,
        },
        "candidate": {
            "rendered": candidate,
            "letters": len(child), "normalized_sha256": fwd, "reverse_sha256": rev,
            "independent_outside_in": exact, "project_validator_exact": project_exact,
            "mechanically_exact": exact["exact"] and project_exact and fwd == rev,
        },
        "shortcut_audit": {
            "new_left_tokens": lt, "new_right_tokens": rt,
            "cross_surface_whole_token_reversal_pairs": [list(x) for x in reverse_pairs],
            "self_palindromic_new_tokens": self_palindromes,
            "repeated_new_tokens": repeated_new,
            "inherited_shortcut_debt": "the preserved 558 central `Leon/Aidan/stops/a rat/Tara/spots/Nadia/Noel` span contains ordinary reversed lexical units and remains shortcut debt; this attempt does not certify a shortcut-clean whole tape.",
            "admission": "not admitted; exactness failed before a reader test",
        },
        "provenance": {"authoring": "one hand-authored linked scene pair for this seam, not catalogue text", "borrowed_catalogue": False,
                       "candidate_fully_rendered": True, "reader_evidence": False},
        "failure_and_pivot": {
            "failure": "the local reverse obligation fails at the recorded cursor; full tape mismatch is independently recorded above.",
            "queued_pivot_preflight": {
                "frontier": "runs/incumbent-498-event-frame-seam-repair-20261002.json / depth39-longest-f1g1h1r (556 letters)",
                "lexical_advantage": "partial-word `i|ts` ownership was the closure key in its event-frame lineage",
                "proposed_operator": "partial-word owner transfer on a distinct wider context, not another 558 center edit",
                "status": "not drafted, per coordinator instruction; requires post-commit lane assignment to avoid overlap",
            },
            "after_pivot": "Do not promote this inexact child. Coordinator will queue a non-overlapping pivot after committing this obstruction.",
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"path": str(OUTPUT_PATH), "letters": len(child), "growth": len(child)-len(p),
                      "first_mismatch": exact["first_mismatch"], "local_cursor": cursor,
                      "local_left_emits": left[cursor:cursor+1], "local_right_requires": reverse_right[cursor:cursor+1],
                      "exact": result["candidate"]["mechanically_exact"]}, indent=2))


if __name__ == "__main__":
    main()
