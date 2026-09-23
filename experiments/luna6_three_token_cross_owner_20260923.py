"""One bounded three-source-token suffix owner map on the pinned 568 tape.

Preflight found related character-level seam/resegmentation work, but none of
the checked committed signatures imposed the specific condition that each
parsed content word itself consume characters from at least two source-token
reversals. This file tests one fresh clause only; it is not a sweep.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs/luna6-three-token-cross-owner-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def pointer(tape: str) -> dict:
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {"exact": i >= j, "letters": len(tape), "matched_pairs": i,
            "first_mismatch": None if i >= j else {
                "offset_left": i, "offset_right": j,
                "left": tape[i], "right": tape[j]},
            "left_residual": tape[i:i + 32],
            "right_reverse_residual": tape[max(0, j - 31):j + 1][::-1]}


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.casefold())


def boundary_audit(left: str, right: str) -> dict:
    ltok, rtok = word_tokens(left), word_tokens(right)
    reverse_pairs = sorted({(a, b) for a in ltok for b in rtok if a[::-1] == b})
    one_letter = sorted({w for w in ltok + rtok if len(w) == 1})
    self_pal = sorted({w for w in ltok + rtok if len(w) > 1 and w == w[::-1]})
    return {"left_tokens": ltok, "right_tokens": rtok,
            "whole_token_reversal_pairs": [list(p) for p in reverse_pairs],
            "one_letter_palindromic_supports": one_letter,
            "multi_letter_self_palindromic_tokens": self_pal,
            "shared_tokens": sorted(set(ltok) & set(rtok)),
            "repeated_content_tokens": sorted((set(ltok) & set(rtok)) - {"a", "an", "the"})}


def main() -> dict:
    parent_data = json.loads(PARENT_PATH.read_text())
    parent = next(r["rendered"] for r in parent_data["rows"]
                  if r["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    from llm_palindrome.validator import is_palindrome
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA256:
        raise AssertionError("Pinned 568 parent identity changed")
    if not pointer(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("Pinned parent failed independent validation")

    # Novelty preflight against committed related artifacts.  These establish
    # neighboring ideas, but none enforces this lane's all-content-words-cross
    # ownership condition at the released outer seam.
    preflight = {
        "checked_committed_artifacts": [
            {"artifact": "experiments/luna6_three_boundary_residual_pivot_20260923.py",
             "result": "11-character local said/Dias closure rejected; changes seam to [91,232)/cut 477; not an all-content-word cross-owner chart"},
            {"artifact": "experiments/incumbent_568_outer_event_resegmentation_20260923.py",
             "result": "outer multi-clause replacement at [0,20)/[548,568), 574/578 exact controls; records shared word boundaries and token reversals rather than requiring every content word to cross owners"},
            {"artifact": "experiments/incumbent_568_bilateral_typed_grammar_chart_20261002.py",
             "result": "online typed chart at internal cuts [194,374], distinct geometry and no all-content-word crossing invariant"},
            {"artifact": "experiments/bounded_cfg_earley_palindrome_intersection_20260920.py",
             "result": "paired-CFG online intersection on short construction space; not a three-source-token owner map on the pinned 568 outer seam"},
        ],
        "signature": "pinned-568-outer-suffix-first|3 adjacent source-token reversals|typed reverse clause|every parsed content word consumes >=2 source-token owners",
        "collision_found": False,
        "distinct_enough_for_one_bounded_attempt": True,
        "scope": "one hand-authored Y and one typed reverse parse only",
    }

    # Fresh authored pair.  It deliberately tests whether a complete typed
    # reverse parse can satisfy the stronger cross-owner rule, not merely close
    # the character equation.
    y = "Mara saw rats."
    y_tape = letters(y)
    reverse_y = y_tape[::-1]
    left_parse = "Star was a ram."
    left_tape = letters(left_parse)
    if reverse_y != left_tape:
        raise AssertionError("The single authored chart parse does not consume reverse(Y)")

    # Explicit source-token spans in reverse(Y): rats -> star [0,4), saw -> was
    # [4,7), Mara -> aram [7,11). The lexical parser's word spans do not cross
    # any of these owner boundaries.
    owner_map = {
        "reverse_Y": reverse_y,
        "source_token_reversals": [
            {"source_token": "rats", "reverse_surface": "star", "span": [0, 4]},
            {"source_token": "saw", "reverse_surface": "was", "span": [4, 7]},
            {"source_token": "Mara", "reverse_surface": "aram", "span": [7, 11]},
        ],
        "typed_parse_spans": [
            {"word": "Star", "role": "subject", "span": [0, 4], "source_token_owners": ["rats"], "crosses_owner_boundary": False},
            {"word": "was", "role": "copula", "span": [4, 7], "source_token_owners": ["saw"], "crosses_owner_boundary": False},
            {"word": "a", "role": "article", "span": [7, 8], "source_token_owners": ["Mara"], "crosses_owner_boundary": False},
            {"word": "ram", "role": "predicative_NP", "span": [8, 11], "source_token_owners": ["Mara"], "crosses_owner_boundary": False},
        ],
        "cursor": 11,
        "residual": "",
        "content_words_crossing_owner_boundary": 0,
        "content_words_total": 3,
    }
    all_content_cross = owner_map["content_words_crossing_owner_boundary"] == owner_map["content_words_total"]
    masks = boundary_audit(left_parse, y)
    rendered = left_parse + " " + parent + " " + y
    candidate_tape = letters(rendered)
    forward, reverse = sha(candidate_tape), sha(candidate_tape[::-1])
    exact_pointer = pointer(candidate_tape)
    project_exact = bool(is_palindrome(rendered))
    exact = exact_pointer["exact"] and project_exact and forward == reverse
    if not exact:
        raise AssertionError("The full exact control failed independent validation")

    admissible = (all_content_cross
                  and not masks["whole_token_reversal_pairs"]
                  and not masks["one_letter_palindromic_supports"]
                  and not masks["multi_letter_self_palindromic_tokens"])
    return {
        "experiment_id": "luna6-three-token-cross-owner-20260923",
        "method": "single suffix-first reverse-clause lexicalization with a strict three-source-token owner rule",
        "parent": {"artifact": str(PARENT_PATH.relative_to(ROOT)), "letters": 568,
                   "sha256": PARENT_SHA256, "independent_exact": True},
        "novelty_preflight": preflight,
        "authored_pair": {"Y": y, "reverse_Y": reverse_y, "typed_reverse_parse": left_parse,
                          "provenance": "fresh hand-authored clause; no seed sentence, catalogue text, or copied line"},
        "owner_map": owner_map,
        "candidate": {"rendered": rendered, "letters": len(candidate_tape), "growth": len(candidate_tape) - 568,
                      "normalized_sha256": forward, "reverse_sha256": reverse,
                      "independent_outside_in": exact_pointer, "project_validator_exact": project_exact,
                      "exact": exact, "admitted": exact and admissible,
                      "shortcut_audit": masks,
                      "rejection": "The character equation closes, but 0/3 content words cross source-token owners; Star/rats and was/saw are whole-token reversal pairs, and `a` is a singleton support."},
        "readability": {"human_ratings": False, "whole_tape_readability_claim": False,
                        "assessment": "The short clauses are syntactically parseable but semantically odd; exactness does not establish readability of the inherited 568-letter interior."},
        "next_topology": "Do not vary the same SVO token bank. Preflight a typed participial-NP start whose first lexical item must consume the entire reversed final token plus a residual from the penultimate source token (e.g. the specific `draw` -> `ward` prefix shape), and require the next typed constituent to own the leftover characters without singleton articles or reversed-token matches. One lexicalization only after proving an English suffix can satisfy both owners.",
    }


if __name__ == "__main__":
    result = main()
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"experiment": result["experiment_id"], "Y": result["authored_pair"]["Y"],
                      "reverse_parse": result["authored_pair"]["typed_reverse_parse"],
                      "cursor": result["owner_map"]["cursor"], "residual": result["owner_map"]["residual"],
                      "crossing_content_words": f"{result['owner_map']['content_words_crossing_owner_boundary']}/{result['owner_map']['content_words_total']}",
                      "length": result["candidate"]["letters"], "exact": result["candidate"]["exact"],
                      "admitted": result["candidate"]["admitted"]}, sort_keys=True))
