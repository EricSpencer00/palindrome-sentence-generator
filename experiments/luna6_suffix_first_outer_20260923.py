"""One suffix-first typed-clause test at the exact 568 outer seam.

This is a single authored equation, not an expansion of the prior 96-row
forward grammar.  It preserves a full exact rendering for audit, but applies
the structural-shortcut gate before treating the child as admissible.
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
OUT_PATH = ROOT / "runs/luna6-suffix-first-outer-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def outside_in(tape: str) -> dict:
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {"exact": i >= j, "letters": len(tape), "matched_outer_pairs": i,
            "first_mismatch": None if i >= j else {
                "offset_from_left": i, "offset_from_right": j,
                "left": tape[i], "right": tape[j]},
            "left_residual": tape[i:i + 32],
            "right_reverse_residual": tape[max(0, j - 31):j + 1][::-1]}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.casefold())


def boundary_audit(left: str, right: str) -> dict:
    ltok, rtok = tokens(left), tokens(right)
    reversals = sorted({(a, b) for a in ltok for b in rtok if a[::-1] == b})
    shared = sorted(set(ltok) & set(rtok))
    self_pal = sorted({w for w in ltok + rtok if len(w) > 1 and w == w[::-1]})
    one_letter = sorted({w for w in ltok + rtok if len(w) == 1})
    function = {"a", "an", "the", "at", "of", "to"}
    repeated_content = sorted(w for w in set(ltok) & set(rtok) if w not in function)
    return {"left_tokens": ltok, "right_tokens": rtok,
            "whole_token_reversal_pairs": [list(x) for x in reversals],
            "shared_tokens": shared, "repeated_content_tokens": repeated_content,
            "multi_letter_self_palindromic_tokens": self_pal,
            "one_letter_palindromic_supports": one_letter}


def main() -> dict:
    parent_data = json.loads(PARENT_PATH.read_text())
    parent = next(r["rendered"] for r in parent_data["rows"]
                  if r["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    from llm_palindrome.validator import is_palindrome
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA256:
        raise AssertionError("Pinned parent identity changed")
    if not outside_in(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("Pinned parent failed independent exact audit")

    # Fresh complete authored Y.  Reverse(Y) has one complete typed parse:
    # Tara (proper-name NP) + was (copula) + Dog (proper-name complement).
    # The subject crosses the terminal-token boundary (rat -> tar + a).
    y = "God saw a rat."
    y_tape = letters(y)
    reverse_y = y_tape[::-1]
    parse = {
        "surface": "Tara was Dog.",
        "normalized": "tarawasdog",
        "grammar": "S -> proper-name NP + copula-past VP + proper-name predicative NP",
        "lexical_roles": {"Tara": "proper-name subject", "was": "past copula",
                          "Dog": "proper-name predicative complement"},
        "character_owners": [
            {"text": "tar", "reverse_Y_offsets": [0, 3], "source_token": "rat", "owner": "subject"},
            {"text": "a", "reverse_Y_offsets": [3, 4], "source_token": "a", "owner": "subject"},
            {"text": "was", "reverse_Y_offsets": [4, 7], "source_token": "saw", "owner": "copula"},
            {"text": "dog", "reverse_Y_offsets": [7, 10], "source_token": "God", "owner": "predicative-NP"},
        ],
        "cursor": 10,
        "residual": "",
    }
    if parse["normalized"] != reverse_y:
        raise AssertionError("Typed parse does not consume the reverse tape")

    rendered = parse["surface"] + " " + parent + " " + y
    tape = letters(rendered)
    fwd, rev = sha(tape), sha(tape[::-1])
    audit = outside_in(tape)
    project_exact = bool(is_palindrome(rendered))
    masks = boundary_audit(parse["surface"], y)
    shortcut_free = not any((masks["whole_token_reversal_pairs"],
                             masks["multi_letter_self_palindromic_tokens"],
                             masks["one_letter_palindromic_supports"]))
    exact = audit["exact"] and project_exact and fwd == rev
    if not exact:
        raise AssertionError("Constructed full rendering failed exact validation")

    return {
        "experiment_id": "luna6-suffix-first-outer-20260923",
        "operator": "one suffix-first two-final-token typed parse, with explicit character ownership",
        "parent": {"artifact": str(PARENT_PATH.relative_to(ROOT)), "letters": len(parent_tape),
                   "sha256": PARENT_SHA256, "independent_exact": True},
        "authored_clause_pair": {"Y": y, "reverse_Y": reverse_y,
                                 "left_parse": parse,
                                 "provenance": "one fresh manually authored clause; no seed phrase, catalogue lookup, or borrowed text"},
        "full_rendered_control": rendered,
        "length": len(tape), "growth": len(tape) - 568,
        "normalized_sha256": fwd, "reverse_sha256": rev,
        "independent_outside_in": audit, "project_validator_exact": project_exact,
        "exact": exact,
        "boundary_shortcut_audit": masks,
        "admissibility": {"shortcut_free": shortcut_free,
                          "admitted": exact and shortcut_free,
                          "reason": "exact child is retained as rejected structural control: saw/was and God/dog are whole-token reversals; `a` is a one-letter palindromic support"},
        "readability": {"envelope_pair": "God saw a rat. / Tara was Dog.",
                        "human_reader_evidence": False,
                        "whole_568_interior_claim": False,
                        "assessment": "The clauses are syntactically parseable but semantically strained; the inherited interior remains rough. No readability claim."},
        "next_operator": "Pivot the seam owner map rather than swapping more final words: allow the clause-initial NP and copular predicate to take characters from three adjacent source-token reversals, while requiring every parsed content word to cross at least one source-token boundary. Preflight this owner topology against the registry/history, then attempt one authored pair; reject any aligned reversal pair or singleton support.",
    }


if __name__ == "__main__":
    result = main()
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"experiment": result["experiment_id"], "Y": result["authored_clause_pair"]["Y"],
                      "left_parse": result["authored_clause_pair"]["left_parse"]["surface"],
                      "length": result["length"], "growth": result["growth"],
                      "exact": result["exact"], "admitted": result["admissibility"]["admitted"],
                      "shortcuts": result["boundary_shortcut_audit"]["whole_token_reversal_pairs"],
                      "one_letter": result["boundary_shortcut_audit"]["one_letter_palindromic_supports"]}, sort_keys=True))
