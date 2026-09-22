"""Bounded authored search for a discourse-linked six-clause ABBA paragraph."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "paragraph-abba-discourse-chain-20261002.json"
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    t = tape(text)
    mismatches = [{"offset": i, "left": t[i], "right": t[-1 - i]}
                  for i in range(len(t) // 2) if t[i] != t[-1 - i]]
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    # Independent two-pointer walk (deliberately not delegated to the validator).
    i, j = 0, len(t) - 1
    while i < j and t[i] == t[j]:
        i += 1
        j -= 1
    exact = bool(t) and i >= j
    return {"letters": len(t), "two_pointer_exact": exact,
            "first_mismatches": mismatches[:8], "forward_sha256": f,
            "reverse_sha256": r, "sha_equal": f == r,
            "project_validator": bool(is_palindrome(text))}


def pair_audit(units: list[str]) -> list[dict[str, object]]:
    out = []
    for i in range(len(units) // 2):
        j = len(units) - 1 - i
        left, right = tape(units[i]), tape(units[j])
        out.append({"left_index": i, "right_index": j,
                    "left_unit": units[i], "right_unit": units[j],
                    "pair_exact": left == right[::-1]})
    return out


def candidate() -> dict[str, object]:
    units = ["Aron saw deer.", "Deer saw mail.", "Noel saw war.",
             "Raw was Leon.", "Liam was Reed.", "Reed was Nora."]
    rendered = " ".join(units)
    a = audit(rendered)
    pairs = pair_audit(units)
    row = {"id": "discourse-chain-64", "kind": "exact_abba_candidate",
           "rendered": rendered, "units": units, "audit": a,
           "seam_certificate": {"pairs": pairs,
                                 "all_pairs_exact": all(x["pair_exact"] for x in pairs)},
           "discourse_links": [
               {"adjacent_units": [0, 1], "shared_referent": "deer",
                "relation": "Aron sees deer; deer is then the observer of mail"},
               {"adjacent_units": [4, 5], "shared_referent": "Reed",
                "relation": "Liam is Reed; Reed is then Nora"}],
           "unit_guard": {"distinct_units": len(set(units)) == len(units),
                          "self_palindromic_units": [u for u in units
                              if tape(u) and tape(u) == tape(u)[::-1]],
                          "word_order_mirror": []},
           "provenance": {"independently_authored_units": True,
                          "finished_tape_reversal": False, "catalogue_text": False,
                          "posthoc_repair": False, "repeated_units": False,
                          "reader_certified": False}}
    assert a["two_pointer_exact"] and a["sha_equal"] and a["project_validator"]
    assert row["seam_certificate"]["all_pairs_exact"] and row["unit_guard"]["distinct_units"]
    assert not row["unit_guard"]["self_palindromic_units"]
    return row


def main() -> dict[str, object]:
    c = candidate()
    return {"experiment_id": "paragraph-abba-discourse-chain-20261002",
            "method": "bounded authored A B C C' B' A' clause construction with referent chaining",
            "hypothesis": "A palindrome can carry a micro-scene when adjacent outer clauses share an event referent.",
            "candidates": [c], "stats": {"exact_abba_candidates": 1,
                "candidate_lengths": [c["audit"]["letters"]], "controls": 2},
            "controls": [{"id": "intact-scene", "rendered": "Aron saw a deer; the deer crossed the mail road.",
                          "audit": audit("Aron saw a deer; the deer crossed the mail road.")},
                         {"id": "seam-break", "rendered": "Aron saw deer. Deer saw mail. Noel saw war. Raw was Leon. Liam was Reed. Nora was Reed.",
                          "audit": audit("Aron saw deer. Deer saw mail. Noel saw war. Raw was Leon. Liam was Reed. Nora was Reed.")}],
            "deepest_seam": {"closed": "deer/reed and mail/Liam boundary carriers",
                "residual": "Noel saw war / Raw was Leon is grammatical but only loosely event-linked",
                "next_repair_operator": "typed event-role substitution: replace the center object with a witnessed event noun whose reverse-name carrier can be an agent in C' while preserving saw/was valency",
                "why_not_applied": "Applying it would be posthoc repair without a pre-authored role inventory."},
            "independent_audits": ["local two-pointer walk", "forward/reverse SHA-256", "llm_palindrome.validator.is_palindrome"],
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "shortcuts_excluded": ["finished-tape reversal", "catalogue text", "posthoc repair", "self-palindromic units"]}}


if __name__ == "__main__":
    OUT.write_text(json.dumps(main(), indent=2) + "\n")
    print(json.dumps(main(), indent=2))
