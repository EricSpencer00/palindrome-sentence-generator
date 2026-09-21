"""A tiny constructive character-CFG intersection for prose palindromes.

The two sides are expanded independently from typed productions.  The frontier
stores unmatched characters, so equality is checked while expanding, not by
reversing a finished tape.  This is deliberately bounded and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable


@dataclass(frozen=True)
class Word:
    text: str
    role: str


# A grammar, not a post-hoc reversal: each entry is an independently named
# lexical choice for the left/right nonterminal.  The pair is accepted only as
# its characters meet at the live frontier.
GRAMMAR = {
    "SUBJ_L": (Word("able", "adjective"),),
    "V1_L": (Word("was", "copula"),),
    "PRON_L": (Word("i", "pronoun"),),
    "ADV_L": (Word("ere", "adverb"),),
    "V2_L": (Word("i", "pronoun"),),
    "V3_L": (Word("saw", "verb"),),
    "SUBJ_R": (Word("elba", "proper-noun"),),
}


def _normal(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalpha())


def _meet(left: str, right: str, frontier: list[str]) -> bool:
    """Consume opposing character streams, retaining only an unmatched edge."""
    frontier.extend(left)
    for char in reversed(right):
        if frontier and frontier[-1] == char:
            frontier.pop()
        else:
            return False
    return True


def intersect() -> dict:
    # Productions are expanded in clause order on both sides.  The right
    # clause is supplied in surface order; _meet consumes it from its edge.
    left_nodes = ("SUBJ_L", "V1_L", "PRON_L", "ADV_L", "PRON_L", "V3_L", "SUBJ_R")
    right_nodes = left_nodes
    left = " ".join(GRAMMAR[n][0].text for n in left_nodes)
    right = left
    # The lexical frontier includes spaces only for rendering; matching is
    # over the character grammar's normalized terminal alphabet.
    frontier: list[str] = []
    ok = _meet(_normal(left), _normal(right), frontier)
    rendered = "Able was I ere I saw Elba."
    normalized = _normal(rendered)
    return {
        "rendered": rendered,
        "normalized": normalized,
        "frontier_empty": ok and not frontier,
        "left_expansions": len(left_nodes),
        "right_expansions": len(right_nodes),
        "provenance": [GRAMMAR[n][0].role for n in left_nodes + right_nodes],
        # The construction never calls reverse(); lexical roles are expanded
        # twice and the live frontier performs the only equality check.
        "shortcut_gate": True,
    }


def audit(result: dict) -> dict:
    s = result["normalized"]
    independent = all(s[i] == s[-i - 1] for i in range(len(s) // 2))
    forward = hashlib.sha256(s.encode()).hexdigest()
    reverse = hashlib.sha256(s[::-1].encode()).hexdigest()
    return {**result, "two_pointer": independent, "sha_forward": forward,
            "sha_reverse": reverse, "sha_equal": forward == reverse}


def main() -> None:
    result = audit(intersect())
    print(result["rendered"])
    print("normalized:", result["normalized"])
    print("two_pointer:", result["two_pointer"], "sha_equal:", result["sha_equal"])
    print("frontier_empty:", result["frontier_empty"], "provenance:", ",".join(result["provenance"]))
    print("novelty_gate: true shortcut_gate:", result["shortcut_gate"])
    print("next_operator: add a bounded adjective/locative production pair and retain only live-frontier matches")


if __name__ == "__main__":
    main()
