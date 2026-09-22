"""Grow exact working palindromes from the 44-letter now/won seam.

This is a bounded result-first run over the existing two-sided overhang engine.
The center is already an exact working palindrome; new words are admitted only
when the live debt closes, while the scorer is a frozen vocabulary prior plus
repayable-debt lookahead.  Generated surfaces remain drafts until a human
reader study confirms ordinary English.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llm_palindrome.overhang import DebtIndex
from llm_palindrome.search import WordTries, unit_letters
from llm_palindrome.centerout import centerout_search
from llm_palindrome.validator import is_palindrome, normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "working-overhang-growth-20260921.json"
VOCAB = ROOT / "tools" / "polaris" / "payload" / "vocab30k.txt"

CENTERS = {
    "now-won-44": "Now, an aide rips nine memos; some men inspire. Diana won.",
    "noel-saga-54": "Was Noel an era, a gas, an item smart? Trams met in a, saga, arena, Leon saw.",
}


def center_unit(text: str) -> str:
    """Make a center acceptable to the word engine without changing its tape."""
    return re.sub(r"[^A-Za-z ]", "", text).lower()


def audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    mismatches = [
        (i, tape[i], tape[-1 - i])
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "normalized": tape,
        "two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
        "validator_exact": is_palindrome(text),
    }


class DebtRankScorer:
    """Frequency-order prior with no per-candidate model call."""

    wants_overhang = True

    def __init__(self, words: list[str], debt: DebtIndex):
        self.rank = {word: index for index, word in enumerate(words)}
        self.debt = debt

    def word_delta(self, left, right, placement, word, growth, overhang=None):
        rank = self.rank.get(word, len(self.rank))
        score = 8.0 - 0.7 * (rank ** 0.5) + 0.08 * len(unit_letters(word))
        if overhang is not None:
            options = self.debt.options(overhang)
            score += 1.5 if options else -18.0
        used = list(left) + list(right)
        score -= 2.0 * used.count(word)
        return score


def render_units(units: list[str]) -> str:
    """Expose generated boundary seams without changing normalized letters."""
    if not units:
        return ""
    if len(units) == 1:
        return units[0]
    left = " ".join(units[:-1])
    return left + " " + units[-1]


def run(*, seeds_per_center: int = 3, min_growth: int = 20,
        beam_width: int = 48, candidate_limit: int = 180,
        max_steps: int = 160) -> dict[str, object]:
    words = [w.strip().lower() for w in VOCAB.read_text().splitlines() if w.strip()]
    tries = WordTries(words)
    debt = DebtIndex(tries, limit=96)
    rows = []
    for center_id, center in CENTERS.items():
        # centerout works on word units and deliberately leaves punctuation to
        # the renderer.  Keep the source sentence as provenance, but pass a
        # letters-and-spaces equivalent so punctuation cannot fake a match.
        center_rendered = center_unit(center)
        center_tape = normalize(center_rendered)
        center_words = set(center_rendered.split())
        scorer = DebtRankScorer(words, debt)

        def allow_word(placement, word, state):
            letters = unit_letters(word)
            if len(letters) <= 1 or letters == letters[::-1]:
                return False
            if word in center_words:
                return False
            existing = list(state.left) + list(state.right)
            return existing.count(word) == 0

        for seed in range(seeds_per_center):
            units = centerout_search(
                tries,
                scorer,
                center=center_rendered,
                min_letters=len(center_tape) + min_growth,
                beam_width=beam_width,
                per_parent=6,
                candidate_limit=candidate_limit,
                max_steps=max_steps,
                seed=seed,
                diversity=0.55,
                max_overhang=24,
                maximize="letters",
                allow_word=allow_word,
            )
            text = render_units(units)
            au = audit(text) if text else {
                "letters": 0,
                "two_pointer_exact": False,
                "validator_exact": False,
            }
            center_pos = units.index(center_rendered) if center_rendered in units else -1
            left = units[:center_pos] if center_pos >= 0 else []
            right = units[center_pos + 1:] if center_pos >= 0 else []
            rows.append({
                "center_id": center_id,
                "seed": seed,
                "rendered": text,
                "letters": au["letters"],
                "growth_over_center": au["letters"] - len(center_tape),
                "left_extension": left,
                "right_extension": right,
                "audit": au,
                "provenance": {
                    "generator": "llm_palindrome.centerout_search",
                    "center_replayed_exact": True,
                    "source_center": center,
                    "render_center": center_rendered,
                    "live_overhang": True,
                    "frozen_frequency_prior": True,
                    "per_candidate_rlaif": False,
                    "finished_tape_reversal": False,
                    "posthoc_character_repair": False,
                    "repeated_generated_unit": False,
                    "reader_certified": False,
                },
                "seam_debt": [
                    "generated boundary words are not yet syntax-realized",
                    "human readability and discourse continuity are untested",
                ],
            })
    exact = [r for r in rows if r["audit"].get("two_pointer_exact")
             and r["audit"].get("sha_equal") and r["audit"].get("validator_exact")
             and r["growth_over_center"] > 0]
    rows.sort(key=lambda r: (-r["letters"], r["center_id"], r["seed"]))
    return {
        "experiment_id": "working-overhang-growth-20260921",
        "method": "center-out live overhang growth from exact working centers with frozen lexical debt lookahead",
        "stats": {
            "centers": len(CENTERS),
            "runs": len(rows),
            "exact_growths": len(exact),
            "longest_letters": max((r["letters"] for r in rows), default=0),
            "longest_growth_over_center": max((r["growth_over_center"] for r in rows), default=0),
        },
        "rows": rows,
        "exact_growths": exact,
        "reader_gate": "closed; exact closure is necessary but no draft is reader-certified",
        "next_growth": "retain the longest exact lineage and replace only its worst boundary word pair with syntax-realized phrase windows; keep the 44- and 54-letter centers as separate branches",
    }


if __name__ == "__main__":
    payload = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
