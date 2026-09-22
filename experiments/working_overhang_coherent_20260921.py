"""Compare a join-aware live-overhang lane against the frequency baseline.

The search remains exact at every transition.  The only change is the
diagnostic ordering prior: observed forward/backward word joins replace the
individual-word frequency rank.  This is a construction experiment, not a
readability certificate; its rows retain the full text and independent audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import centerout_search
from llm_palindrome.scoring import adjacent, first_word, last_word, unit_words
from llm_palindrome.search import WordTries, unit_letters

from experiments.working_overhang_growth_20260921 import (
    CENTERS,
    audit,
    center_unit,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "working-overhang-coherent-20260921.json"
VOCAB = ROOT / "tools" / "polaris" / "payload" / "vocab30k.txt"
BIGRAMS = ROOT / "data" / "count_2w.txt"


class JoinScorer:
    """Dependency-free subset of CoherentScorer for the remote bench."""

    def __init__(self, bg, center: str):
        self.bg = bg
        self.center = center

    def word_delta(self, left, right, placement, word, growth):
        neighbor = adjacent(left, right, placement, growth)
        if neighbor is None:
            neighbor = self.center or (right[0] if placement == "L" and right
                                       else (left[-1] if left else None))
        if growth == "prepend":
            score = self.bg.backward(last_word(word),
                                     first_word(neighbor) if neighbor else None)
        else:
            score = self.bg.forward(last_word(neighbor) if neighbor else None,
                                    first_word(word))
        inner = word.split()
        score += 0.9 * sum(self.bg.forward(a, b) for a, b in zip(inner, inner[1:]))
        existing = unit_words(left) + unit_words(right)
        score -= 2.0 * sum(existing.count(w) for w in inner)
        score += 0.04 * len(unit_letters(word))
        return score


def run(*, seeds_per_center: int = 3, min_growth: int = 20,
        beam_width: int = 48, candidate_limit: int = 180,
        max_steps: int = 160) -> dict[str, object]:
    words = [w.strip().lower() for w in VOCAB.read_text().splitlines() if w.strip()]
    tries = WordTries(words)
    bigrams = BigramModel.from_file(str(BIGRAMS), vocab=words)
    rows = []
    for center_id, source_center in CENTERS.items():
        center = center_unit(source_center)
        center_tape = unit_letters(center)
        center_words = set(center.split())
        scorer = JoinScorer(bigrams, center)

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
                center=center,
                min_letters=len(center_tape) + min_growth,
                beam_width=beam_width,
                per_parent=6,
                candidate_limit=candidate_limit,
                max_steps=max_steps,
                seed=seed,
                diversity=0.55,
                max_overhang=24,
                maximize="score",
                allow_word=allow_word,
            )
            text = " ".join(units)
            au = audit(text) if text else {
                "letters": 0,
                "two_pointer_exact": False,
                "validator_exact": False,
            }
            center_pos = units.index(center) if center in units else -1
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
                    "scorer": "JoinScorer forward/backward count_2w",
                    "center_replayed_exact": True,
                    "source_center": source_center,
                    "render_center": center,
                    "live_overhang": True,
                    "per_candidate_rlaif": False,
                    "finished_tape_reversal": False,
                    "posthoc_character_repair": False,
                    "repeated_generated_unit": False,
                    "reader_certified": False,
                },
                "seam_debt": [
                    "join prior is diagnostic ordering, not a grammar guarantee",
                    "human readability and discourse continuity are untested",
                ],
            })
    exact = [r for r in rows if r["audit"].get("two_pointer_exact")
             and r["audit"].get("sha_equal") and r["audit"].get("validator_exact")
             and r["growth_over_center"] > 0]
    rows.sort(key=lambda r: (-r["letters"], r["center_id"], r["seed"]))
    return {
        "experiment_id": "working-overhang-coherent-20260921",
        "method": "center-out live overhang growth with bidirectional attested-join ordering",
        "stats": {
            "centers": len(CENTERS),
            "runs": len(rows),
            "exact_growths": len(exact),
            "longest_letters": max((r["letters"] for r in rows), default=0),
            "longest_growth_over_center": max((r["growth_over_center"] for r in rows), default=0),
        },
        "rows": rows,
        "exact_growths": exact,
        "reader_gate": "closed; join score is diagnostic and no draft is reader-certified",
        "next_growth": "use the smoothest complete clause window as an ABBA paragraph seam seed, then reopen only its residual character obligation",
        "provenance": {"bigram_sha256": hashlib.sha256(BIGRAMS.read_bytes()).hexdigest()},
    }


if __name__ == "__main__":
    payload = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
