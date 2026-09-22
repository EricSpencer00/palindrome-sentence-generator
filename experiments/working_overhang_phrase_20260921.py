"""Live-overhang growth with complete attested phrase units.

This lane tests the paragraph-seam idea at the lexical level: the search may
cross a boundary with a complete two-word phrase (for example, ``at dusk``),
while exact character debt is still consumed online.  Phrases are never
reversed or repaired after rendering; a phrase is simply another typed unit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import centerout_search
from llm_palindrome.search import WordTries, unit_letters
from llm_palindrome.paragraphs import is_novel_palindrome

from experiments.working_overhang_coherent_20260921 import JoinScorer
from experiments.working_overhang_growth_20260921 import CENTERS, audit, center_unit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "working-overhang-phrase-20260921.json"
VOCAB = ROOT / "tools" / "polaris" / "payload" / "vocab30k.txt"
BIGRAMS = ROOT / "data" / "count_2w.txt"


class PhraseJoinScorer(JoinScorer):
    """Prefer a complete observed phrase when its seam is equally legal."""

    def word_delta(self, left, right, placement, word, growth):
        score = super().word_delta(left, right, placement, word, growth)
        return score + (3.0 if " " in word else 0.0)


def phrase_units(words: list[str], limit: int = 700) -> list[str]:
    known = set(words)
    rows: list[tuple[int, str]] = []
    for line in BIGRAMS.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            pair, count = line.split("\t")
            left, right = pair.split(" ")
            count = int(count)
        except ValueError:
            continue
        if left in known and right in known and left != right:
            unit = f"{left} {right}"
            if unit_letters(unit) != unit_letters(unit)[::-1]:
                rows.append((count, unit))
    rows.sort(key=lambda item: (-item[0], item[1]))
    seen: set[str] = set()
    return [unit for _, unit in rows if not (unit in seen or seen.add(unit))][:limit]


def run(*, seeds_per_center: int = 3, min_growth: int = 20,
        phrase_limit: int = 700, beam_width: int = 56,
        candidate_limit: int = 220, max_steps: int = 160) -> dict[str, object]:
    words = [w.strip().lower() for w in VOCAB.read_text().splitlines() if w.strip()]
    phrases = phrase_units(words, phrase_limit)
    units = words + phrases
    tries = WordTries(units)
    bigrams = BigramModel.from_file(str(BIGRAMS), vocab=set(words))
    rows = []
    for center_id, source_center in CENTERS.items():
        center = center_unit(source_center)
        center_tape = unit_letters(center)
        center_words = set(center.split())
        scorer = PhraseJoinScorer(bigrams, center)

        def allow_unit(placement, unit, state):
            letters = unit_letters(unit)
            if len(letters) <= 1 or letters == letters[::-1]:
                return False
            if unit == center or unit in center_words:
                return False
            if any(part in center_words for part in unit.split()):
                return False
            existing = list(state.left) + list(state.right)
            return existing.count(unit) == 0

        for seed in range(seeds_per_center):
            sequence = centerout_search(
                tries,
                scorer,
                center=center,
                min_letters=len(center_tape) + min_growth,
                beam_width=beam_width,
                per_parent=6,
                candidate_limit=candidate_limit,
                max_steps=max_steps,
                seed=seed,
                diversity=0.7,
                max_overhang=24,
                maximize="score",
                allow_word=allow_unit,
            )
            text = " ".join(sequence)
            au = audit(text) if text else {
                "letters": 0,
                "two_pointer_exact": False,
                "validator_exact": False,
            }
            rows.append({
                "center_id": center_id,
                "seed": seed,
                "rendered": text,
                "units": sequence,
                "letters": au["letters"],
                "growth_over_center": au["letters"] - len(center_tape),
                "audit": au,
                "novelty_preflight": is_novel_palindrome(text) if text else False,
                "provenance": {
                    "generator": "llm_palindrome.centerout_search",
                    "phrase_inventory": "top count_2w complete two-word units",
                    "phrase_inventory_limit": phrase_limit,
                    "source_center": source_center,
                    "render_center": center,
                    "live_overhang": True,
                    "finished_tape_reversal": False,
                    "posthoc_character_repair": False,
                    "repeated_generated_unit": False,
                    "self_palindromic_unit": False,
                    "reader_certified": False,
                },
                "seam_debt": [
                    "phrase units improve local joins but do not enforce clause syntax",
                    "ABBA paragraph role continuity is the next reader-facing test",
                ],
            })
    exact = [row for row in rows if row["audit"].get("two_pointer_exact")
             and row["audit"].get("sha_equal") and row["audit"].get("validator_exact")
             and row["growth_over_center"] > 0]
    rows.sort(key=lambda row: (-row["letters"], row["center_id"], row["seed"]))
    return {
        "experiment_id": "working-overhang-phrase-20260921",
        "method": "center-out live overhang growth with complete attested phrase units",
        "stats": {
            "centers": len(CENTERS),
            "phrase_units": len(phrases),
            "runs": len(rows),
            "exact_growths": len(exact),
            "longest_letters": max((row["letters"] for row in rows), default=0),
            "longest_growth_over_center": max((row["growth_over_center"] for row in rows), default=0),
        },
        "rows": rows,
        "exact_growths": exact,
        "novelty_preflight": {
            "catalogue": "data/known_palindromes.json",
            "novel_exact_growths": sum(bool(row.get("novelty_preflight")) for row in exact),
            "catalogue_text_presented_as_generated": False,
        },
        "reader_gate": "closed; exact phrase-unit drafts require intact prose and shuffled controls",
        "next_reader_test": "blind ABBA paragraph comparison: intact authored A-B-B-A controls versus phrase-unit candidates and word-shuffled controls",
        "provenance": {
            "bigram_sha256": hashlib.sha256(BIGRAMS.read_bytes()).hexdigest(),
            "vocab_sha256": hashlib.sha256(VOCAB.read_bytes()).hexdigest(),
        },
    }


if __name__ == "__main__":
    payload = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
