"""Build and mechanically audit the catalogue-based sentence control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.reader_first import hundred_word_showcase


def audit(showcase: dict) -> dict:
    """Check mechanical facts only; readers, not this audit, judge prose."""
    from experiments.audit_programmatic_readability import BrownBigramModel, item_features

    text = showcase["text"]
    letters = "".join(char.lower() for char in text if char.isalpha())
    recomputed_words = len(text.split())
    recomputed_letters = len(letters)
    if showcase["words"] != recomputed_words or showcase["letters"] != recomputed_letters:
        raise ValueError("showcase counts do not match the rendered text")
    units = showcase["rendered_units"]
    diagnostic = item_features(
        {"id": "reader-first-showcase", "source": "reader_first_baseline",
         "band": "103_words", "letters": showcase["letters"]},
        showcase["text"], BrownBigramModel.from_brown(), seed=20260912, shuffles=32)
    return {
        "whole_exact_palindrome": letters == letters[::-1],
        "word_count": recomputed_words,
        "letter_count": recomputed_letters,
        "complete_curated_units": len(units),
        "distinct_source_units_before_reflection": len(showcase["source_ids"]),
        "source_ids_are_distinct": len(showcase["source_ids"]) == len(set(showcase["source_ids"])),
        "unit_reflection_is_explicit": units == units[::-1],
        "programmatic_language_diagnostics": {
            key: diagnostic[key] for key in (
                "brown_bigram_logprob", "brown_order_gain_vs_own_shuffle",
                "mean_zipf_frequency", "repeated_word_rate", "repeated_bigram_rate",
                "punctuation_segments_per_100_words",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise RuntimeError(
        "reader-first showcase building is retired: it would package a prohibited repeated catalogue control"
    )
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    showcase = hundred_word_showcase()
    result = {
        "status": "complete_catalogue_sentence_legibility_control",
        "showcase": showcase,
        "mechanical_audit": audit(showcase),
        "use": (
            "A catalogue-based repeated-sentence control. It makes sentence boundaries visible, "
            "but does not demonstrate original material or coherent long prose. The programmatic "
            "diagnostics describe local order, lexical familiarity, repetition, and segmentation; "
            "they do not replace reader evidence."
        ),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["mechanical_audit"], indent=2))


if __name__ == "__main__":
    main()
