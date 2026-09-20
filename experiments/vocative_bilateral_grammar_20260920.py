"""Whole-sentence vocative grammar search without nested palindrome spans."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bilateral_grammar_csp_20260920 import bilateral_grammar_csp
from forward_lexicalized_grammar_20260920 import Word, letters

OUT = ROOT / "runs/vocative-bilateral-grammar-20260920.json"

LEXICON = {
    "DET": "a an the some many our his her my no".split(),
    "AGENT": "aide men man poet sailor keeper reader writer artist mason captain singer actor guard child king queen author clerk farmer pilot teacher deer dog god cod doc lever star rats drawer reward".split(),
    "THEME": "memos memo note book poem song tale story page gift message stone door bridge garden deer reed dog god cod doc mood doom star rats drawer reward".split(),
    "V": "rips inspire reads writes sees guards keeps marks draws loves hears carries opens sends gives seeks finds names tells edits emits notes taps stops deliver reviled saw was draw live".split(),
    "PROPN": "diana leon noel elba adam anna ava".split(),
    "ADJ": "old new red raw quiet young bright small good able drab".split(),
}

GRAMMAR = {
    "VOC": (("PROPN",),),
    "CLAUSE": (("SUBJ", "V", "OBJ"),),
    "SUBJ": (("PROPN",), ("AGENT",), ("DET", "AGENT"), ("DET", "ADJ", "AGENT")),
    "OBJ": (("THEME",), ("PROPN",), ("DET", "THEME"), ("THEME", "THEME"), ("DET", "ADJ", "THEME")),
}


def has_nested_word_span(words):
    """Reject any proper word-aligned palindrome nested inside the sentence."""
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            if start == 0 and end == len(words):
                continue
            tape = letters(" ".join(words[start:end]))
            if len(tape) > 1 and tape == tape[::-1]:
                return True
    return False


def run(max_nodes=5_000_000):
    lexicon = tuple(Word(text, pos) for pos, words in LEXICON.items() for text in words)
    result = bilateral_grammar_csp(
        lexicon,
        grammar=GRAMMAR,
        left_symbols=("VOC", "CLAUSE"),
        right_symbols=("CLAUSE", "VOC"),
        max_words=14,
        max_nodes=max_nodes,
    )
    all_paths = list(result["paths"])
    accepted = [row for row in all_paths if not has_nested_word_span(row["words"])]
    rejected = [row for row in all_paths if has_nested_word_span(row["words"])]
    result["paths_before_nested_filter"] = len(all_paths)
    # Keep concrete evidence of the failed construction rather than reducing it
    # to a count: these rows show exactly what the seam was producing.
    result["nested_rejected_examples"] = sorted(
        rejected,
        key=lambda row: (-row["length"], row["rendered"]),
    )[:20]
    result["paths"] = accepted
    result["stats"]["nested_span_rejected"] = result["paths_before_nested_filter"] - len(accepted)
    result["experiment_id"] = "vocative-bilateral-grammar-20260920"
    result["method"] = "whole-sentence vocative plus two-clause grammar with bilateral live character residual"
    result["provenance"].update({
        "grammar": "left VOC+CLAUSE and right CLAUSE+VOC; both sides independently expanded",
        "nested_palindrome_spans_rejected": True,
        "reader_gate": "closed unless a fresh exact row above 38 survives intact-versus-shuffled blinded reading",
        "next_construction": "retain only non-nested closures and add discourse attachment if a closure survives",
    })
    return result


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["paths"][:20]:
        print(row["rendered"])
