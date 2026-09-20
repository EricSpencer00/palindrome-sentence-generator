"""Curated semantic-role bilateral grammar experiment.

This is a search-space construction, not a post-hoc repair: subject and
theme roles, determiner boundaries, and transitive verb slots are chosen while
the left and right clauses consume opposing character residuals live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bilateral_grammar_csp_20260920 import bilateral_grammar_csp
from forward_lexicalized_grammar_20260920 import Word

OUT = ROOT / "runs/semantic-role-bilateral-20260920.json"

LEXICON = {
    "DET": "a an the some many our his her my no".split(),
    "AGENT": "aide men man poet sailor keeper reader writer artist mason captain singer actor guard child king queen author clerk farmer pilot teacher deer dog god cod doc lever star rats".split(),
    "THEME": "memos memo note book poem song tale story page gift message stone door bridge garden deer reed dog god cod doc mood doom star rats".split(),
    "V": "rips inspire reads writes sees guards keeps marks draws loves hears carries opens sends gives seeks finds names tells edits emits notes taps stops deliver reviled saw was draw live".split(),
    "PROPN": "diana leon noel elba adam anna ava".split(),
    "ADJ": "old new red raw quiet young bright small good able drab".split(),
}

GRAMMAR = {
    "S": (("CLAUSE", "CLAUSE"),),
    "CLAUSE": (("SUBJ", "V", "OBJ"),),
    "SUBJ": (("PROPN",), ("AGENT",), ("DET", "AGENT"), ("DET", "ADJ", "AGENT")),
    "OBJ": (("THEME",), ("PROPN",), ("DET", "THEME"), ("THEME", "THEME"), ("DET", "ADJ", "THEME")),
}


def run(max_nodes=3_000_000):
    lexicon = tuple(
        Word(text, pos)
        for pos, words in LEXICON.items()
        for text in words
    )
    result = bilateral_grammar_csp(lexicon, grammar=GRAMMAR, max_words=12, max_nodes=max_nodes)
    result["experiment_id"] = "semantic-role-bilateral-20260920"
    result["method"] = "curated agent/theme transitive grammar with bilateral live character residual"
    result["provenance"].update({
        "lexicon": "hand-audited role vocabulary; no source sentence text",
        "role_constraints": ["subject agent/proper-name", "transitive verb", "theme object"],
        "grammar": "two independent CLAUSE parses with determiner/adjective alternatives",
        "reader_gate": "closed unless a fresh exact row above 38 survives a human-readability screen",
        "next_construction": "add a typed adjunct only if this grammar yields a non-shortcut exact frontier",
    })
    return result


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["paths"][:20]:
        print(row["rendered"])
