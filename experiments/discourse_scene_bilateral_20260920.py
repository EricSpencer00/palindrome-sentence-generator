"""Bounded scene-plan search: setting + event + consequence on both sides.

This is a new topology, not a repair or reranking pass.  The bilateral CSP
matches characters while expanding a small discourse grammar whose clauses
carry distinct scene roles (setting, event, consequence).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bilateral_grammar_csp_20260920 import bilateral_grammar_csp
from forward_lexicalized_grammar_20260920 import Word, letters

OUT = ROOT / "runs/discourse-scene-bilateral-20260920.json"
LEXICON = {
    "DET": "a an the some many our his her my no this that".split(),
    "AGENT": "aide men man poet sailor keeper reader writer artist mason captain singer actor guard child king queen author clerk farmer pilot teacher deer dog god cod doc bard maid lord dame rose son sun rain star rats".split(),
    "THEME": "memos memo note book poem song tale story page gift message stone door bridge garden deer reed dog god cod doc mood doom star rats rose son sun rain".split(),
    "V": "rips inspire reads writes sees guards keeps marks draws loves hears carries opens sends gives seeks finds names tells edits emits notes taps stops delivers reviles saw was draws lives".split(),
    "PROPN": "diana leon noel elba adam anna ava".split(),
    "ADV": "now then softly still here there".split(),
    "CONJ": "and but so".split(),
    "PREP": "in on by near under".split(),
}
GRAMMAR = {
    "SCENE": (("SETTING", "EVENT", "RESULT"),),
    "SETTING": (("DET", "THEME", "PREP", "DET", "THEME"), ("PROPN",)),
    "EVENT": (("SUBJ", "V", "OBJ"),),
    "RESULT": (("CONJ", "SUBJ", "V"), ("ADV", "SUBJ", "V")),
    "SUBJ": (("PROPN",), ("AGENT",), ("DET", "AGENT")),
    "OBJ": (("THEME",), ("PROPN",), ("DET", "THEME")),
}

def run(max_nodes=2_000_000):
    lex = tuple(Word(t, p) for p, ws in LEXICON.items() for t in ws)
    result = bilateral_grammar_csp(lex, grammar=GRAMMAR,
        left_symbols=("SCENE",), right_symbols=("SCENE",),
        max_words=18, max_nodes=max_nodes)
    result["experiment_id"] = "discourse-scene-bilateral-20260920"
    result["method"] = "typed scene-plan (setting/event/result) with bilateral live character CSP"
    result["provenance"].update({
        "scene_roles": "SETTING -> EVENT -> RESULT; independent scene plans on both sides",
        "search_space": "grammar expansion and lexical choices are constrained online by shared character residual",
        "post_hoc_repair": False, "reward_model": False, "catalogue_text": False,
        "reader_gate": "closed until intact-versus-shuffled blinded reading",
        "next_construction": "if no admissible closure, replace RESULT with anaphoric consequence rather than widening lexical beam",
    })
    return result

if __name__ == "__main__":
    r = run(); OUT.write_text(json.dumps(r, indent=2)+"\n")
    print(json.dumps(r["stats"]))
    for x in r["paths"][:20]: print(x["rendered"])
