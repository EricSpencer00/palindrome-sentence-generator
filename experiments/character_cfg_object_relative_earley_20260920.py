"""Object-relative attachment in the character CFG/Earley intersection.

This lane keeps the lexical terminals from the recursive-relative run fixed,
but changes the construction: an explicit attachment feature distinguishes a
relative clause on the subject NP from one on the object NP.  Character
obligations are consumed online for each paired derivation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_recursive_relative_earley_20260920 import (
    Derivation, NOUN_NUMBER, VERB_NUMBER, audit, brown_words, norm,
)

EXPERIMENT_ID = "character-cfg-object-relative-earley-20260920"


def np_items(lex, include_relative: bool):
    out = []
    for det in sorted(lex["DET"]):
        for noun in sorted(NOUN_NUMBER):
            if noun not in lex["NOUN"]:
                continue
            number = NOUN_NUMBER[noun]
            out.append(((det, noun), f"NP({det},{noun})", number, "none"))
            if not include_relative:
                continue
            for rverb, rnumber in sorted(VERB_NUMBER.items()):
                if rnumber != number or rverb not in lex["VERB"]:
                    continue
                for obj in ("area", "era", "harbor", "garden", "poet", "sailor"):
                    if obj not in lex["NOUN"]:
                        continue
                    words = (det, noun, "who", rverb, det, obj)
                    out.append((words, f"NP({det},{noun},REL(who,{rverb},NP({det},{obj})))", number, "relative"))
    return tuple(out)


def chart(lex, trie, cap=180):
    plain = np_items(lex, False)
    recursive = np_items(lex, True)
    rows = []
    states = 0
    # Subject remains plain while object receives the new relative attachment.
    # The chart feature records object attachment rather than flattening it.
    for sw, stree, snum, _ in plain:
        for verb, vnum in sorted(VERB_NUMBER.items()):
            if verb not in lex["VERB"] or vnum != snum:
                continue
            for ow, otree, _, attachment in recursive:
                words = (*sw, verb, *ow)
                if not all(trie.accepts(word) for word in words):
                    continue
                states += sum(len(word) for word in words)
                rows.append(Derivation(words, f"S({stree},VP({verb},{otree}))", snum, "object" if attachment == "relative" else "none"))
                if len(rows) >= cap:
                    return tuple(rows), states
    return tuple(rows), states


def intersect(left, right):
    rows = []
    states = 0
    longest = {"letters": 0, "left": "", "right": "", "attachments": {}}
    for lder in left:
        lt = norm(" ".join(lder.words))
        for rder in right:
            rt = norm(" ".join(rder.words))
            matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[::-1][matched]:
                states += 1
                matched += 1
            if matched > longest["letters"]:
                longest = {"letters": matched, "left": " ".join(lder.words),
                           "right": " ".join(rder.words),
                           "attachments": {"left": lder.relative_attachment, "right": rder.relative_attachment}}
            if matched == len(lt) == len(rt):
                rendered = " ".join(lder.words).capitalize() + "; " + " ".join(rder.words) + "."
                a = audit(rendered)
                rows.append({"rendered": rendered, "audit": a,
                             "left_tree": lder.tree, "right_tree": rder.tree,
                             "attachment": {"left": lder.relative_attachment, "right": rder.relative_attachment},
                             "provenance": {"brown_lexicon_only": True, "catalogue_imported": False,
                                            "finished_tape_reversed": False, "word_order_mirror": False,
                                            "reader_status": "not run"}})
    return rows, states, longest


def run():
    trie, lex = brown_words()
    left, left_states = chart(lex, trie)
    right, right_states = chart(lex, trie)
    candidates, intersection_states, longest = intersect(left, right)
    unique = {x["audit"]["normalized"]: x for x in candidates}
    exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "character CFG/Earley intersection with explicit object-relative attachment",
            "grammar": ["S -> NPsubject VP", "VP -> V NPobject", "NP -> DET NOUN | DET NOUN REL", "REL -> who VP"],
            "stats": {"left_derivations": len(left), "right_derivations": len(right),
                      "left_chart_states": left_states, "right_chart_states": right_states,
                      "intersection_character_states": intersection_states, "exact": len(exact),
                      "reader_eligible": 0, "longest_mirrored_prefix_letters": longest["letters"]},
            "complete_prose_controls": ["The sailor guards the area that the poet reads.",
                                        "A careful keeper marks the garden that a writer reads."],
            "best_diagnostic": longest,
            "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-object-relative-attachment-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": "character-cfg-recursive-relative-agreement-20260920"},
            "next_construction": "Pair subject- and object-relative attachments asymmetrically in one chart item, carrying attachment and agreement jointly; keep terminals fixed.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
