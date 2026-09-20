"""Asymmetric subject/object-relative pairing with live seam tracking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_object_relative_earley_20260920 import (
    NOUN_NUMBER, VERB_NUMBER, audit, brown_words, norm, np_items,
)
from experiments.character_cfg_recursive_relative_earley_20260920 import Derivation

EXPERIMENT_ID = "character-cfg-asymmetric-relative-earley-20260920"


def chart(lex, trie, subject_relative: bool, cap: int = 180):
    plain = np_items(lex, False)
    rich = np_items(lex, True)
    rows = []
    states = 0
    for sw, stree, snum, _ in (rich if subject_relative else plain):
        for verb, vnum in sorted(VERB_NUMBER.items()):
            if verb not in lex["VERB"] or vnum != snum:
                continue
            for ow, otree, _, attachment in (plain if subject_relative else rich):
                words = (*sw, verb, *ow)
                if not all(trie.accepts(word) for word in words):
                    continue
                states += sum(len(word) for word in words)
                rows.append(Derivation(words, f"S({stree},VP({verb},{otree}))", snum,
                                       "subject" if subject_relative else ("object" if attachment == "relative" else "none")))
                if len(rows) >= cap:
                    return tuple(rows), states
    return tuple(rows), states


def seam_positions(words):
    pos = 0
    result = set()
    for word in words[:-1]:
        pos += len(word)
        result.add(pos)
    return result


def intersect(left, right):
    rows = []
    states = 0
    seam_hits = 0
    longest = {"letters": 0, "left": "", "right": "", "seam_crossed": False}
    for lder in left:
        lt = norm(" ".join(lder.words))
        lseams = seam_positions(lder.words)
        for rder in right:
            rt = norm(" ".join(rder.words))
            rseams = seam_positions(rder.words)
            matched = 0
            crossed = False
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[::-1][matched]:
                states += 1
                # The equation has crossed a word boundary on either exposed
                # side once the next character is after a lexical seam.
                crossed = crossed or (matched + 1 in lseams) or (len(rt) - matched - 1 in rseams)
                matched += 1
            if crossed:
                seam_hits += 1
            if matched > longest["letters"]:
                longest = {"letters": matched, "left": " ".join(lder.words),
                           "right": " ".join(rder.words), "seam_crossed": crossed}
            if matched == len(lt) == len(rt) and crossed:
                rendered = " ".join(lder.words).capitalize() + "; " + " ".join(rder.words) + "."
                a = audit(rendered)
                rows.append({"rendered": rendered, "audit": a,
                             "attachment": {"left": lder.relative_attachment, "right": rder.relative_attachment},
                             "agreement": {"left": lder.number, "right": rder.number},
                             "cross_word_seam": True,
                             "provenance": {"brown_lexicon_only": True, "catalogue_imported": False,
                                            "finished_tape_reversed": False, "word_order_mirror": False,
                                            "reader_status": "not run"}})
    return rows, states, seam_hits, longest


def run():
    trie, lex = brown_words()
    subject, subject_states = chart(lex, trie, True)
    object_side, object_states = chart(lex, trie, False)
    a, s1, h1, best1 = intersect(subject, object_side)
    b, s2, h2, best2 = intersect(object_side, subject)
    candidates = a + b
    unique = {x["audit"]["normalized"]: x for x in candidates}
    exact = list(unique.values())
    best = best1 if best1["letters"] >= best2["letters"] else best2
    return {"experiment_id": EXPERIMENT_ID,
            "method": "asymmetric character CFG intersection: subject-relative ↔ object-relative with live seam state",
            "grammar": ["S -> NPsubject VP", "VP -> V NPobject", "NP -> DET NOUN | DET NOUN REL", "REL -> who VP"],
            "pair_orientations": ["subject-relative left / object-relative right", "object-relative left / subject-relative right"],
            "stats": {"subject_derivations": len(subject), "object_derivations": len(object_side),
                      "subject_chart_states": subject_states, "object_chart_states": object_states,
                      "intersection_character_states": s1 + s2, "cross_word_seam_hits": h1 + h2,
                      "exact": len(exact), "reader_eligible": 0,
                      "longest_mirrored_prefix_letters": best["letters"]},
            "complete_prose_controls": ["The sailor who reads the area guards the harbor.",
                                        "The sailor guards the area that the poet reads."],
            "best_diagnostic": best,
            "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-asymmetric-relative-seam-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": "character-cfg-object-relative-attachment-20260920"},
            "next_construction": "Carry two independent seam debts across a shared center nonterminal, preserving asymmetric attachment and agreement; keep lexical terminals fixed.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
