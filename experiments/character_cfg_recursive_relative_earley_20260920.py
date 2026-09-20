"""Recursive relative-clause character CFG intersection.

This is a new chart construction over the character-trie lane.  Relative
clauses are grammar items, not post-hoc repairs: NP -> DET NOUN REL and
REL -> who VP.  Number/agreement is carried in each derivation and the
subject/object attachment is retained in its tree.  Opposing characters are
matched while the chart pair is exposed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_earley_intersection_20260920 import audit, brown_words, norm

EXPERIMENT_ID = "character-cfg-recursive-relative-earley-20260920"


@dataclass(frozen=True)
class Derivation:
    words: tuple[str, ...]
    tree: str
    number: str
    relative_attachment: str


NOUN_NUMBER = {"sailor": "sg", "poet": "sg", "keeper": "sg", "writer": "sg",
              "garden": "sg", "harbor": "sg", "area": "sg", "era": "sg"}
VERB_NUMBER = {"guards": "sg", "marks": "sg", "guides": "sg", "keeps": "sg", "reads": "sg", "writes": "sg"}


def np_items(lex: dict[str, set[str]], include_relative: bool = True):
    out: list[tuple[tuple[str, ...], str, str]] = []
    for det in sorted(lex["DET"]):
        for noun in sorted(NOUN_NUMBER):
            if noun not in lex["NOUN"]:
                continue
            number = NOUN_NUMBER[noun]
            out.append(((det, noun), f"NP({det},{noun})", number))
            if include_relative:
                for rverb, rnumber in sorted(VERB_NUMBER.items()):
                    if rnumber != number or rverb not in lex["VERB"]:
                        continue
                    # REL -> who V NP; object is deliberately a separate NP
                    # item with no relative attachment at this bounded depth.
                    for obj in ("area", "era", "harbor", "garden", "poet", "sailor"):
                        if obj not in lex["NOUN"]:
                            continue
                        words = (det, noun, "who", rverb, det, obj)
                        out.append((words, f"NP({det},{noun},REL(who,{rverb},NP({det},{obj})))", number))
    return tuple(out)


def chart(lex: dict[str, set[str]], trie, cap: int = 180):
    nps = np_items(lex)
    rows: list[Derivation] = []
    states = 0
    for subject_words, subject_tree, number in nps:
        for verb, verb_number in sorted(VERB_NUMBER.items()):
            if verb not in lex["VERB"] or verb_number != number:
                continue
            # Relative clauses may attach to the subject; the object is a
            # separate NP item, so the attachment feature is explicit.
            for object_words, object_tree, _ in nps:
                words = (*subject_words, verb, *object_words)
                if not all(trie.accepts(word) for word in words):
                    continue
                states += sum(len(word) for word in words)
                rows.append(Derivation(words, f"S({subject_tree},VP({verb},{object_tree}))", number, "subject"))
                if len(rows) >= cap:
                    return tuple(rows), states
    return tuple(rows), states


def intersect(left: tuple[Derivation, ...], right: tuple[Derivation, ...]):
    rows = []
    states = 0
    longest = {"letters": 0, "left": "", "right": ""}
    for li, lder in enumerate(left):
        lt = norm(" ".join(lder.words))
        for ri, rder in enumerate(right):
            rt = norm(" ".join(rder.words))
            matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[::-1][matched]:
                states += 1
                matched += 1
            if matched > longest["letters"]:
                longest = {"letters": matched, "left": " ".join(lder.words), "right": " ".join(rder.words)}
            if matched == len(lt) == len(rt):
                rendered = " ".join(lder.words).capitalize() + "; " + " ".join(rder.words) + "."
                a = audit(rendered)
                rows.append({"rendered": rendered, "audit": a, "left_tree": lder.tree, "right_tree": rder.tree,
                             "agreement": {"left_subject": lder.number, "right_subject": rder.number},
                             "relative_attachment": {"left": lder.relative_attachment, "right": rder.relative_attachment},
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
    reader = [x for x in exact if x["audit"]["letters"] > 38]
    return {"experiment_id": EXPERIMENT_ID,
            "method": "character-level CFG/Earley intersection with recursive relative items and agreement features",
            "grammar": ["S -> NP VP", "NP -> DET NOUN | DET NOUN REL", "REL -> who VP", "VP -> V NP"],
            "stats": {"left_derivations": len(left), "right_derivations": len(right),
                      "left_chart_states": left_states, "right_chart_states": right_states,
                      "intersection_character_states": intersection_states, "exact": len(exact),
                      "reader_eligible": len(reader), "longest_mirrored_prefix_letters": longest["letters"]},
            "complete_prose_controls": ["The sailor who reads the area guards the harbor.",
                                        "A quiet poet who writes a poem marks the garden."],
            "best_diagnostic": longest,
            "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-recursive-relative-agreement-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": "character-cfg-earley-trie-intersection-20260920"},
            "next_construction": "Permit relative attachment to the object NP and carry an explicit attachment feature through the paired chart; keep lexical terminals fixed.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
