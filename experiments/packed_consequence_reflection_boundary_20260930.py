"""Typed consequence/reflection boundary search.

This lane changes only the failed boundary of the non-mirrored paragraph
frame.  Consequence and reflection are represented as independently typed
slots (state, subject agreement, and discourse reference), then intersected
character-by-character before any accepting text is rendered.  It does not
materialize complete sentences or import reverse/semordnilap phrase pairs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_palindrome.validator import is_palindrome
from experiments.packed_seam_grammar_20260927 import Grammar, SEED, audit, intersect, norm

ROOT = Path(__file__).resolve().parents[1]
ID = "packed-consequence-reflection-boundary-20260930"


def frame() -> Grammar:
    g = Grammar()
    # The prefix deliberately remains a small ordinary setting/action chart.
    g.slot(("at dawn", "by noon", "after rain", "in spring"), "setting:time")
    g.slot(("a clerk", "an aide", "the nurse", "a writer"), "agent:singular")
    g.slot(("sorts old notes", "reads the memos", "files the forms",
            "marks the page", "opens the letter"), "action:transitive")
    # New boundary: consequence is a finite state with an agreement-carrying
    # subject; reflection refers to that same subject/state, not a reversed
    # lexical phrase.  Punctuation is epsilon to the character matcher.
    g.slot(("; the room grows calm", "; the work is done",
            "; the notes remain", "; a quiet answer follows",
            "; the door stands open", "; the task is clear"),
           "consequence:state:singular")
    g.slot((" and she waits", " and he listens", " and all is well",
            " and the lesson stays", " and the day goes on",
            " and the answer holds", " and the work rests"),
           "reflection:state:singular")
    return g


def controls():
    texts = [
        "At dawn, a clerk sorts old notes; the room grows calm, and she waits.",
        "The nurse reads the memos; the work is done, and he listens.",
    ]
    return [{"rendered": t, "kind": "intact_ordinary_prose", "audit": audit(t)}
            for t in texts]


def run():
    search = intersect(frame(), max_letters=180, cap=100_000)
    candidates = []
    for row in search["candidates"]:
        text = row["rendered"]
        row["audit"]["independent_validator_exact"] = is_palindrome(text)
        row["novel_relative_to_seed"] = norm(text) != norm(SEED)
        row["provenance"] = "typed consequence/reflection boundary; live state-pair intersection"
        row["human_readability_evidence"] = "not collected; requires blinded reader gate"
        candidates.append(row)
    return {
        "experiment_id": ID,
        "method": "live character intersection over typed consequence/reflection slots",
        "frontier_target": {"left_first_classes": ["a", "b", "i"],
                            "right_first_classes": ["l", "n", "s"]},
        "search": {k: v for k, v in search.items() if k != "candidates"},
        "candidates": candidates,
        "controls": controls(),
        "positive_control": {"rendered": SEED, "audit": audit(SEED),
                             "independent_validator_exact": is_palindrome(SEED),
                             "kind": "historical_recovery_control"},
        "provenance": {"complete_sentence_enumeration": False,
                       "reverse_phrase_catalogue": False,
                       "semordnilap_bank": False, "per_candidate_rlaif": False,
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "novelty_preflight": {"novel_algorithm_claim": False,
            "representation_change": "agreement-carrying consequence/reflection seam only",
            "distinct_from": "packed-nonmirror-paragraph-frame-20260930"},
        "reader_test": {"status": "not_collected",
            "protocol": "Novel exact outputs require randomized blinded intact and shuffled controls."},
        "next_repair": {"operator": "Add a copular consequence with anaphoric state noun while retaining both typed subjects.",
                         "reason": "No accepting path should trigger widening unrelated setting/action slots; use the observed boundary mismatch.",
                         "frontier": "Preserve first-character classes and introduce only one agreement-compatible state noun."},
    }


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states": result["search"]["states"],
                      "transitions": result["search"]["transitions"],
                      "candidates": len(result["candidates"]),
                      "max_letters": max((x["audit"]["letters"] for x in result["candidates"]), default=0)}))
