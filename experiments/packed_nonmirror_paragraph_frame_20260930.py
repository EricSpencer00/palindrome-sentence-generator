"""Search a coherent paragraph frame whose reverse side has different roles.

The slots describe a small discourse (setting, agent, action, consequence,
reflection), but no slot is paired with a reversed lexical phrase.  The NFA
product searches equal character obligations while the two sides are still
inside the frame.  This is deliberately a non-mirrored topology experiment:
the semantic roles do not form an ABBA list, even though an accepting tape
would have to be symmetric.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_palindrome.validator import is_palindrome
from experiments.packed_seam_grammar_20260927 import Grammar, SEED, audit, intersect, norm

ROOT = Path(__file__).resolve().parents[1]
ID = "packed-nonmirror-paragraph-frame-20260930"


def paragraph_frame() -> Grammar:
    g = Grammar()
    # One forward discourse frame: setting -> agent -> action -> result ->
    # reflection.  These are independently authored ordinary-English slots;
    # no reverse phrase bank or complete sentence is constructed.
    g.slot(("at dawn", "by noon", "in spring", "after rain"), "setting:time")
    g.slot(("a clerk", "an aide", "the nurse", "a writer"), "agent:singular")
    g.slot(("sorts old notes", "reads the memos", "files the forms",
            "marks the page", "opens the letter"), "action:transitive")
    g.slot(("; the room grows calm", "; the work is done", "; the notes remain",
            "; a quiet answer follows", "; the door stands open"), "consequence")
    g.slot((" and she waits", " and he listens", " and all is well",
            " and the lesson stays", " and the day goes on"), "reflection")
    return g


def intact_controls():
    return [
        {"rendered": "At dawn, a clerk sorts old notes; the room grows calm, and she waits.",
         "audit": audit("At dawn, a clerk sorts old notes; the room grows calm, and she waits."),
         "kind": "intact_ordinary_prose"},
        {"rendered": "The nurse reads the memos; the work is done, and he listens.",
         "audit": audit("The nurse reads the memos; the work is done, and he listens."),
         "kind": "intact_ordinary_prose"},
    ]


def run():
    search = intersect(paragraph_frame(), max_letters=180, cap=100_000)
    candidates = []
    for row in search["candidates"]:
        row["audit"]["independent_validator_exact"] = is_palindrome(row["rendered"])
        row["novel_relative_to_seed"] = norm(row["rendered"]) != norm(SEED)
        row["provenance"] = "non-mirrored typed discourse frame; accepting path from live state product"
        row["human_readability_evidence"] = "not collected; requires blinded reader gate"
        candidates.append(row)
    return {
        "experiment_id": ID,
        "topology": "setting -> agent -> action -> consequence -> reflection (non-mirrored, non-ABBA semantic roles)",
        "method": "live equal-character NFA state-pair intersection before rendering",
        "search": {**{k: v for k, v in search.items() if k != "candidates"}, "candidate_count": len(candidates)},
        "candidates": candidates,
        "controls": intact_controls(),
        "positive_control": {"rendered": SEED, "audit": audit(SEED),
                             "independent_validator_exact": is_palindrome(SEED),
                             "kind": "historical_recovery_control"},
        "provenance": {
            "complete_sentence_enumeration": False,
            "reverse_phrase_catalogue": False,
            "semordnilap_bank": False,
            "per_candidate_rlaif": False,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "novelty_preflight": {
            "novel_algorithm_claim": False,
            "representation_change": "semantic paragraph roles are linear and non-mirrored; character symmetry is imposed only by paired state traversal",
        },
        "reader_test": {
            "status": "not_collected",
            "protocol": "Any novel exact output must be shown in randomized blinded order with intact and shuffled controls; no metric certifies readability.",
        },
        "next_repair": {
            "operator": "Keep the same semantic frame and widen only the consequence/reflection seam with agreement-carrying alternatives.",
            "reason": "The first surviving frontier identifies the character and semantic boundary that needs a concrete construction change.",
        },
    }


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states": result["search"]["states"], "transitions": result["search"]["transitions"],
                      "candidates": len(result["candidates"]),
                      "max_letters": max((x["audit"]["letters"] for x in result["candidates"]), default=0)}))
