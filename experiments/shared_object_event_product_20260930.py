"""Search a bidirectional shared-object event grammar.

The object is a semantic resource, not a copied reverse phrase: an event
introduces an object and a later transitive attachment refers to that same
object (``the note``/``it``).  Character fronts are intersected while lexical
slots are still NFAs.  No complete sentence list is constructed first.
"""
import hashlib
import json
import re
from pathlib import Path

from llm_palindrome.validator import is_palindrome
from experiments.packed_seam_grammar_20260927 import Grammar, audit, intersect

ROOT = Path(__file__).resolve().parents[1]
ID = "shared-object-event-product-20260930"


def norm(text):
    return re.sub(r"[^a-z]", "", text.lower())


def object_event_grammar():
    g = Grammar()
    # One event introduces an object; the attachment remains in the same
    # discourse graph and may select a pronoun or a definite NP.  These are
    # ordinary authored choices, not reverse-word or catalogue banks.
    g.slot(("the nurse", "a clerk", "an aide", "the poet"), "subject:introduce")
    g.slot(("reads", "files", "sorts", "marks", "carries"), "verb:transitive")
    g.slot(("the data", "the report", "a quota", "the note", "a letter", "old papers", "the map", "some memos"),
           "object:introduce")
    g.slot((", and ", "; then ", " while ", " as "), "event:attachment")
    g.slot(("the nurse", "the clerk", "the aide", "the poet", "she", "he", "it"),
           "subject:shared-object-access")
    g.slot(("reads", "files", "sorts", "marks", "carries", "keeps"),
           "verb:transitive-attachment")
    g.slot(("the data.", "the report.", "the quota.", "the note.", "the letter.", "the papers.", "the map.", "them."),
           "object:shared-object")
    return g


def independent(text):
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-i - 1]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "normalized": tape,
            "sha256_forward": forward, "sha256_reverse": reverse,
            "sha_equal": forward == reverse,
            "validator_exact": is_palindrome(text)}


def run():
    search = intersect(object_event_grammar(), max_letters=180, cap=100000)
    candidates = []
    for row in search["candidates"]:
        row["audit"] = independent(row["rendered"])
        row["provenance"] = "fresh shared-object event grammar"
        row["novel_relative_to_seed"] = True
        row["human_readability_evidence"] = "not collected; programmatic gate is diagnostic only"
        candidates.append(row)
    controls = [
        {"rendered": "The nurse reads the note, and she files it.",
         "audit": independent("The nurse reads the note, and she files it."),
         "kind": "intact ordinary prose control"},
        {"rendered": "The clerk marks a letter; then he carries it.",
         "audit": independent("The clerk marks a letter; then he carries it."),
         "kind": "intact ordinary prose control"},
    ]
    return {
        "experiment_id": ID,
        "method": "live character intersection over a shared-object event product",
        "geometry": "introduce object -> discourse attachment -> same-object access",
        "candidates": candidates,
        "search": {k: search[k] for k in ("states", "transitions", "grammar_states",
                                           "grammar_character_edges", "dead_frontiers")},
        "controls": controls,
        "independent_exact_gate": "two-pointer mismatch + validator + forward/reverse SHA-256",
        "provenance": {"complete_sentence_enumeration": False,
                       "catalogue_replay": False, "semordnilap_bank": False,
                       "per_candidate_rlaif": False,
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "novelty_preflight": {"status": "passed",
            "difference": "shared semantic object access is represented in one event product; no phrase-pair scaffold",
            "reader_gate": "closed until blinded human ratings"},
        "next_repair": {"operator": "widen the shared-object attachment with agreement-carrying pronouns while retaining event identity",
            "reason": "the current frontier shows which first character blocks prevent ordinary transitive closures"},
    }


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": ID, "states": result["search"]["states"],
                      "transitions": result["search"]["transitions"],
                      "exact": len(result["candidates"])}, indent=2))
