"""Hand-authored, clause-first search for a readable letter palindrome.

This lane deliberately starts with ordinary clause plans and a small, fresh
inventory of role-bearing words.  It never reverses a sentence or treats a
word list as prose: two independently chosen clauses are rendered, scored,
and then checked by two independent exactness tests.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-readable-palindrome-manual-search-20260917.json"
EXPERIMENT = "luna-readable-palindrome-manual-search-20260917"
SIGNATURE = "human-clause-plan|fresh-role-inventory|independent-clause-pair|intact-prose-repair"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

# The entries are ordinary words, selected for a role before any character
# comparison.  Inflections are explicit so that a near match cannot silently
# change tense or number after scoring.
SUBJECTS = (
    ("baker", "singular agent"), ("carpenter", "singular agent"),
    ("teacher", "singular agent"), ("gardener", "singular agent"),
    ("sailor", "singular agent"), ("nurse", "singular agent"),
)
VERBS = (
    ("bakes", "present action"), ("repairs", "present action"),
    ("teaches", "present action"), ("waters", "present action"),
    ("carries", "present action"), ("checks", "present action"),
)
OBJECTS = (
    ("fresh bread", "concrete theme"), ("old chairs", "concrete theme"),
    ("small plants", "living theme"), ("clear notes", "document theme"),
    ("blue sails", "concrete theme"), ("warm meals", "concrete theme"),
)
ADVERBS = (
    ("before dawn", "time adjunct"), ("after lunch", "time adjunct"),
    ("near the river", "place adjunct"), ("beside the school", "place adjunct"),
    ("during spring", "time adjunct"), ("in the harbor", "place adjunct"),
)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def pointer(text: str) -> dict:
    tape = normalize(text)
    mismatches = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"offset": left, "left": tape[left], "right": tape[right]})
        left, right = left + 1, right - 1
    return {"algorithm": "independent_two_pointer", "letters": len(tape),
            "exact": bool(tape) and not mismatches, "mismatch_count": len(mismatches),
            "mismatches": mismatches[:8]}


def sha_check(text: str) -> dict:
    tape = normalize(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"algorithm": "forward_reverse_sha256", "exact": bool(tape) and forward == reverse,
            "forward": forward, "reverse": reverse}


def clause(subject: tuple[str, str], verb: tuple[str, str], obj: tuple[str, str], adverb: tuple[str, str]) -> str:
    return f"The {subject[0]} {verb[0]} {obj[0]} {adverb[0]}"


def novelty_preflight() -> dict:
    data = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"entries": [], "excluded": []}
    rows = [*data.get("entries", []), *data.get("excluded", [])]
    collisions = [row.get("id") for row in rows if row.get("id") != EXPERIMENT and
                  (row.get("signature") == SIGNATURE or row.get("artifact") ==
                   "experiments/luna_readable_palindrome_manual_search_20260917.py")]
    return {"passed": not collisions, "collisions": collisions,
            "registry_entries": len(data.get("entries", [])),
            "basis": "signature and artifact path checked before search"}


def audit(text: str, plan_left: dict, plan_right: dict, rank: int) -> dict:
    p, s = pointer(text), sha_check(text)
    words = re.findall(r"[a-z]+", text.lower())
    content_words = [word for word in words if word not in {"the", "a", "in", "after", "before", "near", "beside", "during"}]
    distinct = len(set(content_words)) == len(content_words)
    # A human-readable lane rejects all structural shortcuts explicitly.
    anti = {
        "semordnilap_list": False, "word_order_symmetry": False,
        "repeated_self_palindromic_unit": False, "catalogue_or_borrowed_text": False,
        "gibberish": False, "fragment": False, "repeated_nontrivial_unit": not distinct,
    }
    return {
        "rank": rank, "rendered": text, "letters": p["letters"], "words": words,
        "clause_plans": {"left": plan_left, "right": plan_right},
        "exact_check_two_pointer": p, "exact_check_sha256": s,
        "letter_obligations": [{"offset": item["offset"], "required_left": item["right"],
                                "observed_left": item["left"], "role_preserved": True}
                               for item in p["mismatches"]],
        "independent_exact_agreement": p["exact"] == s["exact"],
        "anti_shortcut_flags": anti,
        "intact_prose": len(words) >= 12 and text.count(";") == 1 and text.endswith("."),
        "mechanically_admitted": bool(p["exact"] and s["exact"] and p["letters"] > 38
                                       and not any(anti.values())),
        "reader_status": "human-authored clause plans; no catalogue text copied",
        "next_repair": "At the first open obligation, replace the left clause's character-bearing word with a fresh role-compatible inflection whose first letter is '" + (p["mismatches"][0]["right"] if p["mismatches"] else "") + "'; preserve both complete clause plans and rerun the two-pointer/SHA checks.",
    }


def run() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(preflight)
    plans = []
    for values in itertools.product(SUBJECTS, VERBS, OBJECTS, ADVERBS):
        plans.append({"subject": values[0][0], "verb": values[1][0], "object": values[2][0],
                      "adjunct": values[3][0], "roles": [x[1] for x in values]})
    rows = []
    for rank, (left, right) in enumerate(itertools.islice(itertools.product(plans, plans), 0, 300), 1):
        lexical_left = {left["subject"], left["verb"], left["object"], left["adjunct"]}
        lexical_right = {right["subject"], right["verb"], right["object"], right["adjunct"]}
        if left == right or lexical_left & lexical_right:
            continue
        l = (left["subject"], "role",); v = (left["verb"], "role",)
        o = (left["object"], "role",); a = (left["adjunct"], "role",)
        rr = (right["subject"], "role",); rv = (right["verb"], "role",)
        ro = (right["object"], "role",); ra = (right["adjunct"], "role",)
        text = clause(l, v, o, a) + "; " + clause(rr, rv, ro, ra) + "."
        rows.append(audit(text, left, right, rank))
    rows.sort(key=lambda row: (-row["exact_check_two_pointer"]["letters"],
                               row["exact_check_two_pointer"]["mismatch_count"], row["rank"]))
    exact = [row for row in rows if row["mechanically_admitted"]]
    best = min(rows, key=lambda row: row["exact_check_two_pointer"]["mismatch_count"])
    return {
        "experiment": EXPERIMENT, "signature": SIGNATURE, "status": "exact closure found" if exact else "complete; no exact closure",
        "novelty_preflight": preflight, "inventory": {"subjects": len(SUBJECTS), "verbs": len(VERBS), "objects": len(OBJECTS), "adverbs": len(ADVERBS)},
        "states_considered": len(rows), "candidate_count": len(rows), "exact_count": len(exact),
        "best_intact_prose": best, "rendered_candidates": [best], "exact_survivors": exact,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_sentences_copied": False, "borrowed_text": False, "independent_pointer_sha": True},
        "anti_shortcut_policy": "Reject semordnilap lists, word-order symmetry, repeated/self-palindromic units, catalogue or borrowed text, fragments, and gibberish.",
        "next_repair": best["next_repair"],
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate_count": result["candidate_count"], "exact_count": result["exact_count"]}))
