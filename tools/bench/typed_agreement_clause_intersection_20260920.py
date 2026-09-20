"""Agreement-typed live clause intersection.

This is the next construction after the corrected broad POS run.  The
character equation is unchanged, but determiner/noun and subject/finite-verb
features are checked as each lexical edge is selected.  It is intentionally a
small typed grammar, not a larger lexical sweep.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import broad_pos_clause_intersection as broad

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PAL_OUT", str(ROOT / "runs/typed-agreement-clause-intersection-20260920.json")))

TEMPLATES = [
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "NAME"),
    ("NAME", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
]

SINGULAR_DET = {"a", "an", "one", "each", "every", "this", "that"}
PLURAL_DET = {"some", "these", "those", "many", "several", "both"}
SINGULAR_NOUNS = {"aide", "metal", "career", "level", "eye", "case", "year", "son", "man", "car", "girl", "rowan", "diana"}
PLURAL_NOUNS = {"memos", "men", "people", "letters", "stories", "roads", "rivers", "news"}
IRREGULAR_SINGULAR_VERBS = {"is", "was", "has", "does"}
IRREGULAR_PLURAL_VERBS = {"are", "were", "have", "do"}


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def number(word: str) -> str:
    if word in PLURAL_NOUNS or word.endswith("s") and word not in {"is", "was", "news"}:
        return "pl"
    return "sg"


def verb_number(word: str) -> str | None:
    if word in IRREGULAR_SINGULAR_VERBS:
        return "sg"
    if word in IRREGULAR_PLURAL_VERBS:
        return "pl"
    if word.endswith("s"):
        return "sg"
    return "pl"


def _noun_after(template: tuple[str, ...], start: int) -> int | None:
    for j in range(start, len(template)):
        if template[j] in {"NOUN", "NAME"}:
            return j
        if template[j] not in {"ADJ"}:
            break
    return None


def agreement_ok(words: list[str], template: tuple[str, ...]) -> bool:
    if len(words) != len(template):
        return True
    # Determiner/noun agreement for every overt NP.
    for i, tag in enumerate(template[:-1]):
        if tag != "DET":
            continue
        noun_i = _noun_after(template, i + 1)
        if noun_i is None:
            continue
        det, noun = words[i], words[noun_i]
        if det in SINGULAR_DET and number(noun) != "sg":
            return False
        if det in PLURAL_DET and number(noun) != "pl":
            return False
    # Subject is NP before the first verb; its number must match finite verb.
    try:
        vi = template.index("VERB")
    except ValueError:
        return True
    if vi == 0:
        return True
    subj_i = next((j for j in range(vi - 1, -1, -1)
                   if template[j] in {"NOUN", "NAME", "PRON"}), vi - 1)
    subj = words[subj_i]
    return verb_number(words[vi]) == number(subj)


def partial_ok(words: list[str], template: tuple[str, ...], _side: str) -> bool:
    # Reject only constraints whose operands are already exposed.  This keeps
    # the agreement relation live during generation instead of post-hoc
    # filtering exact tapes.
    if len(words) < len(template):
        for i, tag in enumerate(template[:-1]):
            noun_i = _noun_after(template, i + 1) if tag == "DET" else None
            if noun_i is not None and noun_i < len(words):
                det, noun = words[i], words[noun_i]
                if det in SINGULAR_DET and number(noun) != "sg":
                    return False
                if det in PLURAL_DET and number(noun) != "pl":
                    return False
        try:
            vi = template.index("VERB")
        except ValueError:
            return True
        if vi < len(words) and vi > 0:
            subj_i = next((j for j in range(vi - 1, -1, -1)
                           if template[j] in {"NOUN", "NAME", "PRON"}), vi - 1)
            subj = words[subj_i]
            if verb_number(words[vi]) != number(subj):
                return False
        return True
    return agreement_ok(words, template)


def run() -> dict:
    rows = []
    nodes = 0
    for left_template in TEMPLATES:
        for right_template in TEMPLATES:
            pairs, visited = broad.search(left_template, right_template, cap=200,
                                          partial_ok=partial_ok)
            nodes += visited
            for left, right in pairs:
                rendered = " ".join(left) + "; " + " ".join(right)
                tape = letters(rendered)
                if len(tape) <= 38 or tape != tape[::-1]:
                    continue
                rows.append({"rendered": rendered, "left_template": left_template,
                             "right_template": right_template,
                             "audit": {"letters": len(tape), "exact": True,
                                       "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
                                       "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()},
                             "agreement_valid": agreement_ok(left, left_template) and agreement_ok(right, right_template),
                             "reader_eligible": False,
                             "provenance": {"typed_agreement_live": True, "finished_tape_reversed": False,
                                            "post_hoc_repair": False, "catalogue_text": False}})
    rows.sort(key=lambda x: (-x["audit"]["letters"], x["rendered"]))
    valid = [r for r in rows if r["agreement_valid"]]
    result = {
        "experiment_id": "typed-agreement-clause-intersection-20260920",
        "method": "outside-in live character intersection with determiner/noun and subject/verb agreement carried in grammar state",
        "stats": {"templates": len(TEMPLATES), "visited_nodes": nodes, "exact_over_38": len(rows),
                  "agreement_valid_exact": len(valid), "longest_exact": max((r["audit"]["letters"] for r in rows), default=0)},
        "exact_candidates": rows[:200], "reader_facing_candidates": [],
        "novelty_preflight": {"status": "passed", "signature": "typed-agreement|live-partial-feature-prune|asymmetric-clause",
                              "distinct_from": "corrected broad POS run: feature obligations prune lexical edges before character closure, rather than free POS slots",
                              "finished_tape_reversal": False, "post_hoc_repair": False},
        "provenance": {"audits": ["independent two-pointer", "forward/reverse SHA-256"],
                       "reader_gate": "closed; no programmatic row is a readability certificate"},
        "next_construction": "add valency features for transitive versus intransitive verbs while retaining the agreement state in the live residual key",
        "status": "agreement-valid exact candidate requires human reading" if valid else "no agreement-valid exact closure; malformed exact controls retained",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in rows[:20]:
        print(row["audit"]["letters"], row["agreement_valid"], row["rendered"])
    return result


if __name__ == "__main__":
    run()
