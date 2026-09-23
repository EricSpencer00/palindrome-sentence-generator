"""Bounded grammar/consonant-equation probe at the released 568 outer seam.

The search authors short complete SVO clauses with either an adjective inside
the object NP or a VP-attached PP.  For each Y, a character trie tests whether
reverse(Y) has a complete derivation in the same tiny grammar.  This differs
from spacing an already reversed catalogue tape: both surface clauses are
composed from fresh lexical slots, and all character constraints are checked
before any full rendering is admitted.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs/luna6-grammatical-resegmentation-outer-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"

# One small, authored lexicon.  None of the seed words/phrases is used.
SUBJECTS = [
    {"surface": "Mara", "role": "agent", "number": "singular"},
    {"surface": "A sailor", "role": "agent", "number": "singular"},
]
VERBS = [
    {"surface": "carries", "lemma": "carry", "valency": "transitive", "number": "singular"},
    {"surface": "finds", "lemma": "find", "valency": "transitive", "number": "singular"},
]
ADJECTIVE_OBJECTS = [
    {"surface": "the old map", "attachment": "adjective->object-np", "head": "map"},
    {"surface": "the red map", "attachment": "adjective->object-np", "head": "map"},
    {"surface": "a lost key", "attachment": "adjective->object-np", "head": "key"},
    {"surface": "the old key", "attachment": "adjective->object-np", "head": "key"},
    {"surface": "a red letter", "attachment": "adjective->object-np", "head": "letter"},
    {"surface": "the lost ledger", "attachment": "adjective->object-np", "head": "ledger"},
    {"surface": "an old note", "attachment": "adjective->object-np", "head": "note"},
    {"surface": "a red boat", "attachment": "adjective->object-np", "head": "boat"},
]
PP_OBJECTS = [
    {"surface": "the map", "head": "map"},
    {"surface": "a key", "head": "key"},
    {"surface": "the ledger", "head": "ledger"},
    {"surface": "a letter", "head": "letter"},
]
PPS = [
    {"surface": "at dawn", "attachment": "pp->vp", "relation": "time"},
    {"surface": "after rain", "attachment": "pp->vp", "relation": "time"},
    {"surface": "near the gate", "attachment": "pp->vp", "relation": "place"},
    {"surface": "under a tree", "attachment": "pp->vp", "relation": "place"},
]


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def outside_in(tape: str) -> dict:
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {
        "exact": i >= j,
        "letters": len(tape),
        "matched_outer_pairs": i,
        "first_mismatch": None if i >= j else {
            "offset_from_left": i, "offset_from_right": j,
            "left": tape[i], "right": tape[j]},
        "left_residual": tape[i:i + 40],
        "right_reverse_residual": tape[max(0, j - 39):j + 1][::-1],
    }


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.casefold())


def sentence_bank() -> list[dict]:
    rows = []
    for subject, verb, obj in itertools.product(SUBJECTS, VERBS, ADJECTIVE_OBJECTS):
        rows.append({
            "surface": f"{subject['surface']} {verb['surface']} {obj['surface']}.",
            "subject": subject,
            "verb": verb,
            "object": obj,
            "attachment_state": "adjective attached to object NP",
        })
    for subject, verb, obj, pp in itertools.product(SUBJECTS, VERBS, PP_OBJECTS, PPS):
        rows.append({
            "surface": f"{subject['surface']} {verb['surface']} {obj['surface']} {pp['surface']}.",
            "subject": subject,
            "verb": verb,
            "object": obj,
            "pp": pp,
            "attachment_state": "PP attached to VP after complete object NP",
        })
    assert len(rows) == 96
    assert len({letters(row["surface"]) for row in rows}) == 96
    return rows


def reverse_parser(sentence_tape: str) -> dict | None:
    """Recognize one whole reverse tape; at most 4 subject/verb chart states."""
    residual = sentence_tape[::-1]
    object_forms = {letters(row["surface"]): row for row in ADJECTIVE_OBJECTS + PP_OBJECTS}
    pp_forms = {letters(row["surface"]): row for row in PPS}
    states = 0
    for subject, verb in itertools.product(SUBJECTS, VERBS):
        states += 1
        prefix = letters(subject["surface"] + " " + verb["surface"])
        if not residual.startswith(prefix):
            continue
        tail = residual[len(prefix):]
        # Direct transitive object parse.
        if tail in object_forms:
            return {"surface": f"{subject['surface']} {verb['surface']} {object_forms[tail]['surface']}.",
                    "tokens": tokens(f"{subject['surface']} {verb['surface']} {object_forms[tail]['surface']}"),
                    "parse": {"S": subject, "V": verb, "O": object_forms[tail]},
                    "states_examined": states}
        # VP-attached PP parse; only a complete object followed by a complete PP may close.
        for object_surface, obj in object_forms.items():
            if not tail.startswith(object_surface):
                continue
            pp_tape = tail[len(object_surface):]
            if pp_tape in pp_forms:
                pp = pp_forms[pp_tape]
                return {"surface": f"{subject['surface']} {verb['surface']} {obj['surface']} {pp['surface']}.",
                        "tokens": tokens(f"{subject['surface']} {verb['surface']} {obj['surface']} {pp['surface']}"),
                        "parse": {"S": subject, "V": verb, "O": obj, "PP": pp},
                        "states_examined": states}
    return None


def longest_grammar_prefix(sentence_tape: str) -> dict:
    """Retain the best authored frontier if reverse(Y) does not fully parse."""
    reverse = sentence_tape[::-1]
    prefixes = []
    for subject, verb in itertools.product(SUBJECTS, VERBS):
        candidate = letters(subject["surface"] + " " + verb["surface"])
        matched = 0
        while matched < min(len(candidate), len(reverse)) and candidate[matched] == reverse[matched]:
            matched += 1
        prefixes.append({"matched": matched, "required_prefix": reverse[:min(len(reverse), matched + 12)],
                         "grammar_prefix": candidate, "unconsumed_required": reverse[matched:matched + 24],
                         "next_grammar_char": candidate[matched:matched + 1]})
    return max(prefixes, key=lambda row: row["matched"])


def word_boundary_audit(left_surface: str, right_surface: str) -> dict:
    left, right = tokens(left_surface), tokens(right_surface)
    mirrored = sorted({(a, b) for a in left for b in right if a[::-1] == b})
    repeated = sorted(set(left) & set(right))
    function_words = {"a", "an", "the", "at", "after", "near", "under"}
    repeated_content = sorted(token for token in set(left) & set(right)
                              if token not in function_words)
    self_pal = sorted({w for w in left + right if len(w) > 1 and w == w[::-1]})
    return {"left_tokens": left, "right_tokens": right,
            "whole_token_reversal_pairs": [list(pair) for pair in mirrored],
            "shared_tokens": repeated, "repeated_content_tokens": repeated_content,
            "multi_letter_self_palindromic_tokens": self_pal,
            "one_letter_tokens": sorted({w for w in left + right if len(w) == 1})}


def audit_full_child(parent: str, y_surface: str, left_surface: str) -> dict:
    from llm_palindrome.validator import is_palindrome
    rendered = left_surface + " " + parent + " " + y_surface
    tape = letters(rendered)
    fwd, rev = sha(tape), sha(tape[::-1])
    pointer = outside_in(tape)
    project = bool(is_palindrome(rendered))
    return {"rendered": rendered, "letters": len(tape), "growth": len(tape) - 568,
            "normalized_sha256": fwd, "reverse_sha256": rev,
            "independent_outside_in": pointer, "project_validator_exact": project,
            "sha_equal": fwd == rev,
            "all_exact_checks_agree": pointer["exact"] == project == (fwd == rev)}


def main() -> dict:
    parent_data = json.loads(PARENT_PATH.read_text())
    parent = next(row["rendered"] for row in parent_data["rows"]
                  if row["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    from llm_palindrome.validator import is_palindrome
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA256:
        raise AssertionError("Pinned 568 parent identity changed")
    if not outside_in(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("Pinned parent did not independently validate")

    bank = sentence_bank()
    closures, rejected_shortcuts, frontiers = [], [], []
    chart_states = 0
    for row in bank:
        y_tape = letters(row["surface"])
        left_parse = reverse_parser(y_tape)
        chart_states += 4  # one state for each subject x verb prefix, exactly as in parser
        if left_parse is None:
            frontiers.append({"Y": row["surface"], "Y_tape": y_tape,
                              "reverse_tape": y_tape[::-1],
                              "best_reverse_grammar_prefix": longest_grammar_prefix(y_tape),
                              "attachment_state": row["attachment_state"]})
            continue
        masks = word_boundary_audit(left_parse["surface"], row["surface"])
        constructed = {
            "Y": row["surface"],
            "Y_tape": y_tape,
            "reverse_Y_tape": y_tape[::-1],
            "reverse_sentence_parse": left_parse,
            "attachment_state": row["attachment_state"],
            "word_boundary_audit": masks,
        }
        if (masks["whole_token_reversal_pairs"]
                or masks["multi_letter_self_palindromic_tokens"]
                or masks["one_letter_tokens"]):
            rejected_shortcuts.append(constructed)
            continue
        # The left surface is a grammatical segmentation of reverse(Y), not
        # reverse-word spacing. Its normalized tape is checked before wrapping.
        if letters(left_parse["surface"]) != y_tape[::-1]:
            raise AssertionError("Reverse parser returned a nonmatching surface")
        child = audit_full_child(parent, row["surface"], left_parse["surface"])
        child["independent_exact_audit"] = child.pop("independent_outside_in")
        child["exact"] = child["independent_exact_audit"]["exact"] and child["project_validator_exact"] and child["sha_equal"]
        child["Y"] = row["surface"]
        child["left_sentence"] = left_parse["surface"]
        child["provenance"] = {"fresh_authored_lexical_bank": True,
                                "no_seed_sentence_reused": True,
                                "no_catalogue_text": True,
                                "grammar_slots": row,
                                "reader_evidence": False}
        closures.append(child)

    # Choose best actual controls by grammatical prefix support, not by a
    # readability score. Preserve enough details to inspect residual ownership.
    frontiers.sort(key=lambda x: (-x["best_reverse_grammar_prefix"]["matched"], x["Y"]))
    best_controls = frontiers[:5]
    exact = [row for row in closures if row["exact"]]
    admitted = [row for row in exact if not row["word_boundary_audit"]["whole_token_reversal_pairs"]
                and not row["word_boundary_audit"]["multi_letter_self_palindromic_tokens"]
                and not row["word_boundary_audit"]["one_letter_tokens"]]

    # Novelty source audit: the current full-sentence wrapper probe is known,
    # but this finite grammar/trie intersection is the new operator.
    novelty = {
        "shared_geometry": "released [0,232)/end-568 outer wrapper",
        "prior_full_sentence_wrapper": "runs/luna6-full-sentence-wrapper-probe-20260923.json; one fixed Y, exact but left reversal not readable",
        "distinctive_operator": "jointly compose Y under typed SVO/adjective/PP-attachment grammar and parse reverse(Y) through the same character chart before rendering",
        "catalogue_lookup": False,
        "38_letter_seed_reuse": False,
        "broad_sweep": False,
    }
    result = {
        "experiment_id": "luna6-grammatical-resegmentation-outer-20260923",
        "method": "bounded bidirectional typed grammar intersection at [0,232)/end-568",
        "parent": {"artifact": str(PARENT_PATH.relative_to(ROOT)), "letters": 568,
                   "sha256": PARENT_SHA256, "independent_exact": True, "rendered": parent},
        "novelty_preflight": novelty,
        "search": {
            "candidate_grammar": "S -> NP VP; VP -> V NP or V NP PP; NP object may contain an adjective modifier",
            "candidate_cap": 96,
            "candidate_count": len(bank),
            "reverse_parse_prefix_states": chart_states,
            "state_cap": 384,
            "lexical_slots": {"subjects": len(SUBJECTS), "verbs": len(VERBS),
                              "adjective_object_NPs": len(ADJECTIVE_OBJECTS),
                              "PP_objects": len(PP_OBJECTS), "PP_attachments": len(PPS)},
            "per_candidate_rlaif": False,
        },
        "results": {
            "complete_reverse_grammar_parses": len(closures) + len(rejected_shortcuts),
            "shortcut_rejected_parses": len(rejected_shortcuts),
            "shortcut_clean_exact_children": len(admitted),
            "best_authored_controls": best_controls,
            "rejected_shortcut_examples": rejected_shortcuts[:5],
            "rendered_exact_children": admitted,
            "reader_status": "no human ratings collected; even a grammatical envelope would not certify readability of the inherited 568-letter interior",
        },
        "oversized_pilot_retired": {
            "candidate_strings": 177450,
            "shortcut_clean_reverse_parses": 0,
            "disposition": "pilot exceeded intended local cap; retired and not used to select the bounded 96-row lexical bank",
        },
        "next_operator": (
            "Pivot from forward-Y-then-reverse-parse to a suffix-first, boundary-crossing chart: choose one typed clause-initial production, solve the final two Y token slots backward against its character residual, and permit the reverse parse boundary to cut across the Y token boundary. Preflight that exact topology against registry/history before a single capped realization; do not widen this lexical bank."
            if not admitted else
            "Keep only the exact examples shown; collect blinded readability judgments with intact-prose and shuffled controls before promotion."
        ),
    }
    return result


if __name__ == "__main__":
    payload = main()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"experiment": payload["experiment_id"], **payload["search"], **payload["results"]}, sort_keys=True))
