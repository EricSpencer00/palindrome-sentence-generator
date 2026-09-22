"""Typed two-sentence discourse join from the incumbent's first live seam.

The left grammar begins with ``An aide`` and the reverse-facing right grammar
ends with ``Diana``.  Their character intersection leaves live residual ``e``;
all verbs, arguments, and optional second sentences are then chosen inside one
exact hash join.  The 38-letter incumbent is an oracle only.  Any proper
palindromic span, including an incumbent wrapper, is rejected centrally.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.packed_staggered_paragraph_automaton_20260922 import (
    AGENT, AGENTS, DET, DOCUMENT, INTRANSITIVE, NAMES, PAST, PRESENT,
    PRONOUN, QUANT,
)
from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.paragraph_product import audit_staggered_abba
from llm_palindrome.recursive_product import tape


ID = "seed-open-residual-discourse-join-20260922"
INCUMBENT = "An aide rips nine memos; some men inspire Diana."
LEFT_SUBJECT = "An aide"
RIGHT_PATIENT = "Diana"

SG_PRESENT = (
    "calls", "checks", "edits", "files", "finds", "helps", "keeps",
    "marks", "notes", "opens", "reads", "records", "rips", "saves",
    "sends", "signs", "sorts", "studies", "traces", "writes",
)

DOCUMENT_BY_NUMBER = {
    "SG": tuple(word for word in DOCUMENT if not word.endswith("s")),
    "PL": tuple(word for word in DOCUMENT if word.endswith("s")),
}


def _pointer_audit(text: str) -> dict:
    letters = tape(text)
    left, right = 0, len(letters) - 1
    while left < right and letters[left] == letters[right]:
        left += 1
        right -= 1
    forward = hashlib.sha256(letters.encode()).hexdigest()
    reverse = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {
        "letters": len(letters),
        "exact": left >= right,
        "first_mismatch": None if left >= right else [left, right],
        "sha256": forward,
        "reverse_sha256": reverse,
        "hashes_agree": forward == reverse,
    }


def _object_phrases() -> tuple[tuple[str, str], ...]:
    rows = []
    for determiner in ("a", "one"):
        rows.extend((f"{determiner} {word}", "SG")
                    for word in DOCUMENT_BY_NUMBER["SG"])
    for determiner in ("nine", "some", "many"):
        rows.extend((f"{determiner} {word}", "PL")
                    for word in DOCUMENT_BY_NUMBER["PL"])
    for word in DOCUMENT:
        rows.append((f"the {word}", "PL" if word.endswith("s") else "SG"))
    return tuple(rows)


def _left_events() -> tuple[dict, ...]:
    return tuple({
        "sentence": f"{LEFT_SUBJECT} {verb} {obj}.",
        "roles": {"agent": "aide", "event": verb, "patient": obj},
    } for verb, (obj, _number) in itertools.product(SG_PRESENT,
                                                     _object_phrases()))


def _right_events() -> tuple[dict, ...]:
    rows = []
    for quantifier, agents, verb in itertools.product(QUANT, AGENTS, PRESENT):
        rows.append({
            "sentence": f"{quantifier.capitalize()} {agents} {verb} {RIGHT_PATIENT}.",
            "roles": {"agent": f"{quantifier} {agents}", "event": verb,
                      "patient": "Diana"},
        })
    for name, verb in itertools.product(NAMES, PAST):
        if name.casefold() == RIGHT_PATIENT.casefold():
            continue
        rows.append({
            "sentence": f"{name} {verb} {RIGHT_PATIENT}.",
            "roles": {"agent": name, "event": verb, "patient": "Diana"},
        })
    for pronoun, verb in itertools.product(PRONOUN, PAST):
        rows.append({
            "sentence": f"{pronoun.capitalize()} {verb} {RIGHT_PATIENT}.",
            "roles": {"agent": pronoun, "event": verb, "patient": "Diana"},
        })
    return tuple(rows)


def _bridges() -> tuple[dict, ...]:
    """Small complete sentence bank with explicit discourse roles."""
    rows = [{"sentence": "", "roles": {"kind": "epsilon-control"}}]
    for pronoun, verb in itertools.product(("she", "he", "they"),
                                           INTRANSITIVE):
        rows.append({
            "sentence": f"{pronoun.capitalize()} {verb}.",
            "roles": {"agent": pronoun, "event": verb,
                      "kind": "referential-intransitive"},
        })
    for pronoun, verb, (obj, _number) in itertools.product(
            ("she", "he", "they"), PAST, _object_phrases()):
        rows.append({
            "sentence": f"{pronoun.capitalize()} {verb} {obj}.",
            "roles": {"agent": pronoun, "event": verb, "patient": obj,
                      "kind": "referential-transitive"},
        })
    for name, verb in itertools.product(NAMES, INTRANSITIVE):
        rows.append({
            "sentence": f"{name} {verb}.",
            "roles": {"agent": name, "event": verb,
                      "kind": "named-intransitive"},
        })
    # Deduplicate identical surface tapes while preserving deterministic roles.
    unique = {}
    for row in rows:
        unique.setdefault(tape(row["sentence"]), row)
    return tuple(unique.values())


def _content_words(text: str) -> set[str]:
    from llm_palindrome.admission import REPEATABLE_FUNCTION_WORDS, tokenize
    return {word for word in tokenize(text)
            if word not in REPEATABLE_FUNCTION_WORDS}


def _prefix_trie(rows: tuple[dict, ...]) -> dict:
    """Index reverse-facing event tapes without expanding event/bridge pairs."""
    root: dict = {}
    for row in rows:
        node = root
        for character in tape(row["sentence"])[::-1]:
            node = node.setdefault(character, {})
        node.setdefault("$", []).append(row)
    return root


def _prefix_matches(root: dict, target: str):
    """Yield events whose reverse tape is a prefix of ``target``."""
    node = root
    for offset, character in enumerate(target, start=1):
        node = node.get(character)
        if node is None:
            return
        for row in node.get("$", ()):
            yield row, target[offset:]


def run(*, maximum_rows: int = 200) -> dict:
    left_events = _left_events()
    right_events = _right_events()
    bridges = _bridges()

    # For L_event L_bridge == reverse(R_bridge R_event), the reverse-facing
    # event must prefix the target and the remaining suffix identifies the
    # reverse-facing bridge.  This factors the product rather than storing
    # millions of completed right paths.
    right_event_trie = _prefix_trie(right_events)
    reverse_bridge_index: dict[str, list[dict]] = defaultdict(list)
    for bridge in bridges:
        reverse_bridge_index[tape(bridge["sentence"])[::-1]].append(bridge)
    indexed_right_paths = len(right_events) * len(bridges)

    rows = []
    joined = 0
    for event, bridge in itertools.product(left_events, bridges):
        left_sentences = tuple(x for x in (event["sentence"], bridge["sentence"]) if x)
        left_surface = " ".join(left_sentences)
        target = tape(left_surface)
        for right_event, bridge_residual in _prefix_matches(
                right_event_trie, target):
            for right_bridge in reverse_bridge_index.get(bridge_residual, ()):
                joined += 1
                right_sentences = tuple(x for x in (
                    right_bridge["sentence"], right_event["sentence"]
                ) if x)
                rendered = " ".join(left_sentences + right_sentences)
                exact = _pointer_audit(rendered)
                if not exact["exact"]:
                    raise AssertionError(rendered)
                structural = audit_staggered_abba(left_sentences, right_sentences)
                admission = mechanical_admission_checks(rendered, min_letters=39,
                                                         max_letters=300)
                incumbent_control = tape(rendered) == tape(INCUMBENT)
                has_two_sides = bool(bridge["sentence"] and right_bridge["sentence"])
                semantic_roles = {
                    "left_A": event["roles"], "left_B": bridge["roles"],
                    "right_B_prime": right_bridge["roles"],
                    "right_A_prime": right_event["roles"],
                }
                content_words = [
                    word for sentence in left_sentences + right_sentences
                    for word in _content_words(sentence)
                ]
                repeated_content = len(content_words) != len(set(content_words))
                rows.append({
                    "rendered": rendered,
                    "left_sentences": list(left_sentences),
                    "right_sentences": list(right_sentences),
                    "letters": exact["letters"],
                    "semantic_roles": semantic_roles,
                    "independent_exact_audit": exact,
                    "structural_audit": structural,
                    "mechanical_admission": admission,
                    "incumbent_control": incumbent_control,
                    "has_two_generated_inner_sentences": has_two_sides,
                    "repeated_content_diagnostic": repeated_content,
                    "mechanically_admitted": (
                        not incumbent_control and has_two_sides
                        and structural["cross_sentence_coupled"]
                        and all(admission.values())
                    ),
                    "reader_status": (
                        "not_run; programmatic checks cannot certify readability"
                    ),
                    "provenance": {
                        "method": "joint exact hash intersection of typed discourse paths",
                        "incumbent_use": (
                            "first live residual and oracle only; never inserted "
                            "as a candidate"
                        ),
                        "finished_tape_reversal": False,
                        "catalogue_text": False,
                        "per_candidate_rlaif": False,
                    },
                })
    rows.sort(key=lambda row: (
        not row["mechanically_admitted"], -row["letters"], row["rendered"]
    ))
    retained = rows[:maximum_rows]
    controls = [row for row in rows if row["incumbent_control"]]
    if controls and not any(row["incumbent_control"] for row in retained):
        retained = retained[:max(0, maximum_rows - 1)] + controls[:1]
    admitted = [row for row in retained if row["mechanically_admitted"]]
    return {
        "experiment_id": ID,
        "method": "typed verb/argument/discourse state joined with the incumbent's open character residual",
        "stats": {
            "left_event_paths": len(left_events),
            "right_event_paths": len(right_events),
            "bridge_paths": len(bridges),
            "indexed_right_discourse_paths": indexed_right_paths,
            "exact_joins": joined,
            "retained_rows": len(retained),
            "mechanically_admitted_gt38": len(admitted),
        },
        "rows": retained,
        "mechanically_admitted_candidates": admitted,
        "reader_packet": [],
        "status": (
            "exact mechanically eligible discourses require direct prose review"
            if admitted else "no mechanically admitted paragraph discourse join"
        ),
        "next_discriminator": (
            "direct prose review, then randomized blinded intact/shuffled packet"
            if admitted else
            "retain the deepest nonempty residual and change one discourse production; do not widen every bank"
        ),
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inventories": "finite authored typed roles; no completed sentence catalogue",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-rows", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(maximum_rows=args.maximum_rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
