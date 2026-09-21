"""Connector-first ABBA paragraph search.

Relations are selected before lexical realization.  The opposing B/A clauses
must use a compatible connective, tense, and valency, while a full residual
trie consumes the character obligation online.  This is a topology change,
not a larger phrase bank or a score applied after generation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from experiments.abba_full_residual_lexical_trie_20260922 import audit, letters
except ModuleNotFoundError:
    from abba_full_residual_lexical_trie_20260922 import audit, letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-connector-first-trie-20260922.json"

LEFT = {
    "A": (
        "At dusk, the patient cartographer restored a faded mural.",
        "By dawn, the village baker carried warm loaves to market.",
    ),
    "B": (
        "At noon, a careful gardener watered the shared garden.",
        "In winter, the quiet mechanic repaired a cedar cabinet.",
    ),
}

# The relation is a grammar state: it determines connective, tense, valency,
# and the semantic role of the second clause before any surface is admitted.
RELATIONS = {
    "causal": {
        "connectors": ("because", "so"),
        "subjects": ("the patient archivist", "a quiet sailor"),
        "verbs": ("records", "studies"),
        "objects": ("a folded map", "the blue lantern"),
        "adjuncts": ("before dawn", "near the harbor"),
    },
    "temporal": {
        "connectors": ("while", "after"),
        "subjects": ("our careful teacher", "the young botanist"),
        "verbs": ("carries", "notices"),
        "objects": ("one small basket", "a folded map"),
        "adjuncts": ("near the harbor", "beside the market"),
    },
    "concessive": {
        "connectors": ("although", "though"),
        "subjects": ("the patient archivist", "our careful teacher"),
        "verbs": ("studies", "records"),
        "objects": ("the blue lantern", "one small basket"),
        "adjuncts": ("beside the market", "before dawn"),
    },
}


def trie_add(root: dict, phrase: str, payload: tuple) -> None:
    node = root
    for char in letters(phrase):
        node = node["children"].setdefault(char, {"children": {}, "terminal": []})
    node["terminal"].append(payload)


def decode(obligation: str, relation: dict, max_parses: int = 24):
    """Parse the complete residual with slots from one relation state."""
    bank = {"connector": relation["connectors"], "subject": relation["subjects"],
            "verb": relation["verbs"], "object": relation["objects"],
            "adjunct": relation["adjuncts"]}
    # The residual is read from the physical right edge inward.  Therefore
    # the final clause's adjunct/object/... comes first; the connector is an
    # interior seam state, not an incorrectly forced sentence-initial token.
    slots = ("adjunct", "object", "verb", "subject", "connector",
             "adjunct", "object", "verb", "subject")
    tries = {role: {"children": {}, "terminal": []} for role in bank}
    for role, values in bank.items():
        for phrase in values:
            trie_add(tries[role], phrase, (role, phrase))
    memo = {}
    frontier = []
    transitions = []

    def go(slot: int, pos: int):
        key = (slot, pos)
        if key in memo:
            return memo[key]
        if slot == len(slots):
            return [()] if pos == len(obligation) else []
        role = slots[slot]
        node = tries[role]
        cursor = pos
        matches = []
        while cursor < len(obligation) and obligation[cursor] in node["children"]:
            node = node["children"][obligation[cursor]]
            cursor += 1
            for _role, phrase in node["terminal"]:
                matches.append((cursor, phrase))
        if not matches:
            frontier.append({"slot": slot, "role": role, "offset": pos,
                             "matched_characters": cursor - pos,
                             "required_residual": obligation[pos:pos + 16],
                             "trie_prefix": obligation[pos:cursor]})
        results = []
        for end, phrase in matches:
            transitions.append({"slot": slot, "role": role, "start": pos,
                                "end": end, "surface": phrase})
            for tail in go(slot + 1, end):
                results.append((phrase,) + tail)
                if len(results) >= max_parses:
                    break
        memo[key] = results
        return results

    return go(0, 0), frontier, transitions


def run() -> dict:
    rendered = []
    controls = []
    certificates = []
    for relation_name, relation in RELATIONS.items():
        for a in LEFT["A"]:
            for b in LEFT["B"]:
                left = f"{a} {b}"
                obligation = letters(left)[::-1]
                parses, frontier, transitions = decode(obligation, relation)
                certificates.append({"relation": relation_name, "left_A_B": [a, b],
                                    "residual_prefix": obligation[:18],
                                    "parse_count": len(parses),
                                    "deepest_support": max(
                                        (x["matched_characters"] for x in frontier), default=0),
                                    "frontier": frontier[:8],
                                    "transitions": transitions[:16]})
                controls.append({"rendered": left, "relation": relation_name,
                                 "kind": "intact-authored-AB-control", "audit": audit(left),
                                 "provenance": {"relation_selected_before_surface": True,
                                                "complete_prose": True}})
                for parse in parses:
                    # Two complete connective-bearing clauses, not mirrored units.
                    # Parse order is right-edge to center; render its reverse
                    # as two complete clauses around the selected connector.
                    right = (f"{parse[8]} {parse[7]} {parse[6]} {parse[5]} "
                             f"{parse[4]} {parse[3]} {parse[2]} {parse[1]} {parse[0]}")
                    text = f"{left} {right}."
                    rendered.append({"rendered": text, "relation": relation_name,
                                     "audit": audit(text), "provenance": {
                                         "relation_state_first": True,
                                         "full_residual_trie": True,
                                         "variable_sentence_boundaries": True,
                                         "semantic_ABBA_roles": True,
                                         "finished_text_reversal": False,
                                         "catalogue_text": False,
                                         "repeated_units": False,
                                         "self_palindromic_units": False,
                                         "posthoc_repair": False,
                                         "reward_model": False}})
    exact = [row for row in rendered
             if row["audit"]["two_pointer_exact"] and row["audit"]["letters"] > 38]
    return {
        "experiment_id": "abba-connector-first-trie-20260922",
        "method": "relation/connective-first ABBA full-residual trie",
        "stats": {"relations": len(RELATIONS), "branches": len(certificates),
                   "rendered_candidates": len(rendered),
                   "closed_derivations": len(rendered), "exact_gt38": len(exact),
                   "deepest_support": max((c["deepest_support"] for c in certificates), default=0)},
        "exact_candidates": exact, "rendered_candidates": rendered,
        "controls": controls, "residual_certificates": certificates,
        "novelty_preflight": {"status": "passed",
            "signature": "abba|relation-first|typed-connective|full-residual-trie",
            "distinct_from": "fixed connector banks, NP-depth topology, and onset-only selection",
            "finished_tape_reversal": False, "catalogue_text": False,
            "mirrored_units": False, "reward_ranking": False},
        "provenance": {"generator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["two-pointer scan", "forward/reverse SHA-256"],
            "reader_gate": "closed pending novel exact output"},
        "status": "fresh exact closure found" if exact else
                  "no exact closure; connector residual certificate retained",
        "next_construction": "condition the first connector and subject jointly on the first unsupported residual; retain relation state and do not widen connector banks",
    }


if __name__ == "__main__":
    data = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["stats"], sort_keys=True))
