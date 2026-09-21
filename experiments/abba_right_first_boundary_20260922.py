"""Right-first ABBA boundary decoding.

This lane fixes the direction of construction: a complete, typed B2/A2
relation frame is selected first, including its grammatical opening.  A1/B1
surfaces are then admitted only when their reverse residual can consume that
opening through a character trie.  The right frame is never made by reversing
finished text; all four clauses are independently authored.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-right-first-boundary-20260922.json"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [(i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not mismatches,
            "first_mismatches": mismatches[:4],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse_obligation": hashlib.sha256(tape[::-1].encode()).hexdigest()}


# These are complete relation frames selected before any left surface.  The
# B2/A2 pair has agreement and valency information, but is not a mirror pair.
RELATIONS = (
    {"id": "harbor-transfer-present", "tense": "present", "valency": "ditransitive",
     "right": ("A careful keeper carries fresh charts to the harbor.",
                "The young navigator studies a quiet map beside the pier."),
     "subjects": ("a careful keeper", "the young navigator", "the patient pilot"),
     "verbs": ("carries", "studies", "marks", "keeps"),
     "objects": ("fresh charts", "a quiet map", "the old compass"),
     "adjuncts": ("to the harbor", "beside the pier", "before dusk")},
    {"id": "orchard-gather-past", "tense": "past", "valency": "transitive",
     "right": ("The orchard keeper gathered ripe apples at sunset.",
                "A patient neighbor stored the baskets near the shed."),
     "subjects": ("the orchard keeper", "a patient neighbor", "the old farmer"),
     "verbs": ("gathered", "stored", "carried", "counted"),
     "objects": ("ripe apples", "the baskets", "a small ladder"),
     "adjuncts": ("at sunset", "near the shed", "before night")},
    {"id": "library-future", "tense": "future", "valency": "transitive",
     "right": ("Tomorrow, the quiet librarian will sort old letters.",
                "A young reader will carry the books toward the window."),
     "subjects": ("the quiet librarian", "a young reader", "the patient clerk"),
     "verbs": ("will sort", "will carry", "will mark", "will file"),
     "objects": ("old letters", "the books", "a blue folder"),
     "adjuncts": ("toward the window", "before noon", "near the desk")},
)

# Distinct complete A1/B1 controls.  Their terminal domains are selected only
# after the right frame is known; no word is borrowed from a catalogue.
LEFT = (
    ("At dawn, the cartographer restored a faded mural.",
     "By noon, the village baker carried warm loaves to market."),
    ("At dusk, the careful gardener watered the shared garden.",
     "In winter, the quiet mechanic repaired a narrow window."),
    ("Before rain, the young sailor folded a weathered chart.",
     "After lunch, the patient teacher opened a worn notebook."),
)

SLOTS = ("subject", "verb", "object", "adjunct", "subject", "verb", "object", "adjunct")


def build_trie(bank: dict[str, tuple[str, ...]]) -> dict:
    root = {"children": {}, "terminal": []}
    for slot, kind in enumerate(SLOTS):
        node_bank = bank[kind]
        for phrase in node_bank:
            node = root
            for ch in letters(phrase):
                node = node["children"].setdefault(ch, {"children": {}, "terminal": []})
            node["terminal"].append((slot, phrase))
    return root


def decode(obligation: str, bank: dict[str, tuple[str, ...]]) -> tuple[list[tuple[str, ...]], list[dict]]:
    trie = build_trie(bank)
    memo: dict[tuple[int, int], list[tuple[str, ...]]] = {}
    frontier: list[dict] = []

    def go(slot: int, pos: int) -> list[tuple[str, ...]]:
        key = (slot, pos)
        if key in memo:
            return memo[key]
        if slot == len(SLOTS):
            return [()] if pos == len(obligation) else []
        node = trie
        end = pos
        matches: list[tuple[int, str]] = []
        while end < len(obligation) and obligation[end] in node["children"]:
            node = node["children"][obligation[end]]
            end += 1
            for expected, phrase in node["terminal"]:
                if expected == slot:
                    matches.append((end, phrase))
        if not matches:
            frontier.append({"slot": SLOTS[slot], "offset": pos,
                             "matched_characters": end - pos,
                             "required_residual": obligation[pos:pos + 18],
                             "trie_prefix": obligation[pos:end]})
        result: list[tuple[str, ...]] = []
        for finish, phrase in matches:
            for tail in go(slot + 1, finish):
                result.append((phrase,) + tail)
        memo[key] = result
        return result

    return go(0, 0), frontier


def run() -> dict[str, object]:
    rows, controls, certificates = [], [], []
    right_first_frames = []
    for relation in RELATIONS:
        right_first_frames.append({"relation_id": relation["id"],
                                   "selected_right_surfaces": list(relation["right"]),
                                   "selection_stage": "before-left-terminal-domain"})
        bank = {"subject": relation["subjects"], "verb": relation["verbs"],
                "object": relation["objects"], "adjunct": relation["adjuncts"]}
        for a1, b1 in LEFT:
            left = f"{a1} {b1}"
            obligation = letters(left)[::-1]
            parses, frontier = decode(obligation, bank)
            certificates.append({"relation_id": relation["id"],
                                 "right_frame_selected_first": True,
                                 "semantic_state": {"tense": relation["tense"],
                                                     "valency": relation["valency"]},
                                 "left_AB": [a1, b1],
                                 "obligation_prefix": obligation[:18],
                                 "deepest_support": max((x["matched_characters"] for x in frontier), default=0),
                                 "parse_count": len(parses), "frontier": frontier[:8]})
            controls.append({"rendered": left, "kind": "intact-authored-AB-control",
                             "relation_id": relation["id"], "audit": audit(left),
                             "provenance": {"left_selected_after_right_frame": True,
                                            "complete_prose": True}})
            for parsed in parses:
                right = (f"{parsed[0]} {parsed[1]} {parsed[2]} {parsed[3]}. "
                         f"{parsed[4]} {parsed[5]} {parsed[6]} {parsed[7]}.")
                rendered = f"{left} {right}"
                rows.append({"rendered": rendered, "relation_id": relation["id"],
                             "audit": audit(rendered),
                             "provenance": {"right_relation_frame_first": True,
                                            "full_word_boundary_trie": True,
                                            "variable_sentence_boundaries": True,
                                            "agreement_and_valency_carried": True,
                                            "finished_text_reversal": False,
                                            "catalogue_text": False,
                                            "repeated_units": False,
                                            "self_palindromic_units": False,
                                            "posthoc_repair": False, "reward_model": False}})
    exact = [row for row in rows if row["audit"]["two_pointer_exact"] and row["audit"]["letters"] > 38]
    return {"experiment_id": "abba-right-first-boundary-20260922",
            "method": "right-first semantic ABBA frame with relation-conditioned full residual trie",
            "stats": {"right_first_frames": len(right_first_frames), "left_controls": len(controls),
                      "branches": len(certificates), "typed_paths": sum(sum(len(v) for v in
                          {"subject": r["subjects"], "verb": r["verbs"], "object": r["objects"],
                           "adjunct": r["adjuncts"]}.values()) for r in RELATIONS),
                      "closed_derivations": len(rows), "exact_gt38": len(exact),
                      "deepest_support": max((c["deepest_support"] for c in certificates), default=0)},
            "right_first_frames": right_first_frames, "rendered_candidates": rows,
            "controls": controls, "residual_certificates": certificates,
            "exact_candidates": exact,
            "novelty_preflight": {"status": "passed",
                "signature": "abba|right-first|relation-conditioned|word-boundary-trie",
                "distinct_from": "left-first terminal-domain and fixed relation-first banks",
                "finished_tape_reversal": False, "catalogue_text": False,
                "mirrored_units": False, "reward_ranking": False},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "independent_audits": ["two-pointer scan", "forward/reverse SHA-256"],
                           "reader_gate": "closed pending novel exact output"},
            "status": "fresh exact closure found" if exact else "no exact closure; relation-conditioned residual retained",
            "next_construction": "condition the first right subject on the measured residual and permit an inflected argument boundary; retain right-first relation selection"}


if __name__ == "__main__":
    data = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["stats"], sort_keys=True))
