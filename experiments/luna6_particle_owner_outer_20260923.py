"""One phrasal-particle owner-map attempt at a fresh pinned-568 seam.

The authored pair is a linked packing/recovery event.  A live reflected
character cursor tests it once; a mismatch is retained rather than repaired
by token reversal or a wider lexical sweep.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs/luna6-particle-owner-outer-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def sha(tape: str) -> str:
    return hashlib.sha256(tape.encode("ascii")).hexdigest()


def raw_offset_for_letter(text: str, offset: int) -> int:
    consumed = 0
    for i, ch in enumerate(text):
        if ch.isascii() and ch.isalpha():
            if consumed == offset:
                return i
            consumed += 1
    if consumed == offset:
        return len(text)
    raise ValueError(f"normalized offset {offset} outside {consumed}-letter text")


def outside_in(tape: str) -> dict:
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {"exact": i >= j, "letters": len(tape), "matched_outer_pairs": i,
            "first_mismatch": None if i >= j else {
                "offset_left": i, "offset_right": j,
                "left": tape[i], "right": tape[j]},
            "left_residual": tape[i:i + 32],
            "right_reverse_residual": tape[max(0, j - 31):j + 1][::-1]}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.casefold())


def shortcut_audit(left: str, right: str) -> dict:
    lt, rt = tokens(left), tokens(right)
    pairs = sorted({(a, b) for a in lt for b in rt if a[::-1] == b})
    one = sorted({w for w in lt + rt if len(w) == 1})
    self_pal = sorted({w for w in lt + rt if len(w) > 1 and w == w[::-1]})
    shared = sorted(set(lt) & set(rt))
    return {"left_tokens": lt, "right_tokens": rt,
            "whole_token_reversal_pairs": [list(pair) for pair in pairs],
            "one_letter_palindromic_supports": one,
            "multi_letter_self_palindromic_tokens": self_pal,
            "shared_tokens": shared,
            "repeated_content_tokens": sorted(set(shared) - {"a", "an", "the", "them"}),
            "shortcut_free": not (pairs or one or self_pal or shared)}


def main() -> dict:
    parent_payload = json.loads(PARENT_PATH.read_text())
    parent = next(r["rendered"] for r in parent_payload["rows"]
                  if r["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    from llm_palindrome.validator import is_palindrome
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA256:
        raise AssertionError("Pinned parent identity changed")
    parent_pointer = outside_in(parent_tape)
    parent_project = bool(is_palindrome(parent))
    if not parent_pointer["exact"] or not parent_project:
        raise AssertionError("Pinned parent failed independent exact validation")

    # Search the pinned Git revision, not the mutable worktree, so this script
    # and its own run cannot become novelty self-hits on a rerun.
    head_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   check=True, capture_output=True, text=True).stdout.strip()
    geometry_signature = "pinned-568|insert@81|mirror_insert@487|split-object-particle-owner"
    geometry = subprocess.run(
        ["git", "grep", "-n", "-F", geometry_signature, head_revision, "--",
         "docs", "runs", "experiments", "data"], cwd=ROOT,
        capture_output=True, text=True)
    phrase_search = subprocess.run(
        ["git", "grep", "-n", "-F", "-i", "-e", "Nora packed the red books up.",
         "-e", "Aram found them before dawn.", head_revision, "--",
         "docs", "runs", "experiments", "data"], cwd=ROOT,
        capture_output=True, text=True)
    if geometry.returncode not in (0, 1) or phrase_search.returncode not in (0, 1):
        raise RuntimeError(geometry.stderr + phrase_search.stderr)

    # Nearby but distinct precedent includes typed V_PARTICLE return-frame
    # mining and an internal resegmentation with an odd `set on Siri` reading;
    # neither used this 568 outer insertion geometry or this split-object map.
    preflight = {
        "head_revision": head_revision,
        "parent": {"letters": 568, "sha256": PARENT_SHA256},
        "normalized_insertion_cuts": {"left": 81, "right": 487,
                                      "reflected_relation": "568 - 81 = 487"},
        "raw_context": {"left_cut_after": "A tub?", "right_cut_after": "Eh",
                         "right_continuation_before_insert": ", but a star spots Aram."},
        "geometry_phrase_check": {
            "exact_geometry_signature": geometry_signature,
            "geometry_hits": geometry.stdout.splitlines(),
            "exact_geometry_signature_found_in_committed_head": bool(geometry.stdout),
            "authored_phrase_hits": phrase_search.stdout.splitlines(),
            "left_phrase_found": "Nora packed the red books up.".lower() in phrase_search.stdout.lower(),
            "right_phrase_found": "Aram found them before dawn.".lower() in phrase_search.stdout.lower(),
            "left_authored_clause": "Nora packed the red books up.",
            "right_authored_clause": "Aram found them before dawn.",
            "scope": "tracked experiments, runs, docs, data; exact signature and exact clause literals"},
        "related_but_nonidentical_precedents": [
            {"artifact": "experiments/typed_multiword_return_frames_20260922.py",
             "difference": "mines V_PARTICLE frame types on short return-stack tapes; not a linked event pair at a selected 568 outer seam"},
            {"artifact": "experiments/incumbent_568_internal_event_resegmentation_20260923.py",
             "difference": "includes an unusual `set on Siri` interpretation but has a different seam, no split object-particle ownership rule, and unrelated clause list"},
            {"artifact": "experiments/contraction_lexicalized_seam_20260921.py",
             "difference": "tests contractions/preposition edges, not a transitive verb with intervening object and final particle"},
        ],
        "collision_found": False,
    }

    left_clause = "Nora packed the red books up."
    right_clause = "Aram found them before dawn."
    left_tape, right_tape = letters(left_clause), letters(right_clause)
    required_left = right_tape[::-1]
    cursor = 0
    trace = []
    while cursor < min(len(left_tape), len(required_left)) and left_tape[cursor] == required_left[cursor]:
        trace.append({"cursor": cursor, "left_owner": "left clause", "right_owner": "reverse of right clause",
                      "left": left_tape[cursor], "required": required_left[cursor], "matched": True})
        cursor += 1
    mismatch = None
    if cursor < min(len(left_tape), len(required_left)):
        mismatch = {"cursor": cursor, "left_char": left_tape[cursor],
                    "required_char": required_left[cursor], "left_residual": left_tape[cursor:],
                    "required_residual": required_left[cursor:]}
        trace.append({**mismatch, "matched": False})

    # The split phrasal construction is typed as V + intervening NP + RP.
    left_owners = [
        {"owner": "subject", "surface": "Nora", "span": [0, 4]},
        {"owner": "verb", "surface": "packed", "span": [4, 10]},
        {"owner": "object-determiner", "surface": "the", "span": [10, 13]},
        {"owner": "object-adjective", "surface": "red", "span": [13, 16]},
        {"owner": "object-head", "surface": "books", "span": [16, 21]},
        {"owner": "particle", "surface": "up", "span": [21, 23]},
    ]
    right_owners = []
    offset = 0
    for surface, owner in [("Aram", "subject"), ("found", "verb"), ("them", "anaphoric-object"),
                           ("before", "temporal-adjunct-prep"), ("dawn", "temporal-adjunct-head")]:
        tape = letters(surface)
        right_owners.append({"owner": owner, "surface": surface,
                             "source_span": [offset, offset + len(tape)],
                             "reverse_surface": tape[::-1]})
        offset += len(tape)

    # One insertion at a normalized cut and its reflected partner. Include the
    # attempted complete rendering even though the live equation stops early.
    left_raw = raw_offset_for_letter(parent, 81)
    right_raw = raw_offset_for_letter(parent, 487)
    # Both raw offsets point at the first letter after the semantic cut:
    # the left context already contains a space after `?`, and the right
    # context already contains `, ` after `Eh`. Keep those separators in place.
    left_insert = "Nora packed the red books up. "
    right_insert = "Aram found them before dawn, "
    attempted = parent[:left_raw] + left_insert + parent[left_raw:right_raw] + right_insert + parent[right_raw:]
    attempted_tape = letters(attempted)
    full_audit = outside_in(attempted_tape)
    forward_sha, reverse_sha = sha(attempted_tape), sha(attempted_tape[::-1])
    project_exact = bool(is_palindrome(attempted))
    exact = full_audit["exact"] and project_exact and forward_sha == reverse_sha
    masks = shortcut_audit(left_clause, right_clause)

    return {
        "experiment_id": "luna6-particle-owner-outer-20260923",
        "method": "single linked event pair with V-object-particle order and live reflected cursor",
        "novelty_preflight": preflight,
        "parent": {"artifact": str(PARENT_PATH.relative_to(ROOT)), "letters": 568,
                   "sha256": PARENT_SHA256, "outside_in_exact": parent_pointer["exact"],
                   "project_validator_exact": parent_project},
        "authored_pair": {
            "left_clause": left_clause,
            "left_parse": {"S": "Nora/agent", "V": "packed/past", "O": "the red books/patient",
                           "RP": "up/particle", "frame": "Nora packed [the red books] up"},
            "right_clause": right_clause,
            "right_parse": {"S": "Aram/agent", "V": "found/past", "O": "them/anaphor referring to the red books",
                            "temporal_adjunct": "before dawn"},
            "semantic_relation": "Aram later finds the red books that Nora packed up.",
            "provenance": "one freshly authored linked event pair; exact literal preflight found no tracked occurrence; no catalogue or seed text"},
        "owners": {"left_clause": left_owners, "right_clause_source_order": right_owners,
                   "right_reversed_tape": required_left,
                   "particle_object_boundary": "object ends at left tape offset 21; particle `up` occupies [21,23); equation mismatch occurs earlier at offset 1"},
        "live_equation": {"left_tape": left_tape, "required_left_from_right": required_left,
                          "cursor": cursor, "matched_characters": len(trace) - (1 if mismatch else 0),
                          "residual": mismatch, "trace": trace},
        "attempted_full_rendering": attempted,
        "candidate_audit": {"letters": len(attempted_tape), "growth": len(attempted_tape) - 568,
                            "independent_outside_in": full_audit,
                            "project_validator_exact": project_exact,
                            "forward_sha256": forward_sha, "reverse_sha256": reverse_sha,
                            "sha_equal": forward_sha == reverse_sha, "exact": exact,
                            "shortcut_audit": masks,
                            "admitted": False,
                            "reason": "The clauses pass the anti-shortcut lexical audit, but their live equation contradicts at character 1, before it can consume the object/particle boundary."},
        "next_operator": "Pivot from clause-first phrase selection to suffix-first lexicalization: constrain the partner clause's final multiword owner sequence by the left clause's exact opening residual (`norapacked...`), not by a single guessed ending; realize characters backward from that suffix and continue only if the next owner also matches. Preflight one distinct seam and one typed final phrase, reject direct reversed-token pairs, and then make one realization. This changes construction order rather than widening the particle lexicon.",
        "reader_status": "no exact candidate; no human ratings or readability claim",
    }


if __name__ == "__main__":
    payload = main()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"experiment": payload["experiment_id"],
                      "seam": payload["novelty_preflight"]["normalized_insertion_cuts"],
                      "pair_cursor": payload["live_equation"]["cursor"],
                      "mismatch": payload["live_equation"]["residual"],
                      "full_letters": payload["candidate_audit"]["letters"],
                      "full_exact": payload["candidate_audit"]["exact"],
                      "shortcut_free": payload["candidate_audit"]["shortcut_audit"]["shortcut_free"]}, sort_keys=True))
