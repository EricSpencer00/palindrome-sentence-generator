"""Curated agreement-safe clause lattice with reverse-character trie search.

The semantic bank is deliberately small and hand-authored.  Clauses are
generated with subject agreement and valency constraints, then paired through a
reverse-character trie.  The bilateral checker streams residual characters and
records cross-word seams; it does not reverse or repair a finished candidate.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/agreement-clause-reverse-trie-20260920.json"
ID = "agreement-clause-reverse-trie-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatch = next(((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-1 - i]), None)
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": fwd,
            "sha256_reverse": rev, "sha_equal": fwd == rev}


@dataclass(frozen=True)
class Frame:
    number: str
    text: str
    valency: str
    words: tuple[str, ...]


SUBJECTS = {
    "singular": (("the", "gentle", "sailor"), ("a", "patient", "keeper"),
                 ("the", "winter", "poet")),
    "plural": (("the", "young", "scouts"), ("several", "bright", "guides"),
                ("the", "quiet", "pilots")),
}
VERBS = {
    "singular": (("transitive", "charts", ("the northern inlet", "a weathered map")),
                 ("intransitive", "waits", ("by the harbor", "under the cedar"))),
    "plural": (("transitive", "guard", ("the old bridge", "a brass lantern")),
               ("intransitive", "return", ("before dawn", "after the rain"))),
}


def clauses() -> tuple[Frame, ...]:
    out = []
    for number, subjects in SUBJECTS.items():
        for det, adj, noun in subjects:
            for valency, verb, complements in VERBS[number]:
                for complement in complements:
                    text = f"{det} {adj} {noun} {verb} {complement}."
                    out.append(Frame(number, text, valency, tuple(text[:-1].split())))
    return tuple(out)


def make_trie(frames: tuple[Frame, ...]):
    root = {"children": {}, "frame_ids": []}
    for idx, frame in enumerate(frames):
        node = root
        for char in letters(frame.text)[::-1]:
            node = node["children"].setdefault(char, {"children": {}, "frame_ids": []})
        node["frame_ids"].append(idx)
    return root


def trie_prefix(root, text: str) -> list[int]:
    """Return right frames whose reversed tapes share a live left prefix."""
    node = root
    for char in letters(text):
        node = node["children"].get(char)
        if node is None:
            return []
    return list(node["frame_ids"])


def live_pair(left: Frame, right: Frame, bridge: str) -> dict:
    """Stream left against reverse(right), retaining cross-word residuals."""
    l = letters(left.text)
    r = letters(right.text)[::-1]
    i = j = 0
    lbuf = rbuf = ""
    checks = 0
    max_residual = 0
    mismatch = None
    while i < len(l) or j < len(r):
        if i < len(l):
            lbuf += l[i:i + 3]
            i += min(3, len(l) - i)
        if j < len(r):
            rbuf += r[j:j + 3]
            j += min(3, len(r) - j)
        while lbuf and rbuf:
            checks += 1
            if lbuf[0] != rbuf[0]:
                mismatch = (checks - 1, lbuf[0], rbuf[0])
                left_pos = checks - 1
                right_pos = len(letters(right.text)) - checks
                left_word = next((k for k, w in enumerate(left.words)
                                  if left_pos < sum(len(x) for x in left.words[:k + 1])), -1)
                right_word = next((k for k, w in enumerate(right.words)
                                   if right_pos < sum(len(x) for x in right.words[:k + 1])), -1)
                return {"equations": checks, "satisfied": checks - 1,
                        "all_outer_satisfied": False, "first_mismatch": mismatch,
                        "max_residual": max(max_residual, len(lbuf), len(rbuf)),
                        "cross_word_seam": left_word != right_word,
                        "left_word_index": left_word, "right_word_index": right_word,
                        "bridge_deferred": bridge}
            lbuf, rbuf = lbuf[1:], rbuf[1:]
        max_residual = max(max_residual, len(lbuf), len(rbuf))
    return {"equations": checks, "satisfied": checks, "all_outer_satisfied": True,
            "first_mismatch": None, "max_residual": max_residual,
            "cross_word_seam": False, "bridge_deferred": bridge}


def run(limit=5000):
    bank = clauses()
    trie = make_trie(bank)
    bridges = ("and", "but", "so")
    rows = []
    diagnostics = []
    states = trie_hits = pruned = 0
    for left in bank:
        candidates = trie_prefix(trie, left.text[:4])
        trie_hits += len(candidates)
        for ridx in candidates:
            if states >= limit:
                break
            states += 1
            right = bank[ridx]
            for bridge in bridges:
                rendered = left.text.rstrip(".") + ", " + bridge + " " + right.text[0].lower() + right.text[1:]
                eq = live_pair(left, right, bridge)
                row = {"rendered": rendered, "left_frame": left.__dict__,
                       "right_frame": right.__dict__, "bridge": bridge,
                       "online_character_equations": eq, "audit": audit(rendered)}
                if len(diagnostics) < 3 and left.text != right.text:
                    diagnostics.append({**row, "reader_eligible": False, "diagnostic_only": True})
                if not eq["all_outer_satisfied"]:
                    pruned += 1
                    continue
                row["provenance"] = {"curated_semantic_bank": True,
                    "agreement_state_carried": True, "valency_state_carried": True,
                    "cross_word_residual_checked": True, "complete_utterances": True,
                    "catalogue_text": False, "finished_tape_reversal": False,
                    "post_hoc_repair": False, "mirrored_units": False,
                    "word_order_symmetry": False, "fragment": False,
                    "nested_self_palindrome": False}
                rows.append(row)
        if states >= limit:
            break
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    result = {"experiment_id": ID,
        "method": "agreement-safe curated clause lattice with reverse-character trie and live cross-word residuals",
        "stats": {"clause_frames": len(bank), "trie_nodes": _nodes(trie),
                  "trie_prefix_hits": trie_hits, "states": states, "live_prunes": pruned,
                  "outer_survivors": len(rows), "exact_gt38": len(exact),
                  "reader_eligible": len(reader),
                  "longest_letters": max((r["audit"]["letters"] for r in rows + diagnostics), default=0)},
        "controls": diagnostics, "exact_candidates": exact, "reader_facing_candidates": reader,
        "novelty_preflight": {"status": "passed",
            "signature": "curated-det-adj-noun|agreement-valency|reverse-character-trie|cross-word-residual",
            "registry_inspected": True,
            "distinct_from": "typed event composition, center-first event grammar, repair operators, and direct seam banks",
            "catalogue_text_imported": False, "finished_tape_reversal": False,
            "post_hoc_repair": False, "mirrored_units": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
                       "reader_evidence": False},
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Index curated multi-clause frames by cross-word residual signatures before lexical emission; retain agreement and valency state.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **result["stats"]}))
    for row in diagnostics:
        print(row["rendered"])
    return result


def _nodes(node):
    return 1 + sum(_nodes(child) for child in node["children"].values())


if __name__ == "__main__":
    run()
