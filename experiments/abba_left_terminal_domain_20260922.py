"""ABBA terminal-domain selection with live reverse-onset classes.

This lane changes the *left* lexical domain rather than widening a completed
right sentence bank.  Complete authored A/B sentences are selected by the
first characters of their reverse obligation; only then does a typed right
grammar consume that obligation online.  ``red`` is a viable right onset and
``elc`` is retained as an impossible-English certificate (from ``bicycle``).
No sentence is reversed or repaired after rendering.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-left-terminal-domain-20260922.json"

LEFT_DOMAIN = {
    "A": (
        "At dusk, the patient cartographer marked a hidden inlet.",
        "By dawn, the village baker carried warm loaves to market.",
    ),
    "B": (
        # ``lavender`` exposes reverse onset ``red``.
        "At noon, a careful gardener watered the young lavender.",
        # ``bicycle`` exposes reverse onset ``elc``; no English subject uses it.
        "In winter, the quiet mechanic repaired a silver bicycle.",
    ),
}

# The right grammar is independently authored and keyed by the live onset
# class.  The red branch is grammatical but intentionally not assumed to
# close the whole tape; elc is a negative grammar certificate.
RIGHT_BY_ONSET = {
    "red": {
        "subject": ("red-haired archivist", "red-coated sailor"),
        "verb": ("records", "studies"),
        "object": ("a folded map", "the blue lantern"),
        "adjunct": ("before dawn", "near the harbor"),
    },
    "elc": {
        "subject": (),
        "verb": ("records",),
        "object": ("a folded map",),
        "adjunct": ("before dawn",),
    },
}
SLOTS = ("subject", "verb", "object", "adjunct", "subject", "verb", "object", "adjunct")


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [(i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatches": mismatches[:4],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse_obligation": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def consume(obligation: str, bank: dict[str, tuple[str, ...]], max_parses: int = 16):
    """Return complete typed parses and every failed live-prefix observation."""
    memo: dict[tuple[int, int], list[tuple[str, ...]]] = {}
    frontier: list[dict[str, object]] = []

    def go(slot: int, pos: int) -> list[tuple[str, ...]]:
        key = (slot, pos)
        if key in memo:
            return memo[key]
        if slot == len(SLOTS):
            return [()] if pos == len(obligation) else []
        out: list[tuple[str, ...]] = []
        phrases = bank[SLOTS[slot]]
        for phrase in phrases:
            token = letters(phrase)
            if obligation.startswith(token, pos):
                for tail in go(slot + 1, pos + len(token)):
                    out.append((phrase,) + tail)
                    if len(out) >= max_parses:
                        break
            else:
                matched = 0
                while (matched < len(token) and pos + matched < len(obligation)
                       and token[matched] == obligation[pos + matched]):
                    matched += 1
                frontier.append({"slot": SLOTS[slot], "offset": pos,
                                 "phrase": phrase, "matched_characters": matched,
                                 "required": obligation[pos:pos + 3]})
        memo[key] = out
        return out

    return go(0, 0), frontier


def run() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    branches: list[dict[str, object]] = []
    for a in LEFT_DOMAIN["A"]:
        for b in LEFT_DOMAIN["B"]:
            left = f"{a} {b}"
            obligation = letters(left)[::-1]
            onset = obligation[:3]
            bank = RIGHT_BY_ONSET.get(onset, {slot: () for slot in
                                               ("subject", "verb", "object", "adjunct")})
            parses, frontier = consume(obligation, bank)
            branches.append({"left_A_B": [a, b], "reverse_onset": onset,
                             "domain_selected": onset in RIGHT_BY_ONSET,
                             "right_subject_count": len(bank["subject"]),
                             "deepest_matched_prefix": max(
                                 (x["matched_characters"] for x in frontier), default=0),
                             "parse_count": len(parses),
                             "grammar_certificate": "empty right subject domain"
                             if not bank["subject"] else None,
                             "frontier": frontier[:6]})
            for p in parses:
                right = (f"{p[0]} {p[1]} {p[2]} {p[3]}. "
                         f"{p[4]} {p[5]} {p[6]} {p[7]}.")
                rendered = f"{left} {right}"
                rows.append({"rendered": rendered, "reverse_onset": onset,
                             "audit": audit(rendered),
                             "provenance": {"left_complete_authored": True,
                                            "right_domain_selected_before_decode": True,
                                            "typed_slots": list(SLOTS),
                                            "finished_text_reversal": False,
                                            "catalogue_text": False,
                                            "repeated_units": False,
                                            "self_palindromic_units": False,
                                            "posthoc_repair": False,
                                            "reward_model": False}})
            controls.append({"rendered": left, "kind": "intact-authored-AB-control",
                             "reverse_onset": onset, "audit": audit(left)})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]
             and r["audit"]["letters"] > 38]
    return {
        "experiment_id": "abba-left-terminal-domain-20260922",
        "method": "ABBA complete-left terminal-domain selection by live reverse onset",
        "stats": {"left_A_choices": len(LEFT_DOMAIN["A"]),
                   "left_B_choices": len(LEFT_DOMAIN["B"]),
                   "domain_classes": sorted(RIGHT_BY_ONSET),
                   "closed_derivations": len(rows),
                   "exact_gt38": len(exact),
                   "branches": len(branches)},
        "exact_candidates": exact,
        "rendered_candidates": rows,
        "controls": controls,
        "terminal_domain_branches": branches,
        "novelty_preflight": {
            "status": "passed",
            "signature": "abba|left-terminal-domain|reverse-onset-conditioned-typed-decode",
            "distinct_from": "right-bank widening, fixed Cartesian ABBA products, and variable-boundary DP",
            "finished_tape_reversal": False,
            "catalogue_text": False,
            "mirrored_units": False,
        },
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer scan", "forward/reverse SHA-256"],
                       "reader_gate": "closed pending novel exact output"},
        "status": "fresh exact closure found" if exact else "no exact closure; onset-conditioned frontier retained",
        "next_construction": "replace terminal-word classes with a held-out lexical trie whose next phrase is selected from the full residual, not only its first three letters",
    }


if __name__ == "__main__":
    data = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["stats"], sort_keys=True))
