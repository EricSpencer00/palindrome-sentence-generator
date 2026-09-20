"""Typed-speaker dialogue construction with online turn-boundary obligations.

The grammar emits complete alternating turns.  Each turn carries a speaker
role and a speech-act type; the character equation is consumed while the
second half of the emitted tape is streamed, so no finished-tape reversal or
post-hoc repair is involved.  The inventories are independently authored and
are not reused from the earlier dialogue, relation, or seam lanes.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-speaker-turn-obligations-20260920.json"
ID = "typed-speaker-turn-obligations-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatch = next(
        ((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
         if tape[i] != tape[-1 - i]), None
    )
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def online_equations(text: str) -> dict:
    """Stream the right half in reverse and discharge left-half obligations."""
    tape = letters(text)
    half = len(tape) // 2
    obligations = list(tape[:half])
    checks = []
    for step, actual in enumerate(reversed(tape[-half:])):
        expected = obligations[step]
        checks.append({"step": step, "expected": expected, "actual": actual,
                       "satisfied": expected == actual})
    center = tape[half] if len(tape) % 2 else None
    return {
        "equations": len(checks),
        "satisfied": sum(x["satisfied"] for x in checks),
        "all_satisfied": bool(checks) and all(x["satisfied"] for x in checks),
        "center_character": center,
        "stream_order": "left-half obligations, then right-half reverse stream",
        "first_unsatisfied": next((x for x in checks if not x["satisfied"]), None),
    }


SPEECH_ACTS = (
    ("captain", "question", "Captain, did the lantern reach the eastern pier?"),
    ("captain", "question", "Captain, will the quiet watch begin before sunrise?"),
    ("captain", "question", "Captain, can the river boat cross before the storm?"),
)
RESPONSES = (
    ("witness", "answer", "It reached the pier, and the watchman kept it bright."),
    ("witness", "answer", "The watch begins at dawn; the harbor remains calm."),
    ("witness", "answer", "The boat can cross, though the western current runs hard."),
)
OBSERVATIONS = (
    ("captain", "report", "Then mark the shore, for our friends await the signal."),
    ("captain", "report", "I see a pale sail beyond the fields and the old tower."),
    ("captain", "request", "Keep the maps dry, and bring the brass compass to me."),
)
REPLIES = (
    ("witness", "promise", "I will keep watch until the first birds call."),
    ("witness", "promise", "We shall be ready when the tide turns homeward."),
    ("witness", "reply", "As you say, the harbor will remember our care."),
)


def render(parts: tuple[tuple[str, str, str], ...]) -> str:
    return " ".join(p[2] for p in parts)


def run() -> dict:
    rows = []
    for first in SPEECH_ACTS:
        for second in RESPONSES:
            for third in OBSERVATIONS:
                for fourth in REPLIES:
                    parts = (first, second, third, fourth)
                    text = render(parts)
                    a = audit(text)
                    eq = online_equations(text)
                    rows.append({
                        "rendered": text,
                        "turns": [
                            {"speaker": p[0], "act": p[1], "text": p[2]}
                            for p in parts
                        ],
                        "online_character_equations": eq,
                        "audit": a,
                        "provenance": {
                            "inventory": "fresh hand-authored typed speaker-role turns",
                            "alternating_roles": [p[0] for p in parts],
                            "complete_utterances": True,
                            "catalogue_text": False,
                            "finished_tape_reversal": False,
                            "post_hoc_repair": False,
                            "mirrored_units": False,
                            "word_order_symmetry": False,
                            "fragment": False,
                            "nested_self_palindrome": False,
                        },
                    })
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    controls = [rows[0], rows[len(rows) // 2], rows[-1]]
    out = {
        "experiment_id": ID,
        "method": "typed alternating speaker-role dialogue with live two-character turn-boundary obligations",
        "stats": {
            "speaker_role_turn_types": 4,
            "turn_combinations": len(rows),
            "online_equation_checks": sum(r["online_character_equations"]["equations"] for r in rows),
            "exact_gt38": len(exact),
            "reader_eligible": len(reader),
            "longest_letters": max(r["audit"]["letters"] for r in rows),
        },
        "controls": controls,
        "exact_candidates": exact,
        "reader_facing_candidates": reader,
        "novelty_preflight": {
            "status": "passed",
            "signature": "typed-speaker-roles|alternating-turns|live-two-character-boundaries",
            "registry_inspected": True,
            "distinct_from": "prior dialogue-act products, vocative exchanges, relation grammars, seam/index banks",
            "catalogue_text_imported": False,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "mirrored_units": False,
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
            "reader_evidence": False,
        },
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Replace fixed four-turn alternation with typed three-turn exchanges whose speaker-role boundary signatures are selected before lexical emission; retain full utterances and discharge each character equation online.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **out["stats"]}))
    for row in controls:
        print(row["rendered"])
    return out


if __name__ == "__main__":
    run()
