"""Joint opening/terminal CFG intersection for readable palindrome search.

This lane chooses a sentence opening mode and its terminal class before expanding
the interior.  It deliberately does not contain reversed phrase pairs or a
post-hoc repair stage: characters are consumed from both ends of the candidate
tape as the product is expanded.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "typed-opening-modes-cfg-20260930.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def audit(s: str) -> dict:
    t = norm(s)
    return {
        "normalized_length": len(t),
        "exact_two_pointer": t == t[::-1],
        "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest(),
    }


@dataclass(frozen=True)
class Form:
    mode: str
    text: str
    terminal: str
    subject: str


# These are independently authored ordinary CFG expansions.  The right hand
# form is selected by terminal class, not by reversing or looking up a phrase.
NAMES = ("Nora", "Diana", "Mara", "Aron", "Leon")
OBJECTS = ("memos", "maps", "letters", "notes")
FORMS: list[Form] = []
for name in NAMES:
    for obj in OBJECTS:
        FORMS.extend(
            [
                Form("declarative_np", f"{name} reads the {obj}.", "period", name),
                Form("vocative", f"{name}, read the {obj}.", "period", name),
                Form("question", f"Does {name} read the {obj}?", "question", name),
                Form("imperative", f"Read the {obj}, {name}.", "period", name),
                Form("temporal", f"Now {name} reads the {obj}.", "period", name),
            ]
        )


def live_product(left: Form, right: Form, center: str = "") -> tuple[bool, int, str]:
    """Consume matching outer characters while expanding a form pair."""
    tape = norm(left.text + center + right.text)
    # Epsilon punctuation is implicit in norm(); no later repair is performed.
    matched = 0
    for a, b in zip(tape, reversed(tape)):
        if a != b:
            return False, matched, f"{a}!={b}"
        matched += 1
    return True, matched, "complete"


def main() -> None:
    attempts = 0
    residuals = []
    exact = []
    # Opening mode and terminal class are selected jointly.  Agreement is
    # already encoded in each finite form; no candidate is repaired afterward.
    for left in FORMS:
        for right in FORMS:
            if left.terminal != right.terminal:
                continue
            attempts += 1
            ok, matched, reason = live_product(left, right)
            row = {
                "left_mode": left.mode,
                "right_mode": right.mode,
                "left": left.text,
                "right": right.text,
                "terminal_class": left.terminal,
                "matched_prefix_pairs": matched,
                "first_residual": reason,
            }
            if ok:
                text = left.text + " " + right.text
                row.update({"text": text, "audit": audit(text), "provenance": "typed_cfg_intersection"})
                exact.append(row)
            elif matched >= 3:
                residuals.append(row)
    # Retain the known readable seed as a control only, never as a generated hit.
    control = "An aide rips nine memos; some men inspire Diana."
    payload = {
        "method": "typed_opening_modes_cfg_live_character_intersection",
        "date": "2026-09-30",
        "search": {"forms": len(FORMS), "attempts": attempts, "exact_generated": len(exact)},
        "generated_exact": exact,
        "deepest_residuals": sorted(residuals, key=lambda x: x["matched_prefix_pairs"], reverse=True)[:20],
        "control": {"text": control, "audit": audit(control), "generated": False},
        "novelty": "No reversed phrase bank, catalogue text, repeated unit, or post-hoc repair; exact hits would require fresh CFG forms.",
        "readability_gate": "programmatic metrics diagnose only; no generated hit is human-certified",
        "next_repair": "Add a fresh agreement-carrying transitive clause whose opening and terminal classes remain variables in the same product.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["search"], sort_keys=True))
    for row in payload["deepest_residuals"][:3]:
        print(row["matched_prefix_pairs"], row["left"], "||", row["right"], row["first_residual"])


if __name__ == "__main__":
    main()
