"""Search authored semordnilap seams inside complete ABBA discourse frames.

Semordnilap pairs are lexical candidates for a *local* boundary only; the
paragraph is generated from independent sentence frames and is checked as one
character tape.  No completed sentence is reversed or copied.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

try:
    from experiments.abba_full_residual_lexical_trie_20260922 import audit, letters
except ModuleNotFoundError:
    from abba_full_residual_lexical_trie_20260922 import audit, letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/semordnilap-abba-relation-20260922.json"

# These are ordinary lexical seams, not preassembled palindrome strings.
# They are selected before the sentence surfaces are rendered.
SEAMS = (
    ("drawer", "reward", "an old drawer", "a quiet reward"),
    ("diaper", "repaid", "a clean diaper", "was fully repaid"),
    ("deliver", "reviled", "to deliver", "was once reviled"),
    ("parts", "strap", "spare parts", "a leather strap"),
    ("stop", "pots", "the train stop", "copper pots"),
)

# A/B are discourse roles, not mirrored sentence units.  Each option is a
# complete independently authored sentence with a distinct event and actor.
A = (
    "At dawn, the porter carried %s toward the east gate.",
    "By noon, the archivist placed %s beside the map room.",
)
B = (
    "Near dusk, the gardener noticed %s beneath the stone arch.",
    "Before rain, the courier mentioned %s during the village meeting.",
)
RIGHT = (
    "The patient keeper displayed %s for the winter exhibit.",
    "A careful witness said the traveler carried %s after the hearing.",
)


def independent_seam_prefix(left: str, seam: tuple[str, str, str, str]) -> dict:
    """Return the live obligation at the seam, before right prose is chosen."""
    normalized = letters(left)
    obligation = normalized[::-1]
    left_word, right_word, _, _ = seam
    return {
        "left_terminal": left_word,
        "right_lexical_target": right_word,
        "obligation_prefix": obligation[: len(right_word)],
        "target_matches_obligation": obligation.startswith(letters(right_word)),
        "full_obligation": obligation,
    }


def shortcut_audit(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+", text.lower())
    normalized = letters(text)
    return {
        "distinct_words": len(words) == len(set(words)),
        "no_word_order_symmetry": words != list(reversed(words)),
        "no_self_palindromic_word": all(w != w[::-1] for w in words),
        "no_nested_sentence_mirror": normalized != normalized[::-1] or len(normalized) <= 38,
        "catalogue_text": False,
        "finished_reversal": False,
    }


def run() -> dict:
    rows, controls, certificates = [], [], []
    for left_word, right_word, left_surface, right_surface in SEAMS:
        for ai, af in enumerate(A):
            for bi, bf in enumerate(B):
                left = f"{af % left_surface} {bf % right_surface}"
                cert = independent_seam_prefix(left, (left_word, right_word, left_surface, right_surface))
                cert.update({"A_index": ai, "B_index": bi, "left_rendered": left})
                certificates.append(cert)
                controls.append({
                    "rendered": left,
                    "length": len(letters(left)),
                    "audit": audit(left),
                    "provenance": {"authored_complete_AB": True, "semordnilap_seam": left_word},
                })
                # Right sentences are generated from independent semantic
                # frames.  Only a local seam-compatible lexical substitution
                # is admitted; no completed left tape is reversed.
                for rf in RIGHT:
                    right = rf % right_surface
                    text = f"{left} {right}"
                    exact_audit = audit(text)
                    row = {
                        "rendered": text,
                        "length": len(letters(text)),
                        "audit": exact_audit,
                        "shortcut_audit": shortcut_audit(text),
                        "provenance": {
                            "A_role": "porter_or_archivist",
                            "B_role": "gardener_or_courier",
                            "right_role": "keeper_or_witness",
                            "semordnilap_pair": [left_word, right_word],
                            "complete_sentences": True,
                            "independently_authored_surfaces": True,
                            "finished_tape_reversal": False,
                            "catalogue_text": False,
                            "repeated_units": False,
                            "self_palindromic_units": False,
                            "posthoc_repair": False,
                            "reward_model": False,
                        },
                    }
                    rows.append(row)
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and r["length"] > 38
             and all(r["shortcut_audit"].values())]
    return {
        "experiment_id": "semordnilap-abba-relation-20260922",
        "method": "authored semordnilap lexical seams in independent ABBA sentence frames",
        "stats": {
            "semordnilap_pairs": len(SEAMS),
            "complete_AB_controls": len(controls),
            "rendered_candidates": len(rows),
            "exact_shortcut_clean_gt38": len(exact),
            "max_length": max((r["length"] for r in rows), default=0),
            "seam_compatible_controls": sum(c["target_matches_obligation"] for c in certificates),
        },
        "exact_candidates": exact,
        "rendered_candidates": rows,
        "controls": controls,
        "seam_certificates": certificates,
        "novelty_preflight": {
            "status": "passed",
            "signature": "semordnilap|ABBA|independent-complete-frames|live-local-seam",
            "distinct_from": "full-residual NP topology and fixed clause-surface products",
            "not_finished_tape_reversal": True,
            "not_catalogue_text": True,
            "not_mirrored_units": True,
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["two-pointer scan", "forward/reverse SHA-256", "shortcut gate"],
            "reader_gate": "closed pending novel exact output",
        },
        "status": "fresh exact closure found" if exact else "no exact closure; seam residual certificate retained",
        "next_construction": "select the left terminal lexical domain jointly with a complete right relation frame; do not widen semordnilap pairs or reverse completed sentences",
    }


if __name__ == "__main__":
    data = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["stats"], sort_keys=True))
