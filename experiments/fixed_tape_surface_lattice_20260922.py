"""Punctuation-only reader-surface audit for the fixed 42/44-letter tapes.

This is a closed diagnostic inventory, not a generator or search procedure.
The word tokens are frozen; punctuation and sentence-initial case are the only
surface changes.  No row is a readability certification.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.freshness_register_reader_packet_20261002 import audit, tape, words


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "fixed-tape-surface-lattice-20260922.json"
EXPERIMENT_ID = "fixed-tape-surface-lattice-20260922"

TARGETS = (
    {
        "source_id": "target-42",
        "current_rendering": "No trace. Note: Spot spoons; snoop; stop. Set one carton.",
        "sha256": "1013004f658bdefeaaf7dea69c6d90a5d2c53381dbb5d7290ab7b25a8e5de1c3",
        "variants": (
            (
                "42-current", 1,
                "No trace. Note: Spot spoons; snoop; stop. Set one carton.",
                "[NP no trace] (minor-sentence fragment); [imperative note: [imperative spot [NP spoons]]; [imperative snoop]; [imperative stop]]; [imperative set [NP one carton]].",
                "The determiner phrase 'No trace' has no finite predicate. Treating it as a complete utterance is exactly the fragment reading excluded here.",
                "fragmentary",
            ),
            (
                "42-command-list", 2,
                "No trace; note: spot spoons; snoop; stop; set one carton.",
                "[NP no trace] coordinated with four imperative VPs: note, spot spoons, snoop, stop, and set one carton.",
                "A semicolon cannot coordinate the bare NP 'No trace' as a clause; supplying 'leave' or 'make' would add lexical material.",
                "fragmentary",
            ),
            (
                "42-question-opening", 3,
                "No trace? Note: spot spoons; snoop; stop; set one carton.",
                "[interrogative NP no trace]? followed by an imperative list.",
                "'No trace?' is an elliptical question (for example, 'Is there no trace?'), not a finite interrogative clause.",
                "ellipsis",
            ),
            (
                "42-heading-opening", 4,
                "No trace: note spot spoons; snoop; stop; set one carton.",
                "[heading NP no trace]: [imperative note [NP spot spoons]]; [imperative snoop]; [imperative stop]; [imperative set [NP one carton]].",
                "The colon turns 'No trace' into a heading, not a clause; 'spot spoons' also requires a coerced compound-object reading.",
                "fragmentary",
            ),
            (
                "42-attached-opening", 5,
                "No trace note spot. Spoons snoop; stop; set one carton.",
                "Attempted [clause [NP no trace note] [VP spot]]; [clause spoons snoop]; then two imperatives.",
                "The singular head 'note' requires 'spots', not 'spot'; reading 'spot' as a noun leaves 'No trace note spot' as an NP fragment.",
                "agreement_failure",
            ),
            (
                "42-no-particle", 6,
                "No, trace note spot; spoons snoop; stop; set one carton.",
                "[answer particle no], [imperative trace [NP note spot]]; [clause spoons snoop]; [imperative stop]; [imperative set [NP one carton]].",
                "The standalone answer particle depends on an unexpressed question, so this rendering still relies on ellipsis.",
                "ellipsis",
            ),
        ),
        "global_obstruction": (
            "With the tokens frozen, a boundary after 'trace' strands 'No trace' as a minor sentence, heading, or elliptical question. "
            "Without that boundary, 'note' cannot be the finite predicate of singular 'No trace' (it would be 'notes'); later finite-verb assignments leave agreement or complement debris."
        ),
    },
    {
        "source_id": "target-44",
        "current_rendering": "No trace. Note sleet. Spoons snoop. Steel? Set one carton.",
        "sha256": "96462b7bc06958668e9d13d13ebd0b63e4f682a51939c5bc34d7aa230d39b104",
        "variants": (
            (
                "44-current", 1,
                "No trace. Note sleet. Spoons snoop. Steel? Set one carton.",
                "[NP no trace] (minor-sentence fragment); [imperative note [NP sleet]]; [clause [NP spoons] [VP snoop]]; [interrogative NP steel]?; [imperative set [NP one carton]].",
                "Both 'No trace.' and 'Steel?' lack finite predicates; the latter expands only by silently supplying words such as 'Is it'.",
                "fragmentary",
            ),
            (
                "44-command-list", 2,
                "No trace; note sleet; spoons snoop; steel; set one carton.",
                "[NP no trace]; [imperative note sleet]; [clause spoons snoop]; [NP steel]; [imperative set one carton].",
                "The first and fourth units are bare NPs, not clauses; semicolons do not supply predicates.",
                "fragmentary",
            ),
            (
                "44-object-attachment", 3,
                "No trace: note sleet; spoons snoop steel; set one carton.",
                "[heading NP no trace]: [imperative note sleet]; attempted [clause spoons [VP snoop [NP steel]]]; [imperative set one carton].",
                "The opening remains a heading fragment, and ordinary 'snoop' does not license the direct object 'steel' without a preposition.",
                "valency_failure",
            ),
            (
                "44-question-opening", 4,
                "No trace? Note sleet; spoons snoop. Steel: set one carton.",
                "[interrogative NP no trace]?; [imperative note sleet]; [clause spoons snoop]; [heading NP steel]: [imperative set one carton].",
                "The first unit is an elliptical question and 'Steel:' is a noun heading, not an argument of the following imperative.",
                "ellipsis",
            ),
            (
                "44-attached-opening", 5,
                "No trace note sleet. Spoons snoop steel; set one carton.",
                "Attempted [clause [NP no trace note] [VP sleet]]; attempted [clause spoons [VP snoop steel]]; [imperative set one carton].",
                "Singular 'note' requires 'sleets'; independently, 'snoop' does not take bare 'steel' as a direct object.",
                "agreement_and_valency_failure",
            ),
            (
                "44-vocative-steel", 6,
                "No trace; note sleet spoons snoop. Steel, set one carton.",
                "[NP no trace]; [imperative note [clause sleet spoons snoop]]; [vocative Steel], [imperative set one carton].",
                "The opening is fragmentary, 'sleet spoons' is not an ordinary subject, and capitalizing 'Steel' as a vocative creates the prohibited proper-name rescue.",
                "proper_name_and_fragment",
            ),
        ),
        "global_obstruction": (
            "The opening obstruction is the same as for the 42-letter tape. The later material adds a second invariant: separating 'steel' makes it a noun fragment/elliptical question, while attaching it after 'snoop' violates that verb's ordinary valency."
        ),
    },
)


def build_artifact() -> dict:
    targets = []
    for target in TARGETS:
        current_words = words(target["current_rendering"])
        variants = []
        for variant_id, rank, rendering, parse, obstruction, status in target["variants"]:
            row_audit = audit(rendering)
            variants.append(
                {
                    "variant_id": variant_id,
                    "diagnostic_rank": rank,
                    "rendering": rendering,
                    "ordinary_english_parse_attempt": parse,
                    "parse_status": status,
                    "exact_parse_obstruction": obstruction,
                    "same_frozen_word_tokens": words(rendering) == current_words,
                    "same_normalized_tape": tape(rendering) == tape(target["current_rendering"]),
                    "audit": row_audit,
                    "reader_certified": False,
                    "eligible_for_blinded_packet": False,
                }
            )
        targets.append(
            {
                "source_id": target["source_id"],
                "current_rendering": target["current_rendering"],
                "frozen_word_tokens": current_words,
                "normalized_tape": tape(target["current_rendering"]),
                "expected_sha256": target["sha256"],
                "variants": variants,
                "global_parse_obstruction": target["global_obstruction"],
                "materially_clearer_nonfragmentary_variant": None,
            }
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "scope": "closed punctuation/capitalization audit of exactly two fixed tapes; no generation, lexical search, or resegmentation",
        "surface_policy": {
            "allowed": "punctuation between frozen tokens and sentence-initial capitalization",
            "disallowed": [
                "proper-name rescue",
                "quotation or borrowed-text reading",
                "fragment or ellipsis",
                "punctuation that hides a forbidden span",
                "implied lexical material",
                "hyphenation that invents a compound lexeme",
            ],
        },
        "ranking_policy": "diagnostic closeness to a complete parse only; never a readability score or certification",
        "targets": targets,
        "decision": {
            "preserve_new_rendering": False,
            "emit_blinded_variant_packet": False,
            "reason": "No enumerated rendering is both nonfragmentary and ordinary English under the frozen-token policy.",
        },
    }


def main() -> None:
    artifact = build_artifact()
    OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(OUT), "decision": artifact["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
