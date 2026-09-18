"""Bounded three-region discourse-frame infill with a shared referent.

This lane varies three grammatical regions (setup, anchored relative clause,
and response) while keeping one named referent shared across the frame.  It
does not construct a reverse tape or reuse a finished palindrome.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "three-region-discourse-anchor-20260917"
SETUPS = ["At dawn", "Before rain"]
REFERENTS = [("Mara", "she"), ("the keeper", "the keeper")]
RELATIVES = ["who marked the harbor map", "who carried the cedar key"]
RESPONSES = ["records the quiet route", "remembers the safe path"]


def letters(text):
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text):
    tape = letters(text)
    mismatches = sum(a != b for a, b in zip(tape, tape[::-1]))
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatches == 0,
        "mismatches": mismatches,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def render(setup, referent, relative, response):
    name, pronoun = referent
    # Three intact regions: discourse setup; shared-referent relative clause;
    # anaphoric response.  The referent is semantic glue, not a repeated unit.
    return f"{setup}, {name}, {relative}; {pronoun} {response}."


def run():
    rows = []
    for si, setup in enumerate(SETUPS):
        for ri, referent in enumerate(REFERENTS):
            for ci, relative in enumerate(RELATIVES):
                for pi, response in enumerate(RESPONSES):
                    rendered = render(setup, referent, relative, response)
                    rows.append({
                        "candidate_id": f"s{si}-r{ri}-c{ci}-p{pi}",
                        "rendered": rendered,
                        "regions": ["discourse_setup", "shared_referent_relative", "anaphoric_response"],
                        "mutable_spans": ["setup", "relative_clause", "response"],
                        "shared_referent": referent[0],
                        "audit": audit(rendered),
                        "provenance": {
                            "construction": "three_region_discourse_frame",
                            "catalogue_used": False,
                            "wrapped_seed": False,
                            "finished_tape_reversal": False,
                            "repeated_self_palindromic_unit": False,
                        },
                    })
    best = min(rows, key=lambda row: row["audit"]["mismatches"])
    return {
        "experiment": EXPERIMENT,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rendered_candidates": rows,
        "stats": {
            "rendered": len(rows),
            "exact": sum(row["audit"]["two_pointer_exact"] for row in rows),
            "longest_letters": max(row["audit"]["letters"] for row in rows),
            "best_mismatches": best["audit"]["mismatches"],
        },
        "novelty_preflight": {
            "new_geometry": "three regions coupled by one shared referent",
            "prior_lane_reused": False,
            "duplicate_sweep": False,
        },
        "next_repair": {
            "operator": "jointly inflect the referent and anaphoric response while preserving the relative-clause attachment",
            "reason": "the bounded frame remains grammatical but does not close its character debt",
        },
        "provenance": {"bounded_states": len(rows), "human_readability_certified": False},
    }


if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"]))
