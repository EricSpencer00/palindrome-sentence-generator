"""Agreement-carrying repair for the three-region discourse frame."""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "three-region-agreement-repair-20260917"
SETUPS = ["At dawn", "Before rain"]
REFERENTS = [
    ("Mara", "she", "marks"),
    ("the keeper", "they", "record"),
    ("the keepers", "they", "record"),
    ("a sailor", "they", "carry"),
]
RELATIVES = ["who marked the harbor map", "who carried the cedar key"]
RESPONSES = ["records the quiet route", "remembers the safe path"]

def letters(text):
    return re.sub(r"[^a-z]", "", text.lower())

def audit(text):
    tape = letters(text)
    mismatches = sum(a != b for a, b in zip(tape, tape[::-1]))
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and mismatches == 0,
            "mismatches": mismatches,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def run():
    rows = []
    for si, setup in enumerate(SETUPS):
        for ri, (noun, pronoun, agreement_verb) in enumerate(REFERENTS):
            for ci, relative in enumerate(RELATIVES):
                for pi, response in enumerate(RESPONSES):
                    rendered = f"{setup}, {noun}, {relative}; {pronoun} {response}."
                    rows.append({"candidate_id": f"s{si}-r{ri}-c{ci}-p{pi}",
                        "rendered": rendered,
                        "regions": ["discourse_setup", "shared_referent_relative", "anaphoric_response"],
                        "mutable_spans": ["referent_number", "anaphor", "response"],
                        "agreement": {"referent": noun, "anaphor": pronoun, "typed_verb": agreement_verb},
                        "audit": audit(rendered),
                        "provenance": {"construction": "agreement_carrying_three_region_repair",
                            "catalogue_used": False, "wrapped_seed": False,
                            "finished_tape_reversal": False, "repeated_self_palindromic_unit": False}})
    best = min(rows, key=lambda row: row["audit"]["mismatches"])
    return {"experiment": EXPERIMENT,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rendered_candidates": rows,
        "stats": {"rendered": len(rows), "exact": sum(x["audit"]["two_pointer_exact"] for x in rows),
                   "longest_letters": max(x["audit"]["letters"] for x in rows),
                   "best_mismatches": best["audit"]["mismatches"]},
        "novelty_preflight": {"new_geometry": "typed number/agreement repair across referent and anaphoric response",
                              "prior_lane_reused": False, "duplicate_sweep": False},
        "next_repair": {"operator": "move the mutable boundary across the relative-clause seam and jointly choose agreement",
                         "reason": "agreement repair preserves prose but does not close character debt"},
        "provenance": {"bounded_states": len(rows), "human_readability_certified": False}}

if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"]))
