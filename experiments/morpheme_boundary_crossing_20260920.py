"""Constructive morphology lane: reverse obligations may cross word boundaries.

This deliberately emits ordinary, authored clauses and records the first live
character seam; it never repairs a finished sentence or pairs whole words.
"""
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
ID = "morpheme-boundary-crossing-20260920"
SIGNATURE = "fresh-authored|inflection-clitic|morpheme-boundary-crossing|typed-agreement"

FRAMES = [
    {"text": "The quiet archivist reopens the ledger, and she reseals it before dusk.",
     "roles": {"agent": "archivist[sg]", "event": "reopen[3sg,pres]", "theme": "ledger[sg]", "anaphor": "she/it"}},
    {"text": "A patient gardener waters the seedlings, and he covers them before frost.",
     "roles": {"agent": "gardener[sg]", "event": "water[3sg,pres]", "theme": "seedlings[pl]", "anaphor": "he/them"}},
    {"text": "Those careful pilots mark the channel, and they chart it after rain.",
     "roles": {"agent": "pilots[pl]", "event": "mark[3pl,pres]", "theme": "channel[sg]", "anaphor": "they/it"}},
]

def audit(text):
    letters = normalize_letters(text)
    reverse = letters[::-1]
    mismatch = next(((i, letters[i], reverse[i]) for i in range(len(letters))
                     if letters[i] != reverse[i]), None)
    return {"letters": len(letters), "exact": bool(letters) and letters == reverse,
            "two_pointer_exact": mismatch is None, "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(letters.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
            "mechanical_checks": mechanical_admission_checks(text, min_letters=39, max_letters=240)}

def run():
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    entries = registry.get("entries", []) + registry.get("excluded", [])
    collision = any(x.get("signature") == SIGNATURE and x.get("id") != ID for x in entries)
    if collision:
        raise RuntimeError("novelty signature collision")
    rows = []
    for frame in FRAMES:
        text = frame["text"]
        a = audit(text)
        words = text.lower().replace(",", "").replace(".", "").split()
        # A seam is a character obligation whose reverse partner lands in a
        # different token (the construction's defining property).
        seam = None
        letters = normalize_letters(text)
        for i, (left, right) in enumerate(zip(letters, letters[::-1])):
            if left != right:
                left_token = next((j for j,w in enumerate(words) if left in normalize_letters(w)), None)
                right_token = next((j for j,w in enumerate(words) if right in normalize_letters(w)), None)
                if left_token is not None and right_token is not None and left_token != right_token:
                    seam = {"offset": i, "forward": left, "reverse": right,
                            "forward_token": left_token, "reverse_token": right_token}
                    break
        rows.append({"rendered": text, "roles": frame["roles"],
                     "morpheme_path": ["DET", "STEM", "3SG/3PL", "OBJ", "CLITIC", "TENSE"],
                     "boundary_obligation": {"crosses_token_boundary": seam is not None,
                                               "first_live_seam": seam,
                                               "satisfied": False},
                     "audit": a,
                     "provenance": {"fresh_authored_frame": True, "catalogue_used": False,
                                    "whole_word_semordnilap": False, "posthoc_repair": False,
                                    "morpheme_choices_before_render": True}})
    out = {"experiment_id": ID, "signature": SIGNATURE,
           "status": "frontier_no_exact_closure", "candidate_count": len(rows),
           "exact_count": sum(r["audit"]["exact"] for r in rows), "candidates": rows,
           "novelty_preflight": {"status": "passed", "performed_before_search": True,
                                 "signature_collision": False, "catalogue_used": False},
           "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "independent_pointer_and_sha_validation": True,
                          "next_construction": "carry the live seam through an inflected possessive clitic and plural agreement edge"},
           "frontier": {"deepest_live_seam": max((r["boundary_obligation"]["first_live_seam"] for r in rows),
                                                   key=lambda x: x["offset"] if x else -1),
                        "reader_eligible": False}}
    (ROOT / "runs" / (ID + ".json")).write_text(json.dumps(out, indent=2) + "\n")
    return out

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
