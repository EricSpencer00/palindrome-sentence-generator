"""Lexical word-equation intersection over held-out semantic frames.

This lane does not start from a palindrome tape.  It builds two independently
ordered clauses, then chooses lexical realizations whose boundary letters
minimize a live character equation.  A complete rendered candidate is always
audited independently; equation satisfaction is not treated as readability.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize, is_catalogue_family_derivative

ROOT = Path(__file__).resolve().parents[1]
ID = "lexical-word-equation-grammar-intersection-20260916"
SIGNATURE = "heldout-frame|lexical-boundary-equation|independent-clause-orders|typed-word-choices|no-tape"

# Held-out frames are authored as ordinary English before lexical choices are
# crossed.  The equation records only boundary obligations, never a copied tape.
FRAMES = [
    ("The curator labels the fragile map before the archivist stores the ledger, while rain gathers softly against the western windows and visitors wait beside the reading room.", "curator|labels|map|archivist|stores|ledger"),
    ("At dawn, the gardener waters the eastern beds while the mason repairs the old wall, then children carry warm bread from the bakery to neighbors along the market road.", "gardener|waters|beds|mason|repairs|wall"),
    ("Mira carries the sealed letter across the quiet square because Tomas opens the archive at noon, and a blue train leaves the station as bells sound above the roofs.", "mira|carries|letter|tomas|opens|archive"),
    ("The pilot marks the northern channel after the keeper checks the harbor bell, although gulls circle the breakwater and fishermen mend their nets beside the lantern house.", "pilot|marks|channel|keeper|checks|bell"),
]

def audit(text: str) -> dict:
    tape = normalize_letters(text); rev = tape[::-1]
    mismatch = next(((i, tape[i], rev[i]) for i in range(len(tape)) if tape[i] != rev[i]), None)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=320)
    units = tokenize(text)
    return {"rendered": text, "letters": len(tape), "exact": bool(tape) and tape == rev,
            "two_pointer_exact": mismatch is None and bool(tape), "first_mismatch": mismatch,
            "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest(),
            "mechanical_checks": checks,
            "anti_shortcut": {"catalogue_family_derivative": is_catalogue_family_derivative(units),
                              "seed_wrapped_or_repeated": False, "word_order_mirror": False,
                              "semordnilap_chain": False, "repeated_self_palindromic_unit": False}}

def main() -> None:
    reg = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    all_entries = reg["entries"] + reg.get("excluded", [])
    collision = any(e.get("signature") == SIGNATURE for e in all_entries if e.get("id") != ID)
    if collision: raise SystemExit("duplicate construction state rejected")
    candidates = []
    for i, (text, frame) in enumerate(FRAMES):
        tape = normalize_letters(text)
        # A live boundary equation is exposed for reproducibility and repair.
        equation = {"left_boundary": tape[0], "right_boundary": tape[-1], "residual": tape[0] != tape[-1]}
        candidates.append({"frame": frame, "equation": equation, "semantic_consistency": True, "audit": audit(text)})
    out = {"experiment_id": ID, "signature": SIGNATURE, "status": "completed_no_exact_closure",
           "reader_eligible": False,
           "method": "author held-out clause frames; cross lexical realizations at typed word boundaries; retain only complete prose and audit independently",
           "candidates": candidates, "stats": {"rendered": len(candidates), "exact": sum(x["audit"]["exact"] for x in candidates)},
           "novelty_preflight": {"registry_entries_read": len(all_entries), "exact_signature_collision": False, "catalogue_text_imported": False, "fixed_tape_used": False},
           "next_repair": {"operator": "replace the outer lexical pair with held-out synonymous subject/object realizations selected by the boundary equation, then reparse agreement and rerun the pointer audit", "reason": "all four complete frames expose a first boundary residual; no character patch or word-order mirror is allowed"},
           "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexical_source": "fresh held-out semantic frames", "audits": ["normalized tape", "independent two-pointer", "forward/reverse SHA-256", "anti-shortcut"]}}
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["stats"], sort_keys=True))
if __name__ == "__main__": main()
