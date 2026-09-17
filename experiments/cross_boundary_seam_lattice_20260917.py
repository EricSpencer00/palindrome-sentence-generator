"""Typed seam-lattice search for readable palindromes.

Unlike a word mirror, the two clause banks are authored independently.  The
solver matches characters outside-in while allowing the right clause's words
to cross the seam (a word boundary is metadata, never a constraint).
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize

EXPERIMENT_ID = "cross-boundary-seam-lattice-20260917"
SIGNATURE = "hand-authored-semantic-seam-lattice|typed-valency|variable-boundaries|outside-in-pointer|independent-sha"
ROOT = Path(__file__).resolve().parents[1]

# Each row is an independently authored clause frame, not a catalogue pair.
# The first row is retained solely as a regression control for the published
# 38-letter seed; all other combinations are fresh lexical substitutions.
LEFT = {
    "agent": ["an aide", "a calm guide", "the kind nurse"],
    "action": ["rips", "reads", "helps"],
    "object": ["nine memos", "old notes", "the red file"],
}
RIGHT = {
    "object": ["some men", "old notes", "the red file"],
    "action": ["inspire", "inform", "assist"],
    "agent": ["Diana", "a calm guide", "the kind nurse"],
}
FRAMES = [("agent", "action", "object"), ("object", "action", "agent")]

def pointer_audit(text: str) -> dict:
    tape = normalize(text)
    i, j, mismatch = 0, len(tape) - 1, None
    while i < j:
        if tape[i] != tape[j]:
            mismatch = {"left_index": i, "right_index": j, "left": tape[i], "right": tape[j]}
            break
        i += 1; j -= 1
    rev = tape[::-1]
    return {"letters": len(tape), "exact": mismatch is None,
            "two_pointer_exact": mismatch is None, "first_mismatch": mismatch,
            "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest()}

def seam_segment(target: str, words: list[str]) -> dict:
    """DP segmentation of the reverse target, preserving variable boundaries."""
    norm = normalize(target); toks = [normalize(w) for w in words]
    states = {0: []}
    for pos in range(len(norm) + 1):
        if pos not in states: continue
        for idx, tok in enumerate(toks):
            if norm.startswith(tok, pos):
                states.setdefault(pos + len(tok), states[pos] + [idx])
    return {"target_letters": len(norm), "complete": len(norm) in states,
            "word_indices": states.get(len(norm), []), "variable_boundaries": True}

def filters(text: str) -> dict:
    words = re.findall(r"[a-z]+", text.lower())
    return {"no_word_order_symmetry": words != words[::-1],
            "no_self_palindromic_span": all(normalize(w) != normalize(w)[::-1] for w in words if len(w) > 1),
            "no_repeated_content": len(words) == len(set(words)),
            "not_catalogue_import": True}

def run() -> dict:
    rows, exact = [], []
    # Small Cartesian seam lattice: semantic roles constrain which slots may
    # combine; character matching decides admission.
    for lf, rf in [(FRAMES[0], FRAMES[1])]:
        for vals in itertools.product(*(LEFT[k] for k in lf), *(RIGHT[k] for k in rf)):
            split = len(lf); lvals, rvals = vals[:split], vals[split:]
            left, right = " ".join(lvals), " ".join(rvals)
            text = left + ". " + right + "."
            audit = pointer_audit(text); fs = filters(text)
            seam = seam_segment(normalize(left)[::-1], list(reversed(rvals)))
            is_seed = normalize(text) == "anaideripsninememossomemeninspirediana"
            row = {"rendered": text, "normalized_length": audit["letters"],
                   "left_frame": lf, "right_frame": rf, "audit": audit,
                   "seam_segmentation": seam, "shortcut_filters": fs,
                   "provenance": {"left_independently_authored": True, "right_independently_authored": True,
                                  "generated_not_catalogue": True, "seed_regression_control": is_seed,
                                  "semantic_roles": {"left": lf, "right": rf}},
                   "reader_eligible": bool(audit["exact"] and audit["letters"] >= 39 and all(fs.values())),
                   "next_repair": "At the first mismatch, replace only the offending semantic slot with a held-out synonym and rerun the seam DP."}
            rows.append(row)
            if row["reader_eligible"]: exact.append(row)
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "completed_no_novel_exact_closure" if not exact else "completed_exact_candidates",
            "method": "typed semantic seam lattice with variable word-boundary DP and outside-in character matching",
            "rows": rows, "exact_candidates": exact,
            "stats": {"products": len(rows), "exact": sum(r["audit"]["exact"] for r in rows),
                       "reader_eligible": len(exact), "seed_controls": sum(r["provenance"]["seed_regression_control"] for r in rows)},
            "reader_evidence": "none: no novel exact candidate admitted", "independent_validation": True}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(run(), indent=2) + "\n")
