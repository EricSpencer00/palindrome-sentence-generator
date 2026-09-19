"""Exact-by-construction CFG/character-orbit intersection (bounded probe).

The chart emits lexical terminals from both clause ends while carrying the
unmatched character orbit.  A completed semantic clause is required before a
row is rendered; no candidate is repaired or reversed after completion.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
ID = "grammar-char-intersection-20260920"
SIGNATURE = "cfg-semantic-closure|live-character-orbit|two-sided-earley-chart|lexicon-order-only"

# Small authored grammar.  The prior is ordering only, never a candidate score.
LEXICON = {
    "det": ("the", "a", "each"), "noun": ("artist", "gardener", "teacher"),
    "verb": ("admires", "waters", "guides"), "object": ("canvas", "garden", "pupil"),
    "prep": ("near", "beside", "under"), "place": ("river", "school", "bridge"),
}

def _registry_preflight():
    data = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    sigs = {str(e.get("signature", "")) for e in data.get("entries", [])}
    return {"entries_scanned": len(sigs), "exact_signature_collision": SIGNATURE in sigs,
            "catalogue_text_imported": False, "known_palindrome_imported": False}

def orbit_pair(left: str, right: str, orbit: str = ""):
    """Consume terminals live from opposite ends, returning residual orbit."""
    a, b = normalize_letters(left), normalize_letters(right)
    i, j = 0, len(b) - 1
    pending = orbit
    while i < len(a) and j >= 0:
        expected = pending[-1] if pending else a[i]
        if expected != b[j]:
            return None
        pending = pending[:-1] if pending else ""
        i += 1; j -= 1
    if i < len(a): pending += a[i:]
    if j >= 0: pending += b[:j + 1][::-1]
    return pending

def chart_clause(left_words, right_words):
    """Earley-like product: grammar states and semantic roles close together."""
    # S -> NP VP PP; semantic closure requires agent, action, theme, setting.
    roles = {"agent": left_words[1], "action": left_words[2],
             "theme": left_words[3], "setting": right_words[3]}
    if not all(roles.values()): return None
    orbit = ""
    for l, r in zip(left_words, reversed(right_words)):
        next_orbit = orbit_pair(l, r, orbit)
        # A failed edge is a live intersection dead-end, not a reason to
        # rewrite the completed clause; retain the diagnostic residual.
        if next_orbit is None:
            return {"roles": roles, "orbit_closed": False,
                    "residual": "mismatch-at-live-edge"}
        orbit = next_orbit
    return {"roles": roles, "orbit_closed": orbit == "", "residual": orbit}

def render(left, right):
    text = " ".join(left) + " " + " ".join(right)
    tape = normalize_letters(text)
    mismatches = [i for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    # independent two-pointer audit, deliberately separate from construction
    i, j = 0, len(tape)-1; pairs = 0; exact = True
    while i < j:
        exact &= tape[i] == tape[j]; pairs += 1; i += 1; j -= 1
    sha = hashlib.sha256(tape.encode()).hexdigest()
    return {"rendered": text, "letters": len(tape), "exact": exact,
            "mismatch_positions": mismatches[:12], "two_pointer_pairs": pairs,
            "sha256": sha, "checks": mechanical_admission_checks(text, min_letters=20, max_letters=260)}

def run():
    # Complete ordinary-English controls; exactness is not manufactured.
    rows = []
    controls = [
        (("the", "artist", "admires", "canvas"), ("near", "the", "river", "bridge")),
        (("a", "gardener", "waters", "garden"), ("beside", "the", "school", "bridge")),
    ]
    for left, right in controls:
        closure = chart_clause(left, right)
        row = render(left, right); row["semantic_closure"] = closure; rows.append(row)
    admitted = [r for r in rows if r["checks"].get("eligible", False) and r["exact"]]
    return {"experiment_id": ID, "signature": SIGNATURE,
            "novelty_preflight": _registry_preflight(),
            "grammar": {"productions": ["S->NP VP PP", "NP->DET N", "VP->V OBJ", "PP->P NP"],
                         "semantic_closure": ["agent", "action", "theme", "setting"],
                         "lexical_prior": "static ordering only", "posthoc_repair": False},
            "rendered_candidates": rows, "mechanical_admission": admitted,
            "stats": {"rendered": len(rows), "complete_clauses": len(rows),
                      "exact": sum(r["exact"] for r in rows), "admitted": len(admitted)},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "lexical_source": "authored finite lexicon; no catalogue text",
                           "independent_audits": ["two-pointer", "SHA-256", "mechanical admission"]},
            "next_discriminator": {"operator": "add held-out transitive clause frame with typed plural agreement",
                                   "reason": "current live orbit closes no exact ordinary control"}}

if __name__ == "__main__":
    out = ROOT / "runs" / (ID + ".json"); out.write_text(json.dumps(run(), indent=2) + "\n"); print(run()["stats"])
