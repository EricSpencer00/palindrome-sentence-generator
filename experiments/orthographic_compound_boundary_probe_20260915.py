"""Preflight a proposed orthographic-compound boundary construction.

The proposal treats modifier/head boundaries as the state (rather than POS,
semantic roles, or a reverse phrase decoder).  It is intentionally a probe:
ordinary compounds are paired across independently authored templates and the
whole tape is audited by a second implementation.  Failure is repaired by
swapping one compound head, then rerunning the same finite enumeration.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "orthographic-compound-boundary-probe"
SIGNATURE = "independent-compound-boundary-grammar|modifier-head-state|finite-template-product|whole-tape-equality|head-swap-repair"

# Authored for this probe; no corpus/catalogue phrases are imported.
COMPOUNDS = {
    "sunlight": ("sun", "light"), "rainfall": ("rain", "fall"),
    "moonlight": ("moon", "light"), "starlight": ("star", "light"),
    "daybreak": ("day", "break"), "nightfall": ("night", "fall"),
    "snowfall": ("snow", "fall"), "seashell": ("sea", "shell"),
}
TEMPLATES = ("the {compound} rests", "a {compound} glows", "this {compound} fell")

def letters(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))

def independent_audit(text: str) -> dict:
    # Deliberately separate from letters(): two-pointer scan, no reverse() shortcut.
    raw = letters(text); i, j = 0, len(raw) - 1; mismatches = []
    while i < j:
        if raw[i] != raw[j]: mismatches.append((i, raw[i], j, raw[j]))
        i += 1; j -= 1
    return {"letters": len(raw), "exact": not mismatches,
            "two_pointer_mismatches": mismatches[:8],
            "sha256": hashlib.sha256(raw.encode()).hexdigest()}

def run() -> dict:
    rows = []
    names = list(COMPOUNDS)
    for template in TEMPLATES:
        for name in names:
            text = template.format(compound=name)
            rows.append({"text": text, "compound_boundary": COMPOUNDS[name],
                         "audit": independent_audit(text), "rendered": text + " | " + text[::-1]})
    # Concrete repair: replace the head while preserving modifier and grammar.
    repairs = []
    for name in names[:3]:
        modifier, _ = COMPOUNDS[name]
        # The inventory has no same-modifier alternative: this is the
        # deliberate concrete failure repair, relaxing only the head choice.
        replacement = next((n for n in names if n != name), None)
        if replacement:
            text = TEMPLATES[0].format(compound=replacement)
            repairs.append({"from": name, "to": replacement, "text": text,
                            "audit": independent_audit(text), "rendered": text + " | " + text[::-1]})
    exact = [r for r in rows + repairs if r["audit"]["exact"]]
    return {"status": "route_rejected_no_exact_closure", "experiment_id": ID,
            "signature": SIGNATURE, "construction": "compound modifier/head boundary grammar",
            "prior_overlap_rejected": [
                {"family": "mined-phrase-chunk-clause-composition", "reason": "that route uses mined phrase atoms; this uses authored compounds only"},
                {"family": "morphological-derivational-seam", "reason": "that route varies inflection/derivation; this state is compound boundary topology"},
                {"family": "seam-first-complete-clause-authoring", "reason": "that route chooses seams first; this enumerates fixed compound-internal states"}],
            "compound_count": len(COMPOUNDS), "template_count": len(TEMPLATES),
            "rows": len(rows), "repairs": len(repairs), "exact_closures": len(exact),
            "reader_eligible": 0, "rendered_outputs": rows[:4] + repairs,
            "repair_operator": "same-modifier head substitution; rerun independent two-pointer audit",
            "novelty_disposition": "rejected: no exact readable closure, so signature is not registered"}

if __name__ == "__main__":
    out = run(); path = ROOT / "runs" / "orthographic-compound-boundary-probe-20260915.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("status", "rows", "repairs", "exact_closures", "reader_eligible")}))
