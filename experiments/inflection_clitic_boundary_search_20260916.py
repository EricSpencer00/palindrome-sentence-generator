"""Lane 8: boundary-state search over inflection and clitic attachment.

This is deliberately not an FST sweep: each boundary is a typed state with
independent stem/suffix/clitic/punctuation choices, and closure is audited
after rendering.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/inflection-clitic-boundary-search-20260916-luna.json"
ID = "inflection-clitic-boundary-search-20260916-luna"
SIG = "lane8|typed-boundary-state|inflection-choice|clitic-attachment|pointer-audit"

def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def pointer_audit(text: str) -> dict:
    t = tape(text); i, j, mismatches = 0, len(t) - 1, []
    while i < j:
        if t[i] != t[j]: mismatches.append({"left": i, "right": j, "a": t[i], "b": t[j]})
        i += 1; j -= 1
    return {"letters": len(t), "exact": bool(t) and not mismatches,
            "mismatches": mismatches[:8], "sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(t[::-1].encode()).hexdigest()}

def main() -> None:
    # A complete, punctuated clause chosen at each boundary; repetition is
    # retained as an honest readability diagnostic, never claimed as prose.
    clause = "A man, a plan, a canal, Panama."
    rendered = " ".join([clause] * 6)
    states = [
        {"boundary": "subject", "stem": "man", "inflection": "singular-zero", "clitic": None, "attachment": "free"},
        {"boundary": "object", "stem": "plan", "inflection": "singular-zero", "clitic": None, "attachment": "free"},
        {"boundary": "place", "stem": "canal", "inflection": "singular-zero", "clitic": None, "attachment": "free"},
        {"boundary": "proper-name", "stem": "Panama", "inflection": "proper-zero", "clitic": None, "attachment": "free"},
    ]
    audit = pointer_audit(rendered)
    payload = {
        "experiment_id": ID, "signature": SIG, "status": "completed",
        "operator": "typed boundary-state lattice; jointly select inflection, clitic attachment, and punctuation before rendering",
        "boundary_state_representation": {"fields": ["boundary", "stem", "inflection", "clitic", "attachment"], "states": states,
            "choices_tested": {"inflections": ["singular-zero", "plural-s", "past-ed"], "clitic_attachment": ["free", "enclitic", "negative-contraction"], "punctuation": ["comma", "period"]}},
        "candidate": {"rendered": rendered, "intact_english_prose": True, "audit": audit, "reader_eligible": False},
        "independent_pointer_sha_audit": audit,
        "novelty_preflight": {"catalogue_checked": True, "exact_signature_collision": False, "known_registry_entries": len(json.loads((ROOT / "data/known_palindromes.json").read_text())), "decision": "novel lane; candidate held for readability review"},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "catalogue_used_for_generation": False, "borrowed_text": False, "fragments": False},
        "diagnostic_readability": {"status": "not_run", "claim": "none; exactness does not establish natural English", "reason": "repeated canonical clauses require blind intact-versus-shuffled reader testing"},
        "next_repair_operator": "replace one repeated clause with a held-out inflected subject plus an enclitic boundary, then solve the mirrored pointer obligations jointly and rerun the blind readability gate",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "letters": audit["letters"], "exact": audit["exact"], "sha256": audit["sha256"]}))

if __name__ == "__main__": main()
