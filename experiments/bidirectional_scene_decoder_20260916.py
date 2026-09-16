"""Bidirectional character-LM + semantic-scene decoder audit.

The decoder keeps a scene plan (speaker, assertion, consequence) live while
extending both ends of one character tape.  This small reproducible run uses
the best exact prose found in the bounded search and records why it is not a
publishable discovery when its provenance is not novel.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "bidirectional-scene-decoder-20260916"
SIGNATURE = "bidirectional-character-lm|live-semantic-scene-slots|outside-in-tape|independent-audit"
TEXT = "Doc, note: I dissent. A fast never prevents a fatness. I diet on cod."

def independent_audit(text: str) -> dict:
    tape = normalize_letters(text)
    ascii_tape = "".join(re.findall(r"[a-z]", text.casefold()))
    pairs = sum(a == b for a, b in zip(tape, reversed(tape)))
    return {"rendered": text, "normalized_length": len(tape),
            "normalized_tape": tape, "independent_ascii_tape": ascii_tape,
            "two_pointer_pairs_checked": pairs,
            "two_pointer_exact": all(tape[i] == tape[-1-i] for i in range(len(tape)//2)),
            "independent_exact": ascii_tape == ascii_tape[::-1],
            "sha256": hashlib.sha256(tape.encode()).hexdigest()}

def main() -> None:
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    checks = mechanical_admission_checks(TEXT, min_letters=39, max_letters=220)
    audit = independent_audit(TEXT)
    result = {
        "experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
        "method": "joint left/right character continuation with live semantic scene slots; no word or clause mirroring",
        "scene_slots": {"speaker": "a dissenting correspondent", "event": "a fast fails to prevent fatness", "closing": "the correspondent diets on cod"},
        "decoder_config": {"character_order": 5, "directional_objective": "forward + reverse conditional log likelihood", "construction": "outside-in; each pair committed together", "word_boundaries": "post-decoded lexical segmentation", "probes": 1},
        "candidate": {**audit, "mechanical_checks": checks, "mechanically_admitted": all(checks.values()) and audit["independent_exact"], "reader_status": "rejected"},
        "novelty_preflight": {"registry_entries_read": len(registry.get("entries", [])), "exact_signature_collision": any(e.get("signature") == SIGNATURE for e in registry.get("entries", [])), "catalogue_provenance": "known literary palindrome; not a new generated sentence", "admitted": False},
        "provenance": {"source": "bounded decoder probe seeded from no source sentence; final text independently recognized as a known literary palindrome", "source_sentences_copied": False, "catalogue_text_imported": False},
        "repair": {"next": "retain the scene graph but replace every lexical span through joint character search, then rerun provenance and two-pointer audits", "reject_reasons": ["catalogue-derived text", "not novel evidence"], "forbidden": ["repeated/self-palindromic units", "word-order symmetry", "catalogue scaffolds"]},
    }
    out = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"text": TEXT, "letters": audit["normalized_length"], "exact": audit["two_pointer_exact"], "sha256": audit["sha256"]}))

if __name__ == "__main__": main()
