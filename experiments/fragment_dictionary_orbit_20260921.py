"""Novelty preflight for an unfinished-word-boundary fragment dictionary.

This lane is intentionally stopped before generation: the registry already
contains several materially equivalent live character-orbit / boundary-FSM /
reverse-segmentation methods.  Keeping a machine-readable preflight prevents
relabeling the same search as a new construction.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / "fragment-dictionary-orbit-20260921.json"

PROPOSED = {
    "signature": "unfinished-word-boundary-fragments|finite-state-semantic-tags|center-out-character-orbit",
    "description": "Offline authored fragments retain open-word residuals plus syntactic/semantic states while consuming mirrored characters before complete clauses materialize.",
}

def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    terms = ("fragment", "seam", "orbit", "boundary", "reverse", "transducer", "character")
    hits = []
    for e in registry.get("entries", []):
        hay = json.dumps(e, sort_keys=True).lower()
        score = sum(t in hay for t in terms)
        if score >= 3:
            hits.append({"id": e.get("id"), "signature": e.get("signature"),
                         "artifact": e.get("artifact"), "status": e.get("status"),
                         "matched_terms": [t for t in terms if t in hay]})
    # These are the closest direct precedents, selected from the registry hits
    # rather than inferred from filenames.
    direct_ids = {
        "phrase-boundary-fsm-20260920", "phrase-boundary-live-fsm-20260920",
        "recursive-cfg-orbit-20260920", "luna-char-lm-orbit-20260920",
        "independent-clause-seam-trie-20260920", "constructive-seam-morph-cfg-20260916",
        "word-boundary-position-csp-20260917", "typed-boundary-resegment-shortwords-20260916",
        "pos-bilateral-cfg-orbit-20260920", "semordnilap-crossword-resegmentation-20260918",
        "char-lm-tape-resegment-20260916", "online-grammar-state-char-decoder-20260916",
        "reverse-segmentation-cfg-valency-20260916", "boundary-fst-resegment-20260916",
    }
    direct = [h for h in hits if h["id"] in direct_ids]
    payload = {
        "experiment_id": "fragment-dictionary-orbit-20260921",
        "proposed_method": PROPOSED,
        "registry_entries_scanned": len(registry.get("entries", [])),
        "overlap_decision": "duplicate_stop",
        "decision": "Do not generate candidates: an unfinished fragment dictionary with open-word residuals and live mirrored character consumption is materially covered by the existing boundary-FSM, seam-trie, CFG-orbit, character-LM-orbit, and resegmentation entries.",
        "direct_precedents": direct,
        "broader_hits": hits,
        "generation": {"rendered": 0, "exact": 0, "controls": 0, "reason": "duplicate preflight; no candidate claim permitted"},
        "falsifier": "A future lane is distinct only if it changes the search object beyond open-word residual + mirrored character orbit—for example a different proof-carrying semantic object—and demonstrates that distinction before rendering.",
        "next_construction": "Do not widen this dictionary or rename the existing boundary methods; select a representation outside the registry's character-orbit/boundary family.",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"scanned": len(registry.get("entries", [])), "direct_precedents": len(direct), "decision": payload["overlap_decision"]}))

if __name__ == "__main__":
    main()
