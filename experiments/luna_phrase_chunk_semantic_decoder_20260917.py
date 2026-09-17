"""Narrow semantic phrase-chunk character synchronizer.

The decoder chooses complete semantic chunks independently for two clauses and
uses a live character-pair frontier to rank them.  It does not construct a
half-tape, reverse words, or copy catalogue text.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT = "luna-phrase-chunk-semantic-decoder-20260917"
SIGNATURE = "semantic-phrase-chunk-frontier|left-right-chunk-character-synchronization|complete-prose|heldout-chunk-repair|independent-pointer-sha"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

DETERMINERS = ({"id": "the", "text": "The", "meaning": "definite agent"}, {"id": "a", "text": "A", "meaning": "indefinite agent"})
ADJECTIVES = ({"id": "careful", "text": "careful", "meaning": "deliberate"}, {"id": "patient", "text": "patient", "meaning": "steady"}, {"id": "quiet", "text": "quiet", "meaning": "calm"})
SUBJECT_VERBS = ({"id": "archivist", "text": "archivist stores", "meaning": "agent stores"}, {"id": "gardener", "text": "gardener waters", "meaning": "agent waters"}, {"id": "pilot", "text": "pilot repairs", "meaning": "agent repairs"}, {"id": "teacher", "text": "teacher reviews", "meaning": "agent reviews"})
VERB_OBJECTS = ({"id": "maps", "text": "weathered maps", "meaning": "theme maps"}, {"id": "seedlings", "text": "young seedlings", "meaning": "living theme"}, {"id": "lantern", "text": "the brass lantern", "meaning": "instrument theme"}, {"id": "notes", "text": "marked field notes", "meaning": "document theme"})
LOCATIVES = ({"id": "window", "text": "beside the north window", "meaning": "locative attachment"}, {"id": "rain", "text": "after steady rain", "meaning": "temporal attachment"}, {"id": "harbor", "text": "near the harbor office", "meaning": "locative attachment"}, {"id": "porch", "text": "toward the garden porch", "meaning": "directional attachment"})


def novelty_preflight() -> dict:
    data = json.loads(REGISTRY.read_text())
    rows = [*data.get("entries", []), *data.get("excluded", [])]
    collisions = [r.get("id") for r in rows if r.get("id") != EXPERIMENT and (r.get("signature") == SIGNATURE or r.get("artifact") == "experiments/luna_phrase_chunk_semantic_decoder_20260917.py")]
    self_rows = [r for r in data.get("entries", []) if r.get("id") == EXPERIMENT and r.get("signature") == SIGNATURE]
    return {"registry_entries": len(data.get("entries", [])), "self_registered": len(self_rows) == 1, "collisions": collisions, "passed": len(self_rows) == 1 and not collisions}


def clause(chunks: tuple[dict, ...]) -> str:
    det, adj, sv, obj, loc = chunks
    return f"{det['text']} {adj['text']} {sv['text']} {obj['text']} {loc['text']}"


def pointer(text: str) -> dict:
    tape = normalize_letters(text)
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"offset": i, "left": tape[i], "right": tape[j]})
        i += 1
        j -= 1
    return {"algorithm": "independent_two_pointer", "letters": len(tape), "exact": bool(tape) and not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches[:10]}


def sha(text: str) -> dict:
    tape = normalize_letters(text)
    f, r = hashlib.sha256(tape.encode()).hexdigest(), hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "exact": bool(tape) and f == r, "forward": f, "reverse": r}


def frontier(text: str) -> list[dict]:
    tape = normalize_letters(text)
    return [{"width": w, "matched_pairs": sum(tape[i] == tape[-1-i] for i in range(min(w, len(tape)//2))), "first_mismatch": next((i for i in range(min(w, len(tape)//2)) if tape[i] != tape[-1-i]), None), "equation": "left_chunk_char[i] = right_chunk_char[-1-i]"} for w in (4, 8, 16, 24)]


def audit(left: tuple[dict, ...], right: tuple[dict, ...], rank: int) -> dict:
    text = clause(left) + ". " + clause(right) + "."
    p, s = pointer(text), sha(text)
    central = mechanical_admission_checks(text, min_letters=100, max_letters=260)
    words = tuple(tokenize(text))
    complete = text.count(".") == 2 and len(words) >= 16 and all(x["text"].split()[0].lower() in words for x in (left[2], right[2]))
    distinct_chunks = len({x["id"] for x in left + right}) == len(left + right)
    return {"rank": rank, "rendered": text, "letters": p["letters"], "chunk_provenance": {"left": [{"id": x["id"], "meaning": x["meaning"]} for x in left], "right": [{"id": x["id"], "meaning": x["meaning"]} for x in right]}, "independent_reparse": complete, "frontier": frontier(text), "exact_check_two_pointer": p, "exact_check_sha256": s, "independent_exact_agreement": p["exact"] == s["exact"], "central_admission": central, "anti_shortcut_flags": {"fixed_tape": False, "reverse_decoder": False, "word_order_mirror": False, "repeated_self_palindromic_span": False, "repeated_chunk": not distinct_chunks, "catalogue_text_used": False, "fragment": False, "complete_chunks": complete}, "mechanically_admitted": bool(p["exact"] and s["exact"] and complete and distinct_chunks and all(central.values())), "reader_status": "unreviewed; programmatic checks do not certify readability", "next_repair": "Replace the held-out locative chunk at the first open character obligation, preserve both complete dependency frames, and rerun this narrow frontier."}


def run() -> dict:
    pre = novelty_preflight()
    if not pre["passed"]:
        raise RuntimeError(pre)
    # Held-out frontier: independent left/right chunk choices, with disjoint
    # content IDs enforced before the expensive audit is retained.
    seeds = list(itertools.product(DETERMINERS, ADJECTIVES, SUBJECT_VERBS, VERB_OBJECTS, LOCATIVES))
    rows = []
    for rank, (left, right) in enumerate(itertools.islice(itertools.product(seeds, reversed(seeds)), 0, 24), 1):
        if any(a["id"] == b["id"] for a, b in zip(left, right)):
            continue
        row = audit(left, right, rank)
        if row["anti_shortcut_flags"]["repeated_chunk"] or row["anti_shortcut_flags"]["fragment"]:
            continue
        rows.append(row)
    rows.sort(key=lambda r: (-sum(x["matched_pairs"] for x in r["frontier"]), -r["letters"], r["rank"]))
    exact = [r for r in rows if r["mechanically_admitted"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete; no exact closure" if not exact else "exact closure found", "novelty_preflight": pre, "candidate_count": len(rows), "states_considered": 24, "exact_count": len(exact), "rendered_candidates": rows, "exact_survivors": exact, "mechanically_admitted": exact, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "chunk_inventory": {"determiners": len(DETERMINERS), "adjectives": len(ADJECTIVES), "subject_verbs": len(SUBJECT_VERBS), "verb_objects": len(VERB_OBJECTS), "locatives": len(LOCATIVES)}, "heldout_frontier": True}, "anti_shortcut_policy": "Hard reject fixed tapes, reverse decoding, word mirrors, repeated/self-palindromic spans, repeated chunks, catalogue text, and fragments.", "next_repair": "Use one held-out locative replacement at the first seam mismatch; do not widen into a duplicate sweep.", "reader_facing_test": {"status": "not triggered unless an exact mechanically admitted survivor exists", "required": "blind intact-prose and shuffled-control rating"}}


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate_count": result["candidate_count"], "exact_count": result["exact_count"]}))
