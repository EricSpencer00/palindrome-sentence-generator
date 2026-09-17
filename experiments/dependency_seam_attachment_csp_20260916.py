"""Dependency-tree seam CSP with live semantic attachment choices.

This lane chooses complete, ordinary-order clause realizations by solving a
small dependency CSP: subject/verb/object valency, adjunct attachment, and
agreement are constraints, while the emitted character seam is only an
obligation ledger.  It never decodes or resegments a pre-existing tape.
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

EXPERIMENT = "dependency-seam-attachment-csp-20260916"
SIGNATURE = "dependency-tree-seam-csp|attachment-choice-frontier|role-agreement-complete-prose|live-character-obligation|independent-pointer-sha"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

SUBJECTS = (
    {"id": "archivist", "text": "The careful archivist", "number": "singular", "role": "agent"},
    {"id": "gardeners", "text": "The patient gardeners", "number": "plural", "role": "agent"},
    {"id": "pilot", "text": "A quiet pilot", "number": "singular", "role": "agent"},
    {"id": "sailors", "text": "Two watchful sailors", "number": "plural", "role": "agent"},
)
EVENTS = (
    {"id": "maps", "singular": "stores", "plural": "store", "object": "weathered maps", "role": "theme"},
    {"id": "seedlings", "singular": "waters", "plural": "water", "object": "young seedlings", "role": "theme"},
    {"id": "lantern", "singular": "repairs", "plural": "repair", "object": "the brass lantern", "role": "theme"},
    {"id": "letters", "singular": "carries", "plural": "carry", "object": "sealed letters", "role": "theme"},
)
ATTACHMENTS = (
    {"id": "locative", "prep": "beside", "tail": "the north window", "attach": "event", "meaning": "event occurs beside a window"},
    {"id": "temporal", "prep": "after", "tail": "steady rain", "attach": "event", "meaning": "event follows rain"},
    {"id": "instrument", "prep": "with", "tail": "a brass key", "attach": "verb", "meaning": "agent uses a key"},
    {"id": "directional", "prep": "toward", "tail": "the harbor office", "attach": "event", "meaning": "theme moves toward an office"},
)


def preflight() -> dict:
    data = json.loads(REGISTRY.read_text())
    rows = [*data.get("entries", []), *data.get("excluded", [])]
    collisions = [{"id": r.get("id"), "fields": [f for f, v in (("id", r.get("id") == EXPERIMENT), ("signature", r.get("signature") == SIGNATURE), ("artifact", r.get("artifact") == "experiments/dependency_seam_attachment_csp_20260916.py")) if v]} for r in rows if r.get("id") != EXPERIMENT and (r.get("signature") == SIGNATURE or r.get("artifact") == "experiments/dependency_seam_attachment_csp_20260916.py")]
    self_rows = [r for r in data.get("entries", []) if r.get("id") == EXPERIMENT and r.get("signature") == SIGNATURE]
    return {"registry_entries": len(data.get("entries", [])), "self_registered": len(self_rows) == 1, "collisions": collisions, "passed": len(self_rows) == 1 and not collisions}


def render(subject: dict, event: dict, attachment: dict) -> str:
    verb = event["singular"] if subject["number"] == "singular" else event["plural"]
    return f"{subject['text']} {verb} {event['object']} {attachment['prep']} {attachment['tail']}."


def independent_pointer(text: str) -> dict:
    tape = normalize_letters(text)
    mismatches = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"offset": left, "left": tape[left], "right": tape[right]})
        left += 1
        right -= 1
    return {"algorithm": "independent_two_pointer", "letters": len(tape), "exact": bool(tape) and not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches[:10]}


def independent_sha(text: str) -> dict:
    tape = normalize_letters(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"algorithm": "forward_reverse_sha256", "exact": bool(tape) and forward == reverse, "forward": forward, "reverse": reverse}


def seam_frontier(text: str) -> list[dict]:
    tape = normalize_letters(text)
    rows = []
    for width in (1, 2, 4, 8, 16):
        width = min(width, len(tape) // 2)
        pairs = sum(tape[i] == tape[-1-i] for i in range(width))
        rows.append({"obligation_width": width, "matched_pairs": pairs, "first_open_offset": next((i for i in range(width) if tape[i] != tape[-1-i]), None), "equation": "emitted_left[i] = emitted_right[-1-i]"})
    return rows


def audit(subject: dict, event: dict, attachment: dict, rank: int) -> dict:
    text = render(subject, event, attachment)
    pointer, sha = independent_pointer(text), independent_sha(text)
    words = tuple(tokenize(text))
    parsed = subject["text"].lower().split()[1] in words and event["object"].split()[0] in words and attachment["prep"] in words
    central = mechanical_admission_checks(text, min_letters=45, max_letters=180)
    exact = pointer["exact"] and sha["exact"] and pointer["exact"] == sha["exact"]
    return {"rank": rank, "rendered": text, "letters": pointer["letters"], "dependency_provenance": {"subject": subject["id"], "event": event["id"], "attachment": attachment["id"], "semantic_role": event["role"], "attachment_meaning": attachment["meaning"], "agreement": subject["number"]}, "independent_reparse": parsed, "seam_frontier": seam_frontier(text), "exact_check_two_pointer": pointer, "exact_check_sha256": sha, "independent_exact_agreement": pointer["exact"] == sha["exact"], "central_admission": central, "anti_shortcut_flags": {"fixed_tape": False, "reverse_decoder": False, "word_order_mirror": central["not_word_order_symmetry"], "repeated_palindromic_unit": central["no_self_palindromic_proper_multiword_span"], "catalogue_text_used": False, "complete_dependency_constituent": parsed, "agreement_checked": True}, "mechanically_admitted": bool(exact and parsed and all(central.values())), "reader_status": "unreviewed; programmatic checks do not certify readability", "next_repair": "Replace the held-out attachment realization at the first open seam offset, preserving subject agreement and event valency, then rerun the CSP."}


def run() -> dict:
    novelty = preflight()
    if not novelty["passed"]:
        raise RuntimeError(f"novelty preflight failed: {novelty}")
    rows = [audit(s, e, a, i + 1) for i, (s, e, a) in enumerate(itertools.product(SUBJECTS, EVENTS, ATTACHMENTS))]
    rows.sort(key=lambda r: (-sum(x["matched_pairs"] for x in r["seam_frontier"]), -r["letters"], r["rank"]))
    exact = [r for r in rows if r["mechanically_admitted"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete; no exact closure" if not exact else "exact closure found", "novelty_preflight": novelty, "states_examined": len(rows), "exact_count": len(exact), "rendered_candidates": rows, "exact_survivors": exact, "mechanically_admitted": exact, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "inventory_shape": [len(SUBJECTS), len(EVENTS), len(ATTACHMENTS)], "ordinary_order_rendering": True}, "anti_shortcut_policy": "No fixed tape, resegmentation, reverse decoding, word mirroring, repeated units, catalogue text, or isolated character editing; all outputs are complete dependency-compatible prose.", "next_repair": "Use a held-out attachment and re-solve agreement plus seam obligations at the first mismatch; do not widen by duplicate sweeps.", "reader_facing_test": {"status": "not triggered unless exact admitted survivor exists", "required": "blind intact-prose rating against shuffled controls"}}


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states_examined": result["states_examined"], "exact_count": result["exact_count"]}))
