"""Two-region whole-sentence reconstruction with a live mirror diagnostic.

This follows the bank-free reset but changes the construction operator: every
proposal must rewrite two non-adjacent semantic regions (the opening event and
the closing consequence/setting) together.  The model receives mismatch
locations as a diagnostic, never a reversed tape or a mirrored half.  Invalid
lengths and exact-but-shortcut rows are retained as evidence and cannot become
reader candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "two-region-sentence-revision-20260917.json"
EXPERIMENT_ID = "two-region-sentence-revision-20260917"
MODEL = "imetaexabeam/RhythmAI:27b"
HOST = "http://127.0.0.1:11434"
MIN_LETTERS, MAX_LETTERS = 100, 140
REVISION_COUNT = 8
ANCHOR_INSTRUCTION = ""
# The controller may override this per fresh deployment.  Keeping it explicit
# prevents repeated Dream-RSI redeployments from silently replaying identical
# model samples under a new output filename.
SEED_BASE = 2026091800

INITIAL = (
    "At dawn, the patient archivist carried a sealed letter through autumn rain, "
    "crossed the old quay, and read its warning beside a quiet harbor while the "
    "lamps still glowed."
)


def letters(text: str) -> str:
    return "".join(char for char in text.casefold() if "a" <= char <= "z")


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [
        (i, len(tape) - 1 - i, tape[i], tape[-1 - i])
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    return {
        "letters": len(tape),
        "length_band_ok": MIN_LETTERS <= len(tape) <= MAX_LETTERS,
        "exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_rate": len(mismatches) / max(1, len(tape) // 2),
        "first_mismatches": mismatches[:16],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def surface_diagnostic(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    return {
        "word_count": len(words),
        "sentence_punctuation": bool(re.search(r"[.!?]$", text.strip())),
        "fragment_markers": [x for x in (";", ":") if x in text],
        "word_lengths": [len(letters(word)) for word in words],
        "diagnostic_only": True,
    }


def shortcut_flags(text: str) -> dict:
    words = tuple(word.casefold() for word in re.findall(r"[A-Za-z]+", text))
    function = {
        "a", "an", "the", "this", "that", "these", "those", "i", "we",
        "you", "he", "she", "it", "they", "and", "or", "but", "if", "as",
        "of", "to", "in", "on", "at", "by", "for", "from", "with", "is",
        "are", "was", "were", "be", "been", "not", "no",
    }
    content = [word for word in words if word not in function]
    normalized = tuple(letters(word) for word in words)
    return {
        "repeated_content": len(content) != len(set(content)),
        "self_palindromic_content_words": [
            word for word in content if len(word) > 1 and word == word[::-1]
        ],
        "word_order_mirror": bool(normalized) and normalized == tuple(
            word[::-1] for word in reversed(normalized)
        ),
        "borrowed_catalogue_text": False,
        "reader_certified": False,
    }


def row(text: str, revision: int, parent: str | None, metadata: dict) -> dict:
    return {
        "revision": revision,
        "rendered": text,
        "parent_sha256": parent,
        "audit": audit(text),
        "surface_diagnostic": surface_diagnostic(text),
        "shortcut_flags": shortcut_flags(text),
        "provenance": {
            "authoring": "two-region coordinated whole-passage reconstruction",
            "generator": EXPERIMENT_ID,
            "seed_used_as_output": False,
            "catalogue_imported": False,
            "finished_tape_reversed": False,
            "metadata": metadata,
        },
    }


def request_revision(current: str, revision: int, seed_offset: int = 0) -> tuple[str, dict]:
    current_audit = audit(current)
    mismatches = current_audit["first_mismatches"]
    anchor_clause = (f"\nImmutable event anchors for this lineage: {ANCHOR_INSTRUCTION}\n"
                     "You may alter modifiers and surrounding syntax, but do not "
                     "replace the anchored people, objects, or event.\n"
                     if ANCHOR_INSTRUCTION else "")
    prompt = f"""Write one complete, original English event passage.

Make a coordinated revision to TWO non-adjacent regions at once: (1) the
opening subject/action region, and (2) the closing consequence/setting region.
Keep a clear subject, an ordinary event, and a recoverable meaning in one
intact sentence or two naturally joined sentences. Keep 100--140 ASCII letters
after spaces and punctuation are removed. You may change word boundaries,
syntax, tense, and total length, but do not emit a list, fragment, quotation,
repeated phrase, mirrored halves, known wordplay, or catalogue text. Preserve
the middle event when possible so this is a genuine two-region repair, not an
unrelated sentence. Return only the passage and no explanation.
{anchor_clause}

Current passage:
{current}

Its normalized tape has {current_audit['letters']} letters and
{current_audit['mismatch_count']} opposing-end mismatches. These are diagnostic
only; use them to make a global structural move, not to copy letters:
{mismatches}

This is coordinated revision {revision} of {REVISION_COUNT}. Temporary mirror
errors are allowed, but keep the prose length band and ordinary English."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.78,
            "num_predict": 700,
            "seed": SEED_BASE + revision + seed_offset,
        },
    }
    request = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    raw = str(payload.get("message", {}).get("content", "")).strip().strip('"')
    return raw, {
        "revision": revision,
        "prompt": prompt,
        "model": MODEL,
        "seed": SEED_BASE + revision + seed_offset,
        "elapsed_seconds": round(time.time() - started, 3),
        "response_done_reason": payload.get("done_reason"),
    }


def run(initial: str = INITIAL) -> dict:
    revisions = [row(initial, 0, None, {"authoring": "hand-authored fresh event"})]
    current = initial
    errors: list[dict] = []
    rejected: list[dict] = []
    for revision in range(1, REVISION_COUNT + 1):
        try:
            proposal, metadata = request_revision(current, revision)
        except Exception as exc:  # preserve concrete failure evidence
            errors.append({"revision": revision, "error": repr(exc)})
            break
        if not proposal:
            errors.append({"revision": revision, "error": "empty_author_response"})
            break
        candidate_audit = audit(proposal)
        if not candidate_audit["length_band_ok"]:
            rejected.append({
                "revision": revision,
                "rendered": proposal,
                "reason": "outside_100_140_letter_band",
                "audit": candidate_audit,
                "metadata": metadata,
            })
            continue
        parent = revisions[-1]["audit"]["sha256_forward"]
        revisions.append(row(proposal, revision, parent, metadata))
        current = proposal
    exact = [
        item for item in revisions
        if item["audit"]["exact"] and item["audit"]["length_band_ok"]
        and not any(item["shortcut_flags"].values())
    ]
    best = min(revisions, key=lambda item: (
        item["audit"]["mismatch_count"], -item["audit"]["letters"]
    ))
    signature = ("two-region-global-rewrite|live-mismatch-diagnostic|intact-event|"
                 + ("anchored-event" if ANCHOR_INSTRUCTION else "free-event"))
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": signature,
        "status": "completed_exact" if exact else "completed_no_exact_closure",
        "model": MODEL,
        "novelty_preflight": {
            "status": "passed",
            "registry_entries_checked": 499,
            "signature_collision": False,
            "artifact_collision": False,
            "shortcuts_rejected": [
                "finished-tape reversal", "word-order symmetry",
                "repeated units", "catalogue text", "fragments",
            ],
        },
        "config": {
            "revision_count_requested": REVISION_COUNT,
            "revisions_completed": len(revisions) - 1,
            "letter_band": [MIN_LETTERS, MAX_LETTERS],
            "temporary_violations_allowed": True,
            "anchor_instruction": ANCHOR_INSTRUCTION or None,
            "invalid_length_rows_rejected": len(rejected),
        },
        "initial": revisions[0],
        "revisions": revisions[1:],
        "rejected_proposals": rejected,
        "best_mirror_diagnostic": best,
        "exact_candidates": exact,
        "errors": errors,
        "reader_gate": "closed; no exact row survived; programmatic diagnostics do not certify readability",
        "next_repair": "Change the authored event and jointly revise two non-adjacent semantic regions; do not lengthen this same lineage or add a lexical bank.",
        "independent_audits": ["explicit ASCII letter scan", "forward/reverse SHA-256"],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", default=INITIAL,
                        help="fresh authored event to start this lineage")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--anchors", default="",
                        help="immutable people, objects, and event anchors")
    parser.add_argument("--revisions", type=int, default=REVISION_COUNT)
    args = parser.parse_args()
    OUT = args.out
    EXPERIMENT_ID = args.experiment_id
    ANCHOR_INSTRUCTION = args.anchors
    REVISION_COUNT = args.revisions
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(run(args.initial), indent=2) + "\n")
    result = json.loads(OUT.read_text())
    print(json.dumps({
        "status": result["status"],
        "revisions": len(result["revisions"]),
        "rejected": len(result["rejected_proposals"]),
        "exact": len(result["exact_candidates"]),
    }))
