"""Bank-free whole-passage reconstruction under a live mirror diagnostic.

This is deliberately different from the project's slot and bank products. A
local author rewrites one complete event passage at a time; intermediate
passages may be non-palindromic and may have a worse mismatch count. The
program records the full text and diagnostics, while exactness and shortcut
checks remain hard gates downstream. No text is emitted from a reversed tape.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bank-free-sentence-revision-20260917.json"
EXPERIMENT_ID = "bank-free-sentence-revision-20260917"
MODEL = "gpt-oss:20b"
HOST = "http://127.0.0.1:11434"
MIN_LETTERS, MAX_LETTERS = 100, 140
REVISION_COUNT = 20

INITIAL = (
    "The patient archivist carried a sealed letter through the autumn rain and read "
    "it beside the quiet harbor before dawn, while the lamps still glowed."
)


def letters(text: str) -> str:
    return "".join(char for char in text.casefold() if "a" <= char <= "z")


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [
        (index, len(tape) - 1 - index, tape[index], tape[-1 - index])
        for index in range(len(tape) // 2)
        if tape[index] != tape[-1 - index]
    ]
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_rate": (len(mismatches) / (len(tape) / 2)) if tape else 1.0,
        "first_mismatches": mismatches[:16],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def surface_diagnostic(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    return {
        "word_count": len(words),
        "sentence_punctuation": bool(re.search(r"[.!?]$", text.strip())),
        "has_fragment_marker": any(token in text.casefold() for token in (";", ":")),
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


def request_revision(current: str, revision: int) -> tuple[str, dict]:
    current_audit = audit(current)
    prompt = f"""Rewrite one complete English passage toward an exact letter-level palindrome.

The passage must remain an original, coherent event report of 100--140 letters,
with ordinary intact prose and a recoverable subject and action. Rewrite the
whole passage jointly: you may change clauses, word boundaries, syntax, tense,
and length within that band. Do not make two mirrored halves, do not repeat a
phrase or content word, do not use a known palindrome, quotation, list, or
fragment, and do not mention this instruction. Temporary increases in mirror
error are allowed; preserve meaning while making a structural move that could
reduce the global error. Return only the passage, with no quotation marks or
explanation.

Current passage:
{current}

Its normalized letter tape has {current_audit['letters']} letters and
{current_audit['mismatch_count']} mismatched mirrored pairs. The first
mismatches are (left index, right index, left, right):
{current_audit['first_mismatches']}

This is revision {revision} of {REVISION_COUNT}."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": "low",
        "options": {
            "temperature": 0.72,
            # gpt-oss may spend a substantial hidden-reasoning prefix even
            # with ``think`` disabled; leave enough budget for visible prose.
            "num_predict": 1200,
            "seed": 2026091700 + revision,
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
    raw = str(payload.get("message", {}).get("content", "")).strip()
    # Keep the authoring surface intact but remove only wrapper quotes or
    # accidental leading labels; never repair its letters programmatically.
    raw = raw.strip().strip('"')
    return raw, {
        "revision": revision,
        "prompt": prompt,
        "model": MODEL,
        "seed": 2026091700 + revision,
        "elapsed_seconds": round(time.time() - started, 3),
        "response_done_reason": payload.get("done_reason"),
    }


def row(text: str, *, revision: int, parent: str | None, metadata: dict) -> dict:
    return {
        "revision": revision,
        "rendered": text,
        "parent_sha256": parent,
        "audit": audit(text),
        "surface_diagnostic": surface_diagnostic(text),
        "shortcut_flags": shortcut_flags(text),
        "provenance": {
            "authoring": "bank-free whole-passage reconstruction",
            "generator": EXPERIMENT_ID,
            "seed_used_as_output": False,
            "catalogue_imported": False,
            "finished_tape_reversed": False,
            "metadata": metadata,
        },
    }


def run() -> dict:
    revisions = [row(INITIAL, revision=0, parent=None, metadata={
        "authoring": "hand-authored initial event passage",
    })]
    current = INITIAL
    errors = []
    for revision in range(1, REVISION_COUNT + 1):
        try:
            proposal, metadata = request_revision(current, revision)
        except Exception as exc:  # preserve a concrete run failure and stop
            errors.append({"revision": revision, "error": repr(exc)})
            break
        if not proposal:
            errors.append({"revision": revision, "error": "empty_author_response"})
            break
        previous_sha = revisions[-1]["audit"]["sha256_forward"]
        revisions.append(row(proposal, revision=revision,
                             parent=previous_sha, metadata=metadata))
        current = proposal
    exact = [item for item in revisions if item["audit"]["exact"]]
    best = min(revisions, key=lambda item: (
        item["audit"]["mismatch_count"],
        -item["audit"]["letters"],
    ))
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": "bank-free-whole-passage-reconstruction|temporary-mirror-violations|global-structural-rewrite",
        "status": "completed_exact" if exact else "completed_no_exact_closure",
        "model": MODEL,
        "config": {"revision_count_requested": REVISION_COUNT,
                   "revisions_completed": len(revisions) - 1,
                   "letter_band": [MIN_LETTERS, MAX_LETTERS],
                   "temporary_violations_allowed": True},
        "initial": revisions[0],
        "revisions": revisions[1:],
        "best_mirror_diagnostic": best,
        "exact_candidates": exact,
        "errors": errors,
        "reader_gate": "closed; exactness and diagnostics do not certify readability; use intact and shuffled blinded controls if an admissible exact row appears",
        "next_repair": "If no exact row appears, change the authored event and allow a coordinated two-region rewrite; do not add a bank or simply increase revisions.",
        "independent_audits": ["explicit ASCII letter scan", "forward/reverse SHA-256"],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps({
        "status": json.loads(OUT.read_text())["status"],
        "revisions": len(json.loads(OUT.read_text())["revisions"]),
        "exact": len(json.loads(OUT.read_text())["exact_candidates"]),
    }))
