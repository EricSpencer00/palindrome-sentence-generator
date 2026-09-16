"""Index a fresh model-authored bank of complete clauses for exact joins.

This is a deliberately new construction dimension: the model supplies ordinary
complete clauses independently, then a character index searches one-clause and
two-clause compositions across two deterministic banks.  No clause is reversed
or copied into a reflected slot during generation.  The model is only a source
of prose proposals; exactness is checked twice by this program and readability
remains a human-blinded question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MODEL = "gpt-oss:20b"
HOST = "http://127.0.0.1:11434"
ID = "model-authored-clause-bank-index"
SIGNATURE = (
    "model-authored-clause-bank|semantic-intent-conditioned-proposals|"
    "independent-complete-clause-bank|character-synchronous-reverse-index|"
    "no-catalogue-import|blinded-readability-gate"
)
PROPOSAL_OUT = ROOT / "runs/model-authored-clause-proposals-20260915.json"
OUT = ROOT / "runs/model-authored-clause-bank-index-20260915.json"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LINE_RE = re.compile(r"^(?:[-*]\s*|\d+[.)]\s*)")
PROMPT = """Write 180 original, ordinary English complete sentences, one per line.
Each should be 5 to 10 words and describe observation, making, travelling, or
recording something concrete. Use plain prose, no lists, no palindromes, no
quotations, no explanations, and no numbering. These are independent proposal
sentences for a later exact letter-palindrome search; do not try to mirror any
sentence or its word order."""


def request_json(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        HOST + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def capture_bank(model: str = MODEL, seed: int = 20260915) -> dict:
    metadata = request_json("/api/show", {"name": model})
    response = request_json("/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": "low",
        "options": {"temperature": 0.85, "num_predict": 5000, "seed": seed},
    })
    raw = response["message"]["content"]
    payload = {
        "model": model,
        "seed": seed,
        "prompt": PROMPT,
        "model_metadata": metadata,
        "raw_response": raw,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }
    PROPOSAL_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parse_bank(raw: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for line_number, source in enumerate(raw.splitlines(), 1):
        line = LINE_RE.sub("", source.strip()).strip().strip('"')
        if not line or line.startswith(("Here are", "Sure", "Sentences:")):
            continue
        words = WORD_RE.findall(line)
        normalized = normalize_letters(line)
        if not (5 <= len(words) <= 10 and 20 <= len(normalized) <= 90):
            continue
        if not line[-1] in ".!?":
            line += "."
        if sum(line.count(mark) for mark in ".!?" ) != 1:
            continue
        if not all(ord(char) < 128 for char in line):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        rows.append({
            "source_line": line_number,
            "raw_line": source,
            "text": line,
            "normalized": normalized,
            "words": [word.casefold() for word in words],
            "word_count": len(words),
        })
    return rows


def direct_audit(tape: str) -> bool:
    return bool(tape) and tape == tape[::-1]


def two_pointer_audit(tape: str) -> bool:
    if not tape:
        return False
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return False
        left += 1
        right -= 1
    return True


def word_order_shortcut(left: list[str], right: list[str]) -> bool:
    return left and right and right == list(reversed(left))


def row_for(parts_left: list[dict], parts_right: list[dict], layer: str) -> dict:
    left_tape = "".join(part["normalized"] for part in parts_left)
    right_tape = "".join(part["normalized"] for part in parts_right)
    rendered_parts = [part["text"].rstrip(".!?") for part in parts_left + parts_right]
    rendered = ". ".join(rendered_parts) + "."
    tape = normalize_letters(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=400)
    words_left = [word for part in parts_left for word in part["words"]]
    words_right = [word for part in parts_right for word in part["words"]]
    repeated_clause = len({part["normalized"] for part in parts_left + parts_right}) != len(parts_left + parts_right)
    shortcut = word_order_shortcut(words_left, words_right)
    exact_direct = direct_audit(tape)
    exact_two_pointer = two_pointer_audit(tape)
    failed = [key for key, value in checks.items() if not value]
    if repeated_clause:
        failed.append("repeated_clause_unit")
    if shortcut:
        failed.append("word_order_only_symmetry")
    if exact_direct != exact_two_pointer:
        failed.append("independent_audit_disagreement")
    return {
        "rendered": rendered,
        "letters": len(tape),
        "exact": exact_direct and exact_two_pointer,
        "independent_two_pointer": exact_two_pointer,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "layer": layer,
        "left_sources": [part["source_line"] for part in parts_left],
        "right_sources": [part["source_line"] for part in parts_right],
        "left_clause_count": len(parts_left),
        "right_clause_count": len(parts_right),
        "failed_checks": sorted(set(failed)),
        "admitted": exact_direct and exact_two_pointer and not failed,
        "readability_status": "diagnostic_only_unreviewed",
        "provenance": "fresh model-authored complete-clause bank; reverse index only",
    }


def search(bank: list[dict], pair_limit: int = 50000) -> dict:
    left_bank = [row for row in bank if int(hashlib.sha256(row["normalized"].encode()).hexdigest(), 16) % 2 == 0]
    right_bank = [row for row in bank if row not in left_bank]
    reverse_index = {row["normalized"]: row for row in right_bank}
    rows: list[dict] = []
    for left in left_bank:
        right = reverse_index.get(left["normalized"][::-1])
        if right is not None:
            rows.append(row_for([left], [right], "single-clause-index"))

    # A second layer composes two intact clauses per side.  It is bounded and
    # independently indexed; it does not emit a reflected clause or reverse a
    # word list.  Distinct clause IDs prevent repeated-unit constructions.
    pair_index: dict[str, tuple[dict, dict]] = {}
    for first in right_bank:
        for second in right_bank:
            if first["normalized"] == second["normalized"]:
                continue
            key = first["normalized"] + second["normalized"]
            pair_index.setdefault(key, (first, second))
    pair_probes = 0
    for first in left_bank:
        for second in left_bank:
            if first["normalized"] == second["normalized"]:
                continue
            if pair_probes >= pair_limit:
                break
            pair_probes += 1
            right_pair = pair_index.get((first["normalized"] + second["normalized"])[::-1])
            if right_pair is not None:
                rows.append(row_for([first, second], list(right_pair), "two-clause-composition-index"))
        if pair_probes >= pair_limit:
            break
    unique = {}
    for row in rows:
        unique[row["normalized_sha256"]] = row
    rows = list(unique.values())
    return {
        "bank_count": len(bank),
        "left_bank_count": len(left_bank),
        "right_bank_count": len(right_bank),
        "single_clause_probes": len(left_bank),
        "two_clause_probes": pair_probes,
        "exact_count": sum(row["exact"] for row in rows),
        "admitted_count": sum(row["admitted"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    proposal = capture_bank(args.model, args.seed) if not PROPOSAL_OUT.exists() else json.loads(PROPOSAL_OUT.read_text())
    bank = parse_bank(proposal["raw_response"])
    result = search(bank)
    output = {
        "experiment_id": ID,
        "signature": SIGNATURE,
        "preflight": {
            "registry_entries_before_registration": 72,
            "excluded_families": 6,
            "status": "formal_preflight_before_execution",
            "manual_review_required": False,
        },
        "method": "independent model-authored complete-clause bank with one- and two-clause character reverse indexes",
        "proposal_artifact": str(PROPOSAL_OUT.relative_to(ROOT)),
        "proposal_sha256": proposal["raw_sha256"],
        "model": proposal["model"],
        "prompt": proposal["prompt"],
        "parsed_bank": bank,
        "search": result,
        "rendered_candidates": result["rows"],
        "reader_gate": "No programmatic score certifies readability; any admitted row requires randomized blinded human ratings with intact-prose and shuffled controls.",
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "bank": len(bank), "exact": result["exact_count"], "admitted": result["admitted_count"]}, indent=2))


if __name__ == "__main__":
    main()
