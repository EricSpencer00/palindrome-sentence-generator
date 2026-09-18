"""Freeze model-authored, centre-out English palindrome proposals.

Unlike the rejected word-pair and fixed-tree generators, this run asks for a
single complete utterance while the model reasons from unfinished *character*
phrases at the centre.  The model's text is proposal material only.  This
module owns normalization, exactness, repetition, local-catalogue exclusion,
and the durable raw-proposal record.

The centre-out instruction is inspired by the construction representation in
Peter Norvig's public algorithm description, but this program starts without
a borrowed palindrome seed and rejects every local-catalogue collision.
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


HOST = "http://localhost:11434"
PROPOSALS_PER_CALL = 12
OPTIONS = {"temperature": 0.95, "num_predict": 2800, "seed": 20260912}

PROMPT = """You are a meticulous English palindrome composer.  Produce NEW,
readable English utterances by composing outward from unfinished character
phrases at the centre, as a human palindrome writer would.  Do not start from
or quote a known palindrome.

Each proposed utterance must meet every condition:
- at least 30 letters after lowercasing and deleting nonletters;
- exact letter palindrome under that same normalization;
- a coherent complete English sentence, question, or compact dialogue, not
  two unrelated fragments;
- no repeated word and no word that is a letter palindrome by itself;
- no word-order mirror, repeated unit, filler, or gibberish.

Think privately through the centre-out character construction.  Punctuation
may clarify syntax but must never supply a letter.  We will independently
verify every character, so return only candidates you believe are exact.

Return exactly a JSON object:
{"candidates":["candidate 1", ..., "candidate 12"]}
"""


def request(path: str, payload: dict) -> dict:
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(HOST + path, data=raw,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def parse(raw: str) -> tuple[list[str] | None, str | None]:
    body = raw.strip().removeprefix("```json").removesuffix("```").strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        value = json.loads(body[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != PROPOSALS_PER_CALL:
        return None, f"need_exactly_{PROPOSALS_PER_CALL}_candidates"
    if not all(isinstance(candidate, str) for candidate in candidates):
        return None, "candidate_schema_error"
    return candidates, None


def screen(text: str, *, known: set[str]) -> dict:
    return mechanical_admission_checks(text, local_catalogue=known,
                                       min_letters=30, max_letters=100)


def run(model: str) -> dict:
    known = set(json.loads((ROOT / "data" / "known_palindromes.json").read_text()))
    metadata = request("/api/show", {"name": model})
    reply = request("/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": "low",
        "options": OPTIONS,
    })
    raw = reply.get("message", {}).get("content", "")
    candidates, parse_error = parse(raw)
    rows = []
    for candidate in candidates or []:
        checks = screen(candidate, known=known)
        rows.append({
            "rendered": candidate,
            "letters": len(normalize_letters(candidate)) if not any(
                character.isalpha() and not character.isascii() for character in candidate
            ) else 0,
            "checks": checks,
            "rejection_codes": [name for name, passed in checks.items() if not passed],
            "reader_status": "unreviewed; a mechanical survivor is not a readability claim",
        })
    unique = {}
    for row in rows:
        try:
            key = normalize_letters(row["rendered"])
        except ValueError:
            key = "invalid-unicode:" + row["rendered"]
        unique.setdefault(key, row)
    admitted = [row for row in unique.values() if not row["rejection_codes"]]
    return {
        "status": "complete_model_guided_centreout_proposal_run",
        "model_requested": model,
        "model_metadata": metadata,
        "prompt": PROMPT,
        "options": OPTIONS | {"thinking": "low"},
        "known_catalogue_sha256": hashlib.sha256(
            (ROOT / "data" / "known_palindromes.json").read_bytes()).hexdigest(),
        "raw_reply": raw,
        "parse_error": parse_error,
        "records": rows,
        "mechanically_admitted": admitted,
        "reader_gate": (
            "Any mechanically admitted string must be rendered to blinded readers with intact "
            "prose and shuffled controls before the project calls it readable."
        ),
        "next_construction_operator_if_empty": (
            "Use the exact failures to create a bidirectional character-repair request: freeze "
            "the strongest complete grammatical proposal, expose only its mismatched mirrored "
            "spans, and jointly relexicalize those spans under a dependency witness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-oss:20b")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "parse_error": result["parse_error"],
                      "records": len(result["records"]),
                      "mechanically_admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
