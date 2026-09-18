"""Author short original grammatical mirror-pairs for compositional assembly.

The model proposes only clause pairs.  The host computes both letter tapes,
rejects non-exact or fragmentary pairs, checks novelty and anti-shortcut rules,
and records every response.  A surviving pair is still not a readable long
palindrome: several distinct pairs must be assembled and then shown to blinded
readers with intact and shuffled controls.
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

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

HOST = "http://127.0.0.1:11434"
PROMPT = """Write one NEW original English mirror-pair for a compositional palindrome.
Return JSON only: {{\"left\":\"...\",\"right\":\"...\"}}.
Both left and right must be natural ordinary English clauses of 3 to 7 words,
with an identifiable subject and verb and a concrete everyday meaning.  After
lowercasing and removing nonletters, left must equal the exact reverse of
right.  Do not use a famous palindrome, a reflected word-order trick, a
self-palindromic word, a quotation, a list, or a fragment.  Prefer distinct
content words and no proper names unless required by an ordinary clause.
Theme for this attempt: {theme}
Work out the character ledger privately; output no explanation."""

THEMES = (
    "a person checking a note or message",
    "a baker preparing or sharing food",
    "a child reading or drawing",
    "a worker repairing a tool",
    "a nurse helping a patient",
    "a teacher guiding a class",
    "a farmer tending an animal",
    "a friend carrying a gift",
    "a driver finding a road",
    "a poet writing an observation",
    "a guard watching a gate",
    "a cook serving a meal",
)


def request(theme: str, seed: int, timeout: float) -> dict:
    prompt = PROMPT.format(theme=theme)
    body = {"model": "imetaexabeam/RhythmAI:27b", "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.85, "num_predict": 260, "seed": seed}}
    req = urllib.request.Request(HOST + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.load(response)
    raw = str(payload.get("response", ""))
    start, end = raw.find("{"), raw.rfind("}")
    parsed, parse_error = None, None
    if start >= 0 and end >= start:
        try:
            parsed = json.loads(raw[start:end + 1])
        except json.JSONDecodeError as error:
            parse_error = f"json_error:{error.msg}"
    else:
        parse_error = "reply_has_no_json_object"
    left = parsed.get("left") if isinstance(parsed, dict) else None
    right = parsed.get("right") if isinstance(parsed, dict) else None
    audit = {"mechanically_eligible": False}
    if isinstance(left, str) and isinstance(right, str) and left.strip() and right.strip():
        left, right = left.strip(), right.strip()
        lt, rt = normalize_letters(left), normalize_letters(right)
        checks = mechanical_admission_checks(left + " " + right, min_letters=6, max_letters=80)
        checks["pair_exact_reverse"] = bool(lt) and lt == rt[::-1]
        checks["left_has_subject_verb_shape"] = len(tokenize(left)) >= 3
        checks["right_has_subject_verb_shape"] = len(tokenize(right)) >= 3
        checks["distinct_pair_content"] = len(set(tokenize(left) + tokenize(right))) == len(tokenize(left) + tokenize(right))
        audit = {"left": left, "right": right, "left_letters": len(lt),
                 "right_letters": len(rt), "left_tape": lt, "right_tape": rt,
                 "checks": checks, "mechanically_eligible": all(checks.values())}
    return {"seed": seed, "theme": theme, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "raw_response": payload, "raw_visible_reply": raw,
            "parse_error": parse_error, "audit": audit}


def run(*, attempts: int, timeout: float) -> dict:
    rows = []
    for i, theme in enumerate(THEMES[:attempts]):
        try:
            rows.append(request(theme, 2026091420 + i, timeout))
        except Exception as error:
            rows.append({"seed": 2026091420 + i, "theme": theme,
                         "status": "runtime_failure",
                         "error": f"{type(error).__name__}:{error}"})
    return {"status": "complete_mirror_pair_authoring_probe",
            "model": "imetaexabeam/RhythmAI:27b", "attempts": attempts,
            "records": rows,
            "accepted_pairs": [r["audit"] for r in rows if r.get("audit", {}).get("mechanically_eligible")],
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "model_route": "local Ollama /api/generate completion",
                           "letters_supplied_by_model": True},
            "reader_gate": "Pairs are construction material only; any assembled long surface requires exact independent audit and blinded human readers."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--attempts", type=int, default=len(THEMES))
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(attempts=args.attempts, timeout=args.timeout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "accepted_pairs": len(result["accepted_pairs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
