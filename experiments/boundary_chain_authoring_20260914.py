"""One bounded authoring probe seeded by fresh reverseable lexical chains.

The chains are construction hints, not catalogue prose.  The model must write
one new complete sentence; the host independently checks every character and
records the raw response.  A mechanically eligible surface is still only a
candidate for blinded human reading.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

HOST = "http://127.0.0.1:11434"
PAIRS = (
    ("drawer", "reward"), ("deliver", "reviled"), ("stressed", "desserts"),
    ("deer", "reed"), ("keep", "peek"), ("draw", "ward"),
    ("spots", "stops"), ("devil", "lived"), ("doom", "mood"),
    ("pals", "slap"), ("dial", "laid"), ("spit", "tips"),
)

PROMPT = """Author one NEW, original, grammatical English sentence of 100 to 180 letters.
It must be a single connected thought with an identifiable subject and verb,
ordinary vocabulary, and no list, quotation, famous palindrome, repeated
content word, repeated clause, or individually palindromic content word. Do
not use familiar palindrome scaffolds such as madam, level, radar, civic,
noon, kayak, or a man a plan a canal panama.

The normalized letters (lowercase ASCII letters only) must read identically
backwards. Compose the whole sentence; do not mirror word order and do not
borrow any known palindrome. You may use the following reverseable lexical
pairs as raw construction material, but the final sentence must be newly
composed and semantically coherent:
{pairs}

Return only JSON: {{"text":"..."}}. The host will reject any failure.
"""


def request_json(body: dict, timeout: float) -> dict:
    request = urllib.request.Request(HOST + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_text(raw: str) -> tuple[str | None, str | None]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as error:
        return None, f"json_error:{error.msg}"
    text = value.get("text") if isinstance(value, dict) else None
    return (text.strip(), None) if isinstance(text, str) and text.strip() else (None, "text_schema_error")


def audit(text: str | None) -> dict:
    if text is None:
        return {"mechanically_eligible": False}
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=100, max_letters=180)
    independent = {
        "tape": tape,
        "letters": len(tape),
        "direct_symmetric_position_comparison": all(
            tape[i] == tape[-1 - i] for i in range(len(tape))
        ),
    }
    return {
        "text": text,
        "words": list(tokenize(text)),
        "letters": len(tape),
        "mismatch_count": sum(a != b for a, b in zip(tape, reversed(tape))),
        "shared_mechanical_checks": checks,
        "independent_exactness": independent,
        "mechanically_eligible": all(checks.values()) and all(independent.values()),
        "human_reader_study": "not_run",
    }


def run(*, model: str, seed: int, timeout: float) -> dict:
    prompt = PROMPT.format(pairs="; ".join(f"{left}/{right}" for left, right in PAIRS))
    response = request_json({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.85, "num_predict": 700, "seed": seed},
    }, timeout)
    raw = response.get("message", {}).get("content", "")
    text, parse_error = parse_text(raw)
    result = audit(text)
    return {
        "status": "complete_boundary_chain_authoring_probe",
        "model_requested": model,
        "seed": seed,
        "prompt": prompt,
        "prompt_sha256": sha256(prompt.encode()).hexdigest(),
        "raw_response": response,
        "raw_visible_reply": raw,
        "parse_error": parse_error,
        "audit": result,
        "construction_material": list(PAIRS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="imetaexabeam/RhythmAI:27b")
    parser.add_argument("--seed", type=int, default=2026091412)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(model=args.model, seed=args.seed, timeout=args.timeout)
    except Exception as error:
        result = {"status": "boundary_chain_authoring_runtime_failure",
                  "model_requested": args.model, "seed": args.seed,
                  "error": f"{type(error).__name__}:{error}"}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "status": result["status"],
                      "mechanically_eligible": result.get("audit", {}).get("mechanically_eligible", False)}, indent=2))


if __name__ == "__main__":
    main()
