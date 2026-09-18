"""Query a pinned local model for jointly designed reversible phrase pairs.

Each prompt asks for both lexicalizations together.  The model does not choose
among candidates or certify their quality; every response is frozen, then
screened by :mod:`experiments.joint_dual_lexicalization`.  A resumable
checkpoint preserves raw output after every intent/replicate call.
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

from experiments.joint_dual_lexicalization import (
    INTENTS,
    accepted,
    existing_v3_pairs,
    screen_candidate,
)


HOST = "http://localhost:11434"
CANDIDATES_PER_CALL = 4
REPLICATES_PER_INTENT = 4
GENERATION_OPTIONS = {"temperature": 0.8, "num_predict": 600}
RESPONSE_PREFILL = '{"candidates":['
PROMPT = """Create exactly 4 NEW two-sided phrase pairs for the intended scene below.

Both sides are equally important: design them together, not by writing one side
and mechanically respelling it afterward. Each side must already be a complete
sequence of ordinary English words.

Hard constraint:
letters(left), after removing spaces, must equal the reverse of letters(right).

Additional rules:
- lowercase ASCII letters and single spaces only
- 3--8 words and 15--30 letters per side
- the two sides should describe connected parts of one small scene
- do not use or quote a familiar palindrome
- do not add punctuation or explanations outside the JSON
- every pair must be distinct

Intended scene: {intent}

The response is already prefilled with `{{"candidates":[`. Complete that JSON
object only: begin with the first candidate object, give exactly four objects,
and close the array and object. Do not repeat the prefix or add explanations.
"""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def request_json(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        HOST + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def parse_candidates(raw: str) -> tuple[list[dict] | None, str | None]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != CANDIDATES_PER_CALL:
        return None, f"need_exactly_{CANDIDATES_PER_CALL}_candidates"
    parsed = []
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str)
                                                for key in ("left", "right", "link")):
            return None, "candidate_schema_error"
        parsed.append({key: row[key] for key in ("left", "right", "link")})
    return parsed, None


def partial_result(model: str, metadata: dict, records: list[dict]) -> dict:
    return {
        "status": "incomplete_development_only_joint_proposal_run",
        "model_requested": model,
        "model_metadata": metadata,
        "intents": list(INTENTS),
        "intents_sha256": sha256_bytes("\n".join(INTENTS).encode()),
        "prompt_template": PROMPT,
        "candidates_per_call": CANDIDATES_PER_CALL,
        "calls_per_intent": REPLICATES_PER_INTENT,
        "generation_options": GENERATION_OPTIONS | {"thinking": False},
        "response_prefill": RESPONSE_PREFILL,
        "records": records,
    }


def complete_result(model: str, metadata: dict, records: list[dict]) -> dict:
    known_pairs = existing_v3_pairs()
    screened = []
    for record in records:
        if record["parsed"] is None:
            continue
        for row in record["parsed"]:
            screened.append({"intent_id": record["intent_id"],
                             "replicate": record["replicate"]}
                            | screen_candidate(**row, existing_pairs=known_pairs))
    unique = {}
    for row in screened:
        unique.setdefault((row["left_normalized"], row["right_normalized"]), row)
    accepted_rows = [row for row in unique.values() if accepted(row)]
    return partial_result(model, metadata, records) | {
        "status": "complete_mechanical_screen_not_readability_evidence",
        "parsed_calls": sum(row["parsed"] is not None for row in records),
        "total_calls": len(INTENTS) * REPLICATES_PER_INTENT,
        "screened_candidates": len(screened),
        "unique_candidates": len(unique),
        "accepted": accepted_rows,
        "success_gate": (
            "One distinct accepted pair is a material lead for a blinded human screen, not a claim "
            "of readability. Zero accepts rejects this joint-prompt/model/budget configuration only."
        ),
    }


def run(*, model: str, checkpoint: Path | None, resume: bool) -> dict:
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise ValueError("--resume requires an existing --checkpoint")
        prior = json.loads(checkpoint.read_text())
        expected = {"model_requested": model, "intents": list(INTENTS),
                    "prompt_template": PROMPT, "candidates_per_call": CANDIDATES_PER_CALL,
                    "calls_per_intent": REPLICATES_PER_INTENT,
                    "generation_options": GENERATION_OPTIONS | {"thinking": False},
                    "response_prefill": RESPONSE_PREFILL}
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint does not match the frozen joint proposal run")
        metadata = prior["model_metadata"]
        existing = {(row["intent_id"], row["replicate"]): row for row in prior["records"]}
    else:
        if checkpoint is not None and checkpoint.exists():
            raise ValueError("checkpoint already exists; use --resume")
        metadata = request_json("/api/show", {"name": model})
    records = []
    total = len(INTENTS) * REPLICATES_PER_INTENT
    for index, intent in enumerate(INTENTS, 1):
        intent_id = f"I{index:02d}"
        for replicate in range(REPLICATES_PER_INTENT):
            key = intent_id, replicate
            if key in existing:
                records.append(existing[key])
                continue
            prompt = PROMPT.format(intent=intent)
            completion = request_json("/api/chat", {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": RESPONSE_PREFILL},
                ],
                "stream": False,
                "options": GENERATION_OPTIONS | {"seed": 20260915 + replicate},
            })["message"]["content"]
            raw = RESPONSE_PREFILL + completion
            parsed, parse_error = parse_candidates(raw)
            records.append({"intent_id": intent_id, "intent": intent,
                            "replicate": replicate, "prompt": prompt,
                            "response_prefill": RESPONSE_PREFILL,
                            "raw_reply": raw, "parsed": parsed,
                            "parse_error": parse_error})
            if checkpoint is not None:
                checkpoint.write_text(json.dumps(partial_result(model, metadata, records), indent=2) + "\n")
            print(f"proposed {len(records)}/{total}: {intent_id}:{replicate}", flush=True)
    return complete_result(model, metadata, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(model=args.model, checkpoint=args.checkpoint, resume=args.resume)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "parsed_calls": result["parsed_calls"],
                      "total_calls": result["total_calls"],
                      "accepted": len(result["accepted"])}, indent=2))


if __name__ == "__main__":
    main()
