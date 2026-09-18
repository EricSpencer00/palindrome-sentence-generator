"""Propose independent phrase lists for frozen bilateral infill moves.

The local model is a proposal source only.  It is called once for each side of
each frozen move without seeing the other side's prompt or candidates.  The
mechanical intersection in :mod:`experiments.joint_phrase_infill` decides which
proposals can be applied; the model never decides exactness, novelty, or
readability.  Every raw reply, parsed candidate, rejected candidate, and exact
intersection is retained.  Checkpointing permits a preempted local-model run to
resume without requerying a completed side.
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

from experiments.joint_phrase_infill import Seed, apply_replacements, intersect
from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.validator import normalize


HOST = "http://localhost:11434"
CANDIDATES_PER_SIDE = 8
PROMPT = """You are proposing a replacement for one marked span of a draft.
The surrounding letters and word order are fixed. The draft may be rough.

Propose exactly {count} distinct short English word sequences for [HOLE] that
would help this side express the intended meaning. Do not repair or discuss the
surrounding text. Use only lowercase ASCII letters and single spaces. Do not
include punctuation, numbering, explanations, or a palindrome counterpart.

Intended meaning: {intent}
Draft side:
{context}

Reply with exactly one JSON object:
{{"candidates":["first phrase", "second phrase", ...]}}
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_json(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        HOST + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def parse_candidates(raw: str, count: int = CANDIDATES_PER_SIDE) -> tuple[list[str] | None, str | None]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != count:
        return None, f"need_exactly_{count}_candidates"
    clean, seen = [], set()
    for phrase in candidates:
        if not isinstance(phrase, str) or not re.fullmatch(r"[a-z]+(?: [a-z]+)*", phrase):
            return None, "candidate_has_non_ascii_or_invalid_spacing"
        key = normalize(phrase)
        if key in seen:
            return None, "candidate_normalizations_must_be_distinct"
        seen.add(key)
        clean.append(phrase)
    return clean, None


def seed_lookup(materials: dict) -> dict[str, Seed]:
    return {row["seed_id"]: Seed(row["seed_id"], row["search_seed"],
                                  tuple(row["words"]), row["normalized"],
                                  row["intended_meaning"])
            for row in materials["seeds"]}


def partial_result(materials_path: Path, model: str, metadata: dict,
                   records: list[dict]) -> dict:
    return {
        "status": "incomplete_development_only_bilateral_proposal_run",
        "materials_sha256": sha256(materials_path),
        "model_requested": model,
        "model_metadata": metadata,
        "prompt_template": PROMPT,
        "candidates_per_side": CANDIDATES_PER_SIDE,
        "records": records,
    }


def complete_result(materials_path: Path, materials: dict, model: str,
                    metadata: dict, records: list[dict]) -> dict:
    lookup = seed_lookup(materials)
    by_move: dict[str, dict[str, dict]] = {}
    for record in records:
        by_move.setdefault(record["move_id"], {})[record["side"]] = record
    intersections = []
    for move in materials["moves"]:
        left = by_move.get(move["move_id"], {}).get("left")
        right = by_move.get(move["move_id"], {}).get("right")
        if not left or not right or left["parsed"] is None or right["parsed"] is None:
            continue
        for left_phrase, right_phrase in intersect(left["parsed"], right["parsed"]):
            candidate = apply_replacements(lookup[move["seed_id"]], move,
                                           left_phrase, right_phrase)
            intersections.append({
                "move_id": move["move_id"], "seed_id": move["seed_id"],
                "left_phrase": left_phrase, "right_phrase": right_phrase,
                "normalized_text": candidate,
                "exact": candidate == candidate[::-1],
                "novel_against_catalogue": is_novel_palindrome(candidate),
            })
    parsed_records = [row for row in records if row["parsed"] is not None]
    return {
        "status": "development_only_not_readability_evidence",
        "materials_sha256": sha256(materials_path),
        "model_requested": model,
        "model_metadata": metadata,
        "prompt_template": PROMPT,
        "candidates_per_side": CANDIDATES_PER_SIDE,
        "operator_controls": materials["operator_controls"],
        "records": records,
        "parsed_sides": len(parsed_records),
        "total_sides": len(materials["moves"]) * 2,
        "exact_intersections": intersections,
        "novel_exact_intersections": [row for row in intersections
                                      if row["novel_against_catalogue"]],
        "decision_rule": (
            "A novel exact intersection is a generated material lead, not a readable output. "
            "Zero intersections rejects this feasibility configuration only; it does not end the goal."
        ),
    }


def run(materials_path: Path, *, model: str, checkpoint: Path | None,
        resume: bool) -> dict:
    materials = json.loads(materials_path.read_text())
    existing: dict[tuple[str, str], dict] = {}
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise ValueError("--resume requires an existing --checkpoint")
        prior = json.loads(checkpoint.read_text())
        expected = {"materials_sha256": sha256(materials_path),
                    "model_requested": model, "prompt_template": PROMPT,
                    "candidates_per_side": CANDIDATES_PER_SIDE}
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint does not match frozen materials or settings")
        metadata = prior["model_metadata"]
        existing = {(row["move_id"], row["side"]): row for row in prior["records"]}
    else:
        if checkpoint is not None and checkpoint.exists():
            raise ValueError("checkpoint already exists; use --resume")
        metadata = request_json("/api/show", {"name": model})
    records = []
    total = len(materials["moves"]) * 2
    for move in materials["moves"]:
        for side, context_key in (("left", "left_context"), ("right", "right_context")):
            key = move["move_id"], side
            if key in existing:
                records.append(existing[key])
                continue
            prompt = PROMPT.format(count=CANDIDATES_PER_SIDE,
                                   intent=move["intended_meaning"],
                                   context=move[context_key])
            raw = request_json("/api/chat", {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": "low",
                "options": {"temperature": 0.7, "seed": 20260914},
            })["message"]["content"]
            parsed, parse_error = parse_candidates(raw)
            records.append({"move_id": move["move_id"], "seed_id": move["seed_id"],
                            "side": side, "prompt": prompt, "raw_reply": raw,
                            "parsed": parsed, "parse_error": parse_error})
            if checkpoint is not None:
                checkpoint.write_text(json.dumps(
                    partial_result(materials_path, model, metadata, records), indent=2) + "\n")
            print(f"proposed {len(records)}/{total}: {move['move_id']}:{side}", flush=True)
    return complete_result(materials_path, materials, model, metadata, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = run(args.materials, model=args.model, checkpoint=args.checkpoint,
                 resume=args.resume)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "parsed_sides": result["parsed_sides"],
                      "total_sides": result["total_sides"],
                      "exact_intersections": len(result["exact_intersections"]),
                      "novel_exact_intersections": len(result["novel_exact_intersections"])}, indent=2))


if __name__ == "__main__":
    main()
