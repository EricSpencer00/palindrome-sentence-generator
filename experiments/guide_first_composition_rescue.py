"""Record a local model's development-only order decision for rescue blocks.

The input has both counter-orders of each fixed two-pair block.  The model may
choose an order only when it can name a concrete intended interpretation; it
may return ``neither``.  Its complete prompt, model metadata, and raw replies
are saved.  This is a proposal diagnostic, not a readability score or a
substitute for blinded human evaluation.
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


HOST = "http://localhost:11434"
PROMPT = """You are doing a development-only composition check for exact
palindromes. You may not change, omit, add, or reorder any word.

Two alternatives contain the same words in a different exact-palindrome order.
Choose one only if you can give a concrete one-sentence interpretation for it
as it stands. If neither supports any recoverable interpretation, choose
\"neither\". Do not reward length, symmetry, or unusual form.

Alternative A:
{a}

Alternative B:
{b}

Reply with exactly one JSON object:
{{"choice":"a"|"b"|"neither", "intended_interpretation":string|null,
  "reason":string}}
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


def parse_reply(raw: str) -> dict:
    """Parse one JSON decision without silently coercing malformed output."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model reply has no JSON object")
    value = json.loads(cleaned[start:end + 1])
    if value.get("choice") not in {"a", "b", "neither"}:
        raise ValueError("choice must be a, b, or neither")
    intended = value.get("intended_interpretation")
    if value["choice"] == "neither" and intended not in (None, ""):
        raise ValueError("neither must not claim an intended interpretation")
    if value["choice"] != "neither" and not isinstance(intended, str):
        raise ValueError("a selected order needs an intended interpretation")
    if not isinstance(value.get("reason"), str):
        raise ValueError("decision needs a textual reason")
    return {"choice": value["choice"], "intended_interpretation": intended,
            "reason": value["reason"]}


def blocks(materials: dict) -> list[dict]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in materials["variants"]:
        if row["condition"] not in {"order_a_outer", "order_b_outer"}:
            continue
        grouped.setdefault(row["block_id"], {})[row["condition"]] = row
    out = []
    for block_id, rows in sorted(grouped.items()):
        if set(rows) != {"order_a_outer", "order_b_outer"}:
            raise ValueError(f"{block_id} lacks a counter-order")
        out.append({"block_id": block_id, "a": rows["order_a_outer"],
                    "b": rows["order_b_outer"]})
    return out


def guide(materials_path: Path, model: str) -> dict:
    materials = json.loads(materials_path.read_text())
    metadata = request_json("/api/show", {"name": model})
    decisions = []
    for block in blocks(materials):
        prompt = PROMPT.format(a=block["a"]["plain"], b=block["b"]["plain"])
        reply = request_json("/api/chat", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": "low",
            "options": {"temperature": 0},
        })["message"]["content"]
        try:
            parsed = parse_reply(reply)
            parse_error = None
        except (ValueError, json.JSONDecodeError) as exc:
            parsed = None
            parse_error = f"{type(exc).__name__}: {exc}"
        decisions.append({
            "block_id": block["block_id"],
            "alternative_a_id": block["a"]["id"],
            "alternative_b_id": block["b"]["id"],
            "prompt": prompt,
            "raw_reply": reply,
            "parsed": parsed,
            "parse_error": parse_error,
        })
    return {
        "status": "development_only_not_human_readability_evidence",
        "materials_sha256": sha256(materials_path),
        "model_requested": model,
        "model_metadata": metadata,
        "prompt_template": PROMPT,
        "decisions": decisions,
        "interpretation": (
            "A selection is a frozen model-proposal decision, not a claim that the selected text is readable. "
            "A neither decision is evidence against this fixed material under this prompt."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = guide(args.materials, args.model)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    parsed = [row["parsed"] for row in result["decisions"] if row["parsed"] is not None]
    print(json.dumps({"output": str(args.output), "blocks": len(result["decisions"]),
                      "parseable": len(parsed),
                      "neither": sum(row["choice"] == "neither" for row in parsed)}, indent=2))


if __name__ == "__main__":
    main()
