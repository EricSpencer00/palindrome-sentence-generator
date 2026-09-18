"""Run a recorded, calibrated development screen on frozen boundary material.

This is deliberately narrower than a readability evaluation.  A local model
sees two counter-orders of an exact two-pair block, may add presentation marks
to one complete word run, and chooses it only when it can state a concrete
interpretation.  It sees neither source-arm labels nor provenance.  Its reply
is rejected when it changes letters, word boundaries, or word order.

Six prose/shuffle controls use the same response protocol.  Candidate choices
are reported only when every control is parsed, display-preserving, and has its
pre-specified result.  Even then the output is a reproducible development
triage result -- not human readability evidence and not a selection result for
the paper.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.validator import normalize


HOST = "http://localhost:11434"

PROMPT = """You are screening fixed word runs in a development experiment.
You are not allowed to fix, add, omit, or reorder words. You may select one
alternative only when its exact words, in their existing order, can be shown
as a grammatical English thought with a recoverable subject or clear intent.
It may be strained, poetic, or archaic, but it cannot be a word list.

If one alternative qualifies, choose it and add only capitalization, spaces,
and punctuation to make its display clear. Your display must preserve every
word and its order exactly. State a concrete one-sentence interpretation.
If neither qualifies, choose ``neither`` and use null for both display and
interpretation. Do not reward symmetry, length, rarity, or palindrome form.

Alternative A:
{a}

Alternative B:
{b}

Reply with exactly one JSON object:
{{"choice":"a"|"b"|"neither", "display":string|null,
  "interpretation":string|null, "reason":string}}
"""


@dataclass(frozen=True)
class ScreenItem:
    item_id: str
    kind: str
    expected_choice: str | None
    a: str
    b: str
    source_arm: str | None = None
    block_id: str | None = None


# The controls use exactly the same answer shape and word-preservation check as
# candidates.  Positive choices must beat their own word shuffle; both strings
# in a negative control are word salad.
CONTROLS = (
    ScreenItem("P01", "positive_control", "a", "the dog waited by the door",
               "door the by waited dog the"),
    ScreenItem("P02", "positive_control", "a", "maria repaired the old radio",
               "radio old the repaired maria"),
    ScreenItem("P03", "positive_control", "a", "the level rose after the storm",
               "storm the after rose level the"),
    ScreenItem("N01", "negative_control", "neither", "quickly river of was blue the",
               "blue the quickly was river of"),
    ScreenItem("N02", "negative_control", "neither", "award items draw the",
               "river perhaps table sings from"),
    ScreenItem("N03", "negative_control", "neither", "yesterday put around if man a",
               "levels sings table perhaps from"),
)


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


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def parse_reply(raw: str, item: ScreenItem) -> tuple[dict | None, str | None]:
    """Parse and mechanically validate one response without repairing it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    choice = value.get("choice")
    if choice not in {"a", "b", "neither"}:
        return None, "invalid_choice"
    display, interpretation, reason = (value.get("display"),
                                       value.get("interpretation"),
                                       value.get("reason"))
    if not isinstance(reason, str) or not reason.strip():
        return None, "missing_reason"
    if choice == "neither":
        if display is not None or interpretation is not None:
            return None, "neither_must_not_supply_display_or_interpretation"
    else:
        if not isinstance(display, str) or not isinstance(interpretation, str):
            return None, "choice_needs_display_and_interpretation"
        source = item.a if choice == "a" else item.b
        if normalize(display) != normalize(source):
            return None, "display_changed_normalized_letters"
        if words(display) != words(source):
            return None, "display_changed_words_or_word_order"
    return {"choice": choice, "display": display,
            "interpretation": interpretation, "reason": reason}, None


def candidate_items(materials: dict) -> list[ScreenItem]:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in materials["variants"]:
        grouped.setdefault((row["source_arm"], row["block_id"]), {})[row["condition"]] = row
    out = []
    for (source_arm, block_id), rows in sorted(grouped.items()):
        if set(rows) != {"a", "b"}:
            raise ValueError(f"{source_arm}/{block_id} lacks an alternative")
        out.append(ScreenItem(
            item_id=f"{source_arm}:{block_id}", kind="candidate", expected_choice=None,
            a=rows["a"]["plain"], b=rows["b"]["plain"], source_arm=source_arm,
            block_id=block_id,
        ))
    return out


def partial_result(materials_path: Path, model: str, seed: int, metadata: dict,
                   records: list[dict]) -> dict:
    """Persist only completed model calls, so an interrupted screen resumes exactly."""
    return {
        "status": "incomplete_development_only_screen",
        "materials_sha256": sha256(materials_path),
        "model_requested": model,
        "model_metadata": metadata,
        "seed": seed,
        "prompt_template": PROMPT,
        "control_specification": [item.__dict__ for item in CONTROLS],
        "records": records,
    }


def screen(materials_path: Path, *, model: str, seed: int,
           checkpoint: Path | None = None, resume: bool = False) -> dict:
    materials = json.loads(materials_path.read_text())
    candidates = candidate_items(materials)
    ordered: list[ScreenItem] = [*CONTROLS, *candidates]
    random.Random(seed).shuffle(ordered)
    existing: dict[str, dict] = {}
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise ValueError("--resume requires an existing --checkpoint")
        prior = json.loads(checkpoint.read_text())
        expected = {"materials_sha256": sha256(materials_path),
                    "model_requested": model, "seed": seed,
                    "prompt_template": PROMPT}
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint does not match this frozen screen")
        metadata = prior["model_metadata"]
        existing = {row["item_id"]: row for row in prior["records"]}
    else:
        if checkpoint is not None and checkpoint.exists():
            raise ValueError("checkpoint already exists; use --resume to continue it")
        metadata = request_json("/api/show", {"name": model})
    records = []
    for item in ordered:
        if item.item_id in existing:
            records.append(existing[item.item_id])
            continue
        prompt = PROMPT.format(a=item.a, b=item.b)
        raw = request_json("/api/chat", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": "low",
            "options": {"temperature": 0},
        })["message"]["content"]
        parsed, parse_error = parse_reply(raw, item)
        records.append({
            "item_id": item.item_id,
            "kind": item.kind,
            "source_arm": item.source_arm,
            "block_id": item.block_id,
            "expected_choice": item.expected_choice,
            "alternative_a": item.a,
            "alternative_b": item.b,
            "prompt": prompt,
            "raw_reply": raw,
            "parsed": parsed,
            "parse_error": parse_error,
        })
        if checkpoint is not None:
            checkpoint.write_text(json.dumps(
                partial_result(materials_path, model, seed, metadata, records), indent=2) + "\n")
        print(f"screened {len(records)}/{len(ordered)}: {item.item_id}", flush=True)
    controls = [row for row in records if row["kind"] != "candidate"]
    calibration_valid = all(
        row["parse_error"] is None
        and row["parsed"]["choice"] == row["expected_choice"]
        for row in controls
    )
    by_arm: dict[str, dict[str, int]] = {}
    for row in records:
        if row["kind"] != "candidate" or row["parsed"] is None:
            continue
        arm = row["source_arm"]
        counts = by_arm.setdefault(arm, {"a": 0, "b": 0, "neither": 0,
                                         "unparsed": 0})
        counts[row["parsed"]["choice"]] += 1
    for row in records:
        if row["kind"] == "candidate" and row["parsed"] is None:
            by_arm.setdefault(row["source_arm"], {"a": 0, "b": 0,
                                                    "neither": 0, "unparsed": 0})["unparsed"] += 1
    return {
        "status": "development_only_not_human_readability_evidence",
        "materials_sha256": sha256(materials_path),
        "model_requested": model,
        "model_metadata": metadata,
        "seed": seed,
        "prompt_template": PROMPT,
        "control_specification": [item.__dict__ for item in CONTROLS],
        "calibration_valid": calibration_valid,
        "candidate_choice_counts_by_source_arm": by_arm,
        "records": records,
        "interpretation": (
            "When calibration is valid, a non-neither candidate is only a model-proposed "
            "lead for a later blinded human material screen. Neither a selection nor a "
            "negative result is a human readability finding."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="write a resumable record after every model call")
    parser.add_argument("--resume", action="store_true",
                        help="continue an exact matching checkpoint")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = screen(args.materials, model=args.model, seed=args.seed,
                    checkpoint=args.checkpoint, resume=args.resume)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output),
                      "calibration_valid": result["calibration_valid"],
                      "candidate_choice_counts_by_source_arm": result[
                          "candidate_choice_counts_by_source_arm"]}, indent=2))


if __name__ == "__main__":
    main()
