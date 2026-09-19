"""Post-hoc AI feedback for the v4 frontier.

This is deliberately a reader-facing diagnostic, not a search reward.  The
constructor never calls this evaluator while expanding states.  It is run
only after exact validation and asks a local model to score intact English,
scene coherence, and Shakespearean cadence while ignoring length.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.v4 import BEST_KNOWN_TEXT, DREAM_RSI_REPAIR_FRONTIER, independent_audit

RUN_ID = "v4-rlaif-frontier-eval-20260919"
MODEL = "gpt-oss:20b"
HOST = "http://127.0.0.1:11434/api/chat"

PROMPT = """You are a skeptical blinded English-prose evaluator.

Score this one passage on three independent 0-3 scales. Ignore its length,
palindrome status, and any punctuation trick. Judge the rendered text exactly
as shown:

1. intact_english: 0 word salad, 1 local phrases only, 2 understandable but
   strained, 3 natural connected English;
2. scene_coherence: 0 no recoverable scene, 1 scattered actors/objects,
   2 one recoverable scene, 3 a clear event with relations;
3. shakespearean_cadence: 0 broken, 1 mostly flat, 2 some dramatic movement,
   3 memorable dramatic movement and image.

Return JSON only, with numeric fields intact_english, scene_coherence,
shakespearean_cadence, and a short string repair.

PASSAGE: {text}"""


def ask(text: str, model: str = MODEL) -> dict[str, object]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
            "stream": False,
            "think": "low",
            "keep_alive": "30m",
            "options": {"temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(HOST, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        raw = json.load(response)["message"]["content"]
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"judge did not return JSON: {raw[:200]!r}")
    parsed = json.loads(match.group(0))
    return {
        "raw": raw,
        "scores": {
            key: int(parsed[key])
            for key in ("intact_english", "scene_coherence", "shakespearean_cadence")
        },
        "repair": str(parsed.get("repair", "")),
    }


def candidates() -> list[dict[str, object]]:
    rows = [
        {
            "role": "best_known_reader_plausible",
            "run_id": "half-tape-grammar-csp-20260919",
            "rendered": BEST_KNOWN_TEXT,
        }
    ]
    rows.extend(
        {
            "role": "dream_rsi_repair_frontier",
            "run_id": "dream-rsi-strict-phrase-bank-20260919",
            "rendered": row["rendered"],
        }
        for row in DREAM_RSI_REPAIR_FRONTIER
    )
    return rows


def run(model: str = MODEL) -> dict[str, object]:
    results = []
    for row in candidates():
        audit = independent_audit(str(row["rendered"]))
        feedback = ask(str(row["rendered"]), model=model)
        results.append({**row, "audit": audit, "ai_feedback": feedback})
    return {
        "experiment_id": RUN_ID,
        "model": model,
        "status": "diagnostic_only",
        "search_uses_feedback": False,
        "readability_certified": False,
        "blind_human_readers_required": True,
        "rubric": "0-3 intact English, scene coherence, Shakespearean cadence; length ignored",
        "candidates": results,
        "next_test": "randomized blinded intact-prose versus shuffled-control rating",
    }


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs" / f"{RUN_ID}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["candidates"]:
        print(row["role"], row["audit"]["letters"], row["ai_feedback"]["scores"])
    print(out)
