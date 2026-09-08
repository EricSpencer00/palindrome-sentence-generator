"""Calibrated binary sense filter for v3's Brown-shaped sentence pairs.

This is deliberately not a reward.  The model cannot choose among candidates
or improve a scalar score: it only decides whether each half independently
crosses a fixed sentence-level floor.  Positive and word-salad controls are
mixed into every batch; a batch with failed controls contributes no verdicts.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_palindrome.hierarchy import sentence_pairs
from llm_palindrome.validator import normalize
from server import v3

HOST = "http://localhost:11434/api/chat"

POSITIVE = [
    "The dog waited by the door.",
    "I saw the light across the river.",
    "Maria repaired the old radio.",
    "The level rose after the storm.",
    "Items draw award.",
]
NEGATIVE = [
    "Award items draw the.",
    "Quickly river of was blue the.",
    "Man a put if around yesterday.",
    "Levels perhaps table sings from.",
]

RUBRIC = """Decide whether each item is English a reader can interpret.

PASS when it states, asks, commands, labels, or compresses something
interpretable. Strained, poetic, elliptical, headline-like, archaic, or
unusual palindrome wording may pass. "Items draw award" is intentionally a
PASS: it is compressed, charming, and interpretable.

FAIL when it is only a list of words with no interpretable claim, or when it
cycles/repeats words mechanically (for example "do do do do"). Do not repair
it and do not demand ordinary prose style.

Return one compact JSON object mapping every ID to exactly true or false. No
explanation and no markdown.

ITEMS
{items}
"""


def chat(model: str, prompt: str) -> str:
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "stream": False, "keep_alive": "30m", "think": "low",
               "options": {"temperature": 0}}
    req = urllib.request.Request(HOST, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)["message"]["content"]


def parse_verdicts(raw: str, ids: set[str]) -> dict[str, bool]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {key: value for key, value in obj.items()
            if key in ids and isinstance(value, bool)}


def candidate_pairs(pairs_file: Path | None = None,
                    arm: str = "planned_join0") -> list[dict]:
    if pairs_file is not None:
        blob = json.loads(pairs_file.read_text())
        rows = blob[arm]["pairs"] if arm in blob else blob["pairs"]
        return [{"id": index, "left": row["left"], "right": row["right"],
                 "origin": str(pairs_file)}
                for index, row in enumerate(rows)]
    v3.ensure_loaded()
    if v3._load_error:
        raise RuntimeError(v3._load_error)
    table, shapes, trigrams = v3._tables
    raw = []
    seen = set()
    for row in v3._bank:
        if row["source"] != "generated":
            continue
        got = v3.harvest_pair(row["words"])
        if not got:
            continue
        left, right = got
        keys = normalize(" ".join(left)), normalize(" ".join(right))
        if keys[0] == keys[1] or any(key in seen for key in keys):
            continue
        seen.update(keys)
        raw.append((left, right, row["origin"]))
    shaped = sentence_pairs(raw, table, shapes, trigrams)
    return [{"id": index, "left": " ".join(left), "right": " ".join(right),
             "origin": origin} for index, (left, right, origin) in enumerate(shaped)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--limit", type=int, default=0,
                    help="screen only this many pairs; zero means all")
    ap.add_argument("--batch-pairs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=811)
    ap.add_argument("--pairs-file", type=Path,
                    help="aggregate JSON from sentence_plan_debug")
    ap.add_argument("--arm", default="planned_join0")
    ap.add_argument("--out", type=Path,
                    default=Path("runs/v3_sentence_filter.json"))
    args = ap.parse_args()

    pairs = candidate_pairs(args.pairs_file, args.arm)
    if args.limit:
        pairs = pairs[:args.limit]
    verdicts: dict[str, bool] = {}
    batches = []
    rng = random.Random(args.seed)
    for start in range(0, len(pairs), args.batch_pairs):
        group = pairs[start:start + args.batch_pairs]
        items = []
        for pair in group:
            items += [(f"p{pair['id']}L", pair["left"]),
                      (f"p{pair['id']}R", pair["right"])]
        controls = [(f"c{start}p{i}", text) for i, text in enumerate(POSITIVE)]
        controls += [(f"c{start}n{i}", text) for i, text in enumerate(NEGATIVE)]
        mixed = items + controls
        rng.shuffle(mixed)
        listing = "\n".join(f"{key}: {text}" for key, text in mixed)
        raw = chat(args.model, RUBRIC.format(items=listing))
        ids = {key for key, _ in mixed}
        got = parse_verdicts(raw, ids)
        positive_ok = all(got.get(key) is True for key, _ in controls
                          if "p" in key.split("c", 1)[1])
        negative_ok = all(got.get(key) is False for key, _ in controls
                          if "n" in key.split("c", 1)[1])
        complete = all(key in got for key in ids)
        valid = positive_ok and negative_ok and complete
        if valid:
            verdicts.update({key: got[key] for key, _ in items})
        batches.append({"start": start, "pairs": len(group), "valid": valid,
                        "positive_ok": positive_ok, "negative_ok": negative_ok,
                        "complete": complete, "raw": raw[:500]})
        print(f"batch {start // args.batch_pairs + 1}: "
              f"controls={'pass' if valid else 'FAIL'}", flush=True)

    accepted = []
    for pair in pairs:
        left = verdicts.get(f"p{pair['id']}L")
        right = verdicts.get(f"p{pair['id']}R")
        row = dict(pair) | {"left_pass": left, "right_pass": right,
                            "accepted": left is True and right is True}
        if row["accepted"]:
            accepted.append(row)
    result = {"model": args.model, "rubric": RUBRIC, "pairs": len(pairs),
              "valid_batches": sum(batch["valid"] for batch in batches),
              "batches": batches, "accepted": accepted,
              "accepted_count": len(accepted)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("model", "pairs", "valid_batches", "accepted_count")}),
          flush=True)


if __name__ == "__main__":
    main()
