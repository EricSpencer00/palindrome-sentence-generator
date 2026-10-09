"""Run the existing corpus chunk constructor and time warmed requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.realtime_chunk_construct import ChunkConstructor


def independent_check(text: str) -> dict:
    unsupported = [c for c in text if c.isalpha() and not c.isascii()]
    tape = "".join(c.lower() for c in text if c.isascii() and c.isalpha())
    left, right = 0, len(text) - 1
    raw_exact = True
    while left < right:
        if not (text[left].isascii() and text[left].isalpha()):
            left += 1
            continue
        if not (text[right].isascii() and text[right].isalpha()):
            right -= 1
            continue
        if text[left].lower() != text[right].lower():
            raw_exact = False
            break
        left += 1
        right -= 1
    return {
        "letters": len(tape),
        "normalized_exact": bool(tape) and not unsupported and tape == tape[::-1],
        "raw_two_pointer_exact": bool(tape) and not unsupported and raw_exact,
        "tape_sha256": hashlib.sha256(tape.encode("ascii")).hexdigest(),
        "reverse_sha256": hashlib.sha256(tape[::-1].encode("ascii")).hexdigest(),
        "unsupported_alphabetic": unsupported,
    }


def run(out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    constructor = ChunkConstructor(root=ROOT)
    compile_seconds = time.perf_counter() - started
    configs = []
    for anchor in ("saw", "devil", "memos"):
        configs.append(dict(mode="refrain", anchor=anchor,
                            target_letters=1200, units=3, seed=0))
        for target in (400, 1200, 2500):
            configs.append(dict(mode="refrain", anchor=anchor,
                                target_letters=target, units=31, seed=0))
    configs.append(dict(mode="refrain", anchor="roman",
                        target_letters=1200, units=3, seed=0))
    for target in (400, 1200, 2500):
        for seed in (0, 9, 11):
            configs.append(dict(mode="paired", anchor=None,
                                target_letters=target, units=64, seed=seed))
    rows = []
    for index, config in enumerate(configs):
        at = time.perf_counter()
        result = constructor.construct(**config)
        row = {"id": f"C{index + 1:03}", "request": config,
               "request_seconds": time.perf_counter() - at, "result": result}
        if result.get("text"):
            row["independent_validation"] = independent_check(result["text"])
            check = row["independent_validation"]
            if not (check["normalized_exact"] and check["raw_two_pointer_exact"]
                    and check["letters"] <= config["target_letters"]):
                raise AssertionError(f"invalid construction {row['id']}")
        rows.append(row)
    timings = []
    for mode in ("refrain", "paired"):
        config = dict(mode=mode, target_letters=1200, units=31,
                      anchor="saw" if mode == "refrain" else None, seed=0)
        constructor.construct(**config)
        samples = []
        hashes = set()
        for _ in range(100):
            at = time.perf_counter()
            result = constructor.construct(**config)
            samples.append(time.perf_counter() - at)
            check = independent_check(result.get("text", ""))
            if not check["normalized_exact"] or not check["raw_two_pointer_exact"]:
                raise AssertionError("warm request did not produce an exact tape")
            hashes.add(check["tape_sha256"])
        timings.append({"request": config, "calls": len(samples),
                        "p50_ms": statistics.median(samples) * 1000,
                        "p95_ms": sorted(samples)[94] * 1000,
                        "max_ms": max(samples) * 1000,
                        "seconds": samples, "distinct_tapes": len(hashes),
                        "all_independently_exact": True})
    result = {
        "study": "existing-corpus-realtime-chunks-20261009",
        "runtime": {"hostname": platform.node(), "python": sys.version,
                    "platform": platform.platform()},
        "compile_seconds": compile_seconds, "inventory": constructor.inventory,
        "rows": rows, "warm_requests": timings,
        "total_seconds": time.perf_counter() - started,
        "scope": "Existing verified chunks, existing nesting and theme selection; no model training or new clause templates.",
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "constructions": len(rows),
                      "compile_seconds": compile_seconds,
                      "total_seconds": result["total_seconds"],
                      "warm_ms": {t["request"]["mode"]: t["p50_ms"] for t in timings}}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
