"""Mine live function-word residual cycles for a typed LIFO discourse stack.

This register is intentionally separate from the productive suffix ``s``
experiments.  The debt is an ordinary surface function word and remains live
while phrase frames are pushed; the word is emitted exactly once when the
grammar crosses the centre and the paired frames are popped in reverse order.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "function-word-residual-stack-20260922"
DEFAULT_OUTPUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
RESIDUALS = ("on", "no", "in", "as")
TOKEN = re.compile(r"^([A-Za-z]+(?:-[A-Za-z]+)?)/([^ ]+)$")
ORDINARY_TWO = frozenset(
    "ah am an as at be by do go he if in is it me my no of oh on or ox so to up us we".split()
)


def _tag(tag: str) -> str:
    return tag.casefold().split("-", 1)[0].split("+", 1)[0].rstrip("*")


def corpus_chunks(root: Path):
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name in {"README", "CONTENTS", "cats.txt"}:
            continue
        for line_number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            chunk = []
            for raw in line.split():
                match = TOKEN.match(raw)
                if not match:
                    if chunk:
                        yield path.name, line_number, tuple(chunk)
                        chunk = []
                    continue
                word = match.group(1).replace("-", "").casefold()
                tag = _tag(match.group(2))
                if not word.isalpha() or not word.isascii() or tag.startswith(("np", "fw")):
                    if chunk:
                        yield path.name, line_number, tuple(chunk)
                        chunk = []
                    continue
                chunk.append((word, tag))
            if chunk:
                yield path.name, line_number, tuple(chunk)


def _ordinary(word: str, count: int) -> bool:
    return (
        count >= 3
        and (len(word) > 1 or word in {"a", "i"})
        and (len(word) != 2 or word in ORDINARY_TWO)
    )


def _coarse_role(tags: tuple[str, ...]) -> str:
    if tags and tags[0] in {"in", "to"}:
        return "PP"
    if any(tag.startswith("vb") or tag in {"md", "be", "bed", "ben", "ber", "bez"}
           for tag in tags):
        return "VP_OR_CLAUSE"
    if any(tag.startswith("nn") or tag.startswith("pp") for tag in tags):
        return "NP"
    return "OTHER"


def build_inventory(corpus_dir: Path, *, max_words: int, max_letters: int):
    chunks = list(corpus_chunks(corpus_dir))
    counts = Counter(word for _source, _line, chunk in chunks for word, _tag in chunk)
    by_tape: dict[str, dict[tuple, dict]] = defaultdict(dict)
    accepted = 0
    digest = hashlib.sha256()
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file():
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
    for source, line, chunk in chunks:
        for start in range(len(chunk)):
            for width in range(1, max_words + 1):
                part = chunk[start:start + width]
                if len(part) != width:
                    break
                if not all(_ordinary(word, counts[word]) for word, _tag_value in part):
                    continue
                words = tuple(word for word, _tag_value in part)
                tape = "".join(words)
                if len(tape) > max_letters:
                    break
                tags = tuple(tag for _word, tag in part)
                key = (words, tags)
                accepted += 1
                if key not in by_tape[tape]:
                    by_tape[tape][key] = {
                        "words": list(words), "tags": list(tags),
                        "role": _coarse_role(tags), "count": 1,
                        "first_source": f"{source}:{line}",
                    }
                else:
                    by_tape[tape][key]["count"] += 1
    return by_tape, counts, {
        "chunks": len(chunks), "word_types": len(counts),
        "accepted_span_occurrences": accepted, "distinct_tapes": len(by_tape),
        "brown_raw_sha256": digest.hexdigest(),
    }


def solve_cycles(by_tape: dict[str, dict[tuple, dict]], residual: str) -> tuple[list[dict], dict]:
    rows = []
    counters = Counter()
    reverse_prefix = residual[::-1]
    for y_tape, y_surfaces in by_tape.items():
        counters["y_tapes"] += 1
        if not y_tape.startswith(reverse_prefix):
            continue
        counters["frontier_supported_y_tapes"] += 1
        stream = residual + y_tape[::-1]
        if not stream.endswith(residual):
            raise AssertionError((residual, y_tape))
        x_tape = stream[:-len(residual)]
        if x_tape not in by_tape:
            continue
        counters["equation_tape_hits"] += 1
        for x in by_tape[x_tape].values():
            for y in y_surfaces.values():
                counters["surface_pairs"] += 1
                x_content = {word for word in x["words"] if word not in ORDINARY_TWO}
                y_content = {word for word in y["words"] if word not in ORDINARY_TWO}
                if x_content & y_content:
                    counters["content_overlap_rejections"] += 1
                    continue
                symmetric = x["words"] == y["words"]
                finished_unit = x_tape + residual + y_tape
                finished_palindrome = finished_unit == finished_unit[::-1]
                # Every individual exact cycle is mechanically palindromic.
                # It is kept only as a transition and can never be accepted as
                # a finished unit in the rendered text.
                rows.append({
                    "residual": residual, "x_tape": x_tape, "y_tape": y_tape,
                    "x": x, "y": y,
                    "equation": {
                        "left": x_tape + residual,
                        "right": residual + y_tape[::-1],
                        "holds": x_tape + residual == residual + y_tape[::-1],
                    },
                    "self_or_symmetric": symmetric,
                    "finished_cycle_palindrome": finished_palindrome,
                })
    rows.sort(key=lambda row: (
        -(len(row["x_tape"]) + len(row["y_tape"])),
        -(row["x"]["count"] + row["y"]["count"]),
        row["x"]["words"], row["y"]["words"],
    ))
    return rows, dict(counters)


def solve_carriers(by_tape: dict[str, dict[tuple, dict]], residual: str) -> tuple[list[dict], dict]:
    """Find outer P/Q frames satisfying ``reverse(Q) = P + r``."""
    rows = []
    counters = Counter()
    reverse_prefix = residual[::-1]
    for q_tape, q_surfaces in by_tape.items():
        if not q_tape.startswith(reverse_prefix):
            continue
        counters["frontier_supported_q_tapes"] += 1
        p_tape = q_tape[::-1][:-len(residual)]
        if p_tape not in by_tape:
            continue
        counters["carrier_tape_hits"] += 1
        for p in by_tape[p_tape].values():
            for q in q_surfaces.values():
                counters["surface_pairs"] += 1
                p_content = {word for word in p["words"] if word not in ORDINARY_TWO}
                q_content = {word for word in q["words"] if word not in ORDINARY_TWO}
                if p_content & q_content:
                    counters["content_overlap_rejections"] += 1
                    continue
                rows.append({
                    "residual": residual, "p_tape": p_tape, "q_tape": q_tape,
                    "p": p, "q": q,
                    "equation": {
                        "reverse_q": q_tape[::-1], "p_plus_residual": p_tape + residual,
                        "holds": q_tape[::-1] == p_tape + residual,
                    },
                })
    rows.sort(key=lambda row: (
        -(len(row["p_tape"]) + len(row["q_tape"])),
        -(row["p"]["count"] + row["q"]["count"]),
        row["p"]["words"], row["q"]["words"],
    ))
    return rows, dict(counters)


def run(corpus_dir: Path, *, max_words: int = 5, max_letters: int = 24) -> dict:
    started = time.monotonic()
    by_tape, _counts, inventory = build_inventory(
        corpus_dir, max_words=max_words, max_letters=max_letters,
    )
    residual_rows = []
    for residual in RESIDUALS:
        cycles, stats = solve_cycles(by_tape, residual)
        carriers, carrier_stats = solve_carriers(by_tape, residual)
        residual_rows.append({
            "residual": residual, "stats": stats,
            "cycles": cycles[:200], "cycle_count": len(cycles),
            "carrier_stats": carrier_stats,
            "carriers": carriers[:200], "carrier_count": len(carriers),
        })
    return {
        "experiment_id": EXPERIMENT_ID,
        "phase": "phrase-frontier-probe",
        "decision": "Can a common function word remain as the sole nonempty LIFO debt and be realized once inside a complete connected clause longer than 44 letters?",
        "acceptance_gate": {
            "exact": True, "minimum_letters_exclusive": 44,
            "residual_surface_count": 1, "residual_has_ordinary_function_role": True,
            "complete_clause": True, "agreement": True, "valency": True,
            "connected_event_links": True, "fresh_content_lemmas": True,
            "proper_span_mask": True, "complementary_boundary_mask": True,
            "self_or_symmetric_cycles": False, "finished_cycle_units": False,
            "catalogue_text": False, "fragments": False,
        },
        "fixed_conditions": {
            "residuals": list(RESIDUALS), "stack": "strict LIFO",
            "equation": "T(x) r = r reverse(T(y))",
            "max_phrase_words": max_words, "max_phrase_letters": max_letters,
            "finished_tape_reversal": False, "posthoc_repair": False,
        },
        "inventory": inventory, "residual_frontiers": residual_rows,
        "stats": {"elapsed_seconds": round(time.monotonic() - started, 3)},
        "provenance": {
            "host": os.uname().nodename, "python": sys.version.split()[0],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brown", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-words", type=int, default=5)
    parser.add_argument("--max-letters", type=int, default=24)
    args = parser.parse_args()
    payload = run(args.brown, max_words=args.max_words, max_letters=args.max_letters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "inventory": payload["inventory"],
        "residuals": [
            {"residual": row["residual"], "cycle_count": row["cycle_count"],
             "stats": row["stats"]}
            for row in payload["residual_frontiers"]
        ],
        "elapsed_seconds": payload["stats"]["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
