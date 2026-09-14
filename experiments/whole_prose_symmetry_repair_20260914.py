"""Bounded whole-prose repair pilot for exact English palindromes.

The proposer always receives and returns a complete ordinary sentence.  It may
rewrite words, boundaries, syntax, and length while preserving a frozen scene.
The host records symmetry diagnostics and admits an output only when the
rendered letters pass the shared mechanical gate and an independent scan.
Intermediate prose is evidence about the construction trajectory, never a
readability certificate.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import re
import sys
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

HOST = "http://127.0.0.1:11434"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

SCENES = (
    ("A baker cools a tray of bread, notices a child waiting outside, and shares the warm loaf before closing the shop.",
     "a baker shares warm bread with a waiting child before closing the shop"),
    ("A gardener finds that the young tree is leaning after a storm, braces it with a spare board, and checks it again at dusk.",
     "a gardener braces a storm-bent tree and checks it at dusk"),
    ("A teacher sees that the class misunderstood the experiment, repeats the demonstration slowly, and lets the students explain the result.",
     "a teacher corrects a misunderstanding by repeating an experiment and hearing the students explain it"),
    ("A traveler misses the last train, calls a friend from the quiet station, and walks home after the friend brings a bicycle.",
     "a traveler gets home after missing a train and receiving a bicycle from a friend"),
)

INITIAL_PROMPT = """Write one original, grammatical English sentence of 100 to 160 letters.
It must express the supplied ordinary scene as one connected event, with normal
word choices and no list, quotation, famous phrase, repeated clause, or
palindrome wordplay. Return only JSON {{\"text\":\"...\"}}.

Scene: {scene}
Intent: {intent}
"""

REPAIR_PROMPT = """Rewrite the complete sentence below as one different, grammatical,
ordinary English sentence expressing the same scene and intent. You may change
every word, punctuation mark, boundary, syntax, and total length. Keep it a
single connected thought of 100 to 160 letters; do not use a list, quotation,
famous palindrome, repeated clause, or self-palindromic wordplay.

This is a constructive character constraint: letters counted from the two ends
should agree as often as possible. Repair several interacting mismatches at
once rather than copying or reversing the sentence. The host will check exact
symmetry; do not claim success. Return only JSON {{\"text\":\"...\"}}.

Frozen intent: {intent}
Current sentence: {text}
Normalized letters: {letters}
Mismatching mirrored positions (0-based): {mismatches}
"""


def request_json(path: str, body: dict[str, Any], *, timeout: float = 600) -> dict[str, Any]:
    request = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_text(raw: str) -> tuple[str | None, str | None]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    text = data.get("text") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        return None, "text_schema_error"
    return text.strip(), None


def mismatch_positions(letters: str) -> list[int]:
    return [i for i, (left, right) in enumerate(zip(letters, reversed(letters))) if left != right]


def independent_scan(text: str) -> dict[str, Any]:
    tape = "".join(ch.lower() for ch in text if ch.isascii() and ch.isalpha())
    return {
        "tape": tape,
        "letters": len(tape),
        "exact_letter_palindrome": tape == tape[::-1],
        "direct_symmetric_position_comparison": all(
            tape[i] == tape[-1 - i] for i in range(len(tape))
        ),
    }


def surface_diagnostics(text: str | None, intent: str) -> dict[str, Any]:
    if text is None:
        return {"parseable": False, "intent": intent}
    letters = normalize_letters(text)
    mismatches = mismatch_positions(letters)
    words = [word.lower() for word in WORD_RE.findall(text)]
    shared = mechanical_admission_checks(text, min_letters=100, max_letters=180)
    independent = independent_scan(text)
    return {
        "parseable": True,
        "text": text,
        "intent": intent,
        "letters": len(letters),
        "mismatch_count": len(mismatches),
        "mismatch_rate": (len(mismatches) / len(letters)) if letters else 1.0,
        "mismatch_positions": mismatches[:80],
        "one_sentence": sum(mark in text for mark in ".!?") <= 1,
        "word_count": len(words),
        "shared_mechanical_checks": shared,
        "independent_exactness": independent,
        "mechanically_eligible": all(shared.values()) and all(independent.values()),
        "human_readability": "not_certified",
    }


def better_diagnostic(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    """Return whether a parsed surface is a better exactness frontier point."""
    if not candidate.get("parseable"):
        return False
    if incumbent is None or not incumbent.get("parseable"):
        return True
    return (
        candidate.get("mismatch_count", 10**9),
        candidate.get("mismatch_rate", 1.0),
        abs(candidate.get("letters", 10**9) - 130),
    ) < (
        incumbent.get("mismatch_count", 10**9),
        incumbent.get("mismatch_rate", 1.0),
        abs(incumbent.get("letters", 10**9) - 130),
    )


def run(*, model: str, rounds: int, seed: int, request_timeout: float = 45,
        lineage_limit: int | None = None) -> dict[str, Any]:
    metadata = request_json("/api/show", {"name": model}, timeout=request_timeout)
    lineages: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    scenes = SCENES if lineage_limit is None else SCENES[:lineage_limit]
    for scene_index, (scene, intent) in enumerate(scenes):
        chain: list[dict[str, Any]] = []
        prompt = INITIAL_PROMPT.format(scene=scene, intent=intent)
        best_text: str | None = None
        best_diagnostics: dict[str, Any] | None = None
        best_round: int | None = None
        for round_index in range(rounds + 1):
            call_seed = seed + scene_index * 100 + round_index
            request = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # This pilot measures visible prose revisions; hidden chain of
                # thought is disabled so the fixed call budget cannot be
                # consumed before the JSON surface is returned.
                "think": False,
                "options": {"temperature": 0.75, "num_predict": 360, "seed": call_seed},
            }
            runtime_error = None
            response_error = None
            try:
                response = request_json("/api/chat", request, timeout=request_timeout)
                raw = response.get("message", {}).get("content", "")
                response_error = response.get("error")
            except Exception as error:  # preserve failed generator runs for replay and diagnosis
                raw = ""
                runtime_error = f"{type(error).__name__}:{error}"
            text, error = parse_text(raw)
            diagnostics = surface_diagnostics(text, intent)
            row = {
                "round": round_index,
                "seed": call_seed,
                "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                "raw_reply": raw,
                "text": text,
                "parse_error": error,
                "runtime_error": runtime_error,
                "response_error": response_error,
                "diagnostics": diagnostics,
            }
            chain.append(row)
            if better_diagnostic(diagnostics, best_diagnostics):
                best_text = text
                best_diagnostics = diagnostics
                best_round = round_index
            row["frontier"] = {
                "selected": best_round == round_index,
                "best_round": best_round,
                "best_mismatch_count": (best_diagnostics or {}).get("mismatch_count"),
            }
            if diagnostics.get("mechanically_eligible"):
                accepted.append({"scene_index": scene_index, "round": round_index, **diagnostics})
                break
            if best_text is None:
                prompt = INITIAL_PROMPT.format(scene=scene, intent=intent)
            else:
                prompt = REPAIR_PROMPT.format(
                    intent=intent,
                    text=best_text,
                    letters=normalize_letters(best_text),
                    mismatches=mismatch_positions(normalize_letters(best_text))[:80],
                )
        lineages.append({"scene_index": scene_index, "scene": scene, "intent": intent, "chain": chain})
    return {
        "status": "complete_whole_prose_symmetry_repair_pilot",
        "model_requested": model,
        "model_metadata": metadata,
        "config": {"rounds_per_lineage": rounds, "lineages": len(scenes), "seed": seed,
                   "request_timeout_seconds": request_timeout,
                   "frontier_policy": "retain lowest mismatch count, then mismatch rate, then distance from 130 letters"},
        "lineages": lineages,
        "accepted": accepted,
        "reader_gate": "No programmatic diagnostic certifies readability; any eligible surface requires randomized blinded human reading with intact prose and shuffled controls.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="imetaexabeam/RhythmAI:27b")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026091401)
    parser.add_argument("--request-timeout", type=float, default=45)
    parser.add_argument("--lineages", type=int, default=None,
                        help="run only the first N fixed scenes (for bounded pilots)")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.lineages is not None and not 1 <= args.lineages <= len(SCENES):
        parser.error(f"--lineages must be between 1 and {len(SCENES)}")
    result = run(model=args.model, rounds=args.rounds, seed=args.seed,
                 request_timeout=args.request_timeout, lineage_limit=args.lineages)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "accepted": len(result["accepted"])}, indent=2))


if __name__ == "__main__":
    main()
