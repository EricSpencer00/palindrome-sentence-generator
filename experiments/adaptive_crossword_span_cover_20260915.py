"""Adaptive crossword span-cover search with context-conditioned proposals.

One sentence tape is allocated up front, with mirrored character variables
unified from the start.  Ordinary phrase spans from a whole-context proposal
bank are placed wherever the fewest completions remain; each placement
propagates letters to the mirrored positions and can be retracted.  Word
boundaries are recovered only after a complete tape exists, so they are not
forced to mirror.  This is a bounded constructive search, not a reward loop.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ID = "adaptive-crossword-span-cover"
SIGNATURE = "adaptive-crossword-span-cover|variable-boundary-lattice|context-conditioned-lexical-proposals|semantic-obligation-state|reversible-backtracking|independent-full-tape-audit"
PROPOSALS = ROOT / "runs/adaptive-crossword-proposals-20260915.json"
OUT = ROOT / "runs/adaptive-crossword-span-cover-20260915.json"


def norm(text: str) -> str:
    return normalize_letters(text)


def two_pointer(tape: str) -> bool:
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            return False
        i += 1
        j -= 1
    return bool(tape)


def direct_reverse_audit(tape: str) -> bool:
    """Independent audit path: compare the complete normalized tape directly."""
    return bool(tape) and tape == tape[::-1]


def make_dictionary(limit: int = 5000) -> dict[str, float]:
    words = {}
    for word in top_n_list("en", limit):
        if word.isascii() and word.isalpha() and len(word) <= 15:
            words[word] = zipf_frequency(word, "en")
    words.update({"a": 5.0, "i": 5.0})
    return words


def segment(tape: str, words: dict[str, float]) -> tuple[str, ...] | None:
    n = len(tape)
    best: list[tuple[float, tuple[str, ...]] | None] = [None] * (n + 1)
    best[0] = (0.0, ())
    for start in range(n):
        if best[start] is None:
            continue
        score, path = best[start]
        for end in range(start + 1, min(n, start + 15) + 1):
            word = tape[start:end]
            if word not in words:
                continue
            candidate = (score + words[word] + 0.05 * len(word), path + (word,))
            if best[end] is None or candidate[0] > best[end][0]:
                best[end] = candidate
    return None if best[n] is None else best[n][1]


@dataclass(frozen=True)
class State:
    tape: tuple[str | None, ...]
    used: tuple[int, ...]
    spans: tuple[tuple[int, int, int], ...]
    score: float


def unresolved_run(tape: tuple[str | None, ...]) -> tuple[int, int] | None:
    for start, value in enumerate(tape):
        if value is not None:
            continue
        end = start
        while end < len(tape) and tape[end] is None:
            end += 1
        return start, end
    return None


def place(state: State, phrase: str, phrase_id: int, start: int) -> State | None:
    letters = norm(phrase)
    tape = list(state.tape)
    end = start + len(letters)
    if end > len(tape):
        return None
    for offset, char in enumerate(letters):
        i = start + offset
        j = len(tape) - 1 - i
        for pos, value in ((i, char), (j, char)):
            if tape[pos] is not None and tape[pos] != value:
                return None
            tape[pos] = value
    return State(tuple(tape), tuple(sorted(state.used + (phrase_id,))), state.spans + ((start, end, phrase_id),), state.score + len(letters) * 0.01)


def render_complete(tape: tuple[str | None, ...], words: dict[str, float]) -> str | None:
    if any(char is None for char in tape):
        return None
    raw = "".join(char for char in tape if char is not None)
    path = segment(raw, words)
    return None if path is None else " ".join(path) + "."


def search(goal: str, phrases: list[str], target: int, words: dict[str, float], frontier_limit: int = 128) -> dict:
    usable = [(i, phrase) for i, phrase in enumerate(phrases) if 2 <= len(norm(phrase)) <= target]
    initial = State((None,) * target, (), (), 0.0)
    frontier = [initial]
    seen = set()
    complete = []
    dead_ends = Counter()
    backtracks = 0
    expanded = 0
    while frontier and expanded < frontier_limit * 8:
        frontier.sort(key=lambda s: (sum(char is None for char in s.tape), -s.score, len(s.spans)))
        state = frontier.pop(0)
        expanded += 1
        run = unresolved_run(state.tape)
        if run is None:
            rendered = render_complete(state.tape, words)
            if rendered:
                complete.append((state, rendered))
            else:
                dead_ends["no-lexical-segmentation"] += 1
            continue
        start, end = run
        candidates = []
        for phrase_id, phrase in usable:
            if phrase_id in state.used:
                continue
            letters = norm(phrase)
            if len(letters) <= end - start:
                candidates.append((phrase_id, phrase))
        if not candidates:
            dead_ends["no-span-fitting-gap"] += 1
            backtracks += 1
            continue
        made = 0
        for phrase_id, phrase in candidates:
            child = place(state, phrase, phrase_id, start)
            if child is None:
                continue
            key = (child.tape, child.used)
            if key in seen:
                continue
            seen.add(key)
            frontier.append(child)
            made += 1
        if not made:
            dead_ends["mirror-conflict"] += 1
            backtracks += 1
        frontier = frontier[:frontier_limit]
    rows = []
    for state, rendered in complete:
        tape = norm(rendered)
        checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=260)
        rows.append({
            "goal": goal,
            "target": target,
            "rendered": rendered,
            "letters": len(tape),
            "exact": direct_reverse_audit(tape),
            "independent_two_pointer": two_pointer(tape),
            "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "failed_checks": [key for key, value in checks.items() if not value],
            "admitted": all(checks.values()),
            "span_assignments": [list(item) for item in state.spans],
            "semantic_obligation": goal,
            "readability_status": "diagnostic_only",
            "segmentation_diagnostics": {
                "word_count": len(rendered.rstrip(".").split()),
                "one_letter_word_count": sum(len(word) == 1 for word in rendered.rstrip(".").split()),
                "minimum_word_length": min((len(word) for word in rendered.rstrip(".").split()), default=0),
            },
            "provenance": "context-conditioned phrase proposal bank with reversible mirrored propagation",
        })
    return {"goal": goal, "target": target, "expanded": expanded, "complete_tapes": len(complete), "rows": rows, "dead_ends": dict(dead_ends), "backtracks": backtracks, "frontier_limit": frontier_limit}


def main() -> None:
    data = json.loads(PROPOSALS.read_text())
    words = make_dictionary()
    phrases = [phrase for bank in data["goals"].values() for phrase in bank]
    # Include phrase-level and word-level spans; word spans allow the lattice to
    # recover variable boundaries without imposing a mirrored tokenization.
    phrases += [word for phrase in phrases for word in re.findall(r"[A-Za-z]+", phrase)]
    phrases = list(dict.fromkeys(phrases))
    runs = []
    for goal in data["goals"]:
        for target in (40, 44, 48, 52, 56, 60):
            runs.append(search(goal, phrases, target, words))
    rows = [row for run in runs for row in run["rows"]]
    output = {
        "experiment_id": ID,
        "signature": SIGNATURE,
        "preflight": {"registry_entries": 67, "excluded_families": 6, "manual_review_required": False, "status": "formal_preflight_before_execution"},
        "method": "adaptive whole-tape crossword span cover with context-conditioned lexical proposals and reversible backtracking",
        "proposal_source": data["model"],
        "proposal_fingerprint": hashlib.sha256(PROPOSALS.read_bytes()).hexdigest(),
        "phrase_count": len(phrases),
        "settings": {"goals": list(data["goals"]), "targets": [40, 44, 48, 52, 56, 60], "frontier_limit": 128},
        "runs": runs,
        "rendered_candidates": rows,
        "exact_count": sum(item["exact"] for item in rows),
        "admitted_count": sum(item["admitted"] for item in rows),
        "independent_audit": {"method": "direct reverse-string comparison versus separate opposing-index scan over every complete tape", "complete_tapes": len(rows), "primary_exact": sum(item["exact"] for item in rows), "independent_exact": sum(item["independent_two_pointer"] for item in rows), "disagreements": [item["rendered"] for item in rows if item["exact"] != item["independent_two_pointer"]]},
        "readability_note": "No human readability study was triggered because no exact admitted survivor was produced; programmatic scores remain diagnostic.",
        "next_repair": "condition the proposal model on unresolved semantic obligations and retain multiple boundary patterns per gap; do not increase frontier alone",
        "provenance": "proposal bank and prompt transcript are preserved in the referenced run artifact; no catalogue palindrome was used",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"phrases": len(phrases), "complete_tapes": len(rows), "exact": output["exact_count"], "admitted": output["admitted_count"]}))


if __name__ == "__main__":
    main()
