"""Propose new English material, then repair its reversed letter tape jointly.

This is a construction method, not a readability metric.  A local language
model first proposes short, ordinary English clauses for a fixed micro-intent.
Code freezes each proposal's letters, reverses that exact tape, and enumerates
only dictionary-valid segmentations of the reverse.  The model may then select
one of those exact segmentations as a repair; it cannot add, delete, reorder,
or otherwise alter letters.  Hard gates verify both readings, non-repetition,
novelty, and length before a pair is emitted for a future blinded reader test.

The important difference from asking a model for two phrases is that every
repair is selected from a finite, mechanically generated bilateral lattice.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bidirectional_attested_span_mining import common_lexicon
from experiments.joint_dual_lexicalization import INTENTS, existing_v3_pairs, screen_candidate
from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.validator import is_palindrome, normalize


HOST = "http://127.0.0.1:11434"
PHRASES_PER_INTENT = 12
WORD_RE = re.compile(r"[a-z]+(?: [a-z]+)*")
PROPOSE_PROMPT = """Write exactly {count} distinct, original, ordinary English clauses for this micro-intent:
{intent}

Each clause must have 3--8 lowercase words and 15--30 ASCII letters after
spaces are removed. Use a concrete subject, verb, and object or complement.
Do not use a known palindrome, a quotation, a list, or a fragment. Do not
mention wordplay. Return exactly this JSON object and nothing else:
{{"phrases":["...", "...", ...]}}
"""
REPAIR_PROMPT = """A program reversed the letters of a proposed English clause. Choose at most one candidate
segmentation below that is itself a grammatical, ordinary English clause. The
letters and candidate word boundaries are fixed. Do not improve, paraphrase,
or invent words. Return exactly {{"chosen":"candidate"}} using one listed
candidate, or {{"chosen":null}} if none is an ordinary English clause.

Micro-intent: {intent}
Original clause: {left}
Reversed letter tape: {tape}
Exact candidate segmentations:
{candidates}
"""


def request_json(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        HOST + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def parse_phrases(raw: str, expected: int = PHRASES_PER_INTENT) -> tuple[list[str] | None, str | None]:
    """Parse an exact-sized proposal batch without accepting surrounding prose."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    phrases = data.get("phrases")
    if not isinstance(phrases, list) or len(phrases) != expected:
        return None, f"need_exactly_{expected}_phrases"
    if not all(isinstance(phrase, str) for phrase in phrases):
        return None, "phrase_schema_error"
    return [phrase.strip().lower() for phrase in phrases], None


def parse_choice(raw: str, candidates: set[str]) -> tuple[str | None, str | None]:
    """Accept only a literal choice from the program-generated lattice."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    chosen = data.get("chosen")
    if chosen is None:
        return None, None
    if not isinstance(chosen, str):
        return None, "choice_schema_error"
    chosen = chosen.strip().lower()
    if chosen not in candidates:
        return None, "choice_not_in_lattice"
    return chosen, None


def common_word_set(min_zipf: float) -> set[str]:
    """Use an explicit closed-class allowance with the conservative lexicon."""
    return common_lexicon(min_zipf) | {
        "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if",
        "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "we",
    }


def _trie(words: Iterable[str]) -> dict:
    root: dict = {}
    for word in words:
        if not word.isalpha():
            continue
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node.setdefault("", []).append(word)
    return root


def segmentations(tape: str, vocabulary: set[str], *, min_words: int = 3,
                  max_words: int = 8, limit: int = 64) -> list[str]:
    """Enumerate lexical word-breaks of one fixed tape, deterministically."""
    if not tape.isalpha():
        return []
    trie = _trie(vocabulary)
    at: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    at[0] = [()]
    for start in range(len(tape)):
        if not at[start]:
            continue
        node = trie
        for end in range(start, len(tape)):
            node = node.get(tape[end])
            if node is None:
                break
            if "" not in node:
                continue
            for prefix in at[start]:
                if len(prefix) >= max_words:
                    continue
                for word in node[""]:
                    at[end + 1].append(prefix + (word,))
                    if len(at[end + 1]) > limit * 4:
                        at[end + 1] = at[end + 1][:limit * 4]
    rows = sorted({" ".join(words) for words in at[len(tape)]
                   if min_words <= len(words) <= max_words})
    return rows[:limit]


def proposal_checks(phrase: str, vocabulary: set[str]) -> list[str]:
    """Reject malformed left material before any repair call."""
    if not WORD_RE.fullmatch(phrase):
        return ["ascii_word_form"]
    words = phrase.split()
    failures = []
    if not 3 <= len(words) <= 8:
        failures.append("word_band")
    if not 15 <= len(normalize(phrase)) <= 30:
        failures.append("letter_band")
    if any(word not in vocabulary for word in words):
        failures.append("common_lexicon_words")
    return failures


def run(*, model: str, min_zipf: float, per_intent: int) -> dict:
    """Run proposal -> exact lattice -> fixed-tape repair under recorded prompts."""
    vocabulary = common_word_set(min_zipf)
    metadata = request_json("/api/show", {"name": model})
    records: list[dict] = []
    for index, intent in enumerate(INTENTS, 1):
        prompt = PROPOSE_PROMPT.format(count=per_intent, intent=intent)
        response = request_json("/api/chat", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": "low",
            "options": {"temperature": 0.65, "num_predict": 1200, "seed": 2026091200 + index},
        })
        raw = response["message"]["content"]
        phrases, error = parse_phrases(raw, per_intent)
        records.append({"intent_id": f"I{index:02d}", "intent": intent,
                        "proposal_prompt": prompt, "proposal_raw": raw,
                        "phrases": phrases, "proposal_parse_error": error})

    repair_records: list[dict] = []
    for record in records:
        for left in record["phrases"] or []:
            failures = proposal_checks(left, vocabulary)
            tape = normalize(left)[::-1]
            lattice = segmentations(tape, vocabulary)
            row = {"intent_id": record["intent_id"], "intent": record["intent"],
                   "left": left, "reversed_tape": tape, "proposal_failures": failures,
                   "lattice": lattice}
            if failures or not lattice:
                row["repair_raw"] = None
                row["repair_parse_error"] = None
                row["chosen"] = None
                repair_records.append(row)
                continue
            repair_prompt = REPAIR_PROMPT.format(intent=record["intent"], left=left,
                                                  tape=tape,
                                                  candidates="\n".join(f"- {x}" for x in lattice))
            raw = request_json("/api/chat", {
                "model": model,
                "messages": [{"role": "user", "content": repair_prompt}],
                "stream": False,
                "think": "low",
                "options": {"temperature": 0.0, "num_predict": 200,
                            "seed": 2026092200 + len(repair_records)},
            })["message"]["content"]
            chosen, parse_error = parse_choice(raw, set(lattice))
            row |= {"repair_prompt": repair_prompt, "repair_raw": raw,
                    "repair_parse_error": parse_error, "chosen": chosen}
            repair_records.append(row)

    known_pairs = existing_v3_pairs()
    screened = [screen_candidate(row["left"], row["chosen"], row["intent"],
                                 existing_pairs=known_pairs)
                | {"intent_id": row["intent_id"]}
                for row in repair_records if row["chosen"] is not None]
    accepted = [row for row in screened if not row["rejection_codes"]]
    return {
        "status": "complete_bilateral_fixed_tape_repair_run",
        "model_requested": model,
        "model_metadata": metadata,
        "config": {"min_zipf": min_zipf, "phrases_per_intent": per_intent,
                   "source_proposal_seed": 2026091200,
                   "repair_selection_seed": 2026092200},
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": hashlib.sha256("\n".join(sorted(vocabulary)).encode()).hexdigest(),
        "proposal_records": records,
        "repair_records": repair_records,
        "proposal_count": sum(len(row["phrases"] or []) for row in records),
        "lattice_nonempty_count": sum(bool(row["lattice"]) for row in repair_records),
        "model_selected_count": sum(row["chosen"] is not None for row in repair_records),
        "screened": screened,
        "accepted": accepted,
        "reader_gate": (
            "An accepted pair is only a mechanically exact, novel material lead. It must be "
            "shown with intact prose controls and randomized blinded order to independent readers "
            "before any readability or coherence claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--min-zipf", type=float, default=3.6)
    parser.add_argument("--phrases-per-intent", type=int, default=PHRASES_PER_INTENT)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if not 1 <= args.phrases_per_intent <= 24:
        parser.error("--phrases-per-intent must be 1..24")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(model=args.model, min_zipf=args.min_zipf,
                 per_intent=args.phrases_per_intent)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out), "proposals": result["proposal_count"],
        "lattices": result["lattice_nonempty_count"],
        "model_selected": result["model_selected_count"],
        "accepted": len(result["accepted"]),
    }, indent=2))


if __name__ == "__main__":
    main()
