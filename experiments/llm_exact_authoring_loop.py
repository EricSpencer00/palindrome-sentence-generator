"""Directly author whole English palindromes with mechanical repair feedback.

The local model authors a single complete sentence rather than a reflected
catalogue, word-order mirror, or post-hoc resegmentation.  After every attempt,
code supplies the exact normalized mismatch and rejects it unless the whole
rendered sentence is a novel 100--180-letter palindrome with no repeated or
individually palindromic word units.  The model may revise text, but it never
decides whether its letters qualify.  A surviving result is only a candidate
for a blinded reader screen, never an automatic readability result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.validator import is_palindrome, normalize
from server.v3 import real_words


HOST = "http://127.0.0.1:11434"
WORD_RE = re.compile(r"[A-Za-z]+")
PROMPT = """Author one NEW complete, ordinary English sentence whose letters are an exact palindrome.

Hard requirements:
- after lowercasing and removing every non-letter, it has 100--180 letters and reads identically backwards;
- it is one grammatical sentence with a concrete subject and verb, not a list or fragment;
- it uses no repeated word and no individually palindromic word (including a, I, did, noon);
- it is not a quotation, a famous palindrome, a reflected/repeated unit, or word-order-only symmetry;
- construct the whole sentence, including any word boundary that crosses the palindrome centre; do not
  mirror word order or hide letters in punctuation.

Work out the letter reversal privately. Return only JSON: {{"text":"your candidate"}}.
"""
REPAIR = """Your previous candidate failed a strict mechanical palindrome check.

Previous text: {text}
Normalized letters: {letters}
Required reverse: {reverse}
First mismatch index: {mismatch}

Write a substantially repaired NEW complete ordinary-English sentence. It must satisfy every hard
requirement in the original instruction; do not explain. Return only {{"text":"..."}}.
"""


def request_json(path: str, body: dict) -> dict:
    request = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def parse_text(raw: str) -> tuple[str | None, str | None]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "text_schema_error"
    return text.strip(), None


def first_mismatch(letters: str) -> int | None:
    for index, (left, right) in enumerate(zip(letters, reversed(letters))):
        if left != right:
            return index
    return None


def checks(text: str) -> dict[str, bool]:
    words = [word.lower() for word in WORD_RE.findall(text)]
    shared = mechanical_admission_checks(
        text,
        local_catalogue=set(json.loads((ROOT / "data" / "known_palindromes.json").read_text())),
        min_letters=100,
        max_letters=180,
    )
    return shared | {
        "one_sentence": sum(mark in text for mark in ".!?") <= 1,
        "exact_palindrome": shared["exact_letter_palindrome"],
        "lexicon_words": bool(words and real_words(words)),
        "no_repeated_words": shared["distinct_words"],
        "no_self_palindromic_word_units": shared["no_self_palindromic_word"],
        "novel_catalogue": shared["local_catalogue_absent"] and is_novel_palindrome(text),
    }


def author(*, model: str, attempts: int, repairs: int) -> dict:
    metadata = request_json("/api/show", {"name": model})
    records: list[dict] = []
    accepted: list[dict] = []
    for attempt in range(attempts):
        prompt = PROMPT
        chain = []
        for repair in range(repairs + 1):
            raw = request_json("/api/chat", {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # gpt-oss consumes its response budget in hidden reasoning when
                # ``think=False`` is supplied through this local runtime.  A
                # bounded low-thinking mode leaves enough tokens for the
                # required machine-readable candidate.
                "think": "low",
                "options": {"temperature": 0.85, "num_predict": 600,
                            "seed": 2026093000 + 100 * attempt + repair},
            })["message"]["content"]
            text, error = parse_text(raw)
            letters = normalize(text or "")
            gate = checks(text) if text is not None else {}
            row = {"repair_round": repair, "prompt": prompt, "raw_reply": raw,
                   "text": text, "parse_error": error, "letters": letters,
                   "checks": gate,
                   "rejection_codes": ([key for key, value in gate.items() if not value]
                                       if gate else [error])}
            chain.append(row)
            if text is not None and not row["rejection_codes"]:
                accepted.append({"attempt": attempt, "text": text,
                                 "letters": len(letters), "checks": gate})
                break
            prompt = REPAIR.format(text=text or "(unparseable)", letters=letters,
                                   reverse=letters[::-1], mismatch=first_mismatch(letters))
        records.append({"attempt": attempt, "chain": chain})
    return {
        "status": "complete_direct_whole_sentence_authoring_run",
        "model_requested": model,
        "model_metadata": metadata,
        "config": {"attempts": attempts, "repairs_per_attempt": repairs,
                   "seeds": [2026093000 + 100 * n for n in range(attempts)]},
        "records": records,
        "accepted": accepted,
        "reader_gate": (
            "Mechanical admission does not establish readability. Each candidate requires a frozen, "
            "randomized blinded reader screen with intact prose and matched shuffle controls."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--repairs", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = author(model=args.model, attempts=args.attempts, repairs=args.repairs)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "accepted": len(result["accepted"])}, indent=2))


if __name__ == "__main__":
    main()
