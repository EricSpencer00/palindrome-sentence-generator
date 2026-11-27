"""Audit credited text without changing it or treating corpus text as novel.

Run: python tools/audit_credited_palindromes.py POSTS.json --out DIRECTORY
Writes audit.json and CREDITS.md. Input is a list of credited rows or an object
with a rows list. Word overlap measures spelling constraints, not readability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import quote, urlsplit


WORD = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
REQUIRED = ("text", "author", "source_url", "author_url")


def _valid_credit_url(value: str) -> bool:
    if not value or any(c.isspace() or ord(c) < 32 for c in value):
        return False
    try:
        parsed = urlsplit(value)
        # Accessing port also validates a supplied port number.
        _ = parsed.port
        return (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                and parsed.username is None and parsed.password is None)
    except ValueError:
        return False


def letter_tape(text: str) -> str:
    """Use the project's ASCII-letter rule, rejecting other alphabets first."""
    if any(c.isalpha() and not c.isascii() for c in text):
        raise ValueError("unsupported non-ASCII alphabetic character")
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def mirrored_word_spans(text: str) -> list[dict]:
    """Map word spans onto the mirrored tape; all intervals are half-open.

    Overlaps can include the same word at the center. Apostrophes inside a
    word do not split it, while spaces, hyphens and other punctuation do.
    """
    tape = letter_tape(text)
    spans = []
    cursor = 0
    for match in WORD.finditer(text):
        word = match.group()
        end = cursor + len(letter_tape(word))
        spans.append({"word": word, "span": [cursor, end]})
        cursor = end
    if cursor != len(tape):
        raise AssertionError("word spans do not cover the letter tape")
    result = []
    for index, word in enumerate(spans):
        start, end = word["span"]
        reflected = [len(tape) - end, len(tape) - start]
        overlaps = []
        for other_index, other in enumerate(spans):
            lo = max(reflected[0], other["span"][0])
            hi = min(reflected[1], other["span"][1])
            if lo < hi:
                overlaps.append({"word_index": other_index, **other,
                                 "overlap_span": [lo, hi], "letters": hi - lo})
        result.append({"word_index": index, **word, "letters": end - start,
                       "mirrored_span": reflected, "fanout": len(overlaps),
                       "overlaps": overlaps})
    return result


def audit(payload: dict | list) -> dict:
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("input must be a list or an object with a rows list")
    exact_by_tape: dict[str, dict] = {}
    invalid, unsupported = [], []
    exact_posts = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in REQUIRED):
            raise ValueError(f"row {row_index} requires string fields: {', '.join(REQUIRED)}")
        if not row["author"].strip():
            raise ValueError(f"row {row_index} requires a nonblank author credit")
        for field in ("source_url", "author_url"):
            if not _valid_credit_url(row[field]):
                raise ValueError(f"row {row_index} requires a valid HTTP(S) {field}")
        source = dict(row)
        unsupported_letters = sorted({c for c in row["text"] if c.isalpha() and not c.isascii()})
        if unsupported_letters:
            unsupported.append({**source, "reason": "unsupported_non_ascii_letters",
                                "unsupported_letters": unsupported_letters})
            continue
        tape = letter_tape(row["text"])
        if not tape or tape != tape[::-1]:
            mismatch = next((i for i in range(len(tape) // 2)
                             if tape[i] != tape[-i - 1]), None)
            invalid.append({**source, "normalized": tape, "letters": len(tape),
                            "reason": "empty_letter_tape" if not tape else "not_palindrome",
                            "first_mismatch": None if mismatch is None else {
                                "left_index": mismatch, "right_index": len(tape) - mismatch - 1,
                                "left": tape[mismatch], "right": tape[-mismatch - 1]}})
            continue
        exact_posts += 1
        if tape not in exact_by_tape:
            exact_by_tape[tape] = {
                "text": row["text"], "normalized": tape, "letters": len(tape),
                "sha256": sha256(tape.encode("ascii")).hexdigest(),
                "sources": [], "word_spans": mirrored_word_spans(row["text"]),
            }
        exact_by_tape[tape]["sources"].append(source)

    exact = list(exact_by_tape.values())
    examples = [{"text": item["text"], "source_url": item["sources"][0]["source_url"],
                 **word} for item in exact for word in item["word_spans"]]
    top = max(examples, key=lambda x: (x["fanout"], x["letters"]), default=None)
    histogram = Counter(word["fanout"] for word in examples)
    return {
        "metadata": {k: v for k, v in payload.items() if k != "rows"} if isinstance(payload, dict) else {},
        "summary": {
            "input_posts": len(rows), "exact_posts": exact_posts, "unique_exact": len(exact),
            "duplicate_exact_posts": exact_posts - len(exact),
            "non_palindromes": len(invalid), "unsupported": len(unsupported),
            "max_exact_letters": max((item["letters"] for item in exact), default=0),
            "word_fanout_histogram": dict(sorted(histogram.items())), "top_fanout_example": top,
            "notes": [
                "Punctuation, spaces, digits and case do not count; non-ASCII letters require separate review.",
                "Deduplication uses the letter tape; every source row and its original text are retained.",
                "Word diagnostics describe the first rendering of each distinct tape, not every spelling variant.",
                "Fanout counts words intersecting a reflected letter span, including self-overlap at the center.",
                "Fanout measures a spelling constraint, not readability or a proven cause of poor writing.",
                "Credit identifies the poster; original authorship and permission to reuse are not established.",
                "Credited corpus examples are not independently generated novelty.",
            ],
        },
        "exact": exact, "non_palindromes": invalid, "unsupported": unsupported,
    }


def _escape(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>])", r"\\\1", text).replace("\n", " ")


def _link(label: str, url: str) -> str:
    # Keep raw URLs in JSON, but never turn unsupported schemes into links.
    if urlsplit(url).scheme not in {"http", "https"}:
        return _escape(label)
    destination = quote(url, safe=":/?#[]@!$&'()*+,;=%")
    return f"[{_escape(label)}](<{destination}>)"


def credited_markdown(result: dict) -> str:
    stats = result["summary"]
    lines = ["# Credited palindrome corpus", "",
             f"{stats['unique_exact']} distinct exact palindromes from {stats['exact_posts']} posts; "
             f"{stats['non_palindromes']} non-palindromes and {stats['unsupported']} unsupported texts excluded.", "",
             "Credit identifies the poster, not verified original authorship. These are reference examples, not project-generated originals.", "",
             "## Verified exact palindromes", ""]

    def credit_lines(source):
        return [f"> {_escape(source['text'])}", "",
                f"{_link(source['author'], source['author_url'])} · {_link('Reddit post', source['source_url'])}", ""]

    for item in result["exact"]:
        lines.extend([f"### {item['letters']} letters", ""])
        for source in item["sources"]:
            lines.extend(credit_lines(source))
    if not result["exact"]:
        lines.extend(["None.", ""])
    lines.extend(["## Excluded: not exact", "",
                  "These original titles remain credited but are excluded from the exact corpus.", ""])
    for source in result["non_palindromes"]:
        lines.extend(credit_lines(source))
        reason = "No ASCII letters." if source["reason"] == "empty_letter_tape" else "The normalized letters do not match their reverse."
        lines.extend([reason, ""])
    if not result["non_palindromes"]:
        lines.extend(["None.", ""])
    lines.extend(["## Unsupported alphabetic characters", "",
                  "These original titles require separate alphabet handling; no exactness claim is made.", ""])
    for source in result["unsupported"]:
        lines.extend(credit_lines(source))
    if not result["unsupported"]:
        lines.extend(["None.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("posts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.posts.read_text(encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "CREDITS.md").write_text(credited_markdown(result), encoding="utf-8")
    print(json.dumps({k: v for k, v in result["summary"].items()
                      if k not in {"notes", "top_fanout_example", "word_fanout_histogram"}}))


if __name__ == "__main__":
    main()
