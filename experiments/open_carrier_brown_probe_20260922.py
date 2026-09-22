"""Remote Brown-span probe for open carriers around the live ``s`` seam.

This discovery script indexes ordinary forward corpus spans and intersects them
by ``reverse(T(Q)) = T(P) + s``.  Its output is diagnostic inventory, not a
reader-facing candidate: selected carriers are replayed through the repository
gate by the companion experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


ORDINARY_TWO = frozenset(
    "ah am an as at be by do go he if in is it me my no of oh on or ox so to up us we".split()
)
FUNCTION = frozenset(
    "a an the this that these those i me we us you he him she her it they them "
    "who which whose and or but if as while when after before because though "
    "of to in on at by for from with without near during is are was were be been "
    "do does did can could will would may might should have has had not no some "
    "any each every either neither another all both few many much more most several".split()
)
TOKEN = re.compile(r"^([A-Za-z]+(?:-[A-Za-z]+)?)/([^ ]+)$")


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
                tag = match.group(2).casefold()
                if not word.isalpha() or not word.isascii():
                    if chunk:
                        yield path.name, line_number, tuple(chunk)
                        chunk = []
                    continue
                chunk.append((word, tag))
            if chunk:
                yield path.name, line_number, tuple(chunk)


def ordinary(word: str, tag: str, counts: Counter) -> bool:
    return (
        counts[word] >= 3
        and not tag.startswith("np")
        and not tag.startswith("fw")
        and (len(word) > 1 or word in {"a", "i"})
        and (len(word) != 2 or word in ORDINARY_TWO)
    )


def role(tags: tuple[str, ...]) -> str:
    if any(tag.startswith("vb") or tag in {"md", "be", "bed", "ben", "ber", "bez"}
           for tag in tags):
        return "clause_or_vp"
    if tags and tags[0] in {"in", "to"}:
        return "pp"
    if tags and any(tag.startswith("nn") or tag.startswith("pp") for tag in tags):
        return "np_or_fragment"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brown_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=9)
    parser.add_argument("--min-letters", type=int, default=8)
    parser.add_argument("--max-letters", type=int, default=36)
    args = parser.parse_args()

    chunks = list(corpus_chunks(args.brown_root))
    counts = Counter(word for _path, _line, chunk in chunks for word, _tag in chunk)
    by_tape = defaultdict(dict)
    occurrences = 0
    for source, line, chunk in chunks:
        for start in range(len(chunk)):
            for width in range(args.min_words, args.max_words + 1):
                part = chunk[start:start + width]
                if len(part) != width:
                    break
                if not all(ordinary(word, tag, counts) for word, tag in part):
                    continue
                words = tuple(word for word, _tag in part)
                tape = "".join(words)
                if not args.min_letters <= len(tape) <= args.max_letters:
                    continue
                tags = tuple(tag for _word, tag in part)
                text = " ".join(words)
                key = (text, tags)
                occurrences += 1
                row = by_tape[tape].get(key)
                if row is None:
                    by_tape[tape][key] = {
                        "text": text,
                        "words": list(words),
                        "tags": list(tags),
                        "role": role(tags),
                        "count": 1,
                        "first_source": f"{source}:{line}",
                    }
                else:
                    row["count"] += 1

    matches = []
    for q_tape, q_rows in by_tape.items():
        if not q_tape.startswith("s"):
            continue
        p_tape = q_tape[::-1][:-1]
        if p_tape not in by_tape:
            continue
        for p in by_tape[p_tape].values():
            for q in q_rows.values():
                p_content = {w for w in p["words"] if w not in FUNCTION}
                q_content = {w for w in q["words"] if w not in FUNCTION}
                if p_content & q_content:
                    continue
                matches.append({
                    "p": p,
                    "q": q,
                    "p_tape": p_tape,
                    "q_tape": q_tape,
                    "equation_holds": q_tape[::-1] == p_tape + "s",
                    "carrier_letters": len(p_tape) + len(q_tape),
                    "combined_count": p["count"] + q["count"],
                    "same_role": p["role"] == q["role"],
                })
    matches.sort(key=lambda row: (
        -row["same_role"], -row["carrier_letters"], -row["combined_count"],
        row["p"]["text"], row["q"]["text"],
    ))
    payload = {
        "probe": "open-carrier-brown-probe-20260922",
        "equation": "reverse(T(Q)) = T(P) + s",
        "config": vars(args) | {"brown_root": str(args.brown_root), "out": str(args.out)},
        "stats": {
            "chunks": len(chunks),
            "word_types": len(counts),
            "accepted_span_occurrences": occurrences,
            "distinct_tapes": len(by_tape),
            "matches": len(matches),
            "same_role_matches": sum(row["same_role"] for row in matches),
        },
        "matches": matches[:500],
        "provenance": {
            "source": "NLTK Brown tagged corpus, forward contiguous spans only",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "finished_phrase_reversal_used_for_generation": False,
            "equation_index_only": True,
        },
    }
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in matches[:50]:
        print(row["carrier_letters"], row["combined_count"], row["same_role"],
              "P=", row["p"]["text"], "| Q=", row["q"]["text"])


if __name__ == "__main__":
    main()
