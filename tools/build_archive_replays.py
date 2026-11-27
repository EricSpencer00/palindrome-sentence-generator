"""Save replays of the retired v1-v4 generators for the site.

Queries a local archive server (tools/archive.sh, PALINDROME_ARCHIVE=1) and
writes web/public/archive/vN.json: arrays of poster shapes, each a verified
letter palindrome. Stdlib only.

    python3 tools/build_archive_replays.py [base-url]
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8060"
OUT = Path(__file__).resolve().parent.parent / "web" / "public" / "archive"
FIELDS = ("left", "right", "center", "centerDisplay", "pivot", "pivotOdd",
          "promptCenter", "letters", "words")


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _alpha(w: str) -> int:
    return sum(c.isalpha() for c in w)


# Mirrors server/app.py _split_at_mirror / _display_pivot / _shape.
def split_at_mirror(words):
    n = sum(_alpha(w) for w in words)
    half, odd, start = n // 2, n % 2 == 1, 0
    for i, w in enumerate(words):
        end = start + _alpha(w)
        if not odd and end == half:
            return words[: i + 1], "", words[i + 1:], 0
        if (start <= half < end) if odd else (start < half < end):
            return words[:i], w, words[i + 1:], half - start
        start = end
    return list(words), "", [], 0


def display_pivot(display: str, before: int) -> int:
    seen = 0
    for i, ch in enumerate(display):
        if not ch.isalpha():
            continue
        if seen == before:
            return i
        seen += 1
    return len(display)


def shape(words) -> dict:
    left, center, right, pivot = split_at_mirror(words)
    n = len(letters(" ".join(words)))
    return {"left": left, "right": right, "center": center, "centerDisplay": center,
            "pivot": display_pivot(center, pivot), "pivotOdd": n % 2 == 1,
            "promptCenter": False, "letters": n, "words": len(words)}


def get(path: str, **params) -> str:
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode()


def sse_results(path: str, **params):
    for line in get(path, **params).splitlines():
        if line.startswith("data:"):
            frame = json.loads(line[5:])
            if frame.get("type") == "result":
                yield frame


def from_frame(frame: dict) -> dict:
    s = {k: frame[k] for k in FIELDS}
    s["promptCenter"] = False
    return s


def valid(s: dict) -> bool:
    tape = letters(" ".join(s["left"] + [s["center"]] + s["right"]))
    return bool(tape) and tape == tape[::-1] and len(tape) == s["letters"]


def collect(version: str, shapes, cap: int = 10) -> list:
    out, seen = [], set()
    for s, note in shapes:
        if not valid(s):
            print(f"  {version}: dropped a non-palindrome", file=sys.stderr)
            continue
        key = letters(" ".join(s["left"] + [s["center"]] + s["right"]))
        if key in seen:
            continue
        seen.add(key)
        entry = {**s, "version": version}
        if note:
            entry["note"] = note
        out.append(entry)
        if len(out) >= cap:
            break
    return out


def v1():
    for prompt in ["", "racecar", "level", "noon", "refer", "madam", "stats", "kayak", "civic"]:
        print(f"  v1 prompt={prompt!r}", file=sys.stderr)
        for f in sse_results("/api/generate", prompt=prompt, budget=16):
            yield from_frame(f), None


def v2():
    prompts = ["", "sea", "night", "river", "garden", "war", "music", "stone", "dawn", "bread"]
    for prompt in prompts:
        for f in sse_results("/api/v2/generate", prompt=prompt):
            yield from_frame(f), None
    for prompt in prompts[:4]:
        d = json.loads(get("/api/v2/paragraph", sentences=7, prompt=prompt))
        if d.get("letterPalindrome"):
            yield shape(d["text"].split()), None


def v3():
    for seed in range(1, 9):
        d = json.loads(get("/api/v3/composition", seed=seed, letters=400))
        yield shape(d["text"].split()), None


def v4():
    texts = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and " " in x and len(x) < 400:
            t = letters(x)
            # Reversed-fragment rows ("ll", "sselehtrev...") are quarantined gibberish.
            junk = any(len(w) > 1 and not re.search(r"[aeiouy]", w) or len(w) > 14
                       for w in re.findall(r"[a-z]+", x.lower()))
            if len(t) >= 10 and t == t[::-1] and not junk:
                texts.append(x)

    for path in ("/api/v4/candidate", "/api/v4/evidence"):
        walk(json.loads(get(path)))
    # Prefer the rendered, punctuated form of each letter tape.
    texts.sort(key=lambda t: (not t[:1].isupper(), -sum(not c.isalnum() and c != " " for c in t)))
    for t in texts:
        yield shape(t.split()), None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for version, source in (("v1", v1), ("v2", v2), ("v3", v3), ("v4", v4)):
        entries = collect(version, source())
        (OUT / f"{version}.json").write_text(json.dumps(entries, indent=1) + "\n")
        print(f"{version}: {len(entries)}")


if __name__ == "__main__":
    main()
