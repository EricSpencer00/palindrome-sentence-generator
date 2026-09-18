"""Fresh character-constrained scene authoring probe.

This lane asks the local model to author an intact scene under an exact
letter-tape constraint, then independently audits every response.  It is not a
certificate of readability: only a blinded reader gate can certify that.  The
probe is deliberately separate from the replay controller and uses parallel,
seeded requests so a failed family becomes a concrete repair target rather
than another duplicate sweep.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "direct-character-scene-authoring-20260918.json"
MODEL = "imetaexabeam/RhythmAI:27b"
HOST = "http://127.0.0.1:11434"
SEEDS = tuple(2032000001 + i for i in range(8))
PROMPT = """Write exactly one original, coherent English scene in one sentence or two naturally joined sentences.

Hard construction constraint: after lowercasing and removing every character
other than a-z, the resulting letter tape must be an exact palindrome, with
80--180 letters.  The scene must have a clear subject, ordinary action, and
recoverable meaning.  Use intact prose with normal punctuation.  Do not use a
list, fragment, quotation, repeated phrase, mirrored halves, word-order-only
symmetry, a self-palindromic word as a trick, a known catalogue palindrome, or
borrowed text.  This must be a fresh composition, not a reversed completed
tape.  Return only the prose and no explanation or labels."""


def normalize(text: str) -> str:
    return "".join(c for c in text.casefold() if "a" <= c <= "z")


def audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    mismatches = [
        (i, len(tape) - 1 - i, tape[i], tape[-1 - i])
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    word_tape = tuple(normalize(w) for w in words)
    content = [w.casefold() for w in words if len(normalize(w)) > 2]
    return {
        "letters": len(tape),
        "length_ok": 80 <= len(tape) <= 180,
        "exact": bool(tape) and not mismatches,
        "mismatches": len(mismatches),
        "mismatch_rate": len(mismatches) / max(1, len(tape) // 2),
        "first_mismatches": mismatches[:16],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
        "sha256_equal": hashlib.sha256(tape.encode()).hexdigest()
        == hashlib.sha256(tape[::-1].encode()).hexdigest(),
        "word_count": len(words),
        "intact_surface": bool(re.search(r"[.!?]$", text.strip()))
        and not text.lstrip().startswith(("-", "•", "1.")),
        "word_order_only_symmetry": bool(word_tape)
        and word_tape == tuple(reversed(word_tape)),
        "repeated_content": len(content) != len(set(content)),
        "self_palindromic_content_words": [w for w in content if w == w[::-1]],
    }


def clean(raw: str) -> str:
    text = raw.strip()
    fenced = re.search(r"```(?:text|english)?\s*\n?(.*?)```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and lines[0].lower().startswith(("here", "sure", "palindrome:")):
        lines = lines[1:]
    return " ".join(lines).strip().strip('"')


def request(seed: int) -> dict[str, object]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.88, "num_predict": 500, "seed": seed},
    }
    req = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            payload = json.load(response)
        raw = str(payload.get("message", {}).get("content", ""))
        rendered = clean(raw)
        return {
            "seed": seed,
            "rendered": rendered,
            "raw_response": raw,
            "audit": audit(rendered),
            "provenance": {
                "authoring": "fresh direct character-constrained scene request",
                "model": MODEL,
                "seed": seed,
                "catalogue_imported": False,
                "finished_tape_reversed": False,
                "word_order_only_symmetry": False,
                "reader_certified": False,
            },
        }
    except Exception as exc:
        return {"seed": seed, "error": repr(exc), "rendered": "", "audit": audit("")}


def run() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(request, seed) for seed in SEEDS]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["seed"])
    exact = [
        row for row in rows
        if row["audit"]["exact"] and row["audit"]["length_ok"]
        and row["audit"]["intact_surface"]
        and not row["audit"]["word_order_only_symmetry"]
        and not row["audit"]["repeated_content"]
        and not row["audit"]["self_palindromic_content_words"]
    ]
    ranked = sorted(rows, key=lambda row: (
        row["audit"]["exact"], -row["audit"]["mismatches"], row["audit"]["letters"]
    ), reverse=True)
    result = {
        "experiment_id": "direct-character-scene-authoring-20260918",
        "signature": "direct-character-constrained-scene|parallel-seeded-authoring|independent-pointer-sha|no-tape-reversal",
        "status": "completed_exact_candidate" if exact else "completed_no_exact_closure",
        "method": "parallel local-model authoring under an exact character-tape prompt, followed by independent mechanical and shortcut audits",
        "prompt": PROMPT,
        "seeds": list(SEEDS),
        "rows": rows,
        "exact_candidates": exact,
        "best": ranked[0] if ranked else None,
        "next_repair": (
            "blind-reader test every exact survivor, then extend its authored scene"
            if exact else
            "feed the first residual seam and its semantic scene roles into a constrained lexical realization beam; do not repeat this prompt-only sweep"
        ),
        "reader_gate": "closed; programmatic exactness and surface checks do not certify readability",
        "independent_audits": ["ASCII two-pointer mismatch scan", "forward/reverse SHA-256"],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    best = result["best"]
    print(json.dumps({
        "status": result["status"],
        "exact": len(result["exact_candidates"]),
        "best": best and best["rendered"],
        "best_audit": best and best["audit"],
    }))
