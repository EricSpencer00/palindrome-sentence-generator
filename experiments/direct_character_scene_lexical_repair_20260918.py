"""Seam-targeted lexical repair of an authored scene.

The preceding direct authoring lane produced intact prose but no exact tape.
This operator keeps the scene's semantic roles and asks for a coordinated
lexical repair at the live outer seam.  It is a new construction operator, not
another prompt-only resampling sweep.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "direct-character-scene-lexical-repair-20260918.json"
MODEL = "imetaexabeam/RhythmAI:27b"
HOST = "http://127.0.0.1:11434"
SEEDS = tuple(2032100001 + i for i in range(8))
SOURCE = (
    "As the rain began to fall, Sam saw the silver sparrow fly past the bay, and "
    "the bay’s spray kissed his face as he watched the bird slip back into the fog."
)


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
    content = [normalize(w) for w in words if len(normalize(w)) > 2]
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
        "intact_surface": bool(re.search(r"[.!?]$", text.strip())),
        "repeated_content": len(content) != len(set(content)),
        "self_palindromic_content_words": [w for w in content if w == w[::-1]],
    }


def clean(raw: str) -> str:
    text = raw.strip()
    fenced = re.search(r"```(?:text|english)?\s*\n?(.*?)```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    return " ".join(line.strip() for line in text.splitlines() if line.strip()).strip('"')


def request(seed: int) -> dict[str, object]:
    source_tape = normalize(SOURCE)
    mismatches = [
        (i, len(source_tape) - 1 - i, source_tape[i], source_tape[-1 - i])
        for i in range(len(source_tape) // 2)
        if source_tape[i] != source_tape[-1 - i]
    ][:16]
    prompt = f"""Revise the complete English scene below into one fresh, coherent scene of 80--180 letters after spaces and punctuation are removed.

Keep the semantic roles and causal event recoverable: rain begins, Sam sees a
silver sparrow near a bay, spray reaches Sam, and the bird disappears into fog.
Make coordinated lexical changes in at least two non-adjacent regions.  The
normalized letters must form an exact palindrome.  Mismatch pairs below are
diagnostic only; do not copy or reverse a finished tape.  Use ordinary prose,
normal punctuation, and no list, fragment, quotation, repeated phrase,
word-order-only mirror, self-palindromic-word trick, catalogue text, or
borrowed sentence.  Return only the revised prose.

Current scene:
{SOURCE}

First mismatch pairs (diagnostic): {mismatches}"""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.9, "num_predict": 520, "seed": seed},
    }
    req = urllib.request.Request(
        HOST + "/api/chat", data=json.dumps(body).encode(),
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
                "authoring": "seam-targeted lexical repair of direct authored scene",
                "source_experiment": "direct-character-scene-authoring-20260918",
                "source_sha256": hashlib.sha256(normalize(SOURCE).encode()).hexdigest(),
                "catalogue_imported": False,
                "finished_tape_reversed": False,
                "reader_certified": False,
            },
        }
    except Exception as exc:
        return {"seed": seed, "rendered": "", "error": repr(exc), "audit": audit("")}


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
        and not row["audit"]["repeated_content"]
        and not row["audit"]["self_palindromic_content_words"]
    ]
    ranked = sorted(rows, key=lambda row: (
        row["audit"]["exact"], -row["audit"]["mismatches"], row["audit"]["letters"]
    ), reverse=True)
    result = {
        "experiment_id": "direct-character-scene-lexical-repair-20260918",
        "signature": "direct-character-scene|seam-targeted-lexical-repair|parallel-seeded-authoring|independent-pointer-sha",
        "status": "completed_exact_candidate" if exact else "completed_no_exact_closure",
        "method": "seam-targeted lexical realization beam over a fixed authored scene's semantic roles",
        "source": SOURCE,
        "seeds": list(SEEDS),
        "rows": rows,
        "exact_candidates": exact,
        "best": ranked[0] if ranked else None,
        "next_repair": (
            "blind-reader test every exact survivor, then extend the scene"
            if exact else
            "replace prompt repair with a live character trie over role-compatible lexical alternatives at the first mismatch"
        ),
        "reader_gate": "closed; exactness and surface checks do not certify readability",
        "independent_audits": ["ASCII two-pointer mismatch scan", "forward/reverse SHA-256"],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    best = result["best"]
    print(json.dumps({
        "status": result["status"], "exact": len(result["exact_candidates"]),
        "best": best and best["rendered"], "best_audit": best and best["audit"],
    }))
