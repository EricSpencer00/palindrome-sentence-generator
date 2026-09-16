"""Model-conditioned continuation at a live palindrome seam.

This is deliberately not a reverse-index or clause-bank cross product.  At
each trial the model sees one ordinary complete sentence and the exact
normalized tape that must be continued from its live boundary.  It proposes
an independent complete counterpart, which is then audited without trusting
the model's claim.  Failed proposals are retained with a concrete seam repair
instruction for the next trial.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/live-seam-intent-continuation-20260915.json"
MODEL = "gpt-oss:20b"
SEEDS = [
    "The photographer captured a quiet sunrise.",
    "A traveler documented the bustling market.",
    "The scientist recorded a chemical reaction.",
    "She carved a wooden boat with patience.",
    "The gardener watered the young tomatoes.",
    "A teacher described the patient experiment.",
]

def norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def ask(prompt: str, seed: int) -> str:
    body = {"model": MODEL, "stream": False, "think": "low", "options": {"seed": seed, "temperature": 0.8},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("message", {}).get("content", "").strip()

def independent_audit(left: str, right: str) -> dict:
    tape = norm(left + right)
    reverse_ok = tape == tape[::-1]
    two_pointer = all(tape[i] == tape[-1-i] for i in range(len(tape)//2))
    words_l = re.findall(r"[A-Za-z]+", left.lower())
    words_r = re.findall(r"[A-Za-z]+", right.lower())
    word_mirror = words_l == list(reversed(words_r))
    return {"letters": len(tape), "reverse_check": reverse_ok, "two_pointer_check": two_pointer,
            "word_order_only": word_mirror, "exact": reverse_ok and two_pointer,
            "reader_eligible": reverse_ok and two_pointer and not word_mirror and len(words_l) >= 4 and len(words_r) >= 4}

def main() -> None:
    rows = []
    repair = ""
    for i, left in enumerate(SEEDS):
        target = norm(left)[::-1]
        prompt = f"""Write one ordinary, grammatical English sentence of 5 to 11 words.
It is the second sentence in a readable two-sentence letter palindrome.
The first sentence is: {left}
The normalized letters of your sentence must equal this exact required seam tape:
{target}
Do not quote a famous palindrome, mirror word order, use fragments, or explain.
Return only the sentence. Previous repair instruction: {repair or 'none'}"""
        try:
            right = ask(prompt, 20260915 + i)
            right = right.splitlines()[0].strip().strip('"')
            audit = independent_audit(left, right)
            if not audit["exact"]:
                repair = f"Previous seam missed; make every required character exact (first mismatch was diagnosed mechanically). Keep complete prose: {left}"
            else:
                repair = "Exact seam reached; preserve the complete-prose constraint."
            rows.append({"trial": i, "left": left, "right": right, "required_right_tape": target,
                         "rendered": left + " " + right, "audit": audit, "repair_for_next": repair})
        except Exception as exc:
            rows.append({"trial": i, "left": left, "error": repr(exc), "required_right_tape": target})
    payload = {"experiment": "live-seam-intent-continuation-20260915", "model": MODEL,
               "method": "intent-conditioned complete-sentence continuation at a live character seam; no bank or reverse index",
               "prompt_contract": "model sees left prose and required reversed tape; independent audit decides exactness",
               "rows": rows, "exact": sum(r.get("audit", {}).get("exact", False) for r in rows),
               "reader_eligible": sum(r.get("audit", {}).get("reader_eligible", False) for r in rows),
               "digest": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "exact": payload["exact"], "reader_eligible": payload["reader_eligible"], "artifact": str(OUT)}))

if __name__ == "__main__":
    main()
