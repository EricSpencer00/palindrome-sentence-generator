"""Direct whole-sentence palindrome drafting on the remote Qwen model.

This is a proposal lane, not a readability certifier: the model must draft a
complete sentence in one shot, and every output is independently audited for
letter-level exactness, provenance, and forbidden structural shortcuts.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/qwen-direct-palindrome-20260920.json"


def audit(text: str) -> dict:
    tape = re.sub(r"[^a-z]", "", text.casefold())
    exact = bool(tape) and tape == tape[::-1]
    return {
        "letters": len(tape),
        "two_pointer_exact": exact,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def structural_gate(text: str) -> dict:
    words = re.findall(r"[a-z]+", text.casefold())
    repeated = len(words) != len(set(words))
    self_pal = any(len(w) > 1 and w == w[::-1] for w in words)
    nested = False
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            if i == 0 and j == len(words):
                continue
            span = "".join(words[i:j])
            if len(span) > 1 and span == span[::-1]:
                nested = True
    return {
        "word_count": len(words),
        "no_repeated_words": not repeated,
        "no_self_palindromic_word": not self_pal,
        "no_nested_word_span": not nested,
        "mechanically_admitted": not repeated and not self_pal and not nested,
    }


def extract(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip().strip('`').strip()
        line = re.sub(r"^(?:\d+[.)]|[-*])\s*", "", line)
        line = re.sub(r"^(?:candidate|sentence)\s*\d*\s*[:\-]\s*", "", line, flags=re.I)
        if 1 < len(re.sub(r"[^a-z]", "", line.casefold())) <= 180:
            if line and line not in out:
                out.append(line)
    return out


def build_prompt(batch: int) -> str:
    return f"""Write 20 original, coherent English sentences (batch {batch}) that are exact letter-level palindromes when spaces and punctuation are removed. Aim for 45-120 letters and complete scenes with ordinary modern English. Do not use famous palindromes, names as filler, lists, word salad, repeated clauses, or a sentence plus its reversed copy. Each sentence must have a subject and finite verb and should make sense to a blinded reader. Output one sentence per line and nothing else. Think through the character equation before writing each sentence; silently discard anything that is not exact."""


def main() -> None:
    # Executed on mini-agent where Ollama is available; keep imports local so
    # the audit and tests remain runnable on the coordinator Mac.
    import urllib.request

    rows = []
    prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for batch in range(prompts):
        payload = {
            "model": "qwen2.5:3b",
            "prompt": build_prompt(batch),
            "stream": False,
            "options": {"temperature": 1.15, "top_p": 0.95, "num_predict": 1800},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=240) as response:
            raw = json.loads(response.read()).get("response", "")
        for text in extract(raw):
            a = audit(text)
            gate = structural_gate(text)
            rows.append({"batch": batch, "rendered": text, "audit": a, "structural_gate": gate,
                         "reader_status": "human-unreviewed; programmatic gate is diagnostic"})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    admitted = [r for r in exact if r["structural_gate"]["mechanically_admitted"]]
    result = {
        "experiment_id": "qwen-direct-palindrome-20260920",
        "method": "direct whole-sentence drafting with exact post-generation audit",
        "rows": rows,
        "exact": exact,
        "stats": {"batches": prompts, "rows": len(rows), "exact": len(exact),
                  "mechanically_admitted": len(admitted),
                  "longest_exact": max((r["audit"]["letters"] for r in exact), default=0)},
        "provenance": {"proposal_model": "qwen2.5:3b on Mac mini",
                       "catalogue_text": False, "finished_tape_reversal": False,
                       "post_hoc_repair": False,
                       "readability": "not certified; requires blinded human ratings"},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for r in sorted(exact, key=lambda x: -x["audit"]["letters"]):
        print(r["audit"]["letters"], r["structural_gate"]["mechanically_admitted"], r["rendered"])


if __name__ == "__main__":
    main()
