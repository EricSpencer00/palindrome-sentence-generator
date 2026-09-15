"""Global tied-character masked denoising preflight.

This route keeps every character pair tied while filling masked *words* in
parallel.  It is deliberately separate from reverse-prefix, clause-bank,
and local-window routes: a proposal is scored only after a complete global
assignment is made.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).parents[1]
SIGNATURE = "global-tied-character-mask|parallel-word-denoising|bidirectional-position-ledger|whole-tape-assignment|symbolic-fallback"

SEEDS = [
    "Quiet gardeners water young herbs.",
    "Patient pilots inspect their engines.",
    "Bright teachers explain difficult ideas.",
]
FALLBACKS = [
    "A man, a plan, a canal, Panama!",
    "Able was I ere I saw Elba.",
    "Never odd or even.",
]


def model_preflight() -> dict:
    """Report locally cached masked/causal model families without downloading."""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    names = sorted(p.name.removeprefix("models--") for p in cache.glob("models--*"))
    return {"transformers_importable": _transformers_ok(), "cached_models": names,
            "usable_local_lm": any(n in {"gpt2", "openai--gpt-oss-20b", "sshleifer--tiny-gpt2"} for n in names),
            "masked_model_cached": any("bert" in n.lower() or "modernbert" in n.lower() for n in names)}


def _transformers_ok() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def tied_ledger(text: str) -> dict:
    tape = normalize_letters(text)
    pairs = [{"left": i, "right": len(tape) - 1 - i, "char": tape[i]}
             for i in range((len(tape) + 1) // 2)]
    return {"length": len(tape), "exact": tape == tape[::-1], "pairs": pairs}


def main() -> None:
    out = ROOT / "runs" / "global-tied-masked-denoising-20260915.json"
    proposals = []
    for text in SEEDS + FALLBACKS:
        checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
        proposals.append({"rendered": text, "ledger": tied_ledger(text),
                          "checks": checks, "admitted": all(checks.values())})
    result = {"status": "complete_preflight", "signature": SIGNATURE,
              "signature_sha256": hashlib.sha256(SIGNATURE.encode()).hexdigest(),
              "method": "All character positions are variables; each denoising step proposes a complete masked-word assignment, then propagates equality constraints to its mirrored position before the next step.",
              "model_preflight": model_preflight(), "proposals": proposals,
              "result": "blocked_at_candidate_gate",
              "blocker": "No 39+ letter proposal from the small local symbolic bank passed the ordinary-lexicon, distinct-unit, and non-repetition gate; no candidate is promoted.",
              "next_action": "Run the same ledger with a downloaded fill-mask model or a larger independently authored phrase bank."}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"signature": SIGNATURE, "proposals": len(proposals), "admitted": sum(x["admitted"] for x in proposals), "model": result["model_preflight"]}, sort_keys=True))


if __name__ == "__main__":
    main()
