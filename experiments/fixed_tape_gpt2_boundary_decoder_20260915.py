"""Model-ranked word-boundary recovery on an immutable exact palindrome tape.

The tape is frozen before lexicalization.  A dynamic program enumerates
dictionary segmentations without grammar templates; GPT-2 only reranks those
complete segmentations.  It cannot alter, add, or remove a character, and a
second ASCII/two-pointer audit plus the shared admission gate is mandatory.
This is a repair probe, not a readability certificate.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import re
import sys
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "fixed-tape-gpt2-boundary-decoder-20260915"
SIGNATURE = "fixed-tape-gpt2-boundary-decoder|global-language-model-reranking|exact-immutable-tape|grammar-free-segmentation|independent-audit"
SOURCE_TAPE = "noitcanwonkneebyeknowohsothanitsoperawasseccasesutahtwonewenowthatusesaccessawarepostinahtoshowonkeybeenknownaction"
MIN_LETTERS = 39


def words_for_tape(tape: str) -> set[str]:
    # Include repository words and a deterministic high-frequency external
    # vocabulary.  The source tape, not the vocabulary, is the immutable input.
    local = {
        w.casefold() for w in (ROOT / "data" / "lexicon.txt").read_text().splitlines()
        if w.isascii() and w.isalpha()
    }
    common = {
        w.casefold() for w in top_n_list("en", 80_000)
        if w.isascii() and w.isalpha()
    }
    return {w for w in local | common if 2 <= len(w) <= 18 and w in tape}


def segmentations(tape: str, vocab: set[str], per_position: int = 180) -> list[tuple[float, tuple[str, ...]]]:
    """K-best exact segmentations; score is only a search prior."""
    n = len(tape)
    dp: list[list[tuple[float, tuple[str, ...]]]] = [[] for _ in range(n + 1)]
    dp[0] = [(0.0, ())]
    by_start: list[list[str]] = [[] for _ in range(n)]
    for i in range(n):
        by_start[i] = [tape[i:j] for j in range(i + 2, min(n, i + 18) + 1) if tape[i:j] in vocab]
    for i in range(n):
        if not dp[i]:
            continue
        for w in by_start[i]:
            j = i + len(w)
            gain = zipf_frequency(w, "en") + 0.55 * len(w)
            if len(w) <= 2:
                gain -= 3.0
            for old, path in dp[i]:
                # Repeated content words are not useful candidates, but retain
                # them in the diagnostic beam so the failure is auditable.
                repeat_penalty = -8.0 if w in path and len(w) > 2 else 0.0
                dp[j].append((old + gain + repeat_penalty, path + (w,)))
        if len(dp[i + 1]) > per_position:
            dp[i + 1] = heapq.nlargest(per_position, dp[i + 1], key=lambda x: x[0])
    return heapq.nlargest(per_position * 8, dp[n], key=lambda x: x[0])


def audit(text: str) -> dict:
    tape = normalize_letters(text)
    independent = "".join(re.findall(r"[a-z]", text.casefold()))
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_tape": tape,
        "independent_ascii_tape": independent,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_exact": bool(independent) and independent == independent[::-1],
        "tapes_equal": tape == independent,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def run() -> dict:
    vocab = words_for_tape(SOURCE_TAPE)
    paths = segmentations(SOURCE_TAPE, vocab)
    texts = [" ".join(path).capitalize() + "." for _score, path in paths]
    # Rank only complete, character-valid segmentations with the local model.
    lm_rows = []
    try:
        from llm_palindrome.lm_scoring import GPT2Scorer
        scorer = GPT2Scorer("gpt2", device="cpu")
        lm_rows = scorer.score_details(texts[:1200], batch_size=8)
    except Exception as exc:  # keep the experiment reproducible if model unavailable
        lm_rows = [{"per_letter": -999.0, "error": repr(exc)} for _ in texts[:1200]]
    rows = []
    for (prior, path), lm in zip(paths[:1200], lm_rows):
        text = " ".join(path).capitalize() + "."
        checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=220)
        a = audit(text)
        rows.append({
            "rendered": text,
            "words": list(path),
            "prior_score": prior,
            "lm": lm,
            "audit": a,
            "mechanical_checks": checks,
            "mechanically_admitted": a["independent_exact"] and all(checks.values()),
            "reader_status": "not_run; model score is diagnostic only",
        })
    rows.sort(key=lambda r: (r["mechanically_admitted"], r["lm"].get("per_letter", -999), r["prior_score"]), reverse=True)
    admitted = [r for r in rows if r["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not admitted else "exact_hits_pending_blinded_readers",
        "config": {
            "immutable_source_tape": SOURCE_TAPE,
            "source_letters": len(SOURCE_TAPE),
            "vocabulary_size": len(vocab),
            "k_best_per_position": 180,
            "model": "gpt2-local-cache",
            "grammar": "none; boundaries only",
        },
        "stats": {
            "complete_segmentations": len(paths),
            "scored_segmentations": len(rows),
            "mechanically_admitted": len(admitted),
            "reader_eligible": 0,
        },
        "rendered_candidates": rows[:80],
        "exact_candidates": admitted,
        "next_repair": "Use model-ranked boundary proposals as lexical hints, then add a typed valency transition check without changing the immutable tape.",
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_tape_sha256": hashlib.sha256(SOURCE_TAPE.encode()).hexdigest(),
            "catalogue_text_copied": False,
            "programmatic_readability_claim": False,
        },
    }


if __name__ == "__main__":
    out = run()
    path = ROOT / "runs" / "fixed-tape-gpt2-boundary-decoder-20260915.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "stats": out["stats"], "path": str(path)}, indent=2))
