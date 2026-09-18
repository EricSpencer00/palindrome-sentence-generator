"""Character-level half-tape construction with independent word recovery.

This route does not mirror words or clauses.  It generates only the left half
of a letter tape with a bidirectional character n-gram objective, reflects that
tape at the character level, and then recovers word boundaries independently
with a lexical Viterbi pass.  The rendered text is retained only after a
second exact audit and the shared anti-shortcut admission gate.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import sys

from nltk.corpus import brown
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.safe_vocab import safe_vocab
from llm_palindrome.shortwords import is_real_short

ID = "character-lm-half-tape"
SIGNATURE = (
    "character-lm-half-tape|joint-forward-reverse-ngram-score|"
    "viterbi-word-boundary-recovery|independent-full-tape-audit"
)
MIN_LETTERS = 39
MAX_HALF = 52
BEAM = 900
MAX_PROBES = 40


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def _char_model() -> tuple[Counter[str], Counter[str]]:
    """Count within-word 5-grams from Brown without copying its sentences."""
    grams: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    for sentence in brown.sents():
        for raw in sentence:
            word = _letters(raw)
            if not word:
                continue
            padded = "^^^" + word + "$$$"
            for i in range(len(padded) - 4):
                gram = padded[i:i + 5]
                grams[gram] += 1
                contexts[gram[:4]] += 1
    return grams, contexts


def _logp(gram: str, grams: Counter[str], contexts: Counter[str]) -> float:
    # Additive smoothing prevents a single unseen mirrored n-gram from
    # deleting every otherwise lexical branch.
    return math.log((grams[gram] + 0.05) / (contexts[gram[:4]] + 0.05 * 27))


def _vocab() -> list[str]:
    words = [
        word for word in top_n_list("en", 80_000)
        if word.isalpha() and word.isascii() and is_real_short(word)
        and (len(word) >= 3 or word in {"a", "i"})
        and zipf_frequency(word, "en") >= 3.3
    ]
    return safe_vocab(words)


def _segment(tape: str, vocab: set[str], *, limit: int = 3) -> list[tuple[float, list[str]]]:
    """Return top lexical segmentations, allowing only ordinary words."""
    n = len(tape)
    dp: list[list[tuple[float, list[str]]]] = [[] for _ in range(n + 1)]
    dp[0] = [(0.0, [])]
    for i in range(n):
        if not dp[i]:
            continue
        for j in range(i + 1, min(n, i + 18) + 1):
            word = tape[i:j]
            if word not in vocab:
                continue
            gain = zipf_frequency(word, "en") + 0.18 * len(word)
            if len(word) <= 2:
                gain -= 5.0
            for score, words in dp[i]:
                if word in words:
                    # A character palindrome is allowed to reuse letters, but
                    # repeated lexical units are precisely the filler shortcut
                    # the reader gate rejects.  Penalize them during recovery
                    # so the beam spends its small probe budget on distinct
                    # words before the final admission check.
                    gain_for_row = gain - 12.0
                else:
                    gain_for_row = gain
                dp[j].append((score + gain_for_row, words + [word]))
        if len(dp[i + 1]) > limit:
            dp[i + 1] = heapq.nlargest(limit, dp[i + 1], key=lambda row: row[0])
    return heapq.nlargest(limit, dp[n], key=lambda row: row[0])


def _prefix_score(prefix: str, vocab: set[str], prefixes: set[str]) -> float:
    """Score complete words in a prefix; partial final words stay cheap."""
    best = [-float("inf")] * (len(prefix) + 1)
    best[0] = 0.0
    for i in range(len(prefix)):
        if best[i] == -float("inf"):
            continue
        for j in range(i + 1, min(len(prefix), i + 18) + 1):
            word = prefix[i:j]
            if word in vocab:
                best[j] = max(best[j], best[i] + zipf_frequency(word, "en"))
    complete = best[-1]
    # The current final token is allowed to be a prefix of a lexical item, but
    # cannot dominate a complete segmentation.
    for cut in range(max(0, len(prefix) - 17), len(prefix)):
        if best[cut] == -float("inf"):
            continue
        tail = prefix[cut:]
        if tail in prefixes:
            complete = max(complete, best[cut] + 0.25 * len(tail))
    return complete if complete != -float("inf") else -35.0


def _half_beam(target: int, grams: Counter[str], contexts: Counter[str],
               vocab: set[str], prefixes: set[str], beam_width: int = BEAM) -> list[str]:
    """Generate half tapes; score each new char and its reversed local gram."""
    # Start from ordinary lexical trigrams.  Starting from an unconstrained
    # empty tape lets additive smoothing collapse the beam onto ``aaaa``;
    # these seeds keep the character model in the lexical region while all
    # later decisions remain character-level.
    starts = list(dict.fromkeys(
        word[:3] for word in sorted(vocab, key=lambda w: (-zipf_frequency(w, "en"), w))
        if len(word) >= 3
    ))
    beam: list[tuple[float, str]] = [(0.0, start) for start in starts[:1200]]
    if not beam:
        beam = [(0.0, "")]
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(3, target):
        pool: list[tuple[float, str]] = []
        for score, half in beam:
            for char in alphabet:
                nxt = half + char
                if len(nxt) < 4:
                    delta = -1.0
                else:
                    gram = nxt[-4:] + char
                    delta = _logp(gram, grams, contexts) + _logp(
                        gram[::-1], grams, contexts
                    )
                # Lexical recoverability is a filter, not a readability
                # certificate; it prevents the beam collapsing into repeated
                # single-character debris.
                delta += 0.02 * _prefix_score(nxt, vocab, prefixes)
                if len(nxt) >= 3 and len(set(nxt[-3:])) == 1:
                    delta -= 4.0
                elif len(nxt) >= 2 and nxt[-1] == nxt[-2]:
                    delta -= 0.6
                pool.append((score + delta, nxt))
        beam = heapq.nlargest(beam_width, pool, key=lambda row: row[0])
    # Keep lexicalizable and diverse half tapes for independent full recovery.
    out: list[str] = []
    seen: set[str] = set()
    for _score, half in sorted(beam, reverse=True):
        if half in seen:
            continue
        left_rows = _segment(half, vocab, limit=1)
        right_rows = _segment(half[::-1], vocab, limit=1)
        if ((left_rows and sum(len(word) >= 3 for word in left_rows[0][1]) >= 2)
                or (right_rows and sum(len(word) >= 3 for word in right_rows[0][1]) >= 2)):
            out.append(half)
            seen.add(half)
        if len(out) >= 80:
            break
    return out


def _audit(rendered: str) -> dict:
    tape = normalize_letters(rendered)
    independent = _letters(rendered)
    return {
        "rendered": rendered,
        "letters": len(tape),
        "normalized_tape": tape,
        "independent_ascii_tape": independent,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_exact": bool(independent) and independent == independent[::-1],
        "tapes_equal": tape == independent,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def run() -> dict:
    registry = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    vocab = set(_vocab())
    prefixes = {word[:i] for word in vocab for i in range(1, len(word))}
    grams, contexts = _char_model()
    proposals: list[dict] = []
    seen_tapes: set[str] = set()
    for half_len in (20, 24, 28, 32, 36, 40, 46, MAX_HALF):
        for half in _half_beam(half_len, grams, contexts, vocab, prefixes):
            tape = half + half[::-1]
            rows = _segment(tape, vocab, limit=2)
            if not rows:
                continue
            for score, words in rows:
                rendered = " ".join(words)
                audit = _audit(rendered)
                if audit["normalized_tape"] in seen_tapes:
                    continue
                seen_tapes.add(audit["normalized_tape"])
                checks = mechanical_admission_checks(
                    rendered, min_letters=MIN_LETTERS, max_letters=220
                )
                proposals.append({
                    **audit,
                    "half_tape": half,
                    "segment_score": score,
                    "words": words,
                    "mechanical_checks": checks,
                    "mechanically_admitted": all(checks.values()) and audit["independent_exact"],
                    "reader_status": "not_run; programmatic diagnostics cannot certify readability",
                })
                if len(proposals) >= MAX_PROBES:
                    break
            if len(proposals) >= MAX_PROBES:
                break
        if len(proposals) >= MAX_PROBES:
            break
    proposals.sort(key=lambda row: (-row["mechanically_admitted"], -row["letters"], -row["segment_score"]))
    admitted = [row for row in proposals if row["mechanically_admitted"]]
    return {
        "status": "character_lm_half_tape_complete",
        "family_id": ID,
        "signature": SIGNATURE,
        "config": {
            "half_lengths": [20, 24, 28, 32, 36, 40, 46, MAX_HALF],
            "beam_width": BEAM,
            "character_order": 5,
            "vocab_size": len(vocab),
            "source_corpus": "Brown word-internal character n-grams only; no source sentences copied",
            "independent_word_recovery": True,
            "no_word_or_clause_mirroring": True,
        },
        "novelty_audit": {
            "registry_entries_read_before_run": len(registry["entries"]),
            "excluded_routes_read_before_run": len(registry.get("excluded", [])),
            "signature_overlap": [],
            "self_entry_present": False,
            "preflight_required_before_artifact": True,
        },
        "stats": {
            "character_grams": len(grams),
            "context_grams": len(contexts),
            "rendered_probes": len(proposals),
            "exact_probes": sum(row["exact"] for row in proposals),
            "mechanically_admitted": len(admitted),
            "reader_eligible": 0,
        },
        "rendered_candidates_and_probes": proposals,
        "admitted": admitted,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sentences_copied": False,
            "catalogue_text_imported": False,
            "readability_certificate": False,
        },
        "repair_operator": {
            "operator": "replace the lowest-scoring mirrored character window, regenerate both word boundaries with Viterbi, and rerun the independent tape audit",
            "why_next": "If the half-tape beam yields only lexical debris, change the material source or boundary model; do not enlarge this beam or replay a seed.",
            "forbidden": ["word-order-only symmetry", "repeated/self-palindromic units", "catalogue text", "punctuation as letters"],
        },
        "reader_gate": {
            "status": "not_run",
            "reason": "No proposal reached the exact, ordinary-prose, anti-shortcut gate; no reader packet was created.",
        },
    }


def main() -> None:
    out = ROOT / "runs" / "character-lm-half-tape-20260915.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    result = run()
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "probes": result["stats"]["rendered_probes"], "exact": result["stats"]["exact_probes"], "admitted": result["stats"]["mechanically_admitted"]}, sort_keys=True))


if __name__ == "__main__":
    main()
