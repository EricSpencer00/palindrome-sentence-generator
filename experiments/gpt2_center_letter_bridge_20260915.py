"""Odd-tape center-letter bridge for topic-conditioned palindrome search.

The even-only GPT-2 half decoder missed the construction used by several
longer palindromes: a single center letter can belong to the first word on the
right.  This run samples a natural left clause, tries each alphabetic center,
and independently segments ``center + reverse(left_tape)`` with a strict
lexical vocabulary.  The center is not copied from a right phrase and every
closure is checked by two independent exact audits.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gpt2-center-letter-bridge-20260915.json"
EXPERIMENT_ID = "gpt2-center-letter-bridge-20260915"
SIGNATURE = (
    "gpt2-center-letter-bridge|odd-tape-center-insertion|"
    "topic-conditioned-half-generation|independent-right-grammar-decoding|"
    "exact-audit"
)
MIN_LETTERS = 39
MAX_LETTERS = 220

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import (
    ORDINARY_TWO_LETTER_WORDS,
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
)
from llm_palindrome.lexicon import is_real_word, load_lexicon
from llm_palindrome.respace import respace_k, unigram_score
from experiments.gpt2_topic_half_decoder_20260915 import (
    PROMPTS,
    _clean_generation,
    _generate,
)


def _strict_vocab() -> frozenset[str]:
    from wordfreq import top_n_list

    lexicon = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    words = set()
    for word in top_n_list("en", 100_000):
        if not word.isascii() or not word.isalpha():
            continue
        if len(word) >= 3 and is_real_word(word, lexicon):
            words.add(word)
        elif word in ORDINARY_TWO_LETTER_WORDS or word in {"a", "i"}:
            words.add(word)
    # These function words are ordinary clause material even when a frequency
    # list omits a spelling; no abbreviation or catalogue vocabulary is added.
    words.update("a an the some one this that my our your i we you they he she it me us no not be do can and or but if as of to in on at by for with from near over under".split())
    return frozenset(words)


def _two_pointer(tape: str) -> dict:
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left": i, "right": j, "a": tape[i], "b": tape[j]})
        i += 1
        j -= 1
    return {"exact": bool(tape) and not mismatches,
            "pairs_checked": len(tape) // 2,
            "mismatches": mismatches[:5]}


def _word_order_mirror(words: tuple[str, ...]) -> bool:
    normalized = tuple(normalize_letters(word) for word in words)
    return bool(normalized) and normalized == tuple(word[::-1] for word in reversed(normalized))


def run(*, samples_per_prompt: int = 64, max_rows: int = 160) -> dict:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
    model.eval()
    vocab = _strict_vocab()
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    seen_tapes: set[str] = set()

    for prompt_index, prompt in enumerate(PROMPTS):
        generated = _generate(model, tokenizer, prompt,
                              samples=samples_per_prompt,
                              seed=20260915 + 100 + prompt_index)
        for sample_index, left_text in enumerate(generated):
            stats["model_samples"] += 1
            left_words = tuple(tokenize(_clean_generation(left_text)))
            left_tape = normalize_letters(left_text)
            if not (4 <= len(left_words) <= 10 and 18 <= len(left_tape) <= 90):
                stats["left_shape_reject"] += 1
                continue
            hit_for_sample = False
            for center in "abcdefghijklmnopqrstuvwxyz":
                target = center + left_tape[::-1]
                readings = respace_k(target, vocab, k=40)
                stats["center_decode_calls"] += 1
                if not readings:
                    continue
                for right_list in readings:
                    right_words = tuple(right_list)
                    if len(right_words) < 3:
                        continue
                    if _word_order_mirror(left_words + right_words):
                        stats["word_order_shortcut_reject"] += 1
                        continue
                    content = [word for word in left_words + right_words
                               if word not in {"a", "an", "the", "some", "one", "this", "that", "my", "our", "your", "i", "we", "you", "they", "he", "she", "it", "me", "us", "no", "not", "be", "do", "can", "and", "or", "but", "if", "as", "of", "to", "in", "on", "at", "by", "for", "with", "from", "near", "over", "under"}]
                    if len(content) != len(set(content)):
                        stats["duplicate_content_reject"] += 1
                        continue
                    right_text = " ".join(right_words)
                    rendered = left_text.rstrip(".!? ,;: ") + "; " + right_text + "."
                    tape = normalize_letters(rendered)
                    if len(tape) < MIN_LETTERS or len(tape) > MAX_LETTERS or tape in seen_tapes:
                        continue
                    seen_tapes.add(tape)
                    two_pointer = _two_pointer(tape)
                    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
                    row = {
                        "rendered": rendered,
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "center_letter": center,
                        "left_words": list(left_words),
                        "right_words": list(right_words),
                        "letters": len(tape),
                        "normalized_tape": tape,
                        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
                        "exact": bool(tape) and tape == tape[::-1],
                        "independent_two_pointer": two_pointer,
                        "mechanical_checks": checks,
                        "mechanically_admitted": bool(tape) and tape == tape[::-1] and two_pointer["exact"] and all(checks.values()),
                        "lexical_score_diagnostic": unigram_score(left_words + right_words),
                        "reader_status": "not_run; strict lexical decoding is not human readability evidence",
                        "provenance": {
                            "model": "gpt2",
                            "prompt": prompt,
                            "source_sentences_copied": False,
                            "center_inserted_before_reverse_tape": True,
                            "reverse_words_emitted": False,
                        },
                    }
                    rows.append(row)
                    stats["exact"] += int(row["exact"])
                    stats["mechanically_admitted"] += int(row["mechanically_admitted"])
                    hit_for_sample = True
                    if len(rows) >= max_rows:
                        break
                if len(rows) >= max_rows:
                    break
            if not hit_for_sample and len(probes) < 160:
                probes.append({
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "left": left_text,
                    "left_letters": len(left_tape),
                    "center_letters_tried": 26,
                    "status": "no-strict-independent-right-reading",
                })
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break

    rows.sort(key=lambda row: (-row["mechanically_admitted"], -row["letters"], -row["lexical_score_diagnostic"]))
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "A GPT-2 topic-conditioned left clause is paired with an independently "
            "decoded right tape formed as center-letter + reverse(left).  The odd "
            "center is searched before lexical segmentation, not borrowed from a "
            "known palindrome."
        ),
        "novelty_preflight": {
            "registry_entries_before_run": 87,
            "excluded_routes_before_run": 6,
            "status": "formal_preflight_before_execution",
            "signature_overlap": [],
            "manual_review_required": False,
        },
        "config": {
            "model": "gpt2",
            "model_files": "local_files_only",
            "prompt_count": len(PROMPTS),
            "samples_per_prompt": samples_per_prompt,
            "center_letters": 26,
            "vocabulary_size": len(vocab),
            "reverse_readings_per_center": 40,
            "catalogue_text_imported": False,
            "known_palindromes_imported": False,
        },
        "stats": {**dict(stats), "rendered_candidates": len(rows),
                  "reader_eligible": 0, "mechanically_admitted": len(admitted)},
        "rendered_candidates_and_probes": rows,
        "probes": probes,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source": "GPT-2 local topic prompts plus WordNet-backed strict lexical vocabulary",
            "source_sentences_copied": False,
            "independent_audits": ["normalized-tape-reversal", "ASCII-two-pointer"],
            "readability_certificate": False,
        },
        "next_repair": (
            "Use an online GPT-2 right-edge decoder with the center letter and "
            "syntactic state carried in the beam; preserve this odd-center lexical "
            "bridge as the baseline rather than reverting to even joins."
        ),
        "reader_gate": (
            "No row is reader evidence. A mechanically admitted row must be manually "
            "screened and frozen with randomized intact/shuffled controls before any "
            "human readability claim."
        ),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUT}")
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": result["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
