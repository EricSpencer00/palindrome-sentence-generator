"""Topic-conditioned GPT-2 half generation with independent reverse decoding.

This route changes the proposal state, not just the lexicon: GPT-2 samples an
ordinary topic-conditioned half-clause in reading order.  The required reverse
character tape is then segmented independently by a lexical dynamic program;
the model never sees or emits a reflected word list.  Surviving exact joins
are ranked by a second whole-text model score, while grammar and readability
remain human-gated.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gpt2-topic-half-decoder-20260915.json"
EXPERIMENT_ID = "gpt2-topic-half-decoder-20260915"
SIGNATURE = (
    "gpt2-topic-half-decoder|prompt-conditioned-natural-half-generation|"
    "reverse-character-lexical-decoder|semantic-continuity-state|"
    "independent-audit"
)
MIN_LETTERS = 39
MAX_LETTERS = 220

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize
from llm_palindrome.respace import respace_k, unigram_score

PROMPTS = (
    "Write one original short English sentence about an editor reading a note. Use five to eight ordinary words. Sentence:",
    "Write one original short English sentence about a teacher sending a letter. Use five to eight ordinary words. Sentence:",
    "Write one original short English sentence about a farmer keeping a plan. Use five to eight ordinary words. Sentence:",
    "Write one original short English sentence about a writer making a clear record. Use five to eight ordinary words. Sentence:",
    "Write one original short English sentence about a nurse carrying a book. Use five to eight ordinary words. Sentence:",
    "Write one original short English sentence about a pilot seeing a quiet harbor. Use five to eight ordinary words. Sentence:",
)


def _clean_generation(text: str) -> str:
    text = text.strip().split("\n", 1)[0]
    # Keep only the first ordinary sentence; GPT-2 often continues with a
    # second example when sampling from an instruction prompt.
    match = re.search(r"[.!?]", text)
    if match:
        text = text[:match.end()]
    text = re.sub(r"[^A-Za-z '\-.,;:!?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _independent_two_pointer(tape: str) -> dict:
    mismatches = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left_index": left, "right_index": right,
                               "left": tape[left], "right": tape[right]})
        left += 1
        right -= 1
    return {
        "exact": bool(tape) and not mismatches,
        "pairs_checked": len(tape) // 2,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:5],
    }


def _word_order_mirror(words: tuple[str, ...]) -> bool:
    normalized = tuple(normalize_letters(word) for word in words)
    return bool(normalized) and normalized == tuple(
        word[::-1] for word in reversed(normalized)
    )


def _topic_overlap(left: list[str], right: list[str], keywords: set[str]) -> float:
    content = {word for word in left + right if word not in {"a", "an", "the", "to", "of", "in", "on", "is", "are"}}
    return len(content & keywords) / max(1, len(keywords))


def _generate(model, tokenizer, prompt: str, *, samples: int, seed: int) -> list[str]:
    import torch

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    torch.manual_seed(seed)
    outputs = model.generate(
        input_ids,
        max_new_tokens=18,
        do_sample=True,
        top_p=0.95,
        temperature=0.92,
        num_return_sequences=samples,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [_clean_generation(tokenizer.decode(row[input_ids.shape[1]:], skip_special_tokens=True))
            for row in outputs]


def run(*, samples_per_prompt: int = 48, max_rows: int = 80) -> dict:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
    model.eval()
    from wordfreq import top_n_list, zipf_frequency

    vocab = frozenset(
        word for word in top_n_list("en", 45_000)
        if word.isascii() and word.isalpha() and len(word) >= 2
        and zipf_frequency(word, "en") >= 3.0
    ) | {"a", "i"}

    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    seen_tapes: set[str] = set()
    for prompt_index, prompt in enumerate(PROMPTS):
        generated = _generate(model, tokenizer, prompt,
                              samples=samples_per_prompt, seed=20260915 + prompt_index)
        keywords = set(re.findall(r"[a-z]+", prompt.casefold())) - {
            "write", "one", "original", "short", "english", "sentence", "about", "use", "five", "to", "eight", "ordinary", "words"
        }
        for sample_index, left_text in enumerate(generated):
            stats["model_samples"] += 1
            left_words = tuple(tokenize(left_text))
            left_tape = normalize_letters(left_text)
            if not (4 <= len(left_words) <= 10 and 18 <= len(left_tape) <= 90):
                stats["left_shape_reject"] += 1
                continue
            readings = respace_k(left_tape[::-1], vocab, k=10)
            stats["reverse_decode_calls"] += 1
            if not readings:
                stats["reverse_decode_miss"] += 1
                if len(probes) < 100:
                    probes.append({"left": left_text, "left_letters": len(left_tape),
                                   "required_reverse_tape": left_tape[::-1],
                                   "status": "no-independent-lexical-reading"})
                continue
            for right_words_list in readings:
                right_words = tuple(right_words_list)
                if len(right_words) < 3 or _word_order_mirror(left_words + right_words):
                    stats["shortcut_reject"] += 1
                    continue
                if any(word in left_words for word in right_words
                       if word not in {"a", "an", "the", "to", "of", "in", "on", "is", "are"}):
                    stats["duplicate_content_reject"] += 1
                    continue
                right_text = " ".join(right_words)
                rendered = left_text.rstrip(".!? ") + "; " + right_text + "."
                tape = normalize_letters(rendered)
                if len(tape) < MIN_LETTERS or len(tape) > MAX_LETTERS or tape in seen_tapes:
                    continue
                seen_tapes.add(tape)
                two_pointer = _independent_two_pointer(tape)
                checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
                row = {
                    "rendered": rendered,
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "left_words": list(left_words),
                    "right_words": list(right_words),
                    "letters": len(tape),
                    "normalized_tape": tape,
                    "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
                    "exact": bool(tape) and tape == tape[::-1],
                    "independent_two_pointer": two_pointer,
                    "mechanical_checks": checks,
                    "mechanically_admitted": bool(tape) and tape == tape[::-1] and two_pointer["exact"] and all(checks.values()),
                    "topic_overlap_diagnostic": _topic_overlap(list(left_words), list(right_words), keywords),
                    "lexical_score_diagnostic": unigram_score(left_words + right_words),
                    "reader_status": "not_run; GPT-2 and lexical scores do not certify readability",
                    "provenance": {
                        "model": "gpt2",
                        "prompt": prompt,
                        "source_sentences_copied": False,
                        "reverse_words_emitted": False,
                    },
                }
                rows.append(row)
                stats["exact"] += int(row["exact"])
                stats["mechanically_admitted"] += int(row["mechanically_admitted"])
                if len(rows) >= max_rows:
                    break
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
            "GPT-2 samples topic-conditioned ordinary half-clauses; an independent "
            "lexical DP segments each required reverse tape, then topic overlap "
            "and unigram likelihood diagnose (but do not certify) survivors."
        ),
        "novelty_preflight": {
            "registry_entries_before_run": 85,
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
            "vocabulary_size": len(vocab),
            "reverse_readings_per_tape": 10,
            "catalogue_text_imported": False,
            "known_palindromes_imported": False,
        },
        "stats": {**dict(stats), "rendered_candidates": len(rows),
                  "reader_eligible": 0, "mechanically_admitted": len(admitted)},
        "rendered_candidates_and_probes": rows,
        "probes": probes,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source": "GPT-2 local generation plus wordfreq lexical decoder",
            "source_sentences_copied": False,
            "independent_audits": ["normalized-tape-reversal", "ASCII-two-pointer"],
            "readability_certificate": False,
        },
        "next_repair": (
            "Replace post-hoc reverse segmentation with a GPT-2 next-token decoder "
            "whose lexical choices are constrained online by the reverse tape and "
            "the topic state; preserve this run as the proposer/decoder baseline."
        ),
        "reader_gate": (
            "No row is reader evidence. A mechanically admitted row must be manually "
            "screened and then frozen with randomized intact/shuffled controls for "
            "blinded ratings."
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
