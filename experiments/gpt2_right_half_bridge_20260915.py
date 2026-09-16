"""Orientation-reversed repair of the GPT-2 odd-center bridge.

This is an explicitly labelled repair, not a new claim of a separate prose
family: GPT-2 proposes the natural clause on the right, then an independent
strict lexical decoder solves ``center + reverse(right)`` on the left.  The
orientation is the only changed state; exactness and reader gates are unchanged.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gpt2-right-half-bridge-20260915.json"
EXPERIMENT_ID = "gpt2-right-half-bridge-20260915"
SIGNATURE = (
    "gpt2-right-half-bridge|orientation-reversed-natural-clause-proposal|"
    "odd-center-character-equation|independent-left-grammar-decoding|exact-audit"
)
MIN_LETTERS = 39

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize
from llm_palindrome.lexicon import is_real_word, load_lexicon
from llm_palindrome.respace import respace_k, unigram_score
from experiments.gpt2_topic_half_decoder_20260915 import PROMPTS, _clean_generation, _generate
from experiments.gpt2_center_letter_bridge_20260915 import _two_pointer


def _strict_vocab() -> frozenset[str]:
    from wordfreq import top_n_list
    lexicon = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    words = set()
    for word in top_n_list("en", 100_000):
        if not word.isascii() or not word.isalpha():
            continue
        if len(word) >= 3 and is_real_word(word, lexicon):
            words.add(word)
        elif word in {"a", "i"}:
            words.add(word)
    words.update("a an the some one this that my our your i we you they he she it me us no not be do can and or but if as of to in on at by for with from near over under".split())
    return frozenset(words)


def run(*, samples_per_prompt: int = 64, max_rows: int = 120) -> dict:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
    model.eval()
    vocab = _strict_vocab()
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    seen: set[str] = set()
    for prompt_index, prompt in enumerate(PROMPTS):
        generated = _generate(model, tokenizer, prompt,
                              samples=samples_per_prompt,
                              seed=20260915 + 300 + prompt_index)
        for sample_index, right_text in enumerate(generated):
            stats["model_samples"] += 1
            right_words = tuple(tokenize(_clean_generation(right_text)))
            right_tape = normalize_letters(right_text)
            if not (4 <= len(right_words) <= 10 and 18 <= len(right_tape) <= 90):
                stats["right_shape_reject"] += 1
                continue
            hit = False
            for center in "abcdefghijklmnopqrstuvwxyz":
                stats["center_decode_calls"] += 1
                left_readings = respace_k(center + right_tape[::-1], vocab, k=40)
                if not left_readings:
                    continue
                for left_words in left_readings:
                    if len(left_words) < 3:
                        continue
                    content = [word for word in tuple(left_words) + right_words
                               if word not in {"a", "an", "the", "some", "one", "this", "that", "my", "our", "your", "i", "we", "you", "they", "he", "she", "it", "me", "us", "no", "not", "be", "do", "can", "and", "or", "but", "if", "as", "of", "to", "in", "on", "at", "by", "for", "with", "from", "near", "over", "under"}]
                    if len(content) != len(set(content)):
                        stats["duplicate_content_reject"] += 1
                        continue
                    rendered = " ".join(left_words) + "; " + right_text.rstrip(".!? ,;: ") + "."
                    tape = normalize_letters(rendered)
                    if len(tape) < MIN_LETTERS or tape in seen:
                        continue
                    seen.add(tape)
                    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=220)
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
                        "independent_two_pointer": _two_pointer(tape),
                        "mechanical_checks": checks,
                        "mechanically_admitted": bool(tape) and tape == tape[::-1] and _two_pointer(tape)["exact"] and all(checks.values()),
                        "lexical_score_diagnostic": unigram_score(tuple(left_words) + right_words),
                        "reader_status": "not_run; orientation repair is not human readability evidence",
                        "provenance": {"model": "gpt2", "prompt": prompt, "source_sentences_copied": False, "right_proposed_naturally": True, "left_reverse_words_emitted": False},
                    }
                    rows.append(row)
                    stats["exact"] += int(row["exact"])
                    stats["mechanically_admitted"] += int(row["mechanically_admitted"])
                    hit = True
                    if len(rows) >= max_rows:
                        break
                if len(rows) >= max_rows:
                    break
            if not hit and len(probes) < 160:
                probes.append({"prompt_index": prompt_index, "sample_index": sample_index, "right": right_text, "right_letters": len(right_tape), "center_letters_tried": 26, "status": "no-strict-independent-left-reading"})
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    rows.sort(key=lambda row: (-row["mechanically_admitted"], -row["letters"], row["rendered"]))
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": "Repair orientation: GPT-2 proposes the natural right clause; a strict independent lexical decoder solves center + reverse(right) on the left.",
        "novelty_preflight": {"registry_entries_before_run": 88, "excluded_routes_before_run": 6, "status": "formal_preflight_before_execution", "signature_overlap": [{"id": "gpt2-center-letter-bridge-20260915", "jaccard": 0.4, "disposition": "explicit orientation repair; no new family claim"}], "manual_review_required": True, "counted_as": "repair"},
        "config": {"model": "gpt2", "prompt_count": len(PROMPTS), "samples_per_prompt": samples_per_prompt, "center_letters": 26, "vocabulary_size": len(vocab), "reverse_readings_per_center": 40, "catalogue_text_imported": False, "known_palindromes_imported": False},
        "stats": {**dict(stats), "rendered_candidates": len(rows), "reader_eligible": 0, "mechanically_admitted": sum(row["mechanically_admitted"] for row in rows)},
        "rendered_candidates_and_probes": rows,
        "probes": probes,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "GPT-2 local generation plus strict lexical decoder", "source_sentences_copied": False, "independent_audits": ["normalized-tape-reversal", "ASCII-two-pointer"], "readability_certificate": False},
        "next_repair": "If orientation also produces only fragments, leave the bridge family and redesign the semantic proposal state; do not increase samples or loosen lexical admission.",
        "reader_gate": "No row is reader evidence. Any mechanically admitted row must be manually screened and frozen with randomized intact/shuffled controls before a readability claim.",
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
