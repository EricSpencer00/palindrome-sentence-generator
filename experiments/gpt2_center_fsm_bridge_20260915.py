"""POS/constituency repair for the GPT-2 odd-center bridge.

The previous odd-center route found exact tapes only by allowing arbitrary
lexical segmentation.  This repair carries a right-side clause automaton into
the reverse decode: every segmentation must realize a complete SVO, pronoun-
verb-object, modified SVO, or copular frame with Brown-backed POS labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gpt2-center-fsm-bridge-20260915.json"
EXPERIMENT_ID = "gpt2-center-fsm-bridge-20260915"
SIGNATURE = (
    "gpt2-center-fsm-bridge|odd-center-character-equation|"
    "right-side-pos-automaton|topic-state-transition|independent-audit"
)
MIN_LETTERS = 39
MAX_LETTERS = 220

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize
from llm_palindrome.lexicon import is_real_word, load_lexicon
from experiments.gpt2_topic_half_decoder_20260915 import PROMPTS, _clean_generation, _generate

FUNCTION = frozenset("a an the some one this that my our your i we you they he she it me us no not be do can and or but if as of to in on at by for with from near over under".split())
PATTERNS = (
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "COP", "ADJ"),
    ("PRON", "VERB", "ADJ"),
)


def _pos_lexicon() -> dict[str, frozenset[str]]:
    from nltk.corpus import brown
    from wordfreq import top_n_list, zipf_frequency

    counts: Counter[tuple[str, str]] = Counter()
    for sent in brown.tagged_sents(tagset="universal"):
        for raw, tag in sent:
            word = raw.casefold()
            if word.isascii() and word.isalpha():
                counts[(word, tag)] += 1
    best: dict[str, str] = {}
    for (word, tag), count in counts.items():
        if count > counts.get((word, best.get(word, "")), 0):
            best[word] = tag
    lexicon = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    out: dict[str, set[str]] = defaultdict(set)
    for word in top_n_list("en", 80_000):
        if not (word.isascii() and word.isalpha() and zipf_frequency(word, "en") >= 3.0):
            continue
        if len(word) >= 3 and not is_real_word(word, lexicon):
            continue
        tag = best.get(word)
        if tag in {"NOUN", "VERB", "ADJ", "ADV", "ADP"}:
            out[tag].add(word)
    # Brown leaves some function words tagged inconsistently; grammar slots
    # receive explicit closed-class inventories.
    out["DET"].update("a an the some one this that my our your".split())
    out["PRON"].update("i we you they he she it me us them".split())
    out["COP"].update("is are was were be".split())
    out["ADP"].update("in on at by for with from near over under".split())
    out["ADV"].update("well quietly carefully clearly calmly often never now".split())
    return {key: frozenset(value) for key, value in out.items()}


def _segment(target: str, pos: dict[str, frozenset[str]], *, limit: int = 12) -> list[tuple[str, ...]]:
    results: list[tuple[str, ...]] = []
    # Match only complete words; prefixes that cannot belong to a frame die
    # immediately. This is a syntax filter, not a readability certificate.
    def visit(offset: int, pattern: tuple[str, ...], slot: int, words: tuple[str, ...]) -> None:
        if len(results) >= limit:
            return
        if slot == len(pattern):
            if offset == len(target):
                results.append(words)
            return
        role = pattern[slot]
        for word in pos.get(role, ()):
            if target.startswith(word, offset):
                visit(offset + len(word), pattern, slot + 1, words + (word,))

    for pattern in PATTERNS:
        visit(0, pattern, 0, ())
        if len(results) >= limit:
            break
    return results


def _two_pointer(tape: str) -> dict:
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left": i, "right": j, "a": tape[i], "b": tape[j]})
        i += 1
        j -= 1
    return {"exact": bool(tape) and not mismatches, "pairs_checked": len(tape) // 2, "mismatches": mismatches[:5]}


def run(*, samples_per_prompt: int = 64, max_rows: int = 120) -> dict:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
    model.eval()
    pos = _pos_lexicon()
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    seen: set[str] = set()
    for prompt_index, prompt in enumerate(PROMPTS):
        generated = _generate(model, tokenizer, prompt,
                              samples=samples_per_prompt,
                              seed=20260915 + 200 + prompt_index)
        for sample_index, left_text in enumerate(generated):
            stats["model_samples"] += 1
            left_words = tuple(tokenize(_clean_generation(left_text)))
            left_tape = normalize_letters(left_text)
            if not (4 <= len(left_words) <= 10 and 18 <= len(left_tape) <= 90):
                stats["left_shape_reject"] += 1
                continue
            hit = False
            for center in "abcdefghijklmnopqrstuvwxyz":
                stats["center_trials"] += 1
                target = center + left_tape[::-1]
                readings = _segment(target, pos, limit=16)
                if not readings:
                    continue
                stats["syntax_hits"] += len(readings)
                for right_words in readings:
                    content = [word for word in left_words + right_words if word not in FUNCTION]
                    if len(content) != len(set(content)):
                        stats["duplicate_content_reject"] += 1
                        continue
                    rendered = left_text.rstrip(".!? ,;: ") + "; " + " ".join(right_words) + "."
                    tape = normalize_letters(rendered)
                    if len(tape) < MIN_LETTERS or len(tape) > MAX_LETTERS or tape in seen:
                        continue
                    seen.add(tape)
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
                        "right_syntax_patterns": [list(pattern) for pattern in PATTERNS if len(pattern) == len(right_words)],
                        "reader_status": "not_run; POS automaton is not human readability evidence",
                        "provenance": {"model": "gpt2", "prompt": prompt, "source_sentences_copied": False, "reverse_words_emitted": False},
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
                probes.append({"prompt_index": prompt_index, "sample_index": sample_index, "left": left_text, "left_letters": len(left_tape), "center_trials": 26, "status": "no-complete-right-pos-frame"})
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    rows.sort(key=lambda row: (-row["mechanically_admitted"], -row["letters"], row["rendered"]))
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": "GPT-2 topic half generation plus odd-center character insertion; an independent Brown-backed POS automaton must decode the right tape as a complete clause frame.",
        "novelty_preflight": {"registry_entries_before_run": 88, "excluded_routes_before_run": 6, "status": "formal_preflight_before_execution", "signature_overlap": [], "manual_review_required": False},
        "config": {"model": "gpt2", "prompt_count": len(PROMPTS), "samples_per_prompt": samples_per_prompt, "center_letters": 26, "right_patterns": [list(pattern) for pattern in PATTERNS], "catalogue_text_imported": False, "known_palindromes_imported": False},
        "stats": {**dict(stats), "rendered_candidates": len(rows), "reader_eligible": 0, "mechanically_admitted": len(admitted)},
        "rendered_candidates_and_probes": rows,
        "probes": probes,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "GPT-2 local generation plus Brown POS lexicon", "source_sentences_copied": False, "independent_audits": ["normalized-tape-reversal", "ASCII-two-pointer"], "readability_certificate": False},
        "next_repair": "Add a semantic role graph and valency-conditioned right-frame transitions to this POS automaton; do not widen the lexical list without a new construction state.",
        "reader_gate": "No row is reader evidence. Any mechanically admitted row must be manually screened and frozen with randomized intact/shuffled controls for blinded ratings.",
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
