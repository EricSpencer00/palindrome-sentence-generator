"""Rejected whole-word mirror ablation retained for diagnostic tests.

The language model never has to reverse characters.  It proposes only the
left-side wording from a finite ``word -> reversed-word`` alphabet; this
program deterministically derives the right side in reverse word order.  The
constraint is therefore exact by construction, while the model can spend its
language budget on the actual problem: choosing two word sequences that a
reader could understand.

This construction is prohibited by the acceptance standard: it derives a
whole-word reverse pairing. It can preserve a negative ablation record but
must never produce a mechanically admitted candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bidirectional_attested_span_mining import common_lexicon
from experiments.joint_dual_lexicalization import INTENTS, existing_v3_pairs, pair_identity
from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.reversibles import semordnilaps
from llm_palindrome.validator import is_palindrome, normalize
from server.v3 import real_words


HOST = "http://localhost:11434"
PHRASES_PER_CALL = 4
GENERATION_OPTIONS = {"temperature": 0.75, "num_predict": 1200}
PROMPT = """Work through the reversible word table internally before answering.

Write exactly 4 NEW candidate left-side English phrases for this scene:
{intent}

You may use only the reversible words in this table. A computer will derive
the right-side phrase by reversing the word order and replacing each word with
the word after its arrow. Choose each left phrase so that BOTH it and that
derived right phrase can be read as ordinary English. Use 3--8 words and
15--30 letters per phrase. Avoid known tricks such as "step on" / "no pets".

Reversible word table:
{mapping}

Return exactly this JSON object and nothing else:
{{"left_phrases":["...", "...", "...", "..."]}}
"""


def word_mapping(min_zipf: float = 3.6) -> dict[str, str]:
    """Common word forms whose literal reversal is another common word."""
    vocab = common_lexicon(min_zipf)
    return dict(sorted(semordnilaps(vocab, min_letters=1, min_zipf=0)))


def mapping_text(mapping: dict[str, str]) -> str:
    return "; ".join(f"{word}->{mirror}" for word, mirror in sorted(mapping.items()))


def derive_right(left: str, mapping: dict[str, str]) -> str | None:
    words = left.split()
    if not words or any(word not in mapping for word in words):
        return None
    return " ".join(mapping[word] for word in reversed(words))


def screen_left(left: str, mapping: dict[str, str], *, existing_pairs: set[frozenset[str]],
                novel_checker=is_novel_palindrome) -> dict:
    right = derive_right(left, mapping)
    shape = bool(re.fullmatch(r"[a-z]+(?: [a-z]+)*", left))
    text = f"{left} {right}" if right else left
    left_norm, right_norm = normalize(left), normalize(right or "")
    shared = mechanical_admission_checks(
        text,
        local_catalogue=set(json.loads((ROOT / "data" / "known_palindromes.json").read_text())),
        min_letters=30,
        max_letters=60,
    )
    checks = shared | {
        "ascii_word_form": shape,
        "all_words_reversible": right is not None,
        "reverse_match": bool(right_norm and left_norm == right_norm[::-1]),
        "half_length_band": bool(right_norm and 15 <= len(left_norm) <= 30 and 15 <= len(right_norm) <= 30),
        "word_band": bool(right and 3 <= len(left.split()) <= 8 and 3 <= len(right.split()) <= 8),
        "lexicon_words": bool(right and real_words(left.split() + right.split())),
        "nondegenerate": bool(right_norm and left_norm != right_norm),
        "exact_palindrome": shared["exact_letter_palindrome"],
        "novel_catalogue": bool(right and shared["local_catalogue_absent"] and novel_checker(text)),
        "novel_v3_bank_pair": bool(right and pair_identity(left, right) not in existing_pairs),
    }
    return {
        "left": left,
        "right": right,
        "text": text,
        "checks": checks,
        "rejection_codes": [name for name, ok in checks.items() if not ok],
    }


def parse_left_phrases(raw: str) -> tuple[list[str] | None, str | None]:
    text = raw.strip().removeprefix("```json").removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None, "reply_has_no_json_object"
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc.msg}"
    rows = data.get("left_phrases")
    if not isinstance(rows, list) or len(rows) != PHRASES_PER_CALL:
        return None, f"need_exactly_{PHRASES_PER_CALL}_phrases"
    if not all(isinstance(row, str) for row in rows):
        return None, "phrase_schema_error"
    return rows, None


def request_json(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        HOST + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def run(*, model: str, min_zipf: float) -> dict:
    raise RuntimeError(
        "guided whole-word mirror authoring is retired: its construction is a prohibited shortcut"
    )
    mapping = word_mapping(min_zipf)
    metadata = request_json("/api/show", {"name": model})
    records = []
    for index, intent in enumerate(INTENTS, 1):
        prompt = PROMPT.format(intent=intent, mapping=mapping_text(mapping))
        raw = request_json("/api/chat", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": "low",
            "options": GENERATION_OPTIONS | {"seed": 20260923 + index},
        })["message"]["content"]
        phrases, error = parse_left_phrases(raw)
        records.append({"intent_id": f"I{index:02d}", "intent": intent,
                        "prompt": prompt, "raw_reply": raw, "left_phrases": phrases,
                        "parse_error": error})
        print(f"proposed {index}/{len(INTENTS)}", flush=True)

    known_pairs = existing_v3_pairs()
    screened = [screen_left(left, mapping, existing_pairs=known_pairs)
                | {"intent_id": record["intent_id"]}
                for record in records if record["left_phrases"] is not None
                for left in record["left_phrases"]]
    unique = {}
    for row in screened:
        unique.setdefault((row["left"], row["right"]), row)
    accepted = [row for row in unique.values() if not row["rejection_codes"]]
    return {
        "status": "complete_exact_by_construction_proposal_run",
        "model_requested": model,
        "model_metadata": metadata,
        "min_zipf": min_zipf,
        "mapping": mapping,
        "mapping_sha256": hashlib.sha256(mapping_text(mapping).encode()).hexdigest(),
        "prompt_template": PROMPT,
        "generation_options": GENERATION_OPTIONS | {"thinking": "low"},
        "records": records,
        "parsed_calls": sum(row["left_phrases"] is not None for row in records),
        "total_calls": len(INTENTS),
        "screened_candidates": len(screened),
        "unique_candidates": len(unique),
        "accepted": accepted,
        "reader_gate": (
            "An accepted string is an exact, lexical, novel material lead only. It advances to "
            "a blinded human readability screen with intact prose and shuffled controls before "
            "any claim about readability."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--min-zipf", type=float, default=3.6)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(model=args.model, min_zipf=args.min_zipf)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.out), "parsed_calls": result["parsed_calls"],
        "total_calls": result["total_calls"], "accepted": len(result["accepted"]),
    }, indent=2))


if __name__ == "__main__":
    main()
