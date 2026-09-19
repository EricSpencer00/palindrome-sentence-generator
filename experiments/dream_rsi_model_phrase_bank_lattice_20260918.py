"""Dream-RSI model phrase-bank lattice.

This lane uses a local model only to propose fresh, role-tagged English
phrases.  The model never supplies letters to the palindrome directly: the
host builds a character trie, searches the exact tape with a bounded beam, and
independently audits every rendered closure.  A phrase bank is useful here
because the earlier span lane kept a bootstrap sentence fixed; this lane
reopens the whole semantic inventory while retaining complete phrase units.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "dream-rsi-model-phrase-bank-lattice-20260918"
HOST = "http://127.0.0.1:11434"
MODEL = "gpt-oss:20b"
PHRASE_RE = re.compile(r"[a-z]+(?: [a-z]+){2,8}")

# Running a module by path puts ``experiments/`` first on sys.path.  Keep the
# standalone command reproducible without requiring callers to export
# PYTHONPATH.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.bigram import BigramModel
from llm_palindrome.generate import build_vocab
from llm_palindrome.phrases import build_inventory
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify
from experiments.dream_rsi_model_guided_span_resynthesis_20260918 import audit


PROMPT = (
    "Generate 48 distinct original ordinary-English phrases, each 3 to 8 "
    "words, about an archivist, reader, weather, cooking, travel, or daily "
    "work. Every phrase must be a natural clause or noun/verb phrase, not a "
    "list, quotation, famous palindrome, reversed-word pair, or fragment. "
    "Use lowercase ASCII words. Return JSON only as "
    '{"items":[{"text":"...","role":"subject|verb_phrase|object|adjunct|discourse"}]}.'
)


def request_json(body: dict) -> dict:
    request = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)


def parse_items(raw: str) -> tuple[list[dict], str | None]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return [], "no_json_object"
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        return [], f"json_error:{exc.msg}"
    values = payload.get("items")
    if not isinstance(values, list):
        return [], "items_not_list"
    out, seen = [], set()
    for value in values:
        if isinstance(value, str):
            text, role = value, "unspecified"
        elif isinstance(value, dict):
            text, role = value.get("text", ""), value.get("role", "unspecified")
        else:
            continue
        text = re.sub(r"[^a-z ]", "", str(text).casefold()).strip()
        if not PHRASE_RE.fullmatch(text) or text in seen:
            continue
        seen.add(text)
        out.append({"text": text, "role": str(role)})
    return out, None


def model_phrase_bank() -> tuple[list[dict], dict]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": "low",
        "options": {"temperature": 0.85, "num_predict": 3200, "seed": 2026091801},
    }
    response = request_json(body)
    raw = response.get("message", {}).get("content", "")
    items, parse_error = parse_items(raw)
    return items, {"raw_reply": raw, "parse_error": parse_error, "model": MODEL, "prompt": PROMPT}


def run(*, seeds: int = 4, min_letters: int = 80, max_letters: int = 180) -> dict:
    items, model_record = model_phrase_bank()
    phrases = [item["text"] for item in items]
    local_vocab = build_vocab(16000)
    inventory = build_inventory(str(ROOT / "data" / "count_2w.txt"), vocab=local_vocab, top_n=12000, min_count=3)
    units = list(dict.fromkeys(local_vocab + inventory + phrases))
    tries = WordTries(units)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=set(local_vocab))
    scorer = CoherentScorer(
        bigrams,
        freq_weight=0.10,
        length_weight=0.14,
        phrase_weight=4.0,
        long_bonus=1.2,
        short_penalty=2.0,
        unit_bonus={phrase: 2.0 for phrase in phrases},
    )
    records = []
    for seed in range(seeds):
        units_out = beam_search(
            tries,
            scorer,
            min_letters=min_letters,
            max_steps=220,
            beam_width=180,
            candidate_limit=500,
            per_parent=8,
            seed=seed,
            diversity=1.25,
            max_word_uses=2,
        )
        if not units_out:
            continue
        text = textify(units_out)
        checks = mechanical_admission_checks(text, min_letters=min_letters, max_letters=max_letters)
        records.append(
            {
                "seed": seed,
                "rendered": text,
                "letters": len(normalize_letters(text)),
                "units": units_out,
                "audit": audit(text),
                "mechanical_checks": checks,
                "mechanically_admitted": all(checks.values()),
                "reader_status": "human-unreviewed",
                "provenance": {
                    "fresh_model_phrase_bank": True,
                    "model_phrase_count": len(phrases),
                    "seed_scaffold_in_output": False,
                    "finished_tape_reversed": False,
                    "catalogue_imported": False,
                    "word_order_only": not checks["not_word_order_symmetry"],
                    "human_readability_certified": False,
                },
            }
        )
    exact = [row for row in records if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {
        "experiment": EXPERIMENT,
        "method": "Dream-RSI model phrase bank plus exact role-agnostic phrase lattice",
        "model": model_record,
        "phrase_items": items,
        "records": records,
        "fresh_exact_closures": exact,
        "mechanically_admitted": admitted,
        "stats": {
            "model_phrases": len(phrases),
            "seeds": seeds,
            "closures": len(records),
            "exact": len(exact),
            "mechanically_admitted": len(admitted),
            "longest_letters": max((row["letters"] for row in records), default=0),
        },
        "reader_gate": {
            "status": "closed",
            "reason": "No exact row is reader-eligible until it clears the strict mechanical gate and a blinded intact/shuffled study.",
            "programmatic_metrics_are_diagnostic": True,
        },
        "next_repair": {
            "operator": "carry role labels into the live character trie and reject a branch when either half loses a complete clause parse",
            "reason": "phrase-bank search supplies fresh lexical material, but role-agnostic closures can still be word salad",
            "reader_test": "only a strict exact survivor enters randomized intact/shuffled blinded rating",
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "catalogue_imported": False,
            "human_readability_certified": False,
        },
    }


if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        directory.mkdir(exist_ok=True)
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))
    for row in sorted(payload["fresh_exact_closures"], key=lambda item: -item["letters"]):
        print(f"{row['letters']} letters | {row['rendered']}")
