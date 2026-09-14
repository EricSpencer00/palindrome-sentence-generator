"""Use original model-proposed phrase units inside the exact palindrome lattice.

The model proposes ordinary multiword material only.  The host filters each
phrase against the frozen local corpus, adds the surviving units to the exact
word trie, and owns every letter transition and validation decision.  A phrase
unit is not a readability certificate; accepted surfaces still require an
independent reader screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.bigram import BigramModel
from llm_palindrome.exact_editor import new_state, surface_audit
from llm_palindrome.generate import build_vocab
from llm_palindrome.phrases import build_inventory
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify


HOST = "http://127.0.0.1:11434"
PHRASE_RE = re.compile(r"[a-z]+(?: [a-z]+){2,5}")
MIN_LETTERS = 100
MAX_LETTERS = 160


def request_json(path: str, body: dict) -> dict:
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def parse_phrases(raw: str) -> tuple[list[str], str | None]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return [], "reply_has_no_json_object"
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        return [], f"json_error:{exc.msg}"
    values = value.get("phrases")
    if not isinstance(values, list):
        return [], "phrases_not_list"
    out, seen = [], set()
    for phrase in values:
        if not isinstance(phrase, str):
            continue
        phrase = phrase.strip().casefold()
        if not PHRASE_RE.fullmatch(phrase) or phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
    return out, None


def corpus_phrases() -> set[str]:
    data = json.loads((ROOT / "data" / "ngrams_wikitext2.json").read_text())
    return {phrase.casefold() for key, rows in data.items() if key.isdigit() and int(key) >= 3 for phrase in rows}


def audit(text: str, *, seed: int, phrase_units: list[str]) -> dict:
    tape = normalize_letters(text)
    state = new_state(half_text=tape[: len(tape) // 2], center_text=tape[len(tape) // 2] if len(tape) % 2 else "",
                      intent="model-proposed phrase-unit search", surface_hint=text)
    exact = surface_audit(state, text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    mechanical = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {"seed": seed, "rendered": text, "letters": len(tape),
            "render_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "words": exact["words"], "phrase_units_used": [p for p in phrase_units if p in text.casefold()],
            "exact_editor_audit": exact, "mechanical_checks": mechanical,
            "mechanically_eligible": all(mechanical.values()), "human_reader_study": "not_run"}


def run(*, model: str, phrase_count: int, seeds: int, budget: float) -> dict:
    metadata = request_json("/api/show", {"name": model})
    prompt = (f"Generate {phrase_count} distinct original ordinary-English phrases, each 3 to 6 words, "
              "about research, evidence, reporting, daily work, or observations. Use lowercase ASCII "
              "words and spaces only. Each phrase must be a natural clause or noun/verb phrase, not an "
              "isolated word list, quotation, or famous line. Return only JSON: {\"phrases\":[...]}")
    raw = request_json("/api/chat", {"model": model, "messages": [{"role": "user", "content": prompt}],
                                     "stream": False, "think": "low",
                                     "options": {"temperature": 0.9, "num_predict": 2200, "seed": 2026091301}})["message"]["content"]
    proposed, parse_error = parse_phrases(raw)
    local = corpus_phrases()
    novel = [phrase for phrase in proposed if phrase not in local]
    vocab = build_vocab(18000)
    inventory = build_inventory(str(ROOT / "data" / "count_2w.txt"), vocab=vocab, top_n=16000, min_count=3)
    units = list(dict.fromkeys(vocab + inventory + novel))
    tries = WordTries(units)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=set(vocab))
    scorer = CoherentScorer(bigrams, freq_weight=0.10, length_weight=0.14,
                            phrase_weight=5.0, long_bonus=1.5, short_penalty=3.0)
    records = []
    for seed in range(seeds):
        words = beam_search(tries, scorer, min_letters=MIN_LETTERS, max_steps=220,
                            beam_width=220, candidate_limit=700, seed=seed, diversity=1.5,
                            max_word_uses=2)
        if words:
            records.append(audit(textify(words), seed=seed, phrase_units=novel))
    return {
        "status": "complete_original_model_phrase_unit_search",
        "model_requested": model, "model_metadata": metadata,
        "prompt": prompt, "raw_model_reply": raw, "parse_error": parse_error,
        "proposed_phrases": proposed, "novel_phrase_units": novel,
        "novel_phrase_count": len(novel), "local_corpus_phrase_count": len(local),
        "vocabulary_size": len(vocab), "unit_count": len(units),
        "config": {"seeds": seeds, "budget_seconds_per_seed": budget,
                   "candidate_letter_range": [MIN_LETTERS, MAX_LETTERS],
                   "model_supplies_letters": False, "machine_readability_certification": False},
        "records": records,
        "mechanically_eligible": [row for row in records if row["mechanically_eligible"]],
        "reader_gate": "No programmatic readability claim; any eligible surface requires randomized blinded human readers with intact prose and shuffled controls.",
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "phrase_source": "single recorded local model reply, corpus-filtered", "lexicon": "data/lexicon.txt"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="imetaexabeam/RhythmAI:27b")
    parser.add_argument("--phrase-count", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=10.0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(model=args.model, phrase_count=args.phrase_count, seeds=args.seeds, budget=args.budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "novel_phrase_count": result["novel_phrase_count"],
                      "records": len(result["records"]), "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
