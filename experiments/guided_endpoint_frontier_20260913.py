"""Model-guided exact frontier construction from compatible endpoints.

This is a bounded constructive pilot.  The host enumerates exact legal child
states and assigns opaque IDs.  A local model may retain IDs that fit a frozen
ordinary-English witness, but it never supplies letters, word boundaries, or
acceptance decisions.  Each retained state is then completed by the exact
lattice and independently audited.
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

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.bigram import BigramModel
from llm_palindrome.exact_editor import new_state, surface_audit
from llm_palindrome.generate import build_vocab
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import State, WordTries, _expand, beam_search, consume, unit_letters
from llm_palindrome.textify import textify


HOST = "http://127.0.0.1:11434"
MIN_LETTERS = 100
MAX_LETTERS = 180


def request_json(path: str, body: dict) -> dict:
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def endpoint_state(left: str, right: str) -> State:
    left_tape, right_reverse = unit_letters(left), unit_letters(right)[::-1]
    if right_reverse.startswith(left_tape):
        return State(0.0, (left,), (right,), right_reverse[len(left_tape):], "R", 0.0)
    if left_tape.startswith(right_reverse):
        return State(0.0, (left,), (right,), left_tape[len(right_reverse):], "L", 0.0)
    raise ValueError(f"incompatible endpoints: {left!r}, {right!r}")


def child_state(parent: State, placement: str, word: str, overhang: str, side: str) -> State:
    if placement == "L":
        left, right = parent.left + (word,), parent.right
    else:
        left, right = parent.left, (word,) + parent.right
    return State(0.0, left, right, overhang, side, parent.score)


def parse_ids(raw: str, valid: set[str]) -> list[str]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return []
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    ids = value.get("ids", value.get("selected_ids"))
    if not isinstance(ids, list):
        return []
    return [item for item in ids if isinstance(item, str) and item in valid]


def menu_for(state: State, tries: WordTries, scorer: CoherentScorer, limit: int = 18) -> list[dict]:
    rows = []
    for index, (placement, word, overhang, side) in enumerate(_expand(state, tries, 160)):
        left = state.left + (word,) if placement == "L" else state.left
        right = state.right if placement == "L" else (word,) + state.right
        delta = scorer.word_delta(left, right, placement, word,
                                  "append" if placement == "L" else "prepend")
        rows.append({"id": f"c{index:03d}", "placement": placement, "word": word,
                     "overhang": overhang, "side": side, "left": list(left),
                     "right": list(right), "delta": delta,
                     "surface": " ".join(left) + " [middle] " + " ".join(right)})
    rows.sort(key=lambda row: (-row["delta"], row["word"], row["id"]))
    return rows[:limit]


def call_selector(model: str, witness: str, parent: State, menu: list[dict], seed: int) -> tuple[str, list[str]]:
    prompt = {
        "task": "Select exact legal transitions that can remain part of one ordinary English sentence.",
        "witness": witness,
        "fixed_prefix": " ".join(parent.left),
        "fixed_suffix": " ".join(parent.right),
        "rule": "Choose only IDs listed below. Do not invent words, letters, or a palindrome. Keep up to four IDs; choose [] if none fit.",
        "options": [{"id": row["id"], "word": row["word"], "surface": row["surface"]} for row in menu],
    }
    raw = request_json("/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": json.dumps(prompt)}],
        "stream": False,
        "think": "low",
        "options": {"temperature": 0.3, "num_predict": 250, "seed": seed},
    })["message"]["content"]
    return raw, parse_ids(raw, {row["id"] for row in menu})[:4]


def audit(words: list[str], seed: int, witness: str, endpoint: tuple[str, str]) -> dict:
    text = textify(words)
    tape = normalize_letters(text)
    state = new_state(half_text=tape[: len(tape) // 2], center_text=tape[len(tape) // 2] if len(tape) % 2 else "",
                      intent=witness, surface_hint=text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {"seed": seed, "endpoint": endpoint, "witness": witness, "rendered": text,
            "words": words, "letters": len(tape), "render_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "exact_editor_audit": surface_audit(state, text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
            "mechanical_checks": checks, "mechanically_eligible": all(checks.values()),
            "human_reader_study": "not_run"}


def run(*, model: str, rounds: int, beam: int, seeds: int) -> dict:
    endpoint = ("some", "demos")
    witness = "Some developers carefully presented their research results and discussed the latest demos with colleagues."
    vocabulary = build_vocab(18000)
    tries = WordTries(vocabulary)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=set(vocabulary))
    scorer = CoherentScorer(bigrams, freq_weight=0.1, length_weight=0.14, phrase_weight=1.4,
                            short_penalty=3.0)
    roots = [endpoint_state(*endpoint)]
    traces, frontier = [], roots
    for depth in range(rounds):
        next_frontier = []
        for parent_index, parent in enumerate(frontier):
            menu = menu_for(parent, tries, scorer)
            raw, chosen = call_selector(model, witness, parent, menu, 2026091400 + depth * 100 + parent_index)
            chosen_rows = [row for row in menu if row["id"] in chosen]
            # Empty model selections are a concrete rejection; preserve a
            # deterministic high-scoring fallback so the exact branch can keep
            # moving without letting the model fabricate a transition.
            if not chosen_rows:
                chosen_rows = menu[:2]
            traces.append({"depth": depth, "parent": {"left": parent.left, "right": parent.right,
                           "overhang": parent.overhang, "side": parent.side},
                           "menu": menu, "raw_reply": raw, "selected_ids": chosen,
                           "fallback_used": not bool(chosen)})
            for row in chosen_rows:
                next_frontier.append(child_state(parent, row["placement"], row["word"], row["overhang"], row["side"]))
        frontier = next_frontier[:beam]
        if not frontier:
            break
    records = []
    for index, state in enumerate(frontier):
        for seed in range(seeds):
            words = beam_search(tries, scorer, min_letters=MIN_LETTERS, max_steps=220,
                                beam_width=180, candidate_limit=500, seed=seed + index,
                                diversity=1.3, max_word_uses=2, initial_state=state)
            if words:
                records.append(audit(words, seed + index, witness, endpoint))
    return {"status": "complete_model_guided_exact_frontier_pilot", "model_requested": model,
            "model_metadata": request_json("/api/show", {"name": model}), "endpoint": endpoint,
            "witness": witness, "config": {"rounds": rounds, "beam": beam, "seeds": seeds,
            "min_letters": MIN_LETTERS, "model_supplies_letters": False,
            "machine_readability_certification": False}, "traces": traces, "records": records,
            "mechanically_eligible": [row for row in records if row["mechanically_eligible"]],
            "reader_gate": "No readability claim; any eligible surface requires randomized blinded human readers with intact prose and shuffled controls.",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "lexicon": "data/lexicon.txt", "endpoint_seed": "host_exact_compatibility_v1"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="imetaexabeam/RhythmAI:27b")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(model=args.model, rounds=args.rounds, beam=args.beam, seeds=args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "traces": len(result["traces"]),
                      "records": len(result["records"]), "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
