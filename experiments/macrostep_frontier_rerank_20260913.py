"""Joint macrostep selection and whole-surface reranking for exact palindromes.

The host owns every letter decision.  A macrostep is a legal path of several
outside-in word transitions, so a language model can judge a phrase-sized
context instead of rewarding an isolated filler word.  Selected states are
completed by the exact beam and the model then sees the actual rendered
surfaces, never a fabricated string.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.bigram import BigramModel
from llm_palindrome.exact_editor import new_state, surface_audit
from llm_palindrome.generate import build_vocab
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import State, WordTries, _expand, beam_search, unit_letters
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


def child(parent: State, placement: str, word: str, overhang: str, side: str) -> State:
    left = parent.left + (word,) if placement == "L" else parent.left
    right = parent.right if placement == "L" else (word,) + parent.right
    return State(0.0, left, right, overhang, side, parent.score)


def transition_delta(scorer: CoherentScorer, state: State, placement: str,
                     word: str, growth: str) -> float:
    left = state.left + (word,) if placement == "L" else state.left
    right = state.right if placement == "L" else (word,) + state.right
    return scorer.word_delta(left, right, placement, word, growth)


def macro_paths(root: State, tries: WordTries, scorer: CoherentScorer,
                depth: int = 3, branch: int = 18, limit: int = 16) -> list[dict]:
    """Enumerate phrase-sized legal paths, retaining structurally diverse rows."""
    frontier = [(0.0, root, [])]
    for _ in range(depth):
        next_rows = []
        for score, state, path in frontier:
            choices = []
            for placement, word, overhang, side in _expand(state, tries, 220):
                if len(word.split()) != 1 or len(unit_letters(word)) < 3:
                    continue
                growth = "append" if placement == "L" else "prepend"
                delta = transition_delta(scorer, state, placement, word, growth)
                choices.append((score + delta, child(state, placement, word, overhang, side),
                                path + [{"placement": placement, "word": word,
                                         "overhang": overhang, "side": side,
                                         "growth": growth}]))
            choices.sort(key=lambda row: (-row[0], tuple(x["word"] for x in row[2])))
            next_rows.extend(choices[:branch])
        next_rows.sort(key=lambda row: (-row[0], tuple(x["word"] for x in row[2])))
        frontier = next_rows[: max(limit, branch)]
        if not frontier:
            break
    rows = []
    seen = set()
    for index, (score, state, path) in enumerate(frontier):
        key = (state.left, state.right, state.overhang, state.side)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": f"p{index:03d}", "score": score,
                     "added_words": [step["word"] for step in path],
                     "path": path, "left": list(state.left), "right": list(state.right),
                     "overhang": state.overhang, "side": state.side,
                     "surface": " ".join(state.left) + " [middle] " + " ".join(state.right),
                     "state": state})
    return rows[:limit]


def parse_ids(raw: str, valid: set[str], maximum: int) -> list[str]:
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
    return [item for item in ids if isinstance(item, str) and item in valid][:maximum]


def select_paths(model: str, witness: str, parent: State, menu: list[dict], seed: int) -> tuple[str, list[str]]:
    prompt = {
        "task": "Select phrase-sized legal paths that can remain part of one complete ordinary English sentence.",
        "witness": witness,
        "fixed_prefix": " ".join(parent.left),
        "fixed_suffix": " ".join(parent.right),
        "rule": "Choose only listed IDs. Select up to four; choose [] if no path fits. Do not invent words or letters.",
        "options": [{"id": r["id"], "added_words": r["added_words"],
                     "surface": r["surface"]} for r in menu],
    }
    raw = request_json("/api/chat", {"model": model,
        "messages": [{"role": "user", "content": json.dumps(prompt)}],
        "stream": False, "think": "low",
        "options": {"temperature": 0.25, "num_predict": 300, "seed": seed},
    })["message"]["content"]
    return raw, parse_ids(raw, {r["id"] for r in menu}, 4)


def rerank_surfaces(model: str, witness: str, surfaces: list[str], seed: int) -> tuple[str, list[int]]:
    options = [{"id": f"s{i:03d}", "text": text} for i, text in enumerate(surfaces)]
    prompt = {
        "task": "Identify the rendered passages that read most like ordinary coherent English.",
        "witness": witness,
        "rule": "Rank only the listed IDs. Do not rewrite them. Prefer one connected thought, normal syntax, and no fragments, lists, or repeated content.",
        "options": options,
    }
    raw = request_json("/api/chat", {"model": model,
        "messages": [{"role": "user", "content": json.dumps(prompt)}],
        "stream": False, "think": "low",
        "options": {"temperature": 0.2, "num_predict": 300, "seed": seed},
    })["message"]["content"]
    ids = parse_ids(raw, {r["id"] for r in options}, min(8, len(options)))
    return raw, [int(i[1:]) for i in ids]


def audit(words: list[str], seed: int, witness: str, endpoint: tuple[str, str],
          path_id: str) -> dict:
    rendered = textify(words)
    tape = normalize_letters(rendered)
    state = new_state(half_text=tape[: len(tape) // 2],
                      center_text=tape[len(tape) // 2] if len(tape) % 2 else "",
                      intent=witness, surface_hint=rendered)
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS,
                                         max_letters=MAX_LETTERS)
    return {"seed": seed, "endpoint": endpoint, "path_id": path_id,
            "witness": witness, "rendered": rendered, "words": words,
            "letters": len(tape),
            "render_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "exact_editor_audit": surface_audit(state, rendered,
                                                  min_letters=MIN_LETTERS,
                                                  max_letters=MAX_LETTERS),
            "mechanical_checks": checks,
            "mechanically_eligible": all(checks.values()),
            "human_reader_study": "not_run"}


ENDPOINTS = (
    (("desserts", "stressed"), "Desserts can cheer people who are stressed."),
    (("did it", "i did"), "Did it cause the same trouble last night that I did?"),
    (("some", "memos"), "Some managers stayed late to revise the memos."),
    (("some", "demos"), "Some developers presented their latest demos to colleagues."),
    (("a", "idea"), "A careful analyst explained the idea to the team."),
    (("we", "few"), "We prepared a short report for a few readers."),
)


def run(*, model: str, rounds: int, path_depth: int, seeds: int,
        endpoint_limit: int | None = None) -> dict:
    vocabulary = build_vocab(18000)
    tries = WordTries(vocabulary)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=set(vocabulary))
    scorer = CoherentScorer(bigrams, freq_weight=0.1, length_weight=0.14,
                            phrase_weight=1.4, short_penalty=3.0)
    traces, records, endpoint_summaries = [], [], []
    endpoints = ENDPOINTS if endpoint_limit is None else ENDPOINTS[:endpoint_limit]
    for endpoint, witness in endpoints:
        root = endpoint_state(*endpoint)
        frontier = [root]
        for depth in range(rounds):
            next_frontier = []
            for parent_index, parent in enumerate(frontier):
                menu = macro_paths(parent, tries, scorer, depth=path_depth)
                raw, selected = select_paths(model, witness, parent, menu,
                                             2026091400 + depth * 100 + parent_index)
                rows = [row for row in menu if row["id"] in selected]
                if not rows:
                    rows = menu[:2]
                traces.append({"endpoint": endpoint, "witness": witness,
                               "depth": depth, "parent": {"left": parent.left,
                               "right": parent.right, "overhang": parent.overhang,
                               "side": parent.side}, "menu": [{k: v for k, v in r.items()
                               if k != "state"} for r in menu], "raw_reply": raw,
                               "selected_ids": selected, "fallback_used": not bool(selected)})
                next_frontier.extend(row["state"] for row in rows)
            frontier = next_frontier[:8]
            if not frontier:
                break
        endpoint_summaries.append({"endpoint": endpoint, "witness": witness,
                                   "terminal_states": len(frontier)})
        for index, state in enumerate(frontier):
            for seed in range(seeds):
                words = beam_search(tries, scorer, min_letters=MIN_LETTERS,
                                    max_steps=220, beam_width=180, candidate_limit=500,
                                    seed=seed + index, diversity=1.3, max_word_uses=2,
                                    initial_state=state)
                if words:
                    records.append(audit(words, seed + index, witness, endpoint,
                                         f"{endpoint[0]}-{endpoint[1]}-{index}"))
    surfaces = [row["rendered"] for row in records]
    rerank_reply, ranked = (rerank_surfaces(model, "Rank the candidate passages as written.",
                                             surfaces, 2026091499) if surfaces else ("", []))
    for rank, index in enumerate(ranked):
        if 0 <= index < len(records):
            records[index]["model_surface_rank"] = rank
    return {"status": "complete_macrostep_frontier_rerank_pilot",
            "model_requested": model, "model_metadata": request_json("/api/show", {"name": model}),
            "endpoints": endpoints, "config": {"rounds": rounds, "path_depth": path_depth,
            "seeds": seeds, "min_letters": MIN_LETTERS, "model_supplies_letters": False,
            "machine_readability_certification": False}, "endpoint_summaries": endpoint_summaries,
            "traces": traces, "records": records, "surface_rerank_reply": rerank_reply,
            "surface_ranking": ranked,
            "mechanically_eligible": [r for r in records if r["mechanically_eligible"]],
            "reader_gate": "No readability claim; an eligible surface requires randomized blinded readers with intact prose and shuffled controls.",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "lexicon": "data/lexicon.txt", "construction": "host-enumerated legal macrosteps"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="imetaexabeam/RhythmAI:27b")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--path-depth", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--endpoint-limit", type=int)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(model=args.model, rounds=args.rounds, path_depth=args.path_depth,
                 seeds=args.seeds, endpoint_limit=args.endpoint_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "traces": len(result["traces"]),
                      "records": len(result["records"]),
                      "mechanically_eligible": len(result["mechanically_eligible"]),
                      "surface_ranking": result["surface_ranking"]}, sort_keys=True))


if __name__ == "__main__":
    main()
