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

from llm_palindrome.admission import (REPEATABLE_FUNCTION_WORDS,
    has_distinct_content_words, has_only_ordinary_short_words,
    mechanical_admission_checks, normalize_letters, tokenize)
from llm_palindrome.bigram import BigramModel
from llm_palindrome.exact_editor import new_state, surface_audit
from llm_palindrome.frontier_macros import state_from_anchors, witness_preserves_anchors
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


def assistant_text(response: dict) -> str:
    """Support local reasoning models that put the visible answer in thinking."""
    message = response.get("message", {})
    return message.get("content") or message.get("thinking") or ""


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
                depth: int = 3, branch: int = 18, limit: int = 16,
                witness: str = "") -> list[dict]:
    """Enumerate phrase-sized legal paths, retaining structurally diverse rows."""
    frontier = [(0.0, root, [])]
    for _ in range(depth):
        next_rows = []
        for score, state, path in frontier:
            choices = []
            for placement, word, overhang, side in _expand(state, tries, 220):
                if len(word.split()) != 1:
                    continue
                growth = "append" if placement == "L" else "prepend"
                candidate = child(state, placement, word, overhang, side)
                candidate_words = tuple(unit for block in candidate.left + candidate.right
                                        for unit in block.split())
                if (not has_distinct_content_words(candidate_words)
                        or not has_only_ordinary_short_words(candidate_words)
                        or any(w == w[::-1] and w not in REPEATABLE_FUNCTION_WORDS
                               for w in candidate_words)):
                    continue
                delta = transition_delta(scorer, state, placement, word, growth)
                choices.append((score + delta, candidate,
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
    # Add paths that follow the authored sentence's actual boundary words.
    # This changes only proposal order; exact compatibility is still checked by
    # state_from_anchors and the model must supply a complete witness later.
    witness_words = tokenize(witness)
    left_words = tuple(w for unit in root.left for w in unit.split())
    right_words = tuple(w for unit in root.right for w in unit.split())
    if (witness_words[:len(left_words)] == left_words
            and witness_words[-len(right_words):] == right_words):
        guided = []
        for left_add in range(depth + 1):
            for right_add in range(depth + 1):
                if left_add + right_add < 2:
                    continue
                prefix_words = witness_words[:len(left_words) + left_add]
                suffix_words = witness_words[-(len(right_words) + right_add):]
                prefix, suffix = " ".join(prefix_words), " ".join(suffix_words)
                try:
                    state = state_from_anchors(prefix, suffix)
                except ValueError:
                    continue
                words = tuple(prefix_words + suffix_words)
                if (not has_distinct_content_words(words)
                        or not has_only_ordinary_short_words(words)):
                    continue
                guided.append({"id": f"w{len(guided):03d}", "score": 1000.0 - left_add - right_add,
                               "added_words": prefix_words[len(left_words):] +
                               suffix_words[:-len(right_words)] if right_words else suffix_words,
                               "path": [], "left": list(state.left), "right": list(state.right),
                               "overhang": state.overhang, "side": state.side,
                               "surface": prefix + " [middle] " + suffix, "state": state})
        rows = guided + rows
    return rows[:limit]


def parse_selections(raw: str, menu: list[dict], maximum: int) -> list[dict]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return []
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    selections = value.get("selections")
    if not isinstance(selections, list):
        return []
    by_id = {r["id"]: r for r in menu}
    accepted = []
    for item in selections:
        if not isinstance(item, dict):
            continue
        path_id, text = item.get("id"), item.get("witness")
        if not isinstance(path_id, str) or path_id not in by_id:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        row = by_id[path_id]
        if not witness_preserves_anchors(row["state"], text):
            continue
        words = tuple(tokenize(text))
        if (len(words) < 5 or not has_distinct_content_words(words)
                or not has_only_ordinary_short_words(words)):
            continue
        accepted.append({"id": path_id, "witness": text.strip()})
        if len(accepted) >= maximum:
            break
    return accepted


def select_paths(model: str, witness: str, parent: State, menu: list[dict], seed: int) -> tuple[str, list[dict]]:
    prompt = {
        "task": "For each selected legal path, write one complete ordinary-English sentence.",
        "witness": witness,
        "fixed_prefix": " ".join(parent.left),
        "fixed_suffix": " ".join(parent.right),
        "rule": "Return JSON only as {\"selections\":[{\"id\":\"p...\",\"witness\":\"...\"}]}. Select only listed IDs. A witness must literally begin with the option prefix words and end with its suffix words. Use [] if no option supports a complete sentence. Do not invent an ID.",
        "options": [{"id": r["id"], "prefix": " ".join(r["left"]),
                     "suffix": " ".join(r["right"]),
                     "added_words": r["added_words"]} for r in menu],
    }
    response = request_json("/api/chat", {"model": model,
        "messages": [{"role": "user", "content": json.dumps(prompt)}],
        "stream": False, "think": "low" if not model.startswith("gpt-oss") else False,
        "options": {"temperature": 0.25, "num_predict": 300, "seed": seed},
    })
    raw = assistant_text(response)
    return raw, parse_selections(raw, menu, 4)


def reopen_boundary(parent: State, witness: str) -> list[dict]:
    """Reopen up to two old boundary words while retaining editable prose."""
    left_words = [w for unit in parent.left for w in unit.split()]
    right_words = [w for unit in parent.right for w in unit.split()]
    out = []
    for left_drop in range(min(2, len(left_words) - 1) + 1):
        for right_drop in range(min(2, len(right_words) - 1) + 1):
            if left_drop == right_drop == 0:
                continue
            prefix = " ".join(left_words[:-left_drop] if left_drop else left_words)
            suffix = " ".join(right_words[right_drop:] if right_drop else right_words)
            try:
                state = state_from_anchors(prefix, suffix)
            except ValueError:
                continue
            out.append({"state": state, "witness": witness,
                        "reopened": {"left_drop": left_drop, "right_drop": right_drop},
                        "id": f"reopen-{left_drop}-{right_drop}"})
    return out


def rerank_surfaces(model: str, witness: str, surfaces: list[str], seed: int) -> tuple[str, list[int]]:
    options = [{"id": f"s{i:03d}", "text": text} for i, text in enumerate(surfaces)]
    prompt = {
        "task": "Identify the rendered passages that read most like ordinary coherent English.",
        "witness": witness,
        "rule": "Rank only the listed IDs. Do not rewrite them. Prefer one connected thought, normal syntax, and no fragments, lists, or repeated content.",
        "options": options,
    }
    response = request_json("/api/chat", {"model": model,
        "messages": [{"role": "user", "content": json.dumps(prompt)}],
        "stream": False, "think": "low" if not model.startswith("gpt-oss") else False,
        "options": {"temperature": 0.2, "num_predict": 300, "seed": seed},
    })
    raw = assistant_text(response)
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
    (("desserts", "stressed"), "Desserts to share made the guests feel welcome and not stressed."),
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
        frontier = [(root, witness, "root")]
        for depth in range(rounds):
            next_frontier = []
            for parent_index, (parent, parent_witness, parent_path_id) in enumerate(frontier):
                menu = macro_paths(parent, tries, scorer, depth=path_depth,
                                   witness=parent_witness)
                raw, selected = select_paths(model, parent_witness, parent, menu,
                                             2026091400 + depth * 100 + parent_index)
                selected_by_id = {item["id"]: item for item in selected}
                rows = [row for row in menu if row["id"] in selected_by_id]
                reopened = [] if rows else reopen_boundary(parent, parent_witness)
                traces.append({"endpoint": endpoint, "witness": witness,
                               "depth": depth, "parent": {"left": parent.left,
                               "right": parent.right, "overhang": parent.overhang,
                               "side": parent.side}, "menu": [{k: v for k, v in r.items()
                               if k != "state"} for r in menu], "raw_reply": raw,
                               "selected": selected, "reopened": [{k: v for k, v in r.items()
                               if k != "state"} for r in reopened],
                               "fallback_used": False, "rejected_no_witness": not bool(rows)})
                next_frontier.extend((row["state"], selected_by_id[row["id"]]["witness"],
                                      row["id"]) for row in rows)
                next_frontier.extend((row["state"], row["witness"], row["id"]) for row in reopened)
            frontier = next_frontier[:8]
            if not frontier:
                break
        endpoint_summaries.append({"endpoint": endpoint, "witness": witness,
                                   "terminal_states": len(frontier)})
        for index, (state, construction_witness, path_id) in enumerate(frontier):
            candidate_words = tokenize(construction_witness)
            tape = normalize_letters(construction_witness)
            if len(tape) >= MIN_LETTERS and tape == tape[::-1]:
                records.append(audit(candidate_words, index, construction_witness,
                                     endpoint, path_id))
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
