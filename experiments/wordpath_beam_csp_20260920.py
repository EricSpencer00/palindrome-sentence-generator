"""Beam search over an observed word-path lattice with live palindrome cells.

This is deliberately different from the fixed grammar CSP: both sides may
advance one word at a time, and the beam is scored only by observed adjacent
word transitions.  Character obligations are still hard constraints, so no
finished string is reversed and no repair is applied after a candidate closes.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from experiments.bilateral_grammar_csp_20260920 import _consume
from experiments.forward_lexicalized_grammar_20260920 import independent_audit, letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/wordpath-beam-csp-20260920.json"


@dataclass(frozen=True)
class State:
    left: tuple[str, ...]
    right_rev: tuple[str, ...]
    left_residual: str
    right_residual: str
    score: float


def load_lattice(path=ROOT / "data/count_2w.txt", edge_limit=200_000, vocab_limit=500):
    nexts = defaultdict(dict)
    prevs = defaultdict(dict)
    rows = 0
    with Path(path).open(errors="ignore") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) != 2:
                continue
            pair = fields[0].casefold().split()
            if len(pair) != 2 or not all(re.fullmatch(r"[a-z]+", w) for w in pair):
                continue
            try:
                count = int(fields[1])
            except ValueError:
                continue
            a, b = pair
            nexts[a][b] = max(count, nexts[a].get(b, 0))
            prevs[b][a] = max(count, prevs[b].get(a, 0))
            rows += 1
            if rows >= edge_limit:
                break
    totals = {a: sum(v.values()) for a, v in nexts.items()}
    vocab = sorted(
        set(nexts) | set(prevs),
        key=lambda w: (-(sum(nexts.get(w, {}).values()) + sum(prevs.get(w, {}).values())), w),
    )[:vocab_limit]
    keep = set(vocab)
    next_table = {
        a: tuple(sorted((b for b in bs if b in keep), key=lambda b: -bs[b]))
        for a, bs in nexts.items() if a in keep
    }
    prev_table = {
        b: tuple(sorted((a for a in bs if a in keep), key=lambda a: -bs[a]))
        for b, bs in prevs.items() if b in keep
    }
    logprob = {
        (a, b): math.log(count / totals[a])
        for a, bs in nexts.items() for b, count in bs.items()
        if a in keep and b in keep
    }
    return tuple(vocab), next_table, prev_table, logprob, rows


def _admissible(words: tuple[str, ...]) -> bool:
    if len(words) != len(set(words)):
        return False
    if any(len(letters(word)) > 1 and letters(word) == letters(word)[::-1] for word in words):
        return False
    middle = len(words) // 2
    return not (len(words) % 2 == 0 and words[:middle] == tuple(reversed(words[middle:])))


def search(vocab, nexts, prevs, logprob, *, max_words=14, beam_width=50_000):
    beam = [State((), (), "", "", 0.0)]
    exact = {}
    stats = {"states": 0, "character_prunes": 0, "repeat_prunes": 0, "beam_prunes": 0}
    for _ in range(max_words):
        expanded = []
        for state in beam:
            stats["states"] += 1
            left_choices = vocab if not state.left else nexts.get(state.left[-1], ())
            right_choices = vocab if not state.right_rev else prevs.get(state.right_rev[-1], ())
            if state.left_residual and not state.right_residual:
                pairs = ((None, b) for b in right_choices)
            elif state.right_residual and not state.left_residual:
                pairs = ((a, None) for a in left_choices)
            else:
                pairs = ((a, b) for a in left_choices for b in right_choices
                         if a[0] == b[-1])
            for a, b in pairs:
                chosen = state.left + state.right_rev
                if (a is not None and a in chosen) or (b is not None and b in chosen):
                    stats["repeat_prunes"] += 1
                    continue
                new_left = state.left + ((a,) if a is not None else ())
                new_right = state.right_rev + ((b,) if b is not None else ())
                left_stream = state.left_residual + (letters(a) if a is not None else "")
                right_stream = state.right_residual + (letters(b)[::-1] if b is not None else "")
                residual = _consume(left_stream, right_stream)
                if residual is None:
                    stats["character_prunes"] += 1
                    continue
                edge_score = 0.0
                if a is not None and state.left:
                    edge_score += logprob.get((state.left[-1], a), -20.0)
                if b is not None and state.right_rev:
                    edge_score += logprob.get((b, state.right_rev[-1]), -20.0)
                child = State(new_left, new_right, residual[0], residual[1], state.score + edge_score)
                if not child.left_residual and not child.right_residual:
                    words = child.left + tuple(reversed(child.right_rev))
                    if len(words) >= 4 and _admissible(words):
                        text = " ".join(words)
                        audit = independent_audit(text)
                        if audit["exact"] and audit["letters"] >= 39:
                            exact[text] = {"length": audit["letters"], "rendered": text + ".",
                                            "words": words, "audit": audit,
                                            "score": child.score,
                                            "provenance": {"observed_bigram_lattice": True,
                                                           "one_sided_residual_advancement": True,
                                                           "live_character_residual": True,
                                                           "beam_search": True,
                                                           "finished_tape_reversal": False,
                                                           "post_hoc_repair": False,
                                                           "word_order_mirroring": False,
                                                           "catalogue_text": False}}
                        # Extending an exact closure would make it a nested
                        # word-aligned palindrome, so closures are terminal.
                        continue
                expanded.append(child)
        if not expanded:
            break
        expanded.sort(key=lambda s: (s.score + 0.04 * sum(map(len, s.left + s.right_rev)),
                                     len(s.left) + len(s.right_rev)), reverse=True)
        if len(expanded) > beam_width:
            stats["beam_prunes"] += len(expanded) - beam_width
            expanded = expanded[:beam_width]
        beam = expanded
    stats["final_beam"] = len(beam)
    stats["exact"] = len(exact)
    stats["status"] = "SAT" if exact else "UNSAT_OR_BEAM_EXHAUSTED"
    return {"paths": sorted(exact.values(), key=lambda row: (-row["length"], row["rendered"])),
            "stats": stats}


def run(*, edge_limit=200_000, vocab_limit=500, max_words=14, beam_width=50_000):
    vocab, nexts, prevs, logprob, rows = load_lattice(edge_limit=edge_limit, vocab_limit=vocab_limit)
    result = search(vocab, nexts, prevs, logprob, max_words=max_words, beam_width=beam_width)
    result["experiment_id"] = "wordpath-beam-csp-20260920"
    result["provenance"] = {
        "observed_rows": rows, "vocab": len(vocab), "edge_limit": edge_limit,
        "vocab_limit": vocab_limit, "max_words": max_words, "beam_width": beam_width,
        "method": "variable word paths with one-sided residual advancement and observed-bigram beam",
        "one_sided_residual_advancement": True,
        "reader_gate": "closed unless an exact row survives independent audit and intact-versus-shuffled blinded reading",
        "next_construction": "add a typed clause gate to exact survivors without altering their character tape",
    }
    return result


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["paths"][:20]:
        print(row["rendered"])
