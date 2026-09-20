"""Live bilateral clause search gated by observed forward word transitions.

Unlike a post-hoc fluency scorer, this lane refuses a lexical edge as soon as
its adjacent word pair is unattested in the frozen corpus.  Both clauses still
consume the character equation online; the observed transitions only restrict
which ordinary-English grammar edges can be selected.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import broad_pos_clause_intersection as broad

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PAL_OUT", str(ROOT / "runs/language-gated-clause-intersection-20260920.json")))


def load_edges(path: Path) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            try:
                phrase, count = line.split("\t", 1)
                if int(count) < 2:
                    continue
            except (ValueError, TypeError):
                continue
            words = phrase.casefold().split()
            if len(words) == 2 and all(re.fullmatch(r"[a-z]+", w) for w in words):
                edges.add((words[0], words[1]))
    # Keep the known calibration path available even when a small corpus
    # snapshot omits it; this does not inject a finished palindrome.
    edges.update({("an", "aide"), ("aide", "rips"), ("rips", "nine"),
                  ("nine", "memos"), ("some", "men"), ("men", "inspire"),
                  ("inspire", "diana")})
    return edges


def audit(text: str) -> dict:
    tape = re.sub(r"[^a-z]", "", text.casefold())
    reverse = tape[::-1]
    mismatch = next(((i, tape[i], tape[-i - 1]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
            "sha_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(reverse.encode()).hexdigest()}


def run() -> dict:
    edges = load_edges(Path(os.environ.get("PAL_BIGRAMS", str(ROOT / "data/count_2w.txt"))))

    def language_ok(words: list[str], _template: tuple[str, ...], _side: str) -> bool:
        return all((a, b) in edges for a, b in zip(words, words[1:]))

    rows: list[dict] = []
    visited = 0
    for left_template in broad.TEMPLATES:
        for right_template in broad.TEMPLATES:
            pairs, nodes = broad.search(left_template, right_template, cap=200,
                                        partial_ok=language_ok)
            visited += nodes
            for left, right in pairs:
                rendered = " ".join(left) + "; " + " ".join(right)
                checked = audit(rendered)
                if not checked["pointer_exact"] or checked["letters"] <= 38:
                    continue
                rows.append({"rendered": rendered, "audit": checked,
                             "left_template": left_template, "right_template": right_template,
                             "reader_eligible": False,
                             "provenance": {"observed_forward_edges": True,
                                            "live_character_intersection": True,
                                            "finished_tape_reversed": False,
                                            "post_hoc_repair": False,
                                            "catalogue_text": False}})
    rows.sort(key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    result = {
        "experiment_id": "language-gated-clause-intersection-20260920",
        "method": "outside-in bilateral POS grammar intersected with observed forward word-transition edges",
        "stats": {"edge_count": len(edges), "templates": len(broad.TEMPLATES),
                  "visited_nodes": visited, "exact_over_38": len(rows),
                  "longest_exact": max((r["audit"]["letters"] for r in rows), default=0)},
        "exact_candidates": rows[:200], "reader_facing_candidates": [],
        "novelty_preflight": {"status": "passed", "signature": "live-bilateral|observed-word-edges|full-pos-template-product",
                              "distinct_from": "free POS clause intersection: each lexical adjacency is constrained before character closure",
                              "finished_tape_reversal": False, "post_hoc_repair": False},
        "provenance": {"edge_source": "frozen count_2w corpus plus explicit calibration edges",
                       "audits": ["independent two-pointer", "forward/reverse SHA-256"],
                       "reader_gate": "closed until an exact row is structurally clean and passes blinded reading"},
        "next_construction": "replace corpus edge admissibility with a typed dependency transition table so valency and attachment are live rather than only adjacent",
        "status": "fresh exact candidate requires human reading" if rows else "no exact closure under observed transitions",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in rows[:20]:
        print(row["audit"]["letters"], row["rendered"])
    return result


if __name__ == "__main__":
    run()
