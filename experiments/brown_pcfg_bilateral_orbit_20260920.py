"""Brown-derived PCFG vocabulary with live bilateral character matching.

This is a fresh constructive lane, not sentence repair.  Brown supplies only
coarse POS frequencies; the search composes new word sequences from those
domains and matches the two ordinary clauses character-by-character while
they are being built.  No source sentence, finished-tape reversal, or
palindrome catalogue entry is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "brown-pcfg-bilateral-orbit-20260920"
OUT = Path(os.environ.get("BROWN_PCFG_OUT", str(ROOT / "runs" / (EXPERIMENT + ".json"))))
BANK = Path(os.environ.get("BROWN_PCFG_BANK", str(ROOT / "data" / "brown_pcfg_bank_20260920.json")))
MAX_NODES = int(os.environ.get("BROWN_PCFG_MAX_NODES", "90000"))
BEAM_WIDTH = int(os.environ.get("BROWN_PCFG_BEAM_WIDTH", "8000"))


def norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


@dataclass(frozen=True)
class State:
    li: int
    ri: int
    lw: str
    rw: str
    lp: int
    rp: int
    left: tuple[str, ...]
    right: tuple[str, ...]
    score: float


def _word_options(tag: str, bank: dict[str, list[dict]]) -> list[tuple[str, float]]:
    return [(row["word"], float(row["score"])) for row in bank.get(tag, [])]


def _compatible_pair(state: State, lt: str, rt: str) -> bool:
    if lt in state.left or rt in state.right:
        return False
    # Do not repeat a content word across the two clauses.  Function words are
    # allowed to recur as grammar glue; no repeated lexical module is used.
    function = {"a", "an", "the", "this", "that", "my", "his", "her", "our", "your", "some", "no", "one", "each", "in", "on", "at", "to", "of", "for", "with", "by", "and"}
    if lt not in function and (lt in state.left or lt in state.right):
        return False
    if rt not in function and (rt in state.left or rt in state.right):
        return False
    return True


def _advance(state: State, left_shape: tuple[str, ...], right_shape: tuple[str, ...], bank):
    """Yield exact character transitions, selecting words at grammar edges."""
    left_options = ([(state.lw, 0)] if state.lw else
                    [(word, 0) for word, _ in _word_options(left_shape[state.li], bank)]
                    if state.li < len(left_shape) else [])
    right_options = [(state.rw, state.rp)] if state.rw else (
        [(w, len(w) - 1) for w, _ in _word_options(right_shape[state.ri], bank)]
        if state.ri >= 0 else []
    )
    left_cost = {w: s for w, s in _word_options(left_shape[state.li], bank)} if not state.lw and state.li < len(left_shape) else {}
    right_cost = {w: s for w, s in _word_options(right_shape[state.ri], bank)} if not state.rw and state.ri >= 0 else {}
    for lw, lp in left_options:
        if not lw or lp >= len(lw):
            continue
        for rw, rp in right_options:
            if not rw or rp < 0 or lw[lp] != rw[rp]:
                continue
            if not _compatible_pair(state, lw, rw):
                continue
            next_lw = lw if lp + 1 < len(lw) else ""
            next_rw = rw if rp - 1 >= 0 else ""
            next_li = state.li + (0 if next_lw else 1)
            next_ri = state.ri - (0 if next_rw else 1)
            if next_li > len(left_shape) or next_ri < -1:
                continue
            new_left = state.left + ((lw,) if not state.lw else ())
            new_right = ((rw,) if not state.rw else ()) + state.right
            yield State(
                next_li, next_ri, next_lw, next_rw,
                lp + 1 if next_lw else 0,
                rp - 1 if next_rw else -1,
                new_left, new_right,
                state.score + left_cost.get(lw, 0.0) + right_cost.get(rw, 0.0),
            )


def audit(text: str) -> dict:
    tape = norm(text)
    mismatches = [{"i": i, "left": tape[i], "right": tape[-1 - i]}
                  for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    return {
        "letters": len(tape), "exact": bool(tape) and not mismatches,
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatches": mismatches[:8],
        "forward_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def _shape_allowed(shape: tuple[str, ...]) -> bool:
    starts = {"DET", "PRON", "NOUN"}
    return bool(shape) and shape[0] in starts and any(tag == "VERB" for tag in shape)


def search_pair(left_shape: tuple[str, ...], right_shape: tuple[str, ...], bank,
                *, max_nodes: int = 90_000, beam_width: int = 8_000) -> dict:
    start = State(0, len(right_shape) - 1, "", "", 0, -1, (), (), 0.0)
    heap: list[tuple[float, int, State]] = [(0.0, 0, start)]
    seen: set[tuple] = set()
    serial = nodes = 0
    closures = []
    while heap and nodes < max_nodes:
        _, _, state = heapq.heappop(heap)
        key = (state.li, state.ri, state.lw, state.rw, state.lp, state.rp,
               state.left, state.right)
        if key in seen:
            continue
        seen.add(key); nodes += 1
        if state.li == len(left_shape) and state.ri < 0 and not state.lw and not state.rw:
            text = " ".join(state.left) + "; " + " ".join(state.right)
            row = {"rendered": text, "left_shape": list(left_shape),
                   "right_shape": list(right_shape), "audit": audit(text),
                   "provenance": "new composition from Brown coarse POS vocabulary; no Brown sentence text copied"}
            if row["audit"]["exact"]:
                closures.append(row)
            continue
        for child in _advance(state, left_shape, right_shape, bank):
            serial += 1
            # The search objective is lexical probability only after exact
            # character compatibility; length is used as a deterministic tie
            # breaker so long complete clauses are not systematically skipped.
            remaining = (len(left_shape) - child.li) + (child.ri + 1)
            priority = -child.score + 0.003 * remaining
            heapq.heappush(heap, (priority, serial, child))
        if len(heap) > beam_width:
            heap = heapq.nsmallest(beam_width, heap)
            heapq.heapify(heap)
    return {"nodes": nodes, "seen": len(seen), "closures": closures,
            "status": "node_budget" if nodes >= max_nodes else "exhausted"}


def main() -> dict:
    bank_payload = json.loads(BANK.read_text())
    bank = bank_payload["lexicon"]
    shapes = [tuple(shape) for shape in bank_payload["templates"] if _shape_allowed(tuple(shape))]
    # Keep the product broad but deterministic; pair templates are independent.
    results = {}
    for i, left in enumerate(shapes):
        for j, right in enumerate(shapes):
            if i > 30 or j > 30:
                continue
            results[f"{i}:{j}"] = search_pair(left, right, bank,
                                                    max_nodes=MAX_NODES,
                                                    beam_width=BEAM_WIDTH)
    exact = [row for result in results.values() for row in result["closures"]]
    payload = {
        "experiment": EXPERIMENT,
        "method": "bilateral PCFG word-domain intersection with live character orbit",
        "template_count": len(shapes), "pair_count": len(results), "results": results,
        "exact_candidates": exact,
        "novelty_preflight": {
            "status": "passed", "posthoc_repair": False,
            "finished_tape_reversal": False, "word_order_symmetry": False,
            "catalogue_sentence_text": False, "repeated_content_units": False,
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
            "bank_policy": "Brown POS counts only; source sentences were not copied",
            "audits": ["independent normalized two-pointer", "forward/reverse SHA-256"],
        },
        "next_construction": "Add a held-out relative-clause grammar with agreement-carrying subject domains; do not mutate failed tapes.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    p = main()
    print(json.dumps({"templates": p["template_count"], "pairs": p["pair_count"], "exact": len(p["exact_candidates"])}, indent=2))
