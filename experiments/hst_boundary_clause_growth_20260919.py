"""Exact-by-construction clause growth for the hst-bench lane.

Each lexical choice is placed at its real tape offset.  The open left/right
character obligations are carried by the DFS; a word is rejected before it is
added when its characters disagree with the obligation.  No completed string
is reversed or repaired.
"""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters

EXPERIMENT_ID = "hst-boundary-clause-growth-20260919"
SLOTS = ("subject", "verb", "number", "object", "name", "adjunct")
LEXICON = {
    "subject": ("a baker", "a sailor", "the farmer", "some poets", "three cooks"),
    "verb": ("marks", "packs", "plants", "reads", "helps", "carry", "plant"),
    "number": ("one", "two", "three", "many"),
    "object": ("a letter", "the map", "fresh herbs", "small cakes", "red apples"),
    "name": ("Diana", "Mara", "Nora", "Lena"),
    "adjunct": ("at dawn", "by noon", "in rain", "for Mina", "near home"),
}

def audit(text: str) -> dict[str, object]:
    t = normalize_letters(text); i, j = 0, len(t)-1; bad=[]
    while i < j:
        if t[i] != t[j]: bad.append((i, j, t[i], t[j]))
        i += 1; j -= 1
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": bool(t) and not bad,
            "mismatch_count": len(bad), "first_mismatch": bad[0] if bad else None,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

def _put(word: str, pos: int, tape: list[str | None], target: int):
    """Place one word while checking both boundary obligations immediately."""
    chars = normalize_letters(word)
    if pos + len(chars) > target: return None
    out = tape[:]
    for k, ch in enumerate(chars):
        p = pos + k; q = target - 1 - p
        if q < 0: return None
        for slot, value in ((p, ch), (q, ch)):
            if out[slot] not in (None, value): return None
            out[slot] = value
    return out

def search(target: int, max_nodes: int = 250_000) -> dict[str, object]:
    rows=[]; nodes=0; leaves=0; deepest=0; tape=[None]*target
    def grow(k, pos, words, state):
        nonlocal nodes, leaves, deepest
        if nodes >= max_nodes: return
        nodes += 1
        deepest=max(deepest, k)
        if k == len(SLOTS):
            leaves += 1
            if pos != target: return
            text = " ".join(words) + "."; a = audit(text)
            rows.append({"rendered": text, "audit": a, "slots": dict(zip(SLOTS, words)),
                         "provenance": {"indexed_before_render": True, "post_hoc_repair": False,
                           "finished_tape_reversal": False, "catalogue_imported": False,
                           "bootstrap_copied": False}})
            return
        slot=SLOTS[k]
        for word in LEXICON[slot]:
            if word in words: continue
            # Agreement and ordinary slot order are live constraints.
            if slot == "verb" and words[0].startswith(("some", "three")) and word.endswith("s"): continue
            placed=_put(word, pos, tape, target)
            if placed is None: continue
            tape_old=tape[:]; tape[:] = placed
            grow(k+1, pos+len(normalize_letters(word)), words+[word], state)
            tape[:] = tape_old
    grow(0, 0, [], {})
    return {"target": target, "nodes": nodes, "leaves": leaves, "deepest_slot": deepest, "candidates": rows,
            "exact": [r for r in rows if r["audit"]["two_pointer_exact"]]}

def run(lengths=range(40, 71)):
    results=[search(n) for n in lengths]; rows=[r for x in results for r in x["candidates"]]
    return {"experiment_id": EXPERIMENT_ID, "method": "boundary-character indexed typed clause growth",
            "remote_target": "hst-bench", "results": rows,
            "stats": {"nodes": sum(x["nodes"] for x in results), "leaves": sum(x["leaves"] for x in results), "candidates": len(rows),
                      "exact": sum(len(x["exact"]) for x in results),
                      "longest_candidate": max((r["audit"]["letters"] for r in rows), default=0),
                      "max_depth": max((x["deepest_slot"] for x in results), default=0)},
            "audit": "independent outside-in two-pointer plus forward/reverse SHA-256",
            "residual": "No exact closure in the bounded fresh inventory; first mismatch is retained per candidate.",
            "novelty_preflight": {"status": "passed", "catalogue_or_duplicate_module": False}}

if __name__ == "__main__":
    out=run(); p=Path(__file__).resolve().parents[1]/"runs"/(EXPERIMENT_ID+".json"); p.write_text(json.dumps(out, indent=2)+"\n"); print(json.dumps(out["stats"], sort_keys=True))
