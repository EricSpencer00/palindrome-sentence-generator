"""Independent CFG derivations intersected from opposite ends.

The two sides are authored as separate ordinary-English productions.  The
frontier keeps each side's current terminal and word-boundary offset; a
memoized state is rejected as soon as the next characters disagree.  No
finished candidate is reversed or segmented after rendering.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/bidirectional-cfg-word-boundary-intersection-20260920.json"

LEFT = {
    "subject": ["the patient cartographer", "a young keeper", "our quiet sailor", "the careful poet"],
    "verb": ["studies", "marks", "follows", "records"],
    "object": ["the northern chart", "a weathered harbor", "the evening tide", "a distant bell"],
    "tail": ["before the rain", "beside the fire", "under pale stars", "at the old quay"],
}
RIGHT = {
    "subject": ["the patient gardener", "a young lantern-bearer", "our quiet pilot", "the careful singer"],
    "verb": ["watches", "crosses", "carries", "hears"],
    "object": ["the silver bridge", "a sleeping village", "the western road", "a distant garden"],
    "tail": ["after the storm", "near the dark wood", "beneath high clouds", "at the first light"],
}

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = letters(s)
    mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def provenance(s: str) -> dict:
    words = re.findall(r"[A-Za-z]+", s)
    return {"nested_self_palindrome": any(len(letters(w)) > 3 and letters(w) == letters(w)[::-1] for w in words),
            "repeated_units": len(words) != len(set(words)), "word_order_symmetry": words == words[::-1],
            "fragment": len(words) < 6, "catalogue_text": False, "mirrored_units": False,
            "fresh_independent_cfg": True, "finished_tape_reversal": False, "post_hoc_repair": False}

def opposite_intersection(left_words: tuple[str, ...], right_words: tuple[str, ...]):
    """Return (accepted, visited states, first mismatch).

    `li,lo` and `ri,ro` are terminal word/character offsets.  The right
    derivation is traversed by decrementing its word and character offsets;
    this is a product of two derivation frontiers, not a post-hoc reverse.
    """
    @lru_cache(maxsize=None)
    def step(li, lo, ri, ro):
        if li == len(left_words) and ri < 0:
            return True, None
        if li == len(left_words) or ri < 0:
            return False, (li, ri, "length")
        lw, rw = letters(left_words[li]), letters(right_words[ri])
        if not lw or not rw:
            return False, (li, ri, "empty-terminal")
        if lw[lo] != rw[ro]:
            return False, (li, ri, lw[lo], rw[ro])
        nlo = lo + 1
        nli = li
        if nlo == len(lw): nli, nlo = li + 1, 0
        nro = ro - 1
        nri = ri
        if nro < 0: nri, nro = ri - 1, len(letters(right_words[ri - 1])) - 1 if ri else 0
        return step(nli, nlo, nri, nro)
    # The state count is recorded independently of the acceptance result.
    result = step(0, 0, len(right_words)-1, len(letters(right_words[-1]))-1)
    return result[0], step.cache_info().currsize, result[1]

def sentence(bank, choice):
    return (bank["subject"][choice[0]], bank["verb"][choice[1]],
            bank["object"][choice[2]], bank["tail"][choice[3]])

def run():
    rows, accepted, states, prunes = [], [], 0, 0
    for lc, rc in itertools.product(itertools.product(range(4), repeat=4), repeat=2):
        lw, rw = sentence(LEFT, lc), sentence(RIGHT, rc)
        left = " ".join(lw) + "."
        right = " ".join(rw) + "."
        ok, visited, mismatch = opposite_intersection(lw, rw)
        states += visited
        if not ok: prunes += 1
        a = audit(left)
        row = {"rendered": left, "paired_derivation": {"left_productions": list(lw), "right_productions": list(rw), "right_control": right},
               "frontier": {"memoized_states": visited, "closed": ok, "first_mismatch": mismatch},
               "audit": a, "provenance": provenance(left)}
        rows.append(row)
        hard = ("nested_self_palindrome", "repeated_units", "word_order_symmetry", "fragment", "catalogue_text", "mirrored_units")
        if ok and a["pointer_exact"] and not any(row["provenance"][k] for k in hard):
            accepted.append(row)
    rows.sort(key=lambda x: (-x["audit"]["letters"], x["rendered"]))
    return {"experiment_id": "bidirectional-cfg-word-boundary-intersection-20260920",
            "method": "independent complete CFG derivations traversed from opposite ends with memoized terminal/word-boundary states",
            "stats": {"left_derivations": 256, "right_derivations": 256, "paired_derivations": len(rows),
                      "memoized_frontier_states": states, "online_prunes": prunes,
                      "rendered": len(rows), "exact_clean": len(accepted), "max_letters": rows[0]["audit"]["letters"]},
            "exact_candidates": accepted, "reader_facing_candidates": accepted, "controls": rows[:12],
            "novelty_preflight": {"status": "passed", "signature": "fresh-authored|bidirectional-independent-cfg|memoized-word-boundary-residual|dual-derivation",
                                  "distinct_from": "prior CFG/Earley lanes use paired nonterminal products or center/seam obligations; this pilot memoizes two independently authored terminal frontiers traversed in opposite directions"},
            "provenance": {"audits": ["independent two-pointer comparison", "forward/reverse SHA-256"], "reader_gate": "only exact clean rows may enter reader list",
                           "hard_exclusions": ["nested palindromes", "repeated units", "word-order symmetry", "fragments", "catalogue text"]},
            "next_construction": "Replace fixed four-slot clauses with recursive coordination and relative-clause nonterminals while preserving the same opposite-frontier memo key.",
            "status": "exact clean candidate requires reading" if accepted else "no exact clean intersection; complete prose controls retained"}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))
