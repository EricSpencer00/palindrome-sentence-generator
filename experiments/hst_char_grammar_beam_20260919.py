"""Character-level grammar beam for the hst-bench palindrome lane.

The two clauses are grown in ordinary order, one character at a time.  A
state carries both unfinished clauses and the reverse-tape obligation; words
are only admitted from a small typed lexicon and no completed tape is ever
reversed or repaired.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters

EXPERIMENT_ID = "hst-char-grammar-beam-20260919"
WORDS = {
    "det": ("a", "the", "some"), "adj": ("calm", "red", "old"),
    "subj": ("baker", "sailor", "farmer", "poet"),
    "verb_s": ("marks", "packs", "reads", "helps"),
    "verb_p": ("mark", "pack", "read", "help"),
    "obj": ("map", "bread", "letters", "cakes"),
    "prep": ("at", "by", "in"), "place": ("dawn", "noon", "rain", "home"),
}
FRAMES = (("det", "adj", "subj", "verb", "det", "obj"),
          ("det", "subj", "verb", "prep", "place"))

def audit(text: str) -> dict[str, object]:
    t = normalize_letters(text)
    i, j, bad = 0, len(t) - 1, []
    while i < j:
        if t[i] != t[j]: bad.append((i, j, t[i], t[j]))
        i += 1; j -= 1
    f, r = hashlib.sha256(t.encode()).hexdigest(), hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": bool(t) and not bad,
            "mismatch_count": len(bad), "first_mismatch": bad[0] if bad else None,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

def _choices(kind: str, subject: str | None) -> tuple[str, ...]:
    if kind == "verb": return WORDS["verb_p" if subject in ("some",) else "verb_s"]
    return WORDS[kind]

def search(target: int, beam: int = 400, max_nodes: int = 100_000) -> dict[str, object]:
    # Each state contains two ordinary-order token streams and an indexed tape.
    states = [{"a": [], "b": [], "ia": 0, "ib": 0, "tape": [None] * target,
               "fa": FRAMES[0], "fb": FRAMES[1], "sa": None, "sb": None,
               "slots": set(), "score": 0}]
    nodes = 0; residual = []
    def place(tape, pos, word):
        chars = normalize_letters(word)
        if pos + len(chars) > target: return None
        out = tape[:]
        for k, ch in enumerate(chars):
            p, q = pos + k, target - 1 - (pos + k)
            if out[p] not in (None, ch) or out[q] not in (None, ch): return None
            out[p] = out[q] = ch
        return out
    # Token transitions are atomic but obligations are checked per character.
    while states and nodes < max_nodes:
        nxt = []
        for st in states:
            if st["ia"] == len(st["fa"]) and st["ib"] == len(st["fb"]):
                if sum(x is not None for x in st["tape"]) == target:
                    rendered = " ".join(st["a"] + [";"] + st["b"]) + "."
                    a = audit(rendered)
                    row = {"rendered": rendered, "audit": a, "state": st}
                    if a["two_pointer_exact"]: return {"target": target, "nodes": nodes, "exact": [row], "residual": []}
                    residual.append(row)
                continue
            side = "a" if st["ia"] < len(st["fa"]) else "b"
            idx = st["ia"] if side == "a" else st["ib"]
            kind = st["fa"][idx] if side == "a" else st["fb"][idx]
            choices = _choices(kind, st["sa"] if side == "a" else st["sb"])
            for word in choices:
                if word in (st["a"] if side == "a" else st["b"]): continue
                if kind == "verb" and side == "a" and st["sa"] == "some" and word.endswith("s"): continue
                pos = sum(len(normalize_letters(w)) for w in st[side])
                # Space is grammar state, not part of the tape; its boundary is explicit.
                placed = place(st["tape"], pos + (1 if st[side] else 0), word)
                if placed is None: continue
                ns = dict(st); ns["tape"] = placed; ns[side] = st[side] + [word]
                ns["slots"] = st["slots"] | {f"{side}:{kind}"}; ns["score"] += len(word)
                ns["i" + side] = idx + 1
                if kind == "det": ns["s" + side] = word
                nxt.append(ns); nodes += 1
                if nodes >= max_nodes: break
        states = sorted(nxt, key=lambda x: (-x["score"], len(x["slots"])))[:beam]
    return {"target": target, "nodes": nodes, "exact": [], "residual": residual[:20],
            "residual_reason": "live character obligation or bounded beam exhausted"}

def run(lengths=(38, 39, 40, 42, 45)):
    results = [search(n) for n in lengths]
    return {"experiment_id": EXPERIMENT_ID, "method": "character-level typed grammar beam/DFS",
            "remote_target": "hst-bench", "results": results,
            "stats": {"targets": len(results), "exact": sum(bool(x["exact"]) for x in results),
                      "longest_exact": max((x["target"] for x in results if x["exact"]), default=0),
                      "nodes": sum(x["nodes"] for x in results)},
            "audit": "independent outside-in two-pointer plus forward/reverse SHA-256",
            "provenance": {"catalogue_imported": False, "repeated_units": False,
                           "finished_tape_reversal": False, "post_hoc_repair": False}}

if __name__ == "__main__":
    out = run(); p = Path(__file__).resolve().parents[1] / "runs" / (EXPERIMENT_ID + ".json")
    p.write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out["stats"], sort_keys=True))
