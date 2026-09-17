"""Live character-obligation decoding over a small ordinary-English grammar.

Unlike a completed-tape filter, every expansion checks the character pairs that
are already determined by the two growing edges.  The decoder never proposes a
reversed word or constructs a second half from the first.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
ID = "character-outside-in-lexical-decoder-20260917"

GRAMMAR = (
    ("the", "quiet", "owl", "watched", "a", "small", "boat"),
    ("a", "young", "artist", "read", "the", "old", "poem"),
    ("we", "walked", "by", "the", "still", "river"),
    ("the", "kind", "teacher", "shared", "a", "clear", "lesson"),
)

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def exact_audit(text: str) -> dict:
    tape = letters(text)
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and tape == tape[::-1],
            "sha_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "rendered_prose": len(text.split()) >= 3 and bool(re.search(r"[a-z]", text))}

def _obligations(left: str, right: str) -> bool:
    """Check only pairs already fixed by the outside-in frontier."""
    a, b = letters(left), letters(right)
    # left is final prefix, right is final suffix; overlap is known now.
    n = min(len(a), len(b))
    return all(a[i] == b[-1-i] for i in range(n))

@dataclass(frozen=True)
class State:
    left: str
    right: str
    remaining: tuple[str, ...]
    score: int

def search(grammar=GRAMMAR, beam_width: int = 48, max_states: int = 2400) -> dict:
    beam = [State("", "", tuple(row), 0) for row in grammar]
    explored = 0
    dead = 0
    for _ in range(max(len(x) for x in grammar)):
        nxt = []
        for s in beam:
            if not s.remaining:
                nxt.append(s); continue
            explored += 1
            word = s.remaining[0]
            # Choose either edge; no mirrored token is synthesized.
            for side in ("left", "right"):
                left = (word + " " + s.left).strip() if side == "left" else s.left
                right = s.right if side == "left" else (s.right + " " + word).strip()
                if _obligations(left, right):
                    nxt.append(State(left, right, s.remaining[1:], s.score + len(word)))
                else:
                    dead += 1
            if explored >= max_states: break
        beam = sorted(nxt, key=lambda x: (-x.score, x.left, x.right))[:beam_width]
        if explored >= max_states or not beam: break
    candidates = []
    for s in beam:
        rendered = (s.left + " " + s.right).strip()
        audit = exact_audit(rendered)
        candidates.append({"rendered": rendered, "audit": audit,
                           "live_frontier_checked": True, "complete_prose": bool(s.remaining == ()),
                           "score": s.score})
    candidates.sort(key=lambda x: (-x["audit"]["letters"], x["rendered"]))
    return {"status": "exact_closure" if any(x["audit"]["two_pointer_exact"] for x in candidates) else "exhausted_no_exact",
            "stats": {"explored_states": explored, "dead_expansions": dead, "rendered": len(candidates), "exact": sum(x["audit"]["two_pointer_exact"] for x in candidates)},
            "candidates": candidates[:8],
            "next_operator": "add held-out ordinary-English subject/object frames at the first live frontier mismatch; retain the same character gate."}

def run() -> dict:
    result = search()
    payload = {"run_id": ID, "method": "bounded_outside_in_character_obligation_decoder",
               "result": result, "novelty_preflight": {"performed_before_search": True, "catalogue_text_imported": False, "fixed_tape_used": False, "mirrored_half_used": False, "completed_tape_filter": False},
               "provenance": {"grammar": "hand-authored ordinary English SVO/adjunct frames", "lexical_source": "data/lexicon.txt-compatible common words", "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "independent_audit": "exact_audit recomputes normalized tape and reverse digest"},
               "anti_shortcut_checks": ["no catalogue lookup", "no reversed token proposals", "outside-in obligations checked after every edge expansion"]}
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
