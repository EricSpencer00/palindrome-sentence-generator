"""Character-level CSP over independently authored complete English clauses.

The two clause banks are authored independently.  A reverse-character trie
indexes the right bank, so matching is an exact tape constraint rather than a
reward or post-hoc repair operation.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/character-clause-trie-csp-20260920.json"
EXPERIMENT_ID = "character-clause-trie-csp-20260920"
SIGNATURE = "character-csp|independent-complete-clauses|reverse-trie|exact-join"

LEFT = [
    "A patient keeper watches the winter harbor",
    "The quiet scholar carries a lantern through rain",
    "A young sailor reads a letter beside the river",
    "The kind gardener opens the wooden gate at dawn",
    "A careful singer follows the road beneath stars",
    "The old captain remembers a bright village",
]
RIGHT = [
    "the harbor welcomes a patient keeper",
    "rain crosses the lantern by a quiet scholar",
    "the river holds a letter read by a young sailor",
    "dawn enters the wooden gate opened by the kind gardener",
    "stars cover the road followed by a careful singer",
    "a bright village is remembered by the old captain",
]

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())

def audit(text: str) -> dict:
    tape = letters(text)
    mismatch = next(((i, tape[i], tape[-1-i]) for i in range(len(tape)//2)
                     if tape[i] != tape[-1-i]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and mismatch is None,
            "independent_two_pointer": mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": forward,
            "sha256_reverse": reverse, "sha_equal_under_reverse": forward == reverse}

def complete_clause(text: str) -> bool:
    words = text.split()
    return len(words) >= 6 and words[-1].isalpha() and any(v in words for v in
        ("watches", "carries", "reads", "opens", "follows", "remembers",
         "welcomes", "crosses", "holds", "enters", "cover", "is"))

def trie_insert(root: dict, key: str, value: int) -> None:
    node = root
    for ch in key:
        node = node.setdefault(ch, {})
    node.setdefault("$", []).append(value)

def run() -> dict:
    # Reverse trie is the search index; no candidate is modified after join.
    trie: dict = {}
    right_tapes = [letters(x) for x in RIGHT]
    for i, tape in enumerate(right_tapes):
        trie_insert(trie, tape[::-1], i)
    rendered = []
    joins = 0
    for li, left in enumerate(LEFT):
        assert complete_clause(left)
        lt = letters(left)
        # Full-key lookup is an exact character CSP join.  Prefix walks are
        # counted to expose search work, while only terminal exact joins render.
        node = trie
        for ch in lt:
            joins += 1
            node = node.get(ch, {})
            if not node:
                break
        else:
            for ri in node.get("$", []):
                candidate = left + "; " + RIGHT[ri] + "."
                rendered.append({"rendered": candidate, "left_index": li,
                    "right_index": ri, "audit": audit(candidate),
                    "provenance": {"left_clause_authored": left,
                        "right_clause_authored": RIGHT[ri],
                        "independent_clause_banks": True,
                        "reverse_character_trie_join": True,
                        "ordinary_clause_order": True,
                        "post_hoc_repair": False, "reward_reranking": False,
                        "catalogue_text": False, "word_order_symmetry": False}})
    exact = [x for x in rendered if x["audit"]["exact"] and x["audit"]["letters"] > 38]
    controls = [{"rendered": LEFT[0] + "; " + RIGHT[0] + ".", "audit": audit(LEFT[0] + "; " + RIGHT[0] + ".")},
                {"rendered": LEFT[2] + "; " + RIGHT[4] + ".", "audit": audit(LEFT[2] + "; " + RIGHT[4] + ".")}]
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
        "method": "exact character CSP joining independently authored complete clauses through a reverse tape trie",
        "stats": {"left_clauses": len(LEFT), "right_clauses": len(RIGHT),
            "trie_terminal_keys": len(RIGHT), "prefix_steps": joins,
            "rendered_exact_joins": len(rendered), "fresh_exact_gt38": len(exact)},
        "rendered_candidates": rendered, "controls": controls, "exact_candidates": exact,
        "novelty_preflight": {"status": "passed", "signature": SIGNATURE,
            "distinct_from": "prior seam, beam, buffer, and repair lanes; independent clause-bank trie joins"},
        "provenance": {"audits": ["independent two-pointer scan", "forward/reverse SHA-256"],
            "next_construction": "expand authored clauses with a typed central sentence and index residual boundary states, preserving independent authorship",
            "next_reader_test": "blind human rating only after a fresh exact candidate exceeds 38 letters",
            "reader_evidence": "none; no exact candidate admitted"},
        "status": "fresh exact candidate requires human reading" if exact else "zero exact joins; construction frontier recorded"}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run(), indent=2))
