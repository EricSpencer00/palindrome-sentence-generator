"""Quarantined fixed-clause comparison diagnostic, not a trie search.

The historical name overstates the implementation: every supplied frame is
fully rendered before comparison, and the tries do not choose any characters
or boundaries. Zero closures here are evidence only about that finite bank.
"""
from dataclasses import dataclass
from typing import Iterable
import hashlib
from pathlib import Path
import re

EXPERIMENT_ID = "semantic-frame-trie-equation-20260917"
SIGNATURE = "fixed-frame-cross-product|comparison-only|no-generative-search|exact-audit"

@dataclass(frozen=True)
class Frame:
    agent: str
    event: str
    theme: str
    setting: str
    reporting: str

    def clause(self) -> str:
        return f"{self.agent} {self.event} {self.theme} {self.setting}, {self.reporting}."

# Independent authored banks; no catalogue strings or palindromic phrase list.
FRAMES = (
    Frame("a courier", "reports", "the signal", "at dawn", "a clerk listens"),
    Frame("the mason", "marks", "a narrow arch", "by water", "a scout records"),
    Frame("a pilot", "notes", "the quiet harbor", "before rain", "a keeper replies"),
    Frame("one ranger", "checks", "a red marker", "near camp", "a guide remembers"),
)

class CharTrie:
    def __init__(self, words: Iterable[str] = ()):
        self.next = {}
        self.terminal = False
        for word in words: self.add(word)
    def add(self, word: str):
        node = self
        for ch in word:
            node = node.next.setdefault(ch, CharTrie())
        node.terminal = True
    def walk(self, prefix: str = ""):
        yield prefix, self.terminal
        for ch, child in self.next.items(): yield from child.walk(prefix + ch)

WORD_BANK = tuple(sorted({w for f in FRAMES for w in re.findall(r"[a-z]+", f.clause())}))
TRIE = CharTrie(WORD_BANK)

def normalize(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalpha())

def _word_paths(node, required: str, offset: int = 0):
    """Yield trie words whose characters satisfy the current obligations."""
    for word, terminal in node.walk():
        if terminal and word and all(offset + i >= len(required) or c == required[offset+i]
                                    for i, c in enumerate(word)):
            yield word

def search(frames: Iterable[Frame] = FRAMES, limit: int = 12):
    """Compare fixed complete clauses; retained name is compatibility only."""
    frames = tuple(frames)
    out, frontiers = [], []
    for left in frames:
        for right in frames:
            ls, rs = left.clause(), right.clause()
            ln, rn = normalize(ls), normalize(rs)
            # Observed boundaries only: neither tokenization nor tries search.
            lw, rw = re.findall(r"[a-z]+", ls.lower()), re.findall(r"[a-z]+", rs.lower())
            ltrie, rtrie = CharTrie(lw), CharTrie(rw)
            obligations = ""
            matched = 0
            for i, ch in enumerate(ln):
                obligations = ch + obligations
                matched = i + 1
                # Historical heuristic only; not a valid mirrored residual.
                if i < len(rn) and rn[i] == obligations[-1]: obligations = obligations[:-1]
            exact = ln == rn[::-1]
            row = {"left_frame": left.__dict__, "right_frame": right.__dict__,
                   "left": ls, "right": rs, "normalized_left": ln,
                   "normalized_right": rn, "exact": exact,
                   "boundary_choices": {"left_words": lw, "right_words": rw},
                   "opposite_character_obligations": obligations,
                   "obligations_are_search_invariant": False,
                   "trie_nodes_left": sum(1 for _ in ltrie.walk()),
                   "trie_nodes_right": sum(1 for _ in rtrie.walk())}
            if exact and left != right:
                out.append(row)
            elif len(frontiers) < limit:
                frontiers.append(row)
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "quarantined_no_search_diagnostic",
            "evidence_scope": "Only the supplied fixed frame cross-product was compared; this does not test trie/grammar-product feasibility.",
            "candidates": out[:limit], "frontiers": frontiers,
            "provenance": {"frame_count": len(frames), "vocabulary_size": len(WORD_BANK),
                           "catalogue_read": False, "fixed_tape": True,
                           "fixed_frame_pairs_compared": len(frames) ** 2,
                           "generative_search": False,
                           "trie_constrained_transitions": 0,
                           "boundary_choices_searched": 0,
                           "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "known_reversible_phrases": False}}

def audit(row):
    left, right = row["normalized_left"], row["normalized_right"]
    return {"left_equals_reverse_right": left == right[::-1],
            "right_equals_reverse_left": right == left[::-1],
            "independent_recomputation": normalize(row["left"]) == left and normalize(row["right"]) == right,
            "anti_shortcut": not row.get("left_frame") == row.get("right_frame")}

def run():
    result = search()
    for row in result["candidates"] + result["frontiers"]:
        row["audit"] = audit(row)
    return result

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
