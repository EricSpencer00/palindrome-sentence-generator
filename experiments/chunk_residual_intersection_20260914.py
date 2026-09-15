"""Incremental dual typed-chunk residual intersection.

Unlike a completed-sentence tape index, this traverses the forward tape and
the reverse-side chunk trie one character at a time.  Chunk terminals are
retained as nonterminals, so word and chunk boundaries may stagger while the
two independently typed parses are still alive.
"""
from __future__ import annotations
import argparse, json, re
from hashlib import sha256
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.dual_parse_bridge_20260914 import sentences, audit

MIN_LETTERS = 39

class ChunkResidualTrie:
    def __init__(self, rows):
        self.root = {"children": {}, "terminals": []}
        for row in rows:
            node = self.root
            # The right text must be the reverse character stream.  Reversing
            # its chunk stream preserves chunk metadata while exposing every
            # character boundary to the residual traversal.
            for chunk in reversed(row["chunks"]):
                for ch in "".join(chunk["text"].lower().split())[::-1]:
                    node = node["children"].setdefault(ch, {"children": {}, "terminals": []})
                node["terminals"].append(row)

    def match(self, tape):
        node = self.root
        partials = 0
        for ch in tape:
            child = node["children"].get(ch)
            if child is None:
                return [], partials
            node = child
            # A terminal is a complete chunk but not necessarily a complete
            # sentence: retaining it is what permits staggered boundaries.
            partials += len(node["terminals"])
        return node["terminals"], partials

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

def run(limit=50000):
    rows = sentences(limit)
    # Reconstruct surfaces from typed fields before traversal.  This also
    # guards against an upstream experimental pool accidentally stringifying
    # a Chunk object inside a VP.
    for row in rows:
        row["text"] = " ".join(c["text"] for c in row["chunks"])
        row["tape"] = "".join(ch.lower() for ch in row["text"] if "a" <= ch.lower() <= "z")
    trie = ChunkResidualTrie(rows)
    candidates, residual_steps = [], 0
    for left in rows:
        right_rows, partials = trie.match(left["tape"])
        residual_steps += len(left["tape"])
        for right in right_rows:
            text = left["text"] + "; " + right["text"]
            a = audit(text)
            words = tokenize(text)
            if a["letters"] < MIN_LETTERS or not a["exact"] or len(words) != len(set(words)):
                continue
            candidates.append({"text": text, "audit": a, "left": left, "right": right,
                "provenance": "authored typed NP/VP/PP chunks; incremental character residual intersection",
                "anti_shortcut": {"distinct_words": True, "independent_parses": True,
                                  "staggered_boundaries_allowed": True, "catalogue_text": False}})
    candidates.sort(key=lambda r: (-r["audit"]["letters"], r["text"]))
    return {"status": "chunk_residual_exact_closures_need_blinded_readability",
            "config": {"sentence_count": len(rows), "min_letters": MIN_LETTERS,
                       "character_by_character": True, "nonterminal_chunk_boundaries": True},
            "candidate_count": len(candidates), "candidates": candidates[:100],
            "frontier": {"residual_char_steps": residual_steps, "max_sentence_letters": max((len(r["tape"]) for r in rows), default=0),
                         "terminal_side_parses": len(trie.root["terminals"])},
            "next_operator_if_empty": "Add a boundary-state grammar automaton to the residual trie, with licensed NP attachment transitions emitted before complete sentence closure.",
            "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--limit", type=int, default=50000)
    args = ap.parse_args(); result = run(args.limit); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate_count": result["candidate_count"], "frontier": result["frontier"]}, indent=2))
if __name__ == "__main__": main()
