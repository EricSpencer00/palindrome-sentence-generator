"""Bidirectional typed-clause lexical transducer (bounded constructive run).

Each side is generated from typed English slots.  A character trie is walked
from the left on one clause and from the right on its independently typed
mirror clause; lexical choices are joined only when their emitted characters
agree.  Results retain the actual rendering and an independent audit.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

@dataclass(frozen=True)
class Slot:
    tag: str
    words: tuple[str, ...]

# Small productive inventory: ordinary clauses, not palindrome material.
SLOTS = (
    Slot("adj", ("calm", "kind", "quick", "brisk", "bright", "silent")),
    Slot("subject", ("artist", "baker", "pilot", "teacher", "writer", "sailor")),
    Slot("verb", ("helps", "guides", "calls", "finds", "sees", "keeps")),
    Slot("object", ("a friend", "the child", "a sailor", "the baker", "a bright star")),
)
TEMPLATES = ((0, 1, 2, 3), (1, 2, 3), (0, 1, 2), (1, 2))

def norm(s: str) -> str:
    return normalize_letters(s)

def audit(text: str, *, min_letters: int = 0, max_letters: int = 140) -> dict:
    n = norm(text)
    independent = "".join(c.casefold() for c in text
                           if c.isascii() and c.isalpha())
    checks = mechanical_admission_checks(text, min_letters=min_letters,
                                         max_letters=max_letters)
    return {"rendered": text, "normalized": n, "letters": len(n),
            "exact_palindrome": bool(n) and n == n[::-1],
            "independent_normalized": independent,
            "independent_exact_audit": bool(n) and n == n[::-1] and n == independent,
            "mechanical_checks": checks,
            "mechanically_eligible": bool(all(checks.values())),
            "word_tokens": text.split(), "intact": True,
            "reader_status": "human-unreviewed; programmatic filters do not certify readability"}

class CharTrie:
    def __init__(self): self.children={}; self.ends=[]
    def add(self, key, value):
        node=self
        for ch in key: node=node.children.setdefault(ch, CharTrie())
        node.ends.append(value)
    def prefixes(self, tape, i=0):
        node=self; out=[]
        while i < len(tape) and tape[i] in node.children:
            node=node.children[tape[i]]; i+=1
            out.extend((j,v) for v in node.ends for j in [i])
        return out

def clause_options(max_per_slot=6):
    rows=[]
    for ti,t in enumerate(TEMPLATES):
        pools=[SLOTS[i].words[:max_per_slot] for i in t]
        for words in itertools.product(*pools):
            phrase=" ".join(words).capitalize()+"."
            rows.append({"template":ti,"tags":[SLOTS[i].tag for i in t],"words":list(words),"rendered":phrase,"tape":norm(phrase)})
    return rows

def run(state_limit=50000, min_letters=39, max_letters=90):
    clauses=clause_options()
    # Typed reverse trie: all right-side clauses are indexed by reversed tape.
    rev=CharTrie()
    for row in clauses: rev.add(row["tape"][::-1], row)
    exact=[]; states=0
    # Jointly choose left slots and consume the reverse tape through the right trie.
    for left in clauses:
        # The rendered pair contains both clauses; the trie walk matches one
        # half, so apply length bands to the complete candidate.
        if not min_letters <= 2 * len(left["tape"]) <= max_letters: continue
        for end,right in rev.prefixes(left["tape"]):
            states += 1
            if states > state_limit: break
            if end != len(left["tape"]): continue
            # right tape is reverse(left); render left + right as intact prose.
            text=left["rendered"]+" "+right["rendered"]
            row=audit(text, min_letters=min_letters, max_letters=max_letters)
            row.update({"left":left,"right":right,
                        "operator":"typed_bidirectional_character_trie"})
            if row["exact_palindrome"] and row["letters"] >= min_letters: exact.append(row)
        if states > state_limit: break
    # Include diagnostics for the best near misses, preserving rendered output.
    near=[]
    for left in clauses:
        for right in clauses[: min(200,len(clauses))]:
            text=left["rendered"]+" "+right["rendered"]
            normalized = norm(text)
            # Near misses are diagnostics only; avoid repeatedly loading the
            # full admission catalogue for rows that cannot be candidates.
            near.append({"rendered": text, "normalized": normalized,
                         "letters": len(normalized),
                         "exact_palindrome": bool(normalized) and normalized == normalized[::-1],
                         "mismatch": sum(x != y for x, y in itertools.zip_longest(
                             normalized, normalized[::-1], fillvalue="_")),
                         "left": left["rendered"], "right": right["rendered"]})
    near.sort(key=lambda x:(x["mismatch"],-x["letters"]))
    return {"status":"complete_bounded_search","operator":"typed bidirectional character trie/transducer",
            "config":{"state_limit":state_limit,"min_letters":min_letters,"max_letters":max_letters,"typed_slot_joint_choice":True,"independent_normalization":True},
            "inventory":{"templates":len(TEMPLATES),"clauses":len(clauses),"slots":[asdict(s) for s in SLOTS]},
            "states_examined":states,"exact_candidates":exact,"near_misses":near[:10],
            "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"material":"authored typed English clause inventory; no catalogue or prebuilt palindrome","rendering":"actual clause strings retained in every record"},
            "next_operator":"Expand typed slot lexicons with reverse-trie-compatible content words while preserving joint slot constraints."}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=Path("runs/bidirectional-typed-clause-transducer-20260914/results.json")); ap.add_argument("--state-limit",type=int,default=50000); args=ap.parse_args(); out=run(args.state_limit); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"states":out["states_examined"],"exact":len(out["exact_candidates"]),"output":str(args.output)}))
if __name__ == "__main__": main()
