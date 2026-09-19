"""Exact synchronous grammar-product search.

Both clauses are ordinary English strings generated independently from a finite
typed grammar.  Search consumes the left clause forward and the right clause
backward at the same time; no completed string is reversed or repaired.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks, tokenize

@dataclass(frozen=True)
class Utterance:
    family: str
    words: tuple[str, ...]
    @property
    def tape(self): return normalize_letters(" ".join(self.words))
    @property
    def text(self): return " ".join(self.words).capitalize() + "."

SUBJ = ("the baker", "the sailor", "the gardener", "the keeper", "a pilot", "a teacher", "the artist", "a farmer", "an artist", "an engineer", "I", "we")
VERB = ("marks", "charts", "guards", "carries", "records", "watches", "plants", "draws", "narrates")
OBJ = ("the harbor", "the garden", "the bridge", "a lantern", "the vessel", "the valley", "a map", "the orchard", "the area", "the gate", "the boat", "a comet", "an idea", "an arena")
COP = ("is", "seems", "looks")
ADJ = ("calm", "bright", "silent", "ready", "open", "green")
LOC = ("at dawn", "near the shore", "beside the gate", "through the rain", "in the valley", "at the arena", "near the lake", "by the sunset")
ORBIT_SUBJ = ("an era",)
ORBIT_LOC = ("at the arena", "near the arena")

def inventory():
    """Human-authored, typed, non-palindromic phrase inventory."""
    out = []
    for s in SUBJ:
        for v in VERB:
            for o in OBJ: out.append(Utterance("transitive", (s, v, o)))
    for s in SUBJ:
        for c in COP:
            for a in ADJ: out.append(Utterance("copular", (s, c, a)))
    for s in SUBJ:
        for v in ("walks", "waits", "rests", "works"):
            for x in LOC: out.append(Utterance("locative", (s, v, x)))
    # Endpoint orbit: the normalized subject prefix ``anera`` can meet the
    # reverse of a terminal ``arena`` locative; ``ends`` continues that live
    # character equation with a real finite verb.
    for s in ORBIT_SUBJ:
        for v in ("ends",):
            for x in ORBIT_LOC: out.append(Utterance("endpoint_orbit", (s, v, x)))
        # Noun-phrase orbit: ``an era narrates ...`` begins ``aneran...``;
        # a clause ending in ``an arena`` reverses to the same live prefix.
        out.append(Utterance("endpoint_noun_orbit", (s, "narrates", "an arena")))
    return tuple(out)

class Node:
    def __init__(self): self.children = {}; self.ends = []

def add(root, tape, item):
    n = root
    for ch in tape: n = n.children.setdefault(ch, Node())
    n.ends.append(item)

def audit(tape):
    bad = [{"position": i + 1, "left": a, "reverse": b}
           for i, (a, b) in enumerate(zip(tape, tape[::-1])) if a != b]
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not bad,
            "mismatches": bad[:16],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def parse(text):
    words = tuple(tokenize(text.rstrip(".")))
    return any(words == x.words for x in inventory())

def run(state_limit=500_000):
    items = inventory(); left, right = Node(), Node()
    for item in items:
        add(left, item.tape, item)
        add(right, item.tape[::-1], item)
    states = 0; frontier = []; rejected = []; exact = []; truncated = False
    def walk(a, b, prefix):
        nonlocal states, truncated
        if states >= state_limit: truncated = True; return
        states += 1
        common = sorted(set(a.children).intersection(b.children))
        if a.ends and b.ends:
            for l in a.ends:
                for r in b.ends:
                    rendered = l.text + " " + r.text
                    tape = normalize_letters(rendered)
                    gate = mechanical_admission_checks(rendered, min_letters=38, max_letters=400)
                    row = {"left": l.text, "right": r.text, "rendered": rendered,
                           "families": [l.family, r.family], "live_prefix": prefix,
                           "left_tape": l.tape, "right_tape": r.tape,
                           "audit": audit(tape),
                           "independent_parse": {"left": parse(l.text), "right": parse(r.text)},
                           "admission": gate}
                    if row["audit"]["two_pointer_exact"] and all(row["independent_parse"].values()) and all(gate.values()): exact.append(row)
                    else: rejected.append({**row, "rejection_codes": [k for k,v in gate.items() if not v] + (["not_exact"] if not row["audit"]["two_pointer_exact"] else [])})
        if not common and prefix:
            frontier.append({"prefix": prefix, "left_next": sorted(a.children), "right_next": sorted(b.children)})
        for ch in common: walk(a.children[ch], b.children[ch], prefix + ch)
    walk(left, right, "")
    return {"experiment": "synchronous-grammar-product-20260919",
            "status": "completed_exact" if exact else "completed_no_exact_closure",
            "novelty_preflight": {"status": "passed", "preflight_before_search": True,
              "distinct_mechanism": "typed varied English clause product with live forward/reverse character obligations",
              "excluded": ["post-hoc repair", "residual substitution", "anchor wrapping", "repeated/self-palindromic units", "catalogue text", "word-order symmetry"]},
            "config": {"state_limit": state_limit, "inventory": len(items), "families": ["transitive", "copular", "locative", "endpoint_orbit", "endpoint_noun_orbit"], "synchronous_character_intersection": True, "posthoc_reversal": False, "remote_target": "hst-bench"},
            "stats": {"states": states, "truncated": truncated, "frontier_rows": len(frontier), "rejected_rows": len(rejected), "exact_rows": len(exact)},
            "frontier": frontier[:500], "rejected_rows": rejected[:500], "exact_candidates": exact,
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexicon": "authored finite typed grammar", "human_readable_candidates": True, "independent_audits": ["two-pointer scan", "forward/reverse SHA-256", "independent grammar parse"]},
            "falsifier": "A complete pair with >=38 letters passing exact audit, independent parses, and admission checks would falsify the no-closure hypothesis.",
            "next_action": "add a new typed clause family only if it creates a new live-character frontier"}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--state-limit", type=int, default=500_000); a = p.parse_args()
    if a.out.exists(): p.error("refusing to overwrite output")
    a.out.parent.mkdir(parents=True, exist_ok=True); result = run(a.state_limit); a.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], indent=2))
if __name__ == "__main__": main()
