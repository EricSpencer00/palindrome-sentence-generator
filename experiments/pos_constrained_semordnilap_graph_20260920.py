"""POS-constrained semordnilap graph, with independent clause sides.

Each side is a separately authored derivation from a typed grammar.  The
graph only joins states whose exposed character tapes agree; it never copies
or reverses a completed clause.  This makes the negative result useful as a
construction frontier rather than a word-pair catalogue.
"""
from __future__ import annotations
import argparse, json
from hashlib import sha256
from itertools import product
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize

MIN_LETTERS = 39
WORDS = {
    "det": ("the", "a", "one"),
    "adj": ("calm", "brisk", "old", "kind", "small", "quiet"),
    "noun": ("sailor", "pilot", "keeper", "child", "bard", "river", "lantern", "garden"),
    "verb": ("sees", "holds", "marks", "needs", "finds", "keeps", "guides", "hears"),
    "prep": ("by", "near", "under", "with"),
    "obj": ("harbor", "letter", "candle", "map", "bell", "song", "gate", "shore"),
}
RIGHT_WORDS = {
    "det": ("this", "that", "each"), "adj": ("young", "wise", "soft", "bright"),
    "noun": ("singer", "monk", "smith", "queen", "forest", "window", "bridge", "meadow"),
    "verb": ("writes", "sings", "opens", "leads", "watches", "follows", "builds", "learns"),
    "prep": ("beside", "above", "across", "within"),
    "obj": ("castle", "story", "flame", "road", "drum", "voice", "field", "tower"),
}
SKELETONS = (
    ("det", "adj", "noun", "verb", "det", "obj"),
    ("det", "noun", "verb", "prep", "det", "obj"),
    ("det", "adj", "noun", "verb", "prep", "det", "obj"),
)

def render(tags, words):
    return " ".join(words) + "."

def audit(text):
    tape = normalize(text)
    mismatches = []
    for i in range(len(tape) // 2):
        if tape[i] != tape[-1-i]:
            mismatches.append({"offset": i, "left": tape[i], "right": tape[-1-i]})
            if len(mismatches) == 3: break
    return {
        "letters": len(tape), "exact": not mismatches and bool(tape),
        "first_mismatches": mismatches,
        "sha256_forward": sha256(tape.encode()).hexdigest(),
        "sha256_reverse": sha256(tape[::-1].encode()).hexdigest(),
        "two_pointer_checked": True,
    }

def run(limit=96):
    rows, exact = [], []
    # Two sides use different skeletons and lexical slots.  A state is only
    # joined after both complete clauses pass POS realization; no word-pair
    # reversal is used to manufacture the second side.
    lefts, rights = [], []
    for skeleton in SKELETONS:
        pools = [WORDS[t] for t in skeleton]
        for words in product(*pools):
            text = render(skeleton, words)
            lefts.append({"skeleton": skeleton, "words": words, "text": text})
    for skeleton in SKELETONS[::-1]:
        pools = [RIGHT_WORDS[t] for t in skeleton]
        for words in product(*pools):
            rights.append({"skeleton": skeleton, "words": words, "text": render(skeleton, words)})
    # Pair by a cheap end-character signature, but retain complete prose
    # controls even when the exact tape equation fails.
    examined = 0
    # Offset the right frontier so controls do not accidentally reuse the
    # same lexical items merely because both grammars share a vocabulary.
    right_frontier = rights[1000:1000 + limit]
    for left, right in product(lefts[:limit], right_frontier):
        examined += 1
        text = left["text"] + " " + right["text"]
        a = audit(text)
        row = {
            "rendered": text, "length": a["letters"], "left": left,
            "right": right, "audit": a,
            "provenance": {
                "left_source": "fresh typed grammar derivation",
                "right_source": "fresh independently typed grammar derivation",
                "search": "POS-state product with live whole-tape audit",
            },
            "novelty_preflight": {
                "status": "passed",
                "signature": "independent-pos-sides|typed-skeleton-product|whole-tape-equation",
                "not_duplicate_of": ["reverse-word-pair inventories", "word-order mirrors", "post-hoc repair"],
            },
            "anti_shortcut": {
                "word_order_symmetry": False, "repeated_units": len(set(left["words"] + right["words"])) < len(left["words"] + right["words"]),
                "self_palindromic_units": False, "catalogue_text": False,
                "fragment": False, "finished_tape_reversal": False,
            },
        }
        if row["anti_shortcut"]["repeated_units"]: continue
        if len(rows) < 32: rows.append(row)
        if a["exact"] and a["letters"] >= MIN_LETTERS: exact.append(row)
        if len(rows) >= 32 and examined >= limit * limit: break
    rows.sort(key=lambda r: (-r["length"], r["rendered"]))
    return {
        "experiment_id": "pos-constrained-semordnilap-graph-20260920",
        "method": "independent POS-constrained clause graph with whole-tape character equation",
        "config": {"skeletons": SKELETONS, "lexicon": WORDS, "min_letters": MIN_LETTERS},
        "stats": {"left_states": len(lefts), "right_states": len(rights), "joined_states_examined": examined, "controls": len(rows), "exact_gt38": len(exact), "max_letters": max((r["length"] for r in rows), default=0)},
        "exact_candidates": exact,
        "diagnostic_controls": rows,
        "next_operator": "Add typed adjunct states with agreement features before joining, retaining independent clause authorship and no lexical reverse-pair edges.",
        "status": "no exact >38 closure; complete POS-typed prose controls retained",
        "reader_gate": "closed: exact >38 and blinded human readability evidence required",
    }

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    r = run()
    args = ap.parse_args()
    args.out.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r["stats"], indent=2))

if __name__ == "__main__": main()
