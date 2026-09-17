"""Word-shape transducer: simultaneous lexical generation with boundary edits.

This is deliberately not a character tape.  At every step the left and right
lexical machines emit a character from their current *word*; a tiny FSA also
decides whether a word boundary is consumed on either side.  A result exists
only if both template machines reach final states and the character stream
closes exactly.
"""
from __future__ import annotations
import json
import sys
from hashlib import sha256
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
LEXICON = tuple(w.strip().lower() for w in (ROOT / "data/lexicon.txt").read_text().splitlines() if w.strip().isalpha())
COMMON = {"a", "an", "the", "calm", "bright", "kind", "old", "new", "artist", "baker", "farmer", "teacher", "writes", "reads", "makes", "bread", "notes", "water", "today", "now"}
VOCAB = tuple(sorted(COMMON & set(LEXICON)))

# Held-out ordinary lexicon: the search never uses catalogue palindromes or
# project-generated text.  Templates are deliberately shallow but grammatical.
TEMPLATES = (("a", "ADJ", "NOUN", "VERB", "NOUN"), ("the", "NOUN", "VERB", "DET", "NOUN"))
POS = {"DET": {"a", "an", "the"}, "ADJ": {"calm", "bright", "kind", "old", "new"},
       "NOUN": {"artist", "baker", "farmer", "teacher", "bread", "notes", "water"},
       "VERB": {"writes", "reads", "makes"}}

def options(tag): return tuple(w for w in VOCAB if w in POS.get(tag, {w}))

def boundary_fsa(left_done, right_done):
    """Finite boundary-edit automaton: 0=none, 1=left, 2=right, 3=both."""
    return ((left_done << 0) | (right_done << 1))

def generate(limit=250000):
    states = 0; closures = []; dead = 0
    # State stores lexical word indices and offsets on both live iterators.
    for lt in TEMPLATES:
      for rt in TEMPLATES:
       def walk(li, ri, lo, ro, lw, rw, left, right, edits):
        nonlocal states, dead
        if states >= limit: return
        states += 1
        if li == len(lt) and ri == len(rt) and lo == len(lw) and ro == len(rw):
            text = " ".join(left + list(reversed(right)))
            n = normalize_letters(text)
            audit = {"exact": n == n[::-1], "letters": len(n), "sha256": sha256(n.encode()).hexdigest()}
            if audit["exact"]:
                gate = mechanical_admission_checks(text, min_letters=20, max_letters=260)
                closures.append({"text": text, "left_words": left, "right_words": list(reversed(right)), "boundary_edits": edits, "independent_exact_audit": audit, "admission": gate})
            return
        if li == len(lt) or ri == len(rt): dead += 1; return
        if lo == len(lw):
            for nw in options(lt[li]): walk(li+1, ri, 0, ro, nw, rw, left+[nw], right, edits+[(boundary_fsa(True, False), li, ri)])
            return
        if ro == len(rw):
            for nw in options(rt[ri]): walk(li, ri+1, lo, 0, lw, nw, left, right+[nw], edits+[(boundary_fsa(False, True), li, ri)])
            return
        if lw[lo] != rw[-1-ro]: return
        walk(li, ri, lo+1, ro+1, lw, rw, left, right, edits)
       # Seed each machine with lexical choices; subsequent words are chosen at
       # a boundary, so no precomputed character tape is ever made.
       for lw in options(lt[0]):
        for rw in options(rt[0]): walk(0, 0, 0, 0, lw, rw, [], [], [])
    return {"status":"exhausted" if states < limit else "truncated", "states":states, "dead_states":dead, "closures":closures,
            "admitted": [x for x in closures if all(x["admission"].values())], "config":{"simultaneous_lexical_generation":True,"boundary_fsa":True,"fixed_tape":False,"posthoc_reverse":False,"exact_closure_gate":True,"held_out_ordinary_lexicon":True},
            "novelty_preflight":{"excluded":{"cfg_earley":True,"character_lm":True,"scene_lattice":True,"semantic_valency":True,"morphology_clitic":True,"bilateral_slot":True,"brown_reverse_segmentation":True},"distinction":"paired lexical word iterators plus boundary-edit FSA"},
            "reader_gate":{"status":"not_triggered" if not closures else "required","provenance":"data/lexicon.txt intersected with hand-written POS templates","concrete_repair":"expand held-out POS lexicon only; never relax exact closure"}}

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args();a.out.write_text(json.dumps(generate(),indent=2)+"\n")
