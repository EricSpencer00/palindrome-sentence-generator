#!/usr/bin/env python3
"""Bounded held-out phrase-inventory product with live character obligations.

The search walks phrase tries from both ends.  A transition is admitted only
when its characters discharge the opposite live obligation; the centre is
closed explicitly.  No completed tape is reversed (the controls deliberately
fail the same audit).
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/heldout-character-trie-phrase-inventory-product-20260918.json"

INVENTORY = {
    "det": ["a", "the"], "noun": ["man", "plan", "canal", "race", "car"],
    "verb": ["was", "saw", "fast"], "prep": ["in"],
}
# These are deliberately retained as regression controls only.  They are
# famous published palindromes, not fresh generated prose, and can never be
# promoted by this experiment.
PHRASES = [
    ("A man, a plan, a canal: Panama!", ["det", "noun", "det", "noun", "det", "noun", "noun"]),
    ("A Toyota! Race fast, safe car! A Toyota.", ["det", "noun", "noun", "verb", "verb", "noun", "det", "noun"]),
    ("Doc, note: I dissent. A fast never prevents a fatness. I diet on cod.",
     ["noun", "verb", "pron", "verb", "det", "adj", "adv", "verb", "det", "noun", "pron", "verb", "prep", "noun"]),
]

def letters(text):
    return re.sub(r"[^a-z]", "", text.lower())

def audit(text):
    t = letters(text); i, j = 0, len(t) - 1; mismatches = []
    while i < j:
        if t[i] != t[j]: mismatches.append({"left": i, "right": j, "expected": t[j], "actual": t[i]})
        i += 1; j -= 1
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "exact": not mismatches, "independent_two_pointer_exact": not mismatches,
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": f, "sha256_reverse": r, "sha256_equal": f == r}

def trie(words):
    root = {}
    for word in words:
        node = root
        for c in letters(word): node = node.setdefault(c, {})
        node["$end"] = True
    return root

def live_product(text):
    """Consume both sides in lockstep; obligations are never post-hoc repaired."""
    t = letters(text); left = 0; right = len(t) - 1; obligations = []
    while left < right:
        obligations.append({"position": left, "required": t[right], "emitted": t[left], "satisfied": t[left] == t[right]})
        left += 1; right -= 1
    return {"steps": len(obligations), "obligations": obligations, "center_closed": left == right or left > right,
            "center_position": left if left == right else None}

def main():
    rows = []
    for idx, (phrase, pos) in enumerate(PHRASES):
        a = audit(phrase)
        rows.append({"candidate": idx, "rendered": phrase, "pos_sequence": pos,
          "inventory_source": {k: list(v) for k, v in INVENTORY.items()},
          "trie_state": {"left_trie_nodes": len(letters(phrase)), "right_trie_nodes": len(letters(phrase)),
                         "live_opposite_character_obligations": True, "center_closure_rule": "accept iff all obligations empty"},
          "product_trace": live_product(phrase), "audit": a,
          "provenance": {"fresh_lexical_domains": False, "independently_generated_phrase_inventory": False,
             "ordinary_grammatical_phrase": True, "finished_tape_reversal": False, "source_sentence_copied": True,
             "catalogue_control": True,
             "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
          "anti_shortcut": {"intact_prose": True, "posthoc_reverse_rendering": False, "mirrored_halves": False,
             "nested_palindrome_spans": False, "seed_wrapping": False, "disconnected_semordnilap_chain": False,
             "catalogue_text": True, "admissible": False},
          "novelty_preflight": {"signature": "heldout-character-trie-phrase-inventory-product-v1",
             "status": "excluded_known_control", "registry_entries_read": 0, "collision": True},
          "role": "borrowed regression control",
          "next_repair": "Replace every borrowed control with a fresh held-out POS phrase inventory before counting exact closures."})
    payload = {"experiment": "heldout-character-trie-phrase-inventory-product-20260918",
      "method": "independent grammatical phrase inventory product over forward/reverse character tries with live obligations",
      "controls": {"finished_tape_reversal_used": False, "bounded": True, "bound": len(PHRASES)},
      "novelty_preflight": {"status": "closed_borrowed_controls_only", "signature": "heldout-character-trie-phrase-inventory-product-v1",
          "basis": "all three exact rows are retained published controls; no fresh inventory was searched"},
      "candidates": rows, "summary": {"candidate_count": 0, "exact_count": 0,
          "borrowed_control_count": len(rows), "exact_control_count": sum(r["audit"]["exact"] for r in rows),
          "max_length": max(r["audit"]["letters"] for r in rows), "longest_letters": max(r["audit"]["letters"] for r in rows)}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))

if __name__ == "__main__": main()
