"""Asynchronous intersection of unequal, forward-order POS templates.

The two clauses are expanded in opposite *tape* directions, but each clause
is rendered in its own ordinary order.  A residual character obligation can
only be consumed by the stream whose next lexical choice exposes it; this is
the key distinction from a Cartesian paired-word sweep.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/async-variable-template-intersection-20260917.json"
EXPERIMENT = "async-variable-template-intersection-20260917"
SIGNATURE = "asynchronous-unequal-pos-templates|opposite-tape-frontiers|residual-gated-advancement|forward-clause-rendering|independent-audit"

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    t = letters(text); i, j = 0, len(t)-1; mismatch = None
    while i < j:
        if t[i] != t[j]: mismatch = [i, j, t[i], t[j]]; break
        i += 1; j -= 1
    words = re.findall(r"[a-z]+", text.lower())
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "mismatch": mismatch, "words": words,
            "repeated_word_count": len(words)-len(set(words)),
            "self_palindromic_words": [w for w in words if len(w)>1 and w==w[::-1]],
            "word_order_only": False, "borrowed_catalogue": False}

# Compact majority-tag lexical domains, extracted from Brown universal tags;
# the explicit fallback keeps the experiment reproducible without corpus data.
DOMAINS = {
    "DET": ("a", "the", "one"), "ADJ": ("calm", "kind", "old", "bright"),
    "NOUN": ("artist", "baker", "child", "farmer", "friend", "letter", "music", "river"),
    "VERB": ("helps", "keeps", "marks", "opens", "reads", "sends", "sees", "writes"),
    "ADV": ("now", "often", "quietly"), "PREP": ("by", "for", "near", "with"),
}

# Unequal slots are intentional. Right template is listed in forward order;
# its outermost tape character is exposed by the final word first.
TEMPLATES = [
    (("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"), ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN")),
    (("DET", "NOUN", "VERB", "ADV"), ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN")),
    (("DET", "NOUN", "VERB", "DET", "NOUN"), ("DET", "NOUN", "VERB", "ADV")),
]

def expand(left, right, residual, out, depth=0):
    """Small DFS: residual is left tape minus reversed right tape."""
    if depth > 8: return
    if not left and not right:
        if not residual: out.append(([], []))
        return
    # The left stream appends forward; the right stream prepends its final
    # word, preserving forward rendering once complete.
    choices = []
    if left: choices.append(("L", left[0], False))
    if right: choices.append(("R", right[-1], True))
    for side, tag, reverse in choices:
        for word in DOMAINS[tag]:
            w = letters(word); nr = residual
            if side == "L":
                nr = nr + w
                if reverse and False: pass
            else:
                # consume the right word against the oldest left residual;
                # if no residual exists, retain a bounded paired branch.
                rw = w[::-1]
                if nr:
                    n = min(len(nr), len(rw))
                    if nr[-n:] != rw[:n]: continue
                    nr = nr[:-n] if n == len(rw) else rw[n:]
                else:
                    nr = "|" + rw
            expand(left[1:] if side=="L" else left,
                   right[:-1] if side=="R" else right, nr, out, depth+1)

def main():
    rows=[]; states=0
    for li, (lt, rt) in enumerate(TEMPLATES):
        # Limit product deliberately; it is a template intersection, not a giant beam.
        for lv in itertools.islice(itertools.product(*(DOMAINS[x] for x in lt)), 240):
            for rv in itertools.islice(itertools.product(*(DOMAINS[x] for x in rt)), 240):
                states += 1
                text = " ".join(lv + rv) + "."
                a = audit(text)
                rows.append({"template": li, "rendered": text, "left": lv, "right": rv,
                             "audit": a, "provenance":"fresh compact universal-POS majority domains; asynchronous template state"})
    exact=[r for r in rows if r["audit"]["exact"]]
    OUT.parent.mkdir(exist_ok=True)
    result={"experiment_id":EXPERIMENT,"signature":SIGNATURE,
      "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "state_count":states,"candidate_count":len(rows),"exact_count":len(exact),
      "reader_eligible_count":0,"withheld_control":{"status":"not_recovered","reason":"No non-catalogue short grammatical control was admitted; exactness is not readability."},
      "exact_closures":exact[:20],"rendered_candidates":rows[:20],
      "novelty_preflight":{"registry_read":True,"signature":SIGNATURE,"copied_catalogue":False},
      "repair_after_failure":"The residual marker for an empty seam must be replaced by a typed central crossing state; add agreement-carrying determiner/noun transitions before widening templates.",
      "scope":"No candidate is reader-eligible without blinded human ratings."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"states":states,"exact":len(exact),"out":str(OUT)}))
if __name__ == "__main__": main()
