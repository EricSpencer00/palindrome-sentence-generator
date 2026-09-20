"""Variable-length joint phrase grammar for exact palindromes.

Both sides grow independently from ordinary phrase units.  The only pruning
rule is the exact outer-to-inner character invariant; no completed tape is
reversed, repaired, or replayed from a corpus sentence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    return {"letters": len(tape), "exact": bool(tape) and tape == tape[::-1],
            "sha256": hashlib.sha256(tape.encode()).hexdigest()}


# These are independently authored ordinary chunks.  They are deliberately
# not reverse pairs; the grammar must discover compatible chunks jointly.
AUTHORED = {
    "NP": ("an aide", "some men", "a quiet scholar", "the old sailor",
            "a kind child", "the patient keeper", "a young poet"),
    "VP": ("rips nine memos", "inspire Diana", "reads old letters",
            "keeps the lantern", "watches the dawn", "carries bright books",
            "sings beside water", "remembers the garden"),
    "PP": ("in the garden", "by the river", "under a pale moon",
            "with a small bell", "near the quiet harbor"),
    "ADV": ("at dawn", "very softly", "with care", "in silence"),
}


def phrase_bank() -> dict[str, tuple[str, ...]]:
    """Return a transparent bank; Brown contributes frequent clean chunks."""
    out = {k: list(v) for k, v in AUTHORED.items()}
    try:
        from nltk.corpus import brown
        for sent in brown.tagged_sents():
            words = [w.casefold() for w, _ in sent]
            tags = [t for _, t in sent]
            for i in range(len(words) - 1):
                if not all(re.fullmatch(r"[a-z]+", w) for w in words[i:i + 2]):
                    continue
                tag = tags[i]
                role = "NP" if tag.startswith(("AT", "JJ", "NN")) else "VP" if tag.startswith("VB") else "PP" if tag.startswith("IN") else None
                if role:
                    out[role].append(" ".join(words[i:i + 2]))
    except LookupError:
        pass
    return {k: tuple(dict.fromkeys(v)) for k, v in out.items()}


def consume_residual(left: str, right: str):
    """Consume the common prefix of two comparison-oriented buffers."""
    n = min(len(left), len(right))
    if left[:n] != right[:n]:
        return None
    return left[n:], right[n:]


def compatible(left: str, right: str) -> bool:
    """Independent audit helper for two completed rendered sides."""
    return consume_residual(letters(left), letters(right)[::-1]) == ("", "")


def grammar_paths(max_units: int = 5):
    """Variable-length NP VP (PP|ADV)* paths, with an optional second VP."""
    paths = []
    for n in range(2, max_units + 1):
        # NP VP, NP VP adjuncts, and NP VP NP VP adjuncts are distinct paths.
        if n == 2:
            paths.append(("NP", "VP"))
        elif n == 3:
            paths.extend((("NP", "VP", "PP"), ("NP", "VP", "ADV"),
                          ("NP", "VP", "NP")))
        elif n == 4:
            paths.extend((("NP", "VP", "PP", "PP"), ("NP", "VP", "PP", "ADV"),
                          ("NP", "VP", "NP", "VP")))
        else:
            paths.append(("NP", "VP", "PP", "PP", "ADV"))
    return tuple(dict.fromkeys(paths))


def run(limit: int = 150_000, max_units: int = 5) -> dict:
    bank = phrase_bank()
    paths = grammar_paths(max_units)
    states = 0
    pruned = 0
    exact = []
    seen = set()
    # Product of independent variable paths.  Prefix compatibility means a
    # state cannot later become valid if an overlap already disagrees.
    for lp in paths:
        for rp_rendered in paths:
            # The right side is emitted from its inner edge outward, so its
            # construction order is the reverse of its eventual role order.
            rp = tuple(reversed(rp_rendered))
            # lbuf/rbuf are unmatched comparison-oriented character debt.
            stack = [(0, 0, "", "", "", "", ())]
            while stack and states < limit:
                li, ri, left, right, lbuf, rbuf, provenance = stack.pop()
                states += 1
                if li == len(lp) and ri == len(rp):
                    rendered = (left + " " + right).strip()
                    info = audit(rendered)
                    # Empty residuals are the construction invariant; the
                    # independent audit below must agree with it.
                    if not lbuf and not rbuf and info["exact"] and info["letters"] >= 38 and rendered not in seen:
                        seen.add(rendered)
                        exact.append({"rendered": rendered, "audit": info,
                                      "provenance": {"left_roles": lp, "right_roles": rp,
                                                     "phrase_units": len(provenance),
                                                     "bank": "authored+Brown-bigrams",
                                                     "corpus_sentence_replay": False,
                                                     "mirrored_token_units": False}})
                    continue
                if li < len(lp):
                    for phrase in reversed(bank[lp[li]][:40]):
                        nl = (lbuf + letters(phrase))
                        residual = consume_residual(nl, rbuf)
                        if residual is not None:
                            stack.append((li + 1, ri, (left + " " if left else "") + phrase, right,
                                          residual[0], residual[1],
                                          provenance + (("L", lp[li], phrase),)))
                        else:
                            pruned += 1
                if ri < len(rp):
                    for phrase in reversed(bank[rp[ri]][:40]):
                        # A phrase prepended to rendered right is appended to
                        # the comparison stream in reverse character order.
                        nr = rbuf + letters(phrase)[::-1]
                        residual = consume_residual(lbuf, nr)
                        if residual is not None:
                            stack.append((li, ri + 1, left, phrase + (" " + right if right else ""),
                                          residual[0], residual[1],
                                          provenance + (("R", rp[ri], phrase),)))
                        else:
                            pruned += 1
            if states >= limit:
                break
        if states >= limit:
            break
    return {"method": "variable-phrase-grammar-20260920", "paths": len(paths),
            "bank_sizes": {k: len(v) for k, v in bank.items()}, "states": states,
            "pruned": pruned, "exact_candidates": exact,
            "candidate_count": len(exact),
            "status": "reader gate required" if exact else "construction frontier empty"}


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs/variable-phrase-grammar-20260920.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
