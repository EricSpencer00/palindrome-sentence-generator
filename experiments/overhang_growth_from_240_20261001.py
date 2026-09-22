"""Grow the actual 240-letter seam draft with live partial-word ownership.

Unlike clause-bank probes, this starts from the saved exact draft itself.  The
center-out engine may consume only a prefix/suffix of a word and carries the
remaining character debt into the next step; closed drafts are retained even
when their seams are rough.
"""
from __future__ import annotations

import hashlib, json, re, time
from pathlib import Path

from llm_palindrome.centerout import centerout_search
from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.search import WordTries, unit_letters
from llm_palindrome.validator import is_palindrome, normalize

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/typed-phrase-graph-mara-god-dog-window-20260930.json"
VOCAB = ROOT / "tools/polaris/payload/vocab30k.txt"
OUT = ROOT / "runs/overhang-growth-from-240-20261001.json"

def audit(text: str) -> dict:
    tape = normalize(text)
    mism = next(((i, tape[i], tape[-1-i]) for i in range(len(tape)//2)
                 if tape[i] != tape[-1-i]), None)
    f = hashlib.sha256(tape.encode()).hexdigest()
    r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and mism is None,
            "first_mismatch": mism, "validator_exact": is_palindrome(text),
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

class FrozenDebtPrior:
    wants_overhang = True
    def __init__(self, words):
        self.rank = {w: i for i, w in enumerate(words)}
    def word_delta(self, left, right, placement, word, growth, overhang=None):
        # Frequency prior, plus a small preference for consuming rather than
        # creating debt.  No model call is made for any candidate.
        score = 5.0 - 0.35 * (self.rank.get(word, len(self.rank)) ** .5)
        score += 0.05 * len(unit_letters(word))
        if overhang is not None:
            score += 2.0 if overhang else 0.0
        score -= 2.5 * (list(left)+list(right)).count(word)
        return score

def main():
    parent = json.loads(PARENT.read_text())["selected"]["candidate"]
    parent_n = len(normalize(parent))
    # centerout's unit alphabet is letters plus spaces (spaces are ignored by
    # unit_letters); punctuation belongs only to the display provenance.
    parent_center = re.sub(r"[^A-Za-z ]", "", parent)
    words = [w.strip().lower() for w in VOCAB.read_text().splitlines() if w.strip()]
    tries = WordTries(words)
    scorer = FrozenDebtPrior(words)
    # Keep the incumbent tape as the exact center.  Existing words are not
    # reusable; partial ownership is represented by centerout's overhang.
    used = set(re.findall(r"[a-z]+", parent.lower()))
    rows = []
    deadline = time.monotonic() + 25.0
    for seed in range(4):
        def allow_word(place, word, state):
            letters = unit_letters(word)
            return len(letters) > 1 and letters != letters[::-1] and word not in used
        units = centerout_search(tries, scorer, center=parent_center,
            min_letters=parent_n + 20, beam_width=64, per_parent=8,
            candidate_limit=220, max_steps=90, seed=seed, diversity=.7,
            max_overhang=32, deadline=deadline, maximize="letters",
            allow_word=allow_word)
        text = " ".join(units)
        au = audit(text) if text else {"letters": 0, "two_pointer_exact": False,
                                        "validator_exact": False}
        ci = units.index(parent_center) if parent_center in units else -1
        left = list(units[:ci]) if ci >= 0 else []
        right = list(units[ci+1:]) if ci >= 0 else []
        rows.append({"seed": seed, "rendered": text, "audit": au,
                     "growth_over_parent": au["letters"] - parent_n,
                     "left_added_units": left, "right_added_units": right,
                     "parent_artifact": str(PARENT.relative_to(ROOT)),
                     "seam_debt": ["partial word ownership is carried by live overhang",
                                   "generated boundary syntax is rough and unreviewed"],
                     "provenance": {"engine": "llm_palindrome.centerout_search",
                       "actual_parent_center": True, "frozen_search_prior": True,
                       "per_candidate_rlaif": False, "finished_tape_reversal": False,
                       "catalogue_text": False, "posthoc_character_repair": False,
                       "human_certified": False},
                     "novelty_preflight": is_novel_palindrome(text) if text else False})
    exact = [r for r in rows if r["audit"].get("two_pointer_exact") and
             r["audit"].get("validator_exact") and r["audit"].get("sha_equal") and
             r["growth_over_parent"] > 0]
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["seed"]))
    payload = {"experiment_id": "overhang-growth-from-240-20261001",
      "method": "center-out residual-carrying growth from actual 240-letter incumbent",
      "parent_letters": parent_n, "stats": {"runs": len(rows),
        "exact_growths": len(exact), "longest_letters": max(r["audit"]["letters"] for r in rows),
        "longest_growth": max(r["growth_over_parent"] for r in rows)},
      "rows": rows, "reader_gate": "closed; no blinded human ratings",
      "next_repair": "retain longest closure and hand-edit only its highest-debt seams"}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    if rows: print(rows[0]["rendered"])

if __name__ == "__main__": main()
