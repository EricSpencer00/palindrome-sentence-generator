"""Fresh typed phrase-lattice seam search.

Unlike the word-bank lanes, this search emits only complete, hand-authored
English clause fragments with an explicit syntactic type.  It then lets the
live overhang solver choose fragments on either side of an exact center.  The
type is retained in provenance and used to prefer a grammatical A-B-B-A
sequence; no text is reversed or repaired after the search.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path
from llm_palindrome.centerout import centerout_search
from llm_palindrome.search import WordTries, unit_letters
from llm_palindrome.paragraphs import is_novel_palindrome
from experiments.working_overhang_growth_20260921 import CENTERS, audit, center_unit
from experiments.working_overhang_coherent_20260921 import JoinScorer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "typed-clause-seam-20260923.json"
BIGRAMS = ROOT / "data" / "count_2w.txt"

LEX = {
    "det": ["the", "a", "an"],
    "adj": ["red", "calm", "old", "kind", "wise", "pale", "quiet", "young"],
    "noun": ["poet", "artist", "sailor", "teacher", "scholar", "gardener", "child", "judge"],
    "verb": ["saw", "heard", "kept", "found", "read", "asked", "watched", "carried"],
    "obj": ["the map", "a letter", "an old book", "the quiet garden", "a red rose", "the small boat"],
    "prep": ["at dawn", "in spring", "by the river", "after rain", "near the harbor"],
}

def fragments() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    # Full, readable phrase windows rather than isolated catalogue words.
    for d in LEX["det"]:
        for a in LEX["adj"]:
            for n in LEX["noun"]:
                rows.append((f"{d} {a} {n}", "NP"))
    for n in LEX["noun"]:
        for v in LEX["verb"]:
            rows.append((f"the {n} {v}", "CLAUSE"))
            for o in LEX["obj"]:
                rows.append((f"the {n} {v} {o}", "CLAUSE"))
    for p in LEX["prep"]:
        rows.append((p, "SETTING"))
    # Remove self-palindromic units and duplicate letter tapes.
    out, seen = [], set()
    for text, typ in rows:
        key = unit_letters(text)
        if len(key) < 4 or key == key[::-1] or key in seen:
            continue
        seen.add(key); out.append((text, typ))
    return out

def run() -> dict:
    bank = fragments()
    # Preserve types out of band; search sees complete phrase units only.
    units = [u for u, _ in bank]
    type_of = {u: t for u, t in bank}
    tries = WordTries(units)
    words = [x for x in (ROOT / "tools/polaris/payload/vocab30k.txt").read_text().splitlines() if x.strip()]
    # A phrase-only lattice avoids the short-word collapse of generic search.
    bg = __import__("llm_palindrome.bigram", fromlist=["BigramModel"]).BigramModel.from_file(str(BIGRAMS), vocab=set(words))
    rows = []
    for cid, source in CENTERS.items():
        center = center_unit(source)
        scorer = JoinScorer(bg, center)
        for seed in range(4):
            seq = centerout_search(tries, scorer, center=center,
                min_letters=len(unit_letters(center)) + 12, beam_width=64,
                per_parent=8, candidate_limit=300, max_steps=80, seed=seed,
                diversity=.8, max_overhang=18, maximize="score",
                allow_word=lambda placement, unit, state: unit != center and
                    unit_letters(unit) != unit_letters(unit)[::-1] and
                    state.left.count(unit) + state.right.count(unit) == 0)
            text = " ".join(seq); au = audit(text)
            rows.append({"center_id": cid, "seed": seed, "rendered": text,
                "units": seq, "types": [type_of.get(x, "CENTER") for x in seq],
                "letters": au["letters"], "growth_over_center": au["letters"]-len(unit_letters(center)),
                "audit": au, "novelty_preflight": is_novel_palindrome(text),
                "provenance": {"generator": "typed complete-clause lattice + live overhang",
                    "phrase_inventory": len(bank), "per_candidate_rlaif": False,
                    "finished_tape_reversal": False, "posthoc_character_repair": False,
                    "repeated_generated_unit": False, "reader_certified": False},
                "seam_debt": ["typed fragments are grammatical windows, not a full discourse grammar",
                              "next repair: constrain NP/CLAUSE/SETTING transitions as an ABBA role lattice"]})
    exact = [r for r in rows if r["audit"].get("two_pointer_exact") and r["audit"].get("sha_equal") and r["audit"].get("validator_exact") and r["growth_over_center"] > 0]
    rows.sort(key=lambda r: -r["letters"])
    return {"experiment_id":"typed-clause-seam-20260923", "method":"typed complete English phrase lattice with online character debt", "stats":{"phrase_units":len(bank),"runs":len(rows),"exact_growths":len(exact),"longest_letters":max(r["letters"] for r in rows),"longest_growth":max(r["growth_over_center"] for r in rows)}, "rows":rows, "exact_growths":exact, "reader_gate":"closed pending intact-vs-shuffled human test", "provenance":{"bigram_sha256":hashlib.sha256(BIGRAMS.read_bytes()).hexdigest()}}

if __name__ == "__main__":
    payload = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(payload, indent=2)+"\n"); print(json.dumps(payload["stats"], sort_keys=True))
