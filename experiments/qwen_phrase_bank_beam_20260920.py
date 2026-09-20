"""Use a fresh remote proposal bank only to order exact live search branches."""
from __future__ import annotations

import json
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.bigram import BigramModel
from llm_palindrome.generate import build_vocab
from llm_palindrome.phrases import build_inventory
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify


def audit(text):
    tape = re.sub(r"[^a-z]", "", text.casefold())
    exact = bool(tape) and tape == tape[::-1]
    return {
        "letters": len(tape), "two_pointer_exact": exact,
        "sha_equal_under_reversal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(tape[::-1].encode()).hexdigest(),
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def mechanical_gate(text):
    words = re.findall(r"[a-z]+", text.casefold())
    repeated = len(words) != len(set(words))
    self_pal = any(len(word) > 1 and word == word[::-1] for word in words)
    nested = False
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            if start == 0 and end == len(words):
                continue
            tape = "".join(words[start:end])
            if len(tape) > 1 and tape == tape[::-1]:
                nested = True
    return {"no_repeated_words": not repeated, "no_self_palindromic_word": not self_pal,
            "no_nested_word_span": not nested, "mechanically_admitted": not repeated and not self_pal and not nested}

OUT = ROOT / "runs/qwen-phrase-bank-beam-20260920.json"


def load_phrases(path=ROOT / "qwen-phrases-20260920.json"):
    payload = json.loads(Path(path).read_text())
    raw = payload["message"]["content"]
    items = json.loads(raw)["items"]
    out = []
    for item in items:
        text = item if isinstance(item, str) else item.get("text", "")
        text = " ".join(text.lower().replace("'", "").split())
        if text and text not in out:
            out.append(text)
    return out


def run(*, seeds=10, vocab_limit=8_000, inventory_limit=5_000,
        beam_width=250, max_steps=140, phrase_weight=4.0,
        short_penalty=2.0, phrase_unit_min=0):
    phrases = load_phrases()
    vocab = build_vocab(vocab_limit)
    inventory = build_inventory(str(ROOT / "data/count_2w.txt"), vocab=set(vocab),
                                top_n=inventory_limit, min_count=3)
    units = list(dict.fromkeys(vocab + inventory + phrases))
    tries = WordTries(units)
    bigrams = BigramModel.from_file(str(ROOT / "data/count_2w.txt"), vocab=set(vocab))
    scorer = CoherentScorer(
        bigrams, freq_weight=0.10, length_weight=0.14, phrase_weight=phrase_weight,
        long_bonus=1.2, short_penalty=short_penalty, unit_bonus={phrase: 3.0 for phrase in phrases},
    )
    rows = []
    for seed in range(seeds):
        units_out = beam_search(
            tries, scorer, min_letters=39, max_steps=max_steps,
            beam_width=beam_width, candidate_limit=400, per_parent=8,
            seed=seed, diversity=1.25, max_word_uses=2,
        )
        if not units_out:
            continue
        text = textify(units_out)
        rows.append({"seed": seed, "rendered": text, "units": units_out,
                     "proposal_units": [unit for unit in units_out if unit in phrases],
                     "audit": audit(text), "mechanical_gate": mechanical_gate(text),
                     "reader_status": "human-unreviewed; programmatic scores are diagnostic"})
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]
             and len(row["proposal_units"]) >= phrase_unit_min]
    admitted = [row for row in exact if row["mechanical_gate"]["mechanically_admitted"]]
    return {
        "experiment_id": "qwen-phrase-bank-beam-20260920",
        "method": "fresh remote proposal phrases as ranked units in exact outside-in beam",
        "phrase_count": len(phrases), "rows": rows, "exact": exact,
        "stats": {"seeds": seeds, "rows": len(rows), "exact": len(exact),
                  "mechanically_admitted": len(admitted),
                  "longest": max((row["audit"]["letters"] for row in rows), default=0)},
        "provenance": {"proposal_model": "qwen2.5:3b on Mac mini", "catalogue_text": False,
                       "finished_tape_reversal": False, "post_hoc_repair": False,
                       "candidate_acceptance": "independent pointer/SHA audit; reader gate remains closed"},
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in sorted(result["rows"], key=lambda x: -x["audit"]["letters"]):
        print(row["audit"]["letters"], row["audit"]["two_pointer_exact"], row["rendered"])
