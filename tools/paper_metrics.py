"""Count-based metrics for the paper and the explainer page.

usage: python tools/paper_metrics.py OUT.json NAME=PATH.jsonl [NAME=PATH.jsonl ...]

Each input line is a JSON object with either "text" or the v1/v5 stream shape
(left, center/centerDisplay, right). Writes per-corpus summaries plus per-text
rows so figures can be drawn from one file.
"""
import json
import re
import statistics as st
import sys

from llm_palindrome.prose_lm import ProseLM
from llm_palindrome.spectrum import Spectrum, Tables
from llm_palindrome.validator import is_palindrome


def text_of(o: dict) -> str:
    if "text" in o:
        return o["text"]
    mid = o.get("centerDisplay") or o.get("center") or ""
    return " ".join([*o["left"], *([mid] if mid else []), *o["right"]])


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+|[.,;:!?]", text.lower())


def words_in(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def row(text: str, sp: Spectrum, lm: ProseLM) -> dict:
    toks = tokens(text)
    words = [t for t in toks if t.isalpha()]
    bis = list(zip(words, words[1:]))
    sents = [s for s in re.split(r"[.!?]", text) if s.strip()]
    # The verifier judges sentences, so it is averaged over sentences.
    judged = [sp.assess(x) for x in sents if len(words_in(x)) >= 2] or [sp.assess(text)]
    r = [j["readable"] for j in judged]
    return {
        "letters": sum(len(w) for w in words),
        "words": len(words),
        "exact": is_palindrome(text),
        "short_share": sum(len(w) <= 2 for w in words) / max(1, len(words)),
        "mean_word_len": sum(map(len, words)) / max(1, len(words)),
        "type_token": len(set(words)) / max(1, len(words)),
        "distinct_bigrams": len(set(bis)) / max(1, len(bis)),
        "sentence_words": len(words) / max(1, len(sents)),
        "lm_per_token": lm.score(toks),
        "readable": st.mean(r),
        "readable_share": sum(x >= sp.R_OK for x in r) / len(r),
        "garbage_share": sum(x < sp.R_LO for x in r) / len(r),
        "ism": st.mean(j["ism"] for j in judged),
        "score": st.mean(j["score"] for j in judged),
    }


def main() -> None:
    out, specs = sys.argv[1], sys.argv[2:]
    tb, lm = Tables.load(), ProseLM.load()
    sp = Spectrum(tb)
    report = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        rows = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            items = o if isinstance(o, list) else [o]
            for it in items:
                t = text_of(it) if isinstance(it, dict) else it
                rows.append({"text": t, **row(t, sp, lm)})
        keys = [k for k in rows[0] if k != "text"]
        summary = {k: (sum(r[k] for r in rows) / len(rows) if k == "exact"
                       else round(st.mean(r[k] for r in rows), 4)) for k in keys}
        summary["n"] = len(rows)
        report[name] = {"summary": summary, "rows": rows}
        print(name, json.dumps(summary), flush=True)
    json.dump(report, open(out, "w"))


if __name__ == "__main__":
    main()
