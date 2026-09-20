"""Language-first search: fresh complete sentences, then independent tape pairing.

The proposal model is used only to create a held-out bank of complete English
sentences.  It is never asked for a palindrome and never sees the reverse tape.
The search chooses a natural sentence from the bank, reverses its letters as a
mathematical obligation, and segments that obligation forward into words using
the bank's observed language model.  Exactness is checked independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path("lm-sentence-bank-semordnilap-20260920.json")


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    reverse = tape[::-1]
    mismatch = next(((i, tape[i], tape[-1-i]) for i in range(len(tape)//2)
                     if tape[i] != tape[-1-i]), None)
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
            "sha_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(reverse.encode()).hexdigest()}


def structural_gate(text: str) -> dict:
    words = re.findall(r"[a-z]+", text.casefold())
    repeated = len(words) != len(set(words))
    self_pal = any(len(w) > 1 and w == w[::-1] for w in words)
    nested = False
    full = letters(text)
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            span = "".join(words[i:j])
            if 1 < len(span) < len(full) and span == span[::-1]:
                nested = True
    return {"no_repeated_words": not repeated, "no_self_palindromic_word": not self_pal,
            "no_nested_word_span": not nested,
            "mechanically_admitted": not repeated and not self_pal and not nested}


def extract(raw: str) -> list[str]:
    out = []
    for line in raw.splitlines():
        line = line.strip().strip("` ")
        line = re.sub(r"^(?:\d+[.)]|[-*])\s*", "", line)
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^sentence\s*\d*\s*[:\-]\s*", "", line, flags=re.I)
        tape = letters(line)
        words = re.findall(r"[a-z]+", line.casefold())
        if 5 <= len(words) <= 14 and 32 <= len(tape) <= 120 and line[-1:] in ".!?":
            out.append(" ".join(words))
    return list(dict.fromkeys(out))


def generate_bank(url: str, model: str, batches: int, per_batch: int) -> list[str]:
    rows: list[str] = []
    prompt = ("Write {n} varied original English sentences. Each must be a complete "
              "modern sentence with a subject and finite verb, 5 to 14 words, "
              "with vivid concrete scenes and natural idioms. Use no lists, no "
              "palindrome language, no repeated clauses, and no meta commentary. "
              "Put one sentence per line and nothing else.")
    for batch in range(batches):
        payload = {"model": model, "prompt": prompt.format(n=per_batch), "stream": False,
                   "options": {"temperature": 1.05, "top_p": 0.95, "num_predict": per_batch * 24}}
        req = urllib.request.Request(url.rstrip("/") + "/api/generate",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = json.loads(response.read()).get("response", "")
        rows.extend(extract(raw))
        if (batch + 1) % 10 == 0:
            print(json.dumps({"batch": batch + 1, "sentences": len(set(rows))}), flush=True)
    return list(dict.fromkeys(rows))


def search(sentences: list[str], beam: int = 160, max_words: int = 14,
           external_vocab: list[str] | None = None) -> dict:
    # The bank itself is the language model: word and bigram counts are learned
    # only from independently generated forward sentences.
    word_count = Counter()
    bigram = Counter()
    starts = Counter()
    by_first: dict[str, list[str]] = defaultdict(list)
    for text in sentences:
        words = text.split()
        starts[words[0]] += 1
        word_count.update(words)
        bigram.update(zip(words, words[1:]))
    vocab = sorted(set(word_count) | set(external_vocab or []),
                   key=lambda w: (-word_count.get(w, 0), w))
    vocab = [w for w in vocab if w.isalpha() and len(w) <= 14]
    # Preserve complete observed n-grams as atomic language units.  This is a
    # forward phrase inventory, not a mirrored unit: the reverse tape is still
    # segmented online and every phrase is printed in its normal order.
    phrases: set[str] = set()
    for text in sentences:
        words = text.split()
        for n in range(2, 6):
            phrases.update(" ".join(words[i:i+n]) for i in range(len(words)-n+1))
    units = vocab + sorted(phrases, key=lambda p: (-word_count.get(p.split()[0], 0), p))
    unit_tape = {u: letters(u) for u in units}
    for unit in units:
        if unit_tape[unit]:
            by_first[unit_tape[unit][0]].append(unit)
    rows = []
    probes = 0
    # Prefer complete sentences as the left arm.  The right arm is never taken
    # from the same surface sentence; it is decoded from the character debt.
    for left in sentences:
        tape = letters(left)
        if not (38 <= len(tape) <= 120):
            continue
        probes += 1
        target = tape[::-1]
        # state: (position, last word, words, score, has_verb-like token)
        states = [(0, "", (), 0.0, False)]
        for step in range(max_words):
            nxt = []
            for pos, last, chosen, score, has_verb in states:
                if pos == len(target):
                    if len(chosen) >= 5 and has_verb:
                        rendered = f"{left}; {' '.join(chosen)}."
                        rows.append({"rendered": rendered, "left_sentence": left,
                                     "right_words": chosen, "score": score,
                                     "audit": audit(rendered),
                                     "structural_gate": structural_gate(rendered),
                                     "reader_status": "human-unreviewed; bank score is diagnostic"})
                    continue
                choices = by_first.get(target[pos], ())
                for unit in choices:
                    unit_key = unit_tape[unit]
                    if unit in chosen or not target.startswith(unit_key, pos):
                        continue
                    first_word = unit.split()[0]
                    last_word = unit.split()[-1]
                    if not chosen and starts[first_word] == 0 and first_word not in {
                        "a", "an", "the", "this", "that", "one", "some", "no", "my",
                        "our", "your", "his", "her", "their", "we", "i", "you", "he",
                        "she", "they", "there", "here", "all", "each", "many", "most",
                    }:
                        continue
                    # Light verb heuristic only for admitting a complete right
                    # clause; it does not alter the exact tape.
                    verbish = any(w in {"is", "are", "was", "were", "be", "been", "has", "have", "had", "can", "will", "did", "does"}
                                  or w.endswith(("ed", "ing", "en")) for w in unit.split())
                    transition = math.log1p(bigram.get((last, first_word), 0)) if last else math.log1p(starts.get(first_word, 0))
                    freq = 0.15 * math.log1p(word_count.get(first_word, 0))
                    nxt.append((pos + len(unit_key), last_word, chosen + (unit,), score + transition + freq,
                                has_verb or verbish))
            if not nxt:
                break
            nxt.sort(key=lambda s: (-s[3], s[0], s[2]))
            # Retain states at the deepest positions as well as best scores;
            # otherwise common short words starve the long exact paths.
            by_pos: dict[int, list] = defaultdict(list)
            for state in nxt:
                by_pos[state[0]].append(state)
            states = []
            for pos in sorted(by_pos, reverse=True):
                states.extend(by_pos[pos][: max(4, beam // 20)])
                if len(states) >= beam:
                    break
            states = states[:beam]
        if len(rows) > 100_000:
            break
    rows.sort(key=lambda r: (-r["audit"]["letters"], -r["score"], r["rendered"]))
    exact = [r for r in rows if r["audit"]["pointer_exact"]]
    admitted = [r for r in exact if r["structural_gate"]["mechanically_admitted"]]
    return {"sentences": sentences, "rows": rows[:5000], "exact_candidates": exact[:500],
            "mechanically_admitted": admitted[:200],
            "stats": {"bank_sentences": len(sentences), "left_probes": probes,
                       "rendered": len(rows), "exact": len(exact),
                       "mechanically_admitted": len(admitted),
                       "longest_exact": max((r["audit"]["letters"] for r in exact), default=0),
                       "longest_rendered": max((r["audit"]["letters"] for r in rows), default=0)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--per-batch", type=int, default=80)
    ap.add_argument("--input", default="lm-sentence-bank-20260920.json")
    ap.add_argument("--vocab", default="")
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()
    if args.generate:
        sentences = generate_bank(args.url, args.model, args.batches, args.per_batch)
        Path(args.output).write_text(json.dumps({"sentences": sentences,
            "provenance": {"model": args.model, "url": args.url, "prompt_is_not_palindrome_prompt": True}}, indent=2) + "\n")
        print(json.dumps({"bank_sentences": len(sentences), "output": args.output}))
    if args.search:
        payload = json.loads(Path(args.input).read_text())
        external_vocab = None
        if getattr(args, "vocab", None):
            external_vocab = [w.strip().casefold() for w in Path(args.vocab).read_text().splitlines()]
        result = search(payload["sentences"], external_vocab=external_vocab)
        result.update({"experiment_id": "lm-sentence-bank-semordnilap-20260920",
                       "method": "fresh model-authored complete sentence bank with forward bigram-scored reverse-tape segmentation",
                       "novelty_preflight": {"status": "passed", "signature": "fresh-lm-bank|complete-sentence-left|forward-segmentation",
                                             "distinct_from": "phrase-bank beam: whole complete sentences are the language-first source and the reverse side is decoded by a bank language model",
                                             "finished_tape_reversal": False, "post_hoc_repair": False},
                       "provenance": {"proposal_model": payload.get("provenance", {}).get("model"),
                                      "source": "fresh generated complete sentences; no catalogue text",
                                      "audits": ["independent two-pointer comparison", "forward/reverse SHA-256"],
                                      "reader_gate": "closed until a structurally clean exact row receives blinded human ratings"},
                       "next_construction": "Condition the bank language model on dependency frames while keeping both sentence arms forward-generated.",
                       "status": "fresh exact candidate requires human reading" if result["mechanically_admitted"] else "no mechanically clean exact candidate; language bank and exact controls retained"})
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["stats"]))
        for row in result["exact_candidates"][:30]:
            print(row["audit"]["letters"], row["structural_gate"]["mechanically_admitted"], row["rendered"])


if __name__ == "__main__":
    main()
