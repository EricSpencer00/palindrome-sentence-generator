"""An absolute coherence score, which is what a search loop can actually use.

The pairwise experiment in `oss_judge.py` asks a model to reproduce a human
preference. That is the right test of whether a model agrees with blind
judging, and the wrong shape for search: ranking candidates against each other
needs a number per candidate, not a tournament.

So this asks one question per text -- is this real English that vaguely makes
sense? -- on a 0-3 scale, and validates the scale with controls rather than
trusting it.

  prose      WikiText spans, printed as they were written. Must score high or
             the scale is broken.
  prose_pres the SAME spans put through OUR presenter. The first run of this
             experiment had only this arm and it scored 0.45 against a shuffle
             at 0.05, which is not a positive control -- it is evidence that
             the presenter destroys real English. Both arms are kept so that
             cost is measured rather than silently charged to every other arm.
  shuffle    the SAME spans with the word order destroyed, raw.
  shuf_pres  and presented, so the presenter's cost is measured at both ends.
             prose-minus-shuffle is the power of the measurement: a model that
             cannot separate these cannot be believed about anything else.
  human      catalogued palindromes people wrote, including the 51-letter
             record. These are the target. A scale that does not rank them
             above our own output is not measuring what we want. They go
             through OUR presenter, so their authors' punctuation is gone and
             the comparison is content against content -- which is the one we
             want, and does mean this arm is handicapped relative to how those
             palindromes are usually printed.
  chunk      one of our verified chunks, presented.
  k2 k4 k8   our nests at two, four and eight chunks.

The rubric deliberately allows archaic and strained phrasing, and spellings
like `canon` where a reader expects `cannon`. The palindrome constraint forces
those and penalising them would measure orthography, not sense.
"""
from __future__ import annotations

import glob, json, os, random, re, sys, urllib.request

sys.path.insert(0, ".")
from llm_palindrome.present import present
from llm_palindrome.syntax import brown_tables
from llm_palindrome.validator import is_palindrome, normalize
from server.v3 import harvest_pair

HOST = "http://localhost:11434/api/chat"

RUBRIC = """Rate this passage on how much it reads as real English that makes
some kind of sense.

3 - clear, natural English. You could meet this sentence in a book.
2 - real English that vaguely makes sense. It may be strained, archaic,
    poetic or odd, and that is fine at this level.
1 - the phrases hold together locally but the passage as a whole does not
    say anything.
0 - word salad. Words in a row with no structure.

Judge sense and grammar only. Do NOT penalise:
  - archaic or Shakespearean phrasing
  - unusual proper nouns
  - a spelling that is off by a letter, such as "canon" where you would
    expect "cannon"
  - length, in either direction

Passage:
{text}

Answer with exactly one digit, 0 1 2 or 3. Nothing else."""


def chat(model, prompt, think=True):
    """One turn. `think` is only accepted by reasoning models -- llama and
    mistral answer a request carrying it with a 400 -- so it is dropped and
    retried rather than restricting the judges to one model family."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0},
    }
    if think:
        payload["think"] = "low"
    req = urllib.request.Request(HOST, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["message"]["content"]
    except urllib.error.HTTPError as e:
        if think and e.code == 400:
            return chat(model, prompt, think=False)
        raise


def ask(model, text):
    msg = chat(model, RUBRIC.format(text=text))
    hits = re.findall(r"\b([0-3])\b", msg)
    # A refusal or a sentence is recorded as a miss rather than coerced to a
    # number, so an unusable reply cannot be averaged in as a low score.
    return (int(hits[-1]) if hits else None), msg.strip()[:80]


def prose_spans_raw(n, words, rng):
    """Returns (normalised word lists, original text spans) for the same spans,
    so the presented and raw arms are the same English and differ only in what
    this repository did to it."""
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/"
        "wikitext-2-raw-v1/train-*.parquet"))
    import pyarrow.parquet as pq
    lines = [t.strip() for t in pq.read_table(hits[0], columns=["text"])
             .column("text").to_pylist()
             if t.strip() and not t.strip().startswith("=")]
    rng.shuffle(lines)
    out, orig, buf, obuf = [], [], [], []
    for line in lines:
        for tok in line.split():
            x = normalize(tok)
            if not x:
                continue
            buf.append(x); obuf.append(tok)
            if len(buf) >= words:
                out.append(buf); orig.append(" ".join(obuf))
                buf, obuf = [], []
                if len(out) == n:
                    return out, orig
    return out, orig


def build(seed=91):
    rng = random.Random(seed)
    t_, s_, g_ = brown_tables()
    items = []

    # Raw spans keep the source's own words joined by spaces. `normalize`
    # already stripped case and punctuation upstream, so "raw" here means
    # "our presenter did not touch the word order or the marks", which is the
    # comparison the presented arms need.
    spans, raw = prose_spans_raw(20, 24, rng)
    for w in raw:
        items.append(("prose", w))
    for w in spans:
        items.append(("prose_pres", present(list(w), t_, s_, g_)))
    for w in spans:
        sh = list(w); rng.shuffle(sh)
        items.append(("shuffle", " ".join(sh)))
        items.append(("shuf_pres", present(sh, t_, s_, g_)))

    bank = json.load(open("data/v3_bank.json"))
    human = [r for r in bank if r.get("source") == "catalogue"
             and len(normalize(r["text"])) >= 20]
    rng.shuffle(human)
    for r in human[:20]:
        items.append(("human", present(r["text"].split(), t_, s_, g_)))

    pool = []
    for r in bank:
        got = harvest_pair(r["text"].split())
        if got and normalize(" ".join(got[0])) != normalize(" ".join(got[1])):
            pool.append(got)
    rng.shuffle(pool)

    def nest(ps):
        w = []
        for l, _ in ps: w += l
        for _, r in reversed(ps): w += r
        return w

    i = 0
    for kind, k in (("chunk", 1), ("k2", 2), ("k4", 4), ("k8", 8)):
        for _ in range(20):
            if i + k > len(pool):
                pool += pool                      # the bank is small; reuse
            got = pool[i:i + k]; i += k
            w = nest(got)
            assert is_palindrome(" ".join(w))
            items.append((kind, present(w, t_, s_, g_)))

    rng.shuffle(items)
    return items


if __name__ == "__main__":
    model, tag = sys.argv[1], sys.argv[2]
    items = build()
    print(f"{len(items)} items", flush=True)
    rows = []
    for n, (kind, text) in enumerate(items):
        score, raw = ask(model, text)
        rows.append({"kind": kind, "score": score, "text": text})
        print(f"{n:4d} {kind:<8} {score}  {text[:70]!r}", flush=True)
    json.dump(rows, open(f"runs/punct/scale_{tag}.json", "w"), indent=1)
    print("wrote", f"runs/punct/scale_{tag}.json")
