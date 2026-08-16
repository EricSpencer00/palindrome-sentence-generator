"""A bigram table a model believes, rather than one a corpus happened to record.

    PYTHONPATH=. python3 training/build_join_table.py --top 400

Constraining the walk to attested joins was measured against the material it is
supposed to find, and it fails that test. Of the 29 catalogued mirror-pairs —
the readable ones, the whole reason the construction is known to work — the
mean weaker half has **0.21** of its joins in `data/count_2w.txt`, and only 10%
have all of them. "Evil rats on || no star live" has none. Neither does "faced
no devil". A 1M-word corpus does no better: Brown has none of them either.

The joins in a readable palindrome are ordinary English that no corpus of that
size records, because palindromic English is odd English. So attestation is a
FREQUENCY test wearing a grammaticality test's clothes, and a walk constrained
by it steers away from exactly the region the catalogue lives in.

A language model answers the question that was actually being asked. p(b | a)
is defined for every ordered pair, including "no star" and "faced no", and the
table below is that distribution thresholded: for each word, the followers the
model ranks in its top `top`. One forward pass per vocabulary word, so the
whole table costs about as much as scoring a few thousand candidates once.

The first token of a word stands in for the word, which is exact for the
single-token majority of the vocabulary and an approximation elsewhere — the
same approximation `directional.py` documents and measures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--vocab", type=int, default=60000)
    ap.add_argument("--min-zipf", type=float, default=3.0)
    ap.add_argument("--top", type=int, default=400,
                    help="followers kept per word. The table is a constraint, "
                         "not a ranking, so it wants to be generous: 400 of "
                         "50,257 tokens is roughly 'the model would not be "
                         "surprised'.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="data/joins_gpt2.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from wordfreq import zipf_frequency

    from llm_palindrome.generate import build_vocab
    from llm_palindrome.lexicon import is_real_word, load_lexicon
    from llm_palindrome.pairs import pair_vocabulary

    lexicon = load_lexicon("data/lexicon.txt")
    cache: dict[str, float] = {}

    def zipf(word: str) -> float:
        hit = cache.get(word)
        if hit is None:
            hit = cache[word] = zipf_frequency(word, "en")
        return hit

    vocab = sorted(set(pair_vocabulary(build_vocab(args.vocab), zipf, lexicon,
                                       args.min_zipf)))
    print(f"{len(vocab)} words")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    # Every word reached by its first token, with the leading space GPT-2 spells
    # a mid-sentence word with.
    first = {w: tok(" " + w, add_special_tokens=False)["input_ids"][0]
             for w in vocab}
    by_token: dict[int, list[str]] = {}
    for word, tid in first.items():
        by_token.setdefault(tid, []).append(word)

    table: dict[str, list[str]] = {}
    with torch.no_grad():
        for i in range(0, len(vocab), args.batch):
            chunk = vocab[i:i + args.batch]
            ids = torch.tensor([[first[w]] for w in chunk], device=device)
            logits = model(input_ids=ids).logits[:, -1]
            best = torch.topk(logits, args.top, dim=-1).indices.tolist()
            for word, tokens in zip(chunk, best):
                followers = [f for t in tokens for f in by_token.get(t, ())]
                table[word] = sorted(followers)
            if i % (args.batch * 20) == 0:
                print(f"  {i}/{len(vocab)}")

    Path(args.out).write_text(json.dumps(table))
    edges = sum(len(v) for v in table.values())
    print(f"{edges} joins over {len(table)} words -> {args.out}")


if __name__ == "__main__":
    main()
