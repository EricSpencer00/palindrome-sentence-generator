"""Word-aligned mirror-pairs, exhaustively. The answer is four.

Half the catalogued pairs are word-aligned chains of reversible words —
"lived on decaf" mirrors as "faced no devil" because lived/devil, on/no and
decaf/faced are each other's reverses, one word at a time. That is a closed
form: no search, nothing discarded, and the whole space is small enough to
walk in a second.

Walking it settles what the form is worth.

    392 words whose reverse is also a word (both above zipf 3.0)
     57 ordered pairs where the join AND its mirror are both attested
      4 novel chains of three words or more, up to flips

        step on was || saw no pets
        live on was || saw no evil
        spit on was || saw no tips
        maps on was || saw no spam

None is a sentence, and the four share a shape — "X on was || saw no Y" — which
is one template wearing four disguises. The constraint that makes the form
closed is also what starves it: every word must reverse into a word, and 392 of
those exist against 40,000 ordinary ones.

So word-alignment is finished as a source. The walk in `llm_palindrome/pairs.py`
is not word-aligned — the two halves are segmented independently — which is
where the rest of the catalogue lives ("go hang a salami || ima lasagna hog")
and where new material has to come from.

    python experiments/reversible_chains.py
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-zipf", type=float, default=3.0)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=6)
    ap.add_argument("--min-join-count", type=int, default=1)
    args = ap.parse_args()

    from wordfreq import zipf_frequency

    from llm_palindrome.generate import build_vocab
    from llm_palindrome.lexicon import is_real_word, load_lexicon
    from llm_palindrome.mining import bigram_counts
    from llm_palindrome.paragraphs import is_novel_palindrome
    from llm_palindrome.reversibles import chains, mirror_consistent_edges

    lexicon = load_lexicon("data/lexicon.txt")
    vocab = {w for w in build_vocab(60000)
             if is_real_word(w, lexicon)} | {"a", "i"}

    def common(word: str) -> bool:
        return zipf_frequency(word, "en") >= args.min_zipf

    # Palindromic words map to themselves, which is what makes "a", "i" and
    # "did" usable as the connective tissue of a chain.
    reversible = {w: w[::-1] for w in vocab
                  if w[::-1] in vocab and common(w) and common(w[::-1])
                  and (len(w) > 1 or w in ("a", "i"))}
    attested = bigram_counts("data/count_2w.txt",
                             min_count=args.min_join_count)
    edges = mirror_consistent_edges(reversible, attested)

    found, seen = [], set()
    for left, right in chains(reversible, edges, min_words=args.min_words,
                              max_words=args.max_words):
        # A chain and its flip are the same material printed twice: every
        # mirror-consistent path reversed is another mirror-consistent path.
        key = frozenset((" ".join(left), " ".join(right)))
        if key in seen or not is_novel_palindrome(" ".join(left + right)):
            continue
        seen.add(key)
        found.append((left, right))

    print(f"{len(reversible)} reversible words")
    print(f"{sum(len(v) for v in edges.values())} mirror-consistent joins")
    print(f"{len(found)} novel chains of {args.min_words}+ words, up to flips\n")
    for left, right in found:
        print(f"  {' '.join(left)} || {' '.join(right)}")


if __name__ == "__main__":
    main()
