"""Mine the mirror-pair inventory the paragraph assembler runs on.

    PYTHONPATH=. python3 training/build_pairs.py

Writes data/mirror_pairs.json, ranked so a caller that takes the first N gets
the most readable N.

Why this exists: assembly was never the bottleneck — `render` has asserted
`is_palindrome` all along — the material was. The canon supplies 23 usable
pairs, and the exhaustive hunts supply 20,000 that read "no it cab action".
Mining takes left halves from attested English bigrams and keeps the ones whose
mirror also segments into real words, which is a different bet: the left half
reads because English attested it, and the right half is filtered rather than
ranked.

Three filters, each answering a different question:

  build_vocab   is this safe to generate?   (mining invents phrases)
  lexicon       is this a word at all?      ("utc", "ips", "evo" are not)
  bigram score  does the right half read?   (used to rank, never to accept)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("data/mirror_pairs.json")
COUNTS = "data/count_2w.txt"
LEXICON = "data/lexicon.txt"
NGRAMS = "data/ngrams_wikitext2.json"


def main() -> int:
    from llm_palindrome.bigram import BigramModel
    from llm_palindrome.generate import build_vocab
    from llm_palindrome.lexicon import is_real_word, load_lexicon
    from llm_palindrome.mining import (attested_bigrams, attested_ngrams,
                                       attested_phrases, mine_pairs,
                                       reads_as_attested)

    lexicon = load_lexicon(LEXICON)
    vocab = [w for w in build_vocab(30000) if is_real_word(w, lexicon)]
    print(f"vocabulary: {len(vocab)} words after the dictionary filter")

    phrases = list(attested_phrases(COUNTS, vocab))
    print(f"attested phrases: {len(phrases)}")

    attested = attested_bigrams(COUNTS)

    # Trigrams as well as bigrams: a half is at most as long as the phrase it
    # came from, so bigrams alone cap the paragraph at two-word fragments.
    # 4-grams are not mined — measured, they yield 0 both-attested pairs, which
    # is the mirror cost arriving as a length curve (131 / 27 / 0).
    phrases += list(attested_ngrams(NGRAMS, vocab, 3))
    print(f"  + attested trigrams -> {len(phrases)} phrases")

    # ONE orientation. Mining each phrase as a right half as well was tried and
    # measured: all 7,704 entries came back with their own flip and not one new
    # unit, because swapping which half is given cannot change which readings
    # the segmenter produces. A pair is still usable either way round — that is
    # sequencing's business, not the inventory's.
    #
    # The letter bounds are wide because the both-attested pairs are short:
    # "went on || not new" is 6 letters a side and a floor of 8 would cut it.
    pairs = list(mine_pairs(phrases, vocab, min_letters=4, max_letters=24,
                            min_words=2, min_word_letters=1,
                            prefer_attested=attested))
    print(f"mined pairs: {len(pairs)}")

    both = sum(1 for left, right in pairs
               if reads_as_attested(right, attested)
               and reads_as_attested(left, attested))
    print(f"  of which BOTH halves attested: {both}")

    bigrams = BigramModel.from_file(COUNTS)

    def reads(words: list[str]) -> float:
        if len(words) < 2:
            return 0.0
        return sum(bigrams.forward(a, b)
                   for a, b in zip(words, words[1:])) / (len(words) - 1)

    # Rank on the RIGHT half only. The left half came from an attested bigram,
    # so it already reads and scoring it again would just re-rank by frequency.
    ranked = sorted(pairs, key=lambda p: -reads(p[1]))

    # `attested` is the quality signal a caller should filter on; `reads` only
    # orders what is left. Both are recorded so neither has to be recomputed.
    OUT.write_text(json.dumps(
        [{"left": left, "right": right, "reads": round(reads(right), 4),
          "attested": (reads_as_attested(left, attested)
                       and reads_as_attested(right, attested))}
         for left, right in ranked], indent=1) + "\n")
    print(f"wrote {len(ranked)} pairs to {OUT}")
    for left, right in ranked[:10]:
        print(f"  {' '.join(left):<20} || {' '.join(right)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
