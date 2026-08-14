"""CLI: generate a long, multi-sentence, LLM-ranked character palindrome.

Pipeline: wordfreq vocabulary -> Tromp-style overhang beam search (many seeds)
-> GPT-2 rerank of closed palindromes -> sentence formatting -> validation.
"""
from __future__ import annotations

import argparse

from wordfreq import top_n_list, zipf_frequency

from .search import WordTries, beam_search
from .textify import textify
from .validator import is_palindrome, normalize


class ZipfScorer:
    """Real-English word frequencies guide the search before the LM reranks."""

    def word_delta(self, left: tuple, right: tuple, placement: str, word: str) -> float:
        # left/right already include the new word; penalize every prior use of
        # it anywhere in the palindrome so cheap mirror-cycles can't dominate.
        uses = left.count(word) + right.count(word) - 1
        seq = left if placement == "L" else right
        neighbor = seq[-2] if placement == "L" and len(seq) >= 2 else (
            seq[1] if placement == "R" and len(seq) >= 2 else None)
        bigram_repeat = -4.0 if neighbor == word else 0.0
        return (zipf_frequency(word, "en") + 0.3 * len(word)
                - 2.0 * uses + bigram_repeat)


REAL_SINGLE_LETTERS = {"a", "i", "o"}


def build_vocab(n: int = 30000) -> list[str]:
    """Frequent English words, minus stray single letters. Lone letters are
    perfect overhang filler, so the search leans on them until they're banned."""
    return [w for w in top_n_list("en", n)
            if w.isalpha() and w.isascii()
            and (len(w) > 1 or w in REAL_SINGLE_LETTERS)]


def make_lm_prune(lm, textify_fn, keep: int):
    """Rescore the beam with the language model and keep the most fluent."""
    def prune(states):
        if len(states) <= keep:
            return states
        texts = [textify_fn(list(s.left) + list(s.right)) for s in states]
        scores = lm.score_texts(texts)
        order = sorted(range(len(states)), key=lambda i: -scores[i])
        return [states[i] for i in order[:keep]]
    return prune


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-letters", type=int, default=120)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=24, help="independent search runs")
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--model", default="gpt2", help="HF model for reranking ('' to skip)")
    ap.add_argument("--words-per-sentence", type=int, default=7)
    ap.add_argument("--lm-in-loop", action="store_true",
                    help="let the LM prune the beam during search, not just at the end")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tries = WordTries(build_vocab(args.vocab))
    scorer = ZipfScorer()

    lm = None
    if args.model:
        from .lm_scoring import GPT2Scorer
        lm = GPT2Scorer(args.model)

    prune = None
    if lm is not None and args.lm_in_loop:
        prune = make_lm_prune(lm, lambda ws: " ".join(ws), keep=max(8, args.beam // 3))

    candidates: list[list[str]] = []
    for seed in range(args.seeds):
        words = beam_search(tries, scorer, min_letters=args.min_letters,
                            beam_width=args.beam, seed=seed, prune=prune)
        if words:
            candidates.append(words)
    if not candidates:
        raise SystemExit("no closed palindrome found; lower --min-letters or raise --beam")

    texts = [textify(w, args.words_per_sentence) for w in candidates]
    if lm is not None:
        scores = lm.score_texts(texts)
        ranked = sorted(zip(scores, texts), key=lambda p: -p[0])
    else:
        ranked = [(0.0, t) for t in texts]

    best_score, best = ranked[0]
    assert is_palindrome(best), "internal error: candidate failed validation"
    print(best)
    print(f"\n[letters={len(normalize(best))} sentences={best.count('.')} "
          f"candidates={len(texts)} lm_score={best_score:.3f}]")
    if args.out:
        with open(args.out, "w") as f:
            f.write(best + "\n")


if __name__ == "__main__":
    main()
