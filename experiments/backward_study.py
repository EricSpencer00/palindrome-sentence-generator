"""Does a backward language model close the gap between a palindrome's halves?

The measurement this answers is from docs/architecture.md: the half built by
prepending scores +0.368 worse than the half built by appending, in both search
directions, so the penalty belongs to backward construction. If that diagnosis
is right, giving the search a model that reads backwards should shrink the gap.
If the gap does not move, the dual-head architecture is built on a wrong
premise and should not be attempted.

Three arms at matched budgets:

  zipf  frequency scoring only — the search as it stands
  fwd   a forward LM term on the appended half, none on the prepended half
  bwd   the same forward term, plus a backward LM term on the prepended half

`fwd` is the control that matters. It separates "a language model in the loop
helps" from "a *backward* language model helps", which is the actual claim.

Every arm is judged by the same unchanged metric: forward GPT-2 on the finished
text in natural reading order. The judge never sees which arm produced what,
and no arm can improve its score by changing how it is measured.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from llm_palindrome.directional import DirectionalScorer, ForwardOnlyScorer
from llm_palindrome.generate import ZipfScorer, build_vocab
from llm_palindrome.lm_scoring import GPT2Scorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify
from llm_palindrome.validator import is_palindrome


def split_halves(words: list[str]) -> tuple[str, str]:
    """Split the word sequence at the midpoint of its letters."""
    total = sum(len(w) for w in words)
    acc, cut = 0, len(words)
    for i, w in enumerate(words):
        acc += len(w)
        if acc >= total / 2:
            cut = i + 1
            break
    return " ".join(words[:cut]), " ".join(words[cut:])


def adjacent_repeat_rate(words: list[str]) -> float:
    """Share of adjacent word pairs that are the same word.

    Reported because the scorer's repeat penalty depends on knowing which end
    of a half just grew; a broken adjacency contract shows up here first.
    """
    if len(words) < 2:
        return 0.0
    return sum(a == b for a, b in zip(words, words[1:])) / (len(words) - 1)


def run_arm(name, scorer, tries, judge, seeds, min_letters, beam) -> dict:
    t0 = time.time()
    runs = []
    for seed in range(seeds):
        w = beam_search(tries, scorer, min_letters=min_letters,
                        beam_width=beam, seed=seed)
        if w:
            runs.append(w)
    elapsed = time.time() - t0

    if not runs:
        return {"arm": name, "closed": 0, "seeds": seeds, "seconds": elapsed}

    texts = [textify(w) for w in runs]
    lefts, rights = zip(*(split_halves(w) for w in runs))
    # Outside-in: the left half is appended, the right half prepended.
    appended = judge.score_texts([textify(t.split()) for t in lefts])
    prepended = judge.score_texts([textify(t.split()) for t in rights])
    whole = judge.score_texts(texts)

    gap = statistics.mean(appended) - statistics.mean(prepended)
    wins = sum(1 for a, p in zip(appended, prepended) if a > p)
    return {
        "arm": name,
        "closed": len(runs),
        "seeds": seeds,
        "seconds": round(elapsed, 1),
        "seconds_per_close": round(elapsed / len(runs), 2),
        "letters_mean": round(statistics.mean(sum(len(x) for x in w) for w in runs), 1),
        "score_best": round(max(whole), 4),
        "score_mean": round(statistics.mean(whole), 4),
        "appended_half": round(statistics.mean(appended), 4),
        "prepended_half": round(statistics.mean(prepended), 4),
        "gap": round(gap, 4),
        "appended_half_wins": f"{wins}/{len(runs)}",
        "adjacent_repeat_rate": round(statistics.mean(
            adjacent_repeat_rate(w) for w in runs), 4),
        "all_valid": all(is_palindrome(t) for t in texts),
        "sample": texts[0][:220],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backward", required=True,
                    help="path to the backward fine-tune")
    ap.add_argument("--forward", default="gpt2",
                    help="path to the forward fine-tune; the matched control")
    ap.add_argument("--judge", default="gpt2",
                    help="metric model, deliberately not a fine-tuned one")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--min-letters", type=int, default=200)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--weights", default="0.25,0.5,1.0",
                    help="LM term weights to sweep; fwd and bwd share each one")
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--arms", default="zipf,fwd,bwd")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    vocab = build_vocab(args.vocab)
    tries = WordTries(vocab)
    judge = GPT2Scorer(args.judge)

    def make(name, weight):
        base = ZipfScorer()
        if name == "zipf":
            return base
        if name == "fwd":
            return ForwardOnlyScorer(base, forward_path=args.forward,
                                     appends="left", weight=weight, vocab=vocab)
        return DirectionalScorer(base, forward_path=args.forward,
                                 backward_path=args.backward,
                                 appends="left", weight=weight, vocab=vocab)

    # The LM term is swept rather than assumed. A term strong enough to change
    # the reading is also strong enough to starve the beam of branches that can
    # close, and where that line falls is not knowable in advance. fwd and bwd
    # always share a weight, so the comparison stays like-for-like.
    arms = args.arms.split(",")
    weights = [float(w) for w in args.weights.split(",")]
    plan = [(a, None) for a in arms if a == "zipf"]
    plan += [(a, w) for w in weights for a in arms if a != "zipf"]

    results = []
    for name, weight in plan:
        scorer = make(name, weight or 0.0)
        label = name if weight is None else f"{name}@{weight:g}"
        if name == "bwd" and weight == weights[0]:
            frac = scorer.single_token_fraction(vocab)
            print(f"[bwd] single-token scoring is exact for "
                  f"{frac:.1%} of the {len(vocab)}-word vocabulary")
        row = run_arm(label, scorer, tries, judge, args.seeds,
                      args.min_letters, args.beam)
        row["weight"] = weight
        if hasattr(scorer, "passes"):
            row["model_passes"] = scorer.passes
            row["cache_misses"] = scorer.misses
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

    by = {r["arm"]: r for r in results}
    print("\n arm          closed  gap      score   letters")
    for r in results:
        print(f" {r['arm']:12s} {r['closed']:>3}/{r['seeds']:<3} "
              f"{r.get('gap', float('nan')):+.3f}  {r.get('score_mean', float('nan')):+.3f}  "
              f"{r.get('letters_mean', 0):.0f}")

    for w in weights:
        f, b = by.get(f"fwd@{w:g}"), by.get(f"bwd@{w:g}")
        if f and b and f.get("gap") is not None and b.get("gap") is not None:
            print(f"\nweight {w:g}: gap {f['gap']:+.3f} -> {b['gap']:+.3f} "
                  f"(narrowed by {f['gap'] - b['gap']:+.3f}); "
                  f"score {f['score_mean']:+.3f} -> {b['score_mean']:+.3f}; "
                  f"closed {f['closed']} -> {b['closed']}")

    if args.out:
        args.out.write_text(json.dumps({"config": vars(args) | {"out": str(args.out)},
                                        "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
