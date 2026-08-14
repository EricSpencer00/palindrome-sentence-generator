"""Best-of-N oracle experiment: ranking problem or search-space problem?

Best-of-N with a good judge is an upper bound on what any reranker could ever
achieve on this search space. If best-of-2000 reads no better than best-of-24,
ranking is exhausted and no future scorer work can help.

Generation mirrors generate.py:main with no LM in the loop: Zipf-scored
overhang beam search, one closed palindrome per seed, seeds 0..N-1. Scoring
uses a modern small base model (Qwen/Qwen2.5-0.5B, falling back to gpt2-large)
and reports BOTH normalizations — total token logprob per token and per letter.
This repo has been burned twice by per-letter artifacts; per token is the
number to believe, and both are recorded.

Usage:
    python3 experiments/oracle_bound.py            # generate + score + report
    python3 experiments/oracle_bound.py --rescore  # rescore cached texts only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_palindrome.generate import ZipfScorer, build_vocab
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify
from llm_palindrome.validator import is_palindrome

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(REPO, "runs", "oracle_bound_texts.json")
OUT_PATH = os.path.join(REPO, "runs", "oracle_bound.json")

CURVE_NS = [1, 10, 24, 100, 500, 1000, 2000]


# ---------------------------------------------------------------- generation

def generate(n_seeds: int, min_letters: int, beam_width: int,
             vocab_size: int) -> list[dict]:
    tries = WordTries(build_vocab(vocab_size))
    scorer = ZipfScorer()
    results: list[dict] = []
    skipped = 0
    t0 = time.time()
    for seed in range(n_seeds):
        words = beam_search(tries, scorer, min_letters=min_letters,
                            beam_width=beam_width, seed=seed)
        if not words:
            skipped += 1
        else:
            text = textify(words)
            # Validity comes from the search; a failure here is a bug, not a
            # bad sample. Abort loudly.
            assert is_palindrome(text), (
                f"FATAL: seed {seed} produced a non-palindrome: {text!r}")
            results.append({"seed": seed, "text": text,
                            "letters": sum(c.isalpha() for c in text)})
        if (seed + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  [generate] seed {seed + 1}/{n_seeds}  "
                  f"closed={len(results)} skipped={skipped}  "
                  f"{el:.0f}s elapsed, ~{el / (seed + 1) * (n_seeds - seed - 1):.0f}s left",
                  flush=True)
    print(f"  [generate] done: {len(results)}/{n_seeds} closed, "
          f"{skipped} skipped, {time.time() - t0:.0f}s", flush=True)
    return results


# ------------------------------------------------------------------- scoring

def load_scorer_model(preferred: str, fallback: str):
    """Try the preferred model; fall back and say so out loud."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for name in (preferred, fallback):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name).to(device)
            model.eval()
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            print(f"  [score] using model: {name} on {device}", flush=True)
            return name, tok, model, device
        except Exception as e:  # noqa: BLE001 — download/load failure
            print(f"  [score] could not load {name}: {e}", flush=True)
    raise SystemExit("no scoring model could be loaded")


def score_texts(texts: list[str], tok, model, device,
                batch_size: int = 16) -> list[dict]:
    """Total token logprob, normalized per token AND per letter.

    Follows the batching pattern of llm_palindrome/lm_scoring.py: right-padded
    batch, logprob of each target token, padding masked out. Token count is the
    number of *scored* (target) tokens, i.e. sequence length minus one.
    """
    import torch

    out: list[dict] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
            logits = model(**enc).logits
            logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            targets = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:]
            tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask
            for j, text in enumerate(batch):
                total = tok_lp[j].sum().item()
                n_tok = int(mask[j].sum().item())
                letters = max(1, sum(c.isalpha() for c in text))
                out.append({
                    "total_logprob": total,
                    "tokens": n_tok,
                    "letters": letters,
                    "per_token": total / max(1, n_tok),
                    "per_letter": total / letters,
                })
            if (i // batch_size) % 10 == 0:
                print(f"  [score] {min(i + batch_size, len(texts))}/{len(texts)}",
                      flush=True)
    return out


# -------------------------------------------------------------------- report

def best_of_n_curve(scores: list[float], ns: list[int],
                    min_estimates: int = 20, rng_seed: int = 0) -> dict:
    """Mean of subset maxima over disjoint random subsets of size N.

    One shuffle partitions the pool into total//N disjoint subsets; extra
    shuffles (each internally disjoint) are added until at least min_estimates
    subset maxima exist, so no small-N point is a single lucky draw. At
    N == len(scores) the value is simply the max.
    """
    rng = random.Random(rng_seed)
    total = len(scores)
    curve = {}
    for n in ns:
        if n > total:
            continue
        if n == total:
            curve[n] = {"mean_best": max(scores), "n_subsets": 1, "sd_best": 0.0}
            continue
        k_per_shuffle = total // n
        n_shuffles = max(1, math.ceil(min_estimates / k_per_shuffle))
        maxima = []
        for _ in range(n_shuffles):
            idx = list(range(total))
            rng.shuffle(idx)
            for s in range(k_per_shuffle):
                subset = idx[s * n:(s + 1) * n]
                maxima.append(max(scores[i] for i in subset))
        curve[n] = {
            "mean_best": statistics.mean(maxima),
            "sd_best": statistics.stdev(maxima) if len(maxima) > 1 else 0.0,
            "n_subsets": len(maxima),
        }
    return curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--min-letters", type=int, default=200)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--fallback-model", default="gpt2-large")
    ap.add_argument("--cache", default=CACHE_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--rescore", action="store_true",
                    help="skip generation, rescore the cached texts")
    args = ap.parse_args()

    # ---- generation phase (cached) ----
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cached = json.load(f)
        if cached.get("params") == {"seeds": args.seeds,
                                    "min_letters": args.min_letters,
                                    "beam": args.beam, "vocab": args.vocab}:
            print(f"[generate] using cached texts: {args.cache} "
                  f"({len(cached['palindromes'])} palindromes)", flush=True)
            results = cached["palindromes"]
        else:
            print("[generate] cache params differ; regenerating", flush=True)
            results = None
    else:
        results = None
    if results is None:
        if args.rescore:
            raise SystemExit("--rescore given but no matching cache exists")
        print(f"[generate] {args.seeds} seeds, min_letters={args.min_letters}, "
              f"beam={args.beam}, no LM in the loop", flush=True)
        results = generate(args.seeds, args.min_letters, args.beam, args.vocab)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        with open(args.cache, "w") as f:
            json.dump({"params": {"seeds": args.seeds,
                                  "min_letters": args.min_letters,
                                  "beam": args.beam, "vocab": args.vocab},
                       "palindromes": results}, f, indent=1)
        print(f"[generate] cached to {args.cache}", flush=True)

    for r in results:  # belt and braces: revalidate whatever we score
        assert is_palindrome(r["text"]), f"cache holds a non-palindrome (seed {r['seed']})"

    # ---- scoring phase ----
    texts = [r["text"] for r in results]
    model_used, tok, model, device = load_scorer_model(args.model,
                                                      args.fallback_model)
    t0 = time.time()
    scores = score_texts(texts, tok, model, device)
    print(f"  [score] {len(texts)} texts in {time.time() - t0:.0f}s", flush=True)

    per_token = [s["per_token"] for s in scores]
    per_letter = [s["per_letter"] for s in scores]

    # ---- report ----
    curve_tok = best_of_n_curve(per_token, CURVE_NS)
    curve_let = best_of_n_curve(per_letter, CURVE_NS)

    order = sorted(range(len(scores)), key=lambda i: -per_token[i])
    top5 = [{"rank": k + 1, "seed": results[i]["seed"],
             "per_token": per_token[i], "per_letter": per_letter[i],
             "letters": scores[i]["letters"], "tokens": scores[i]["tokens"],
             "text": texts[i]} for k, i in enumerate(order[:5])]
    mid = len(order) // 2
    median3 = [{"rank": k + 1, "seed": results[i]["seed"],
                "per_token": per_token[i], "per_letter": per_letter[i],
                "letters": scores[i]["letters"], "tokens": scores[i]["tokens"],
                "text": texts[i]}
               for k, i in zip(range(mid - 1, mid + 2), order[mid - 1:mid + 2])]

    report = {
        "question": ("best-of-N oracle bound: if best-of-2000 reads no better "
                     "than best-of-24, ranking is exhausted on this search space"),
        "params": {"seeds": args.seeds, "min_letters": args.min_letters,
                   "beam": args.beam, "vocab": args.vocab,
                   "closed": len(results)},
        "judge_model": model_used,
        "device": device,
        "per_token_stats": {
            "mean": statistics.mean(per_token),
            "sd": statistics.stdev(per_token),
            "min": min(per_token),
            "max": max(per_token),
        },
        "per_letter_stats": {
            "mean": statistics.mean(per_letter),
            "sd": statistics.stdev(per_letter),
            "min": min(per_letter),
            "max": max(per_letter),
        },
        "best_of_n_per_token": curve_tok,
        "best_of_n_per_letter": curve_let,
        "top5_by_per_token": top5,
        "median3_by_per_token": median3,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    # ---- printed summary ----
    print(f"\n=== oracle bound: {len(results)} closed palindromes, "
          f"judged by {model_used} ===")
    st = report["per_token_stats"]
    print(f"per-token scores:  mean {st['mean']:.4f}  sd {st['sd']:.4f}  "
          f"min {st['min']:.4f}  max {st['max']:.4f}")
    print("\nbest-of-N curve (per TOKEN — the number to believe):")
    print(f"{'N':>6} {'mean best':>10} {'sd':>7} {'subsets':>8}")
    for n, row in curve_tok.items():
        print(f"{n:>6} {row['mean_best']:>10.4f} {row['sd_best']:>7.4f} "
              f"{row['n_subsets']:>8}")
    print("\nbest-of-N curve (per letter — artifact-prone, reported for the record):")
    print(f"{'N':>6} {'mean best':>10} {'sd':>7} {'subsets':>8}")
    for n, row in curve_let.items():
        print(f"{n:>6} {row['mean_best']:>10.4f} {row['sd_best']:>7.4f} "
              f"{row['n_subsets']:>8}")
    print("\n--- top 5 by per-token score ---")
    for t in top5:
        print(f"\n[#{t['rank']} seed={t['seed']} per_token={t['per_token']:.4f} "
              f"per_letter={t['per_letter']:.4f} letters={t['letters']}]")
        print(t["text"])
    print("\n--- 3 around the median ---")
    for t in median3:
        print(f"\n[rank {t['rank']}/{len(order)} seed={t['seed']} "
              f"per_token={t['per_token']:.4f} per_letter={t['per_letter']:.4f}]")
        print(t["text"])
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
