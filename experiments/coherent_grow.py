"""Grow a readable palindrome outward without letting it stop reading.

`experiments/extend_ops.py` proved growth is mechanically free: 750/750 seeds
grew, mean +36 letters. Blind judging then preferred the ungrown seed 20/20,
because growth was accepted on edge-parse and length, and length is not the
scarce thing. This reruns the same operations with a different acceptance
test: a wrap is kept only if a coherence judge scores the grown text no worse
than the text it grew from. The judge filters; it does not compose.

Judge: local gpt-oss:20b, absolute 0-3 scale from
`experiments/coherence_scale.py` -- the instrument that passed the power gate
(prose-shuffle gap +0.85, salad acceptance 0.00) where three other local
models failed it. Scores are taken at temperature 0 on the bare word run, so
presentation cannot confound the comparison.

Search: beam over wrap operations. Each round proposes WRAP-PAIR and GLUE
wraps for every surviving text, keeps candidates whose score holds the floor,
and prunes to the beam width by (score, letters). A text that cannot be grown
without dropping its score is finished, and reported with the trajectory that
built it.
"""
from __future__ import annotations

import argparse, json, sys, time

sys.path.insert(0, ".")
from experiments.coherence_scale import ask
from experiments.extend_ops import glue_pairs, material, wrap_glue, wrap_pair  # noqa: E501
from llm_palindrome.validator import is_palindrome, normalize

MODEL = "gpt-oss:20b"


def judge(text: str, cache: dict) -> int:
    if text not in cache:
        score, _ = ask(MODEL, text)
        cache[text] = -1 if score is None else score
    return cache[text]


def proposals(words, pairs, glue, cap):
    """Wraps only -- the operations that pay the mirror inside the move."""
    out = []
    for a, b in pairs[:cap]:
        out.append(wrap_pair(words, a, b))
    for left, right, *_ in glue[:cap]:
        out.append(wrap_glue(words, left, right))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", default="reno sir parasites set i sara prisoner")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--cap", type=int, default=120,
                    help="wrap proposals per family per round")
    ap.add_argument("--floor-slack", type=int, default=0,
                    help="how far below the seed score a candidate may fall")
    ap.add_argument("--out", default="runs/coherent_grow.jsonl")
    args = ap.parse_args()

    words = args.seed.split()
    assert is_palindrome(" ".join(words))
    from llm_palindrome.generate import build_vocab
    selfpal, pairs = material(build_vocab(30000))
    glue = glue_pairs()
    cache: dict[str, int] = {}

    base = judge(" ".join(words), cache)
    floor = base - args.floor_slack
    print(f"seed [{base}] {' '.join(words)}", flush=True)
    beam = [(base, words)]
    log = [{"round": 0, "score": base, "letters": len(normalize(" ".join(words))),
            "text": " ".join(words)}]

    t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        cands = []
        seen = set()
        for score, ws in beam:
            for nw in proposals(ws, pairs, glue, args.cap):
                text = " ".join(nw)
                key = normalize(text)
                if key in seen or not is_palindrome(text):
                    continue
                seen.add(key)
                cands.append(nw)
        # Cheap structural cut before the judge sees anything: a wrap whose
        # words never appear in the Brown table cannot read. The judge is the
        # expensive resource; spend it on survivors.
        scored = []
        for nw in cands:
            s = judge(" ".join(nw), cache)
            if s >= floor:
                scored.append((s, nw))
        scored.sort(key=lambda t: (-t[0], -len(normalize(" ".join(t[1])))))
        if not scored:
            print(f"round {rnd}: nothing holds the floor; stopping", flush=True)
            break
        beam = scored[:args.beam]
        best = beam[0]
        print(f"round {rnd}: {len(scored)}/{len(cands)} hold floor; "
              f"best [{best[0]}] {len(normalize(' '.join(best[1])))} letters: "
              f"{' '.join(best[1])}", flush=True)
        log.append({"round": rnd, "score": best[0],
                    "letters": len(normalize(" ".join(best[1]))),
                    "text": " ".join(best[1]),
                    "held_floor": len(scored), "proposed": len(cands)})

    out = {"seed": args.seed, "base_score": base, "floor": floor,
           "rounds": log, "judge_calls": len(cache),
           "seconds": round(time.time() - t0, 1)}
    with open(args.out, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rounds"}), flush=True)


if __name__ == "__main__":
    main()
