"""Record one real search session so the mock can replay it.

The page is animated by a sequence of append-only frames, not by a single
answer, so a fixture that is just a finished palindrome exercises almost none
of the frontend. Recording an actual session captures the thing that is hard to
fabricate: a run of palindromes where each one extends the last outward from the
mirror, with the mirror landing inside a word as often as it lands in a gap.

The session is cut at the first frame past `--letters`, so the fixture is a real
search stopped early rather than a short search — which is what the page will
see in production when the budget runs out mid-growth.

    python server/record_fixture.py --letters 400
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PALINDROME_NO_WARM", "1")  # no GPT-2 for a recording

from llm_palindrome.bigram import BigramModel          # noqa: E402
from llm_palindrome.centerout import centerout_search  # noqa: E402
from llm_palindrome.generate import build_vocab        # noqa: E402
from llm_palindrome.scoring import CoherentScorer      # noqa: E402
from llm_palindrome.search import WordTries            # noqa: E402
from llm_palindrome.validator import is_palindrome, normalize  # noqa: E402

from server.app import _coverage, _shape, _wanted_words, longest_palindromic_center  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--letters", type=int, default=400,
                    help="cut the session at the first frame past this length")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--floor", type=int, default=400)
    ap.add_argument("--commit-every", type=float, default=0.75)
    ap.add_argument("--candidates", type=int, default=800)
    ap.add_argument("--bigrams", default="data/count_2w.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("server/fixtures/session.json"))
    args = ap.parse_args()

    vocab = build_vocab()
    tries = WordTries(vocab)
    bigrams = BigramModel.from_file(args.bigrams, vocab=set(vocab))

    # _coverage reads the service's module-level bigram table, which is never
    # populated here because the warm thread is disabled. Without this the
    # fixture records a coherence of 0.0 and the page has nothing to show.
    import server.app as service
    service._bigrams = bigrams
    center = longest_palindromic_center(args.prompt)
    scorer = CoherentScorer(bigrams, center=center,
                            wanted=_wanted_words(args.prompt))

    frames: list[dict] = []
    seen = {"letters": 0}
    t0 = time.monotonic()

    def relay(words: list[str]) -> None:
        if sum(len(w) for w in words) <= seen["letters"]:
            return
        shape = _shape(words, args.prompt)
        seen["letters"] = shape["letters"]
        # `_shape` puts a word COUNT in "words", so the list is kept under a
        # different key and stripped before the fixture is written.
        frames.append({"at": round(time.monotonic() - t0, 3),
                       **shape, "wordList": list(words)})

    centerout_search(
        tries, scorer, min_letters=args.floor, beam_width=60, seed=args.seed,
        center=center, max_steps=10 ** 6, maximize="letters",
        candidate_limit=args.candidates,
        deadline=time.monotonic() + args.budget,
        on_closed=relay, commit_every=args.commit_every,
    )

    if not frames:
        raise SystemExit("nothing closed; raise --budget")

    # Cut at the first frame past the requested length. Every frame is a
    # palindrome in its own right, so the cut leaves a valid session.
    keep = [f for f in frames if f["letters"] <= args.letters]
    tail = [f for f in frames if f["letters"] > args.letters]
    if tail:
        keep.append(tail[0])
    if not keep:
        keep = frames[:1]

    final = keep[-1]
    text = " ".join(final["wordList"])
    if not is_palindrome(text):
        raise SystemExit("recorded a non-palindrome; refusing to write it")

    session = {
        "recorded": time.strftime("%Y-%m-%d"),
        "prompt": args.prompt,
        "letters": final["letters"],
        "normalized": normalize(text),
        "frames": [{k: v for k, v in f.items() if k != "wordList"} for f in keep],
        "result": {
            "type": "result",
            **{k: v for k, v in final.items() if k not in ("at", "wordList")},
            "lm": None,
            "usedPrompt": bool(center) or bool(scorer.wanted),
            "coherence": _coverage(final["wordList"]),
            "seconds": final["at"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(session, indent=2))

    print(f"{len(keep)} frames, {final['letters']} letters, "
          f"{len(final['wordList'])} words -> {args.out}")
    print(f"palindrome verified: {is_palindrome(text)}")
    print(text[:200] + ("..." if len(text) > 200 else ""))


if __name__ == "__main__":
    main()
