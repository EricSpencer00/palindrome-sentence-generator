"""Palindrome generation service.

Center-out search on this box, GPT-2 reranking on this box, streamed to the
browser over SSE. The site animates the result outward from the center, which
is the order the search actually builds it in.

Endpoints
  GET /health            liveness + whether the LM finished loading
  GET /api/generate      SSE: status heartbeats during the search, then a result
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import centerout_search
from llm_palindrome.generate import build_vocab
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries
from llm_palindrome.validator import is_palindrome, normalize

MAX_BUDGET = float(os.environ.get("PALINDROME_MAX_BUDGET", "20"))
DEFAULT_BUDGET = float(os.environ.get("PALINDROME_BUDGET", "14"))
# The length dial. 400 overshoots to ~1200 letters / ~370 words at 60% attested
# bigrams; pushing it higher buys length by spending coherence (800 -> 48%).
LENGTH_FLOOR = int(os.environ.get("PALINDROME_FLOOR", "400"))
SEEDS = int(os.environ.get("PALINDROME_SEEDS", "2"))
# Wider candidate lists are what let the bigram model actually choose: at 200 the
# letter constraint leaves almost nothing to pick between (52% of joins attested),
# at 800 it leaves enough (74%).
CANDIDATE_LIMIT = int(os.environ.get("PALINDROME_CANDIDATES", "800"))
BIGRAMS_PATH = os.environ.get("PALINDROME_BIGRAMS", "data/count_2w.txt")

app = FastAPI(title="palindrome")

_tries: Optional[WordTries] = None
_bigrams: Optional[BigramModel] = None
_lm = None
_lm_error: Optional[str] = None


def _warm() -> None:
    """Vocabulary and model load once, off the request path."""
    global _tries, _bigrams, _lm, _lm_error
    vocab = build_vocab()
    _tries = WordTries(vocab)
    _bigrams = BigramModel.from_file(BIGRAMS_PATH, vocab=set(vocab))
    try:
        from llm_palindrome.lm_scoring import GPT2Scorer
        _lm = GPT2Scorer("gpt2", device="cpu")
    except Exception as exc:  # scoring is optional; search still works
        _lm_error = f"{type(exc).__name__}: {exc}"


threading.Thread(target=_warm, daemon=True).start()


def _wanted_words(prompt: str) -> set[str]:
    """Prompt words the search should favour where letters allow."""
    return {w.lower() for w in prompt.split() if w.isalpha() and len(w) > 1}


def _coverage(words: list[str]) -> float:
    """Share of adjacent pairs that are attested English bigrams."""
    if _bigrams is None or len(words) < 2:
        return 0.0
    hits = sum(1 for a, b in zip(words, words[1:]) if _bigrams._fwd.get(a, {}).get(b))
    return round(hits / (len(words) - 1), 3)


def longest_palindromic_center(prompt: str) -> str:
    """Use the visitor's own text as the mirror point when it can be one."""
    s = normalize(prompt)
    if not s:
        return ""
    best = ""
    for i in range(len(s)):
        for j in range(i + len(best) + 1, len(s) + 1):
            chunk = s[i:j]
            if chunk == chunk[::-1] and len(chunk) > len(best):
                best = chunk
    return best if len(best) >= 3 else ""


def _search(prompt: str, budget: float) -> Optional[dict]:
    """Grow for `budget` seconds and keep the longest palindrome that closed.

    Length is bounded by time rather than by a target, so a loaded box returns
    a shorter palindrome instead of making the visitor wait. The budget is
    split across seeds and GPT-2 picks between the survivors.
    """
    center = longest_palindromic_center(prompt)
    scorer = CoherentScorer(_bigrams, center=center, wanted=_wanted_words(prompt))
    per_seed = max(6.0, budget / SEEDS)

    candidates: list[list[str]] = []
    for seed in range(SEEDS):
        words = centerout_search(
            _tries, scorer, min_letters=LENGTH_FLOOR, beam_width=60, seed=seed,
            center=center, max_steps=10**6, maximize="letters",
            candidate_limit=CANDIDATE_LIMIT,
            deadline=time.monotonic() + per_seed,
        )
        if words and is_palindrome(" ".join(words)):
            candidates.append(words)
    if not candidates:
        return None

    # GPT-2 chooses; ties and failures fall back to the longest.
    best_words = max(candidates, key=lambda w: len(normalize(" ".join(w))))
    lm_score = None
    if _lm is not None and len(candidates) > 1:
        try:
            texts = [" ".join(w) for w in candidates]
            scores = _lm.score_texts(texts)
            i = max(range(len(texts)), key=lambda k: scores[k])
            best_words, lm_score = candidates[i], round(scores[i], 3)
        except Exception:
            pass

    text = " ".join(best_words)
    if not is_palindrome(text):  # never serve a broken palindrome
        return None

    if center and center in best_words:
        i = best_words.index(center)
        left, right = best_words[:i], best_words[i + 1:]
    else:
        mid = len(best_words) // 2
        left, right, center = best_words[:mid], best_words[mid:], ""

    # Spaces are invisible to the palindrome, so the visitor's phrase can be shown
    # at the mirror point exactly as they typed it.
    center_display = center
    if center and center == normalize(prompt):
        center_display = " ".join(prompt.split())

    return {
        "type": "result",
        "center": center,
        "centerDisplay": center_display,
        "left": left,       # reading order; animate from its END outward
        "right": right,     # reading order; animate from its START outward
        "letters": len(normalize(text)),
        "words": len(best_words),
        "lm": lm_score,
        "usedPrompt": bool(center) or bool(scorer.wanted),
        "coherence": _coverage(best_words),
    }


@app.get("/health")
def health():
    return {"ok": True, "vocab": _tries is not None,
            "lm": _lm is not None, "bigrams": _bigrams is not None, "lm_error": _lm_error}


@app.get("/api/generate")
async def generate(prompt: str = Query("", max_length=200),
                   budget: float = Query(DEFAULT_BUDGET)):
    budget = max(1.0, min(float(budget), MAX_BUDGET))

    async def stream():
        # Search must not start while the model is still loading: it competes for
        # the same cores and the budget expires before anything closes.
        if _tries is None or _bigrams is None or (_lm is None and _lm_error is None):
            yield _sse({"type": "status", "phase": "warming"})
            for _ in range(240):
                await asyncio.sleep(0.5)
                if _tries and _bigrams and (_lm is not None or _lm_error):
                    break
            if _tries is None or _bigrams is None:
                yield _sse({"type": "error", "message": "still warming up, try again shortly"})
                return

        t0 = time.time()
        task = asyncio.create_task(asyncio.to_thread(_search, prompt, budget))
        # Heartbeats keep the tunnel and the browser from timing out a long search.
        while not task.done():
            yield _sse({"type": "status", "phase": "searching",
                        "elapsed": round(time.time() - t0, 1)})
            await asyncio.sleep(1.0)
        try:
            result = task.result()
        except Exception as exc:
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            return
        if result is None:
            yield _sse({"type": "error", "message": "no palindrome closed; try a shorter length"})
            return
        result["seconds"] = round(time.time() - t0, 1)
        yield _sse(result)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
