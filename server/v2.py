"""Palindrome generation service, v2: whole sentences placed atomically.

v1 builds from single words and a bigram scorer, and what it produces is
locally fluent and globally about nothing — measured, not guessed:
`experiments/length_sweep.py` puts its long-range coherence on the word-salad
line at every length from 71 letters to 1197, and a blinded judge rejected 25
of 25 of its sentences.

v2 changes the UNIT. The trie holds ~1300 complete sentences mined from
Wikipedia alongside the 29k words, and the search places one whole or not at
all. A sentence that goes in comes out intact, which is why a judge that
rejected everything v1 produced passes these — and it is also the honest limit
of the thing: **v2 quotes, it does not compose.** The search chooses which
sentences to place and closes the mirror around them; it did not write them.

That makes attribution part of the response rather than a footnote. Wikipedia
text is CC BY-SA 4.0, and an endpoint serving it to the public has to say so
and say which spans are not its own.

Endpoints
  GET /api/v2/health      liveness + whether the inventory finished loading
  GET /api/v2/generate    SSE, same frame contract as v1, plus sentences
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from typing import Optional, Sequence

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import centerout_search
from llm_palindrome.generate import build_vocab
from llm_palindrome.phrases import build_inventory, build_units
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries
from llm_palindrome.paragraphs import is_word_palindrome, word_assemble
from llm_palindrome.reversal import drop_worst
from llm_palindrome.textify import segment_at_units
from llm_palindrome.validator import is_palindrome, normalize

router = APIRouter(prefix="/api/v2")

# The configuration that met the acceptance criterion — 5/5 generations
# carrying a judged-coherent English sentence, against a judge calibrated in
# the same blinded batch (real 5/5 pass, salad 0/5 pass). Changing any of these
# invalidates that result, so they are named rather than inlined.
MAX_BUDGET = float(os.environ.get("PALINDROME_V2_MAX_BUDGET", "20"))
DEFAULT_BUDGET = float(os.environ.get("PALINDROME_V2_BUDGET", "12"))
# 200 closed in about a second, which left the page one frame to animate.
# The floor is also the frame budget: the search overshoots it and stops at
# 3x, so a higher floor buys both a longer text and more of the outward
# draw the page is built around.
LENGTH_FLOOR = int(os.environ.get("PALINDROME_V2_FLOOR", "400"))
# A whole-sentence unit placed against an empty overhang owes ALL of its
# letters at once. v1's cap of 24 rejects every one of them before it is
# scored, which is why v1 could never have placed a sentence whatever its
# vocabulary.
MAX_OVERHANG = int(os.environ.get("PALINDROME_V2_OVERHANG", "48"))
LONG_BONUS = float(os.environ.get("PALINDROME_V2_LONG_BONUS", "25"))
PHRASE_WEIGHT = float(os.environ.get("PALINDROME_V2_PHRASE_WEIGHT", "1.0"))
BIGRAM_UNITS = int(os.environ.get("PALINDROME_V2_BIGRAMS", "20000"))
SENTENCE_ORDERS = os.environ.get("PALINDROME_V2_SENTENCES", "sent6,sent8")
PER_ORDER = int(os.environ.get("PALINDROME_V2_PER_ORDER", "2000"))
NGRAMS_PATH = os.environ.get("PALINDROME_V2_NGRAMS", "data/ngrams_wikitext2.json")
BIGRAMS_PATH = os.environ.get("PALINDROME_BIGRAMS", "data/count_2w.txt")
WORD_BANKS_PATH = os.environ.get("PALINDROME_WORD_BANKS", "data/word_banks.json")

ATTRIBUTION = {
    "source": "Wikipedia, via the WikiText-2 corpus",
    "license": "CC BY-SA 4.0",
    "url": "https://creativecommons.org/licenses/by-sa/4.0/",
    "note": "Sentences marked as quoted are reproduced verbatim from Wikipedia. "
            "The palindrome around them is generated; the quoted sentences are not.",
}

_tries: Optional[WordTries] = None
_bigrams: Optional[BigramModel] = None
_sentences: frozenset[str] = frozenset()
_load_error: Optional[str] = None


def _warm() -> None:
    global _tries, _bigrams, _sentences, _load_error
    try:
        vocab = build_vocab()
        _bigrams = BigramModel.from_file(BIGRAMS_PATH, vocab=set(vocab))
        inventory = build_inventory(BIGRAMS_PATH, vocab=vocab, top_n=BIGRAM_UNITS)
        data = json.loads(open(NGRAMS_PATH, encoding="utf-8").read())
        quoted = []
        for order in SENTENCE_ORDERS.split(","):
            quoted.extend(data.get(order.strip(), [])[:PER_ORDER])
        _sentences = frozenset(quoted)
        _tries = WordTries(build_units(vocab, inventory + quoted))
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"


if os.environ.get("PALINDROME_NO_WARM") != "1":
    threading.Thread(target=_warm, daemon=True).start()


def quoted_units(units: Sequence[str], inventory: frozenset[str] | set[str]) -> list[str]:
    """The units in this palindrome that were lifted whole from the corpus.

    These are the spans the response has to attribute: the search placed them,
    it did not build them.
    """
    seen, out = set(), []
    for unit in units:
        if unit in inventory and unit not in seen:
            seen.add(unit)
            out.append(unit)
    return out


def sentence_payload(units: Sequence[str],
                     inventory: frozenset[str] | set[str]) -> list[dict]:
    """The text as sentences, each flagged for whether it is a quote.

    Segmentation follows the units rather than a word count: a quoted sentence
    is exactly one sentence, and the filler between quotes is grouped into its
    own. Punctuation and case are invisible to the palindrome, so this costs
    the mirror nothing.
    """
    out: list[dict] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            out.append({"text": " ".join(buffer).capitalize() + ".", "quoted": False})
            buffer.clear()

    for unit in units:
        if unit in inventory:
            flush()
            out.append({"text": unit.capitalize() + ".", "quoted": True})
        else:
            buffer.extend(unit.split())
    flush()
    return out


def request_seed(prompt: str, nonce: int) -> int:
    """A different search per request.

    The sentence inventory is small — about 1300 units — so the beam converges
    on the same handful whenever it starts from the same place. Varying the
    seed is what stops every visitor being served the same five sentences.
    """
    return abs(hash((prompt, nonce))) % (2 ** 31)


def seed_sequence(prompt: str, nonce: int, attempts: int) -> list[int]:
    """Distinct seeds to try, in order."""
    return [request_seed(prompt, nonce + i) for i in range(attempts)]


ATTEMPTS = int(os.environ.get("PALINDROME_V2_ATTEMPTS", "4"))


def _search(prompt: str, budget: float, on_partial=None,
            nonce: int = 0) -> Optional[dict]:
    """Grow until something closes or the budget runs out.

    A single search at this floor closes on about 85% of random seeds, so a
    visitor who drew one of the other 15% got an error page. Each attempt costs
    roughly two seconds against a twelve-second budget, which is room for
    several — and the deadline is shared, so a slow first attempt simply leaves
    less for the rest rather than overrunning the promise.
    """
    deadline = time.monotonic() + budget
    for seed in seed_sequence(prompt, nonce, ATTEMPTS):
        if time.monotonic() >= deadline:
            break
        found = _attempt(prompt, seed, deadline, on_partial)
        if found is not None:
            return found
    return None


def _attempt(prompt: str, seed: int, deadline: float, on_partial=None) -> Optional[dict]:
    from server.app import _shape, _wanted_words, longest_palindromic_center

    center = longest_palindromic_center(prompt)
    scorer = CoherentScorer(_bigrams, center=center, wanted=_wanted_words(prompt),
                            phrase_weight=PHRASE_WEIGHT, long_bonus=LONG_BONUS)

    seen = {"letters": 0}

    def relay(units: list[str]) -> None:
        if on_partial is None:
            return
        try:
            words = [w for u in units for w in u.split()]
            if sum(len(w) for w in words) <= seen["letters"]:
                return
            shape = _shape(words, prompt)
            seen["letters"] = shape["letters"]
            on_partial({"type": "partial", **shape})
        except Exception:
            pass

    units = centerout_search(
        _tries, scorer, min_letters=LENGTH_FLOOR, beam_width=60, seed=seed,
        center=center, max_steps=10**6, maximize="letters",
        candidate_limit=800, max_overhang=MAX_OVERHANG,
        # A varying seed buys nothing at the default 0.4: `docs/training.md`
        # measured that the jitter is too small to reorder the leading
        # candidates, so 1975 of 2000 searches walked into the same opening.
        # Varying the seed and leaving diversity alone would have served every
        # visitor the same palindrome anyway.
        diversity=float(os.environ.get("PALINDROME_V2_DIVERSITY", "1.0")),
        deadline=deadline,
        on_closed=relay if on_partial is not None else None,
        commit_every=0.75 if on_partial is not None else None,
    )
    if not units:
        return None

    words = [w for u in units for w in u.split()]
    text = " ".join(words)
    if not is_palindrome(text):   # never serve a broken palindrome
        return None

    quotes = quoted_units(units, _sentences)
    return {
        "type": "result",
        **_shape(words, prompt),
        "sentences": sentence_payload(units, _sentences),
        "quoted": quotes,
        "quotedCount": len(quotes),
        "attribution": ATTRIBUTION,
        "usedPrompt": bool(center),
        "promptWordsPlaced": sorted(_wanted_words(prompt) & set(words)),
    }


CENTRES_PATH = "data/centres.json"
_centres: Optional[list] = None


def _load_centres() -> list:
    global _centres
    if _centres is None:
        import json as _json
        from pathlib import Path as _Path
        _centres = _json.loads(_Path(CENTRES_PATH).read_text())
    return _centres


def letter_paragraph(sentences: int = 7, prompt: str = "") -> dict:
    """A paragraph whose LETTERS mirror, assembled from whole sentences.

    The mirror cost of 3.296 bits per free letter forbids free-running
    palindromic prose past about thirty letters, so the constraint is paid
    inside each unit and the paragraph is nested around a centre. What took a
    hundred iterations to find is that the units have to be SENTENCES:
    two-word halves mirror just as correctly and carry no subject, so nothing
    can be about anything, and every attempt at a through-line failed on
    material rather than on the constraint.

    The canon's self-palindromic sentences do carry subjects. `best_cluster`
    finds the ones that share one, `order_for_refrain` seats questions outward
    and the firmest statement on the turn, and `refrain` mirrors the sequence —
    palindromic by construction at any length.
    """
    import re

    from llm_palindrome.paragraphs import refrain
    from llm_palindrome.themes import (best_cluster, content_words,
                                       order_for_refrain, trim_to_theme)

    pool = list(_load_centres())
    asked = {w for w in prompt.lower().split() if w.isalpha()}
    if asked:
        # Steer, do not filter: a prompt matching two centres would otherwise
        # return two sentences and call it a paragraph.
        pool.sort(key=lambda c: -len(asked & content_words(c)))
        head = [c for c in pool if asked & content_words(c)]
        chosen = (head + best_cluster([c for c in pool if c not in head],
                                      max(0, sentences - len(head))))[:sentences]
        # The prompt names the theme. Left to find its own anchor the trim
        # picks the most recurrent word, which for a "basil" request was still
        # "saw" — discarding the one sentence the visitor came for.
        anchor = next((w for w in asked
                       if any(w in content_words(c) for c in head)), None)
    else:
        chosen, anchor = best_cluster(pool, sentences), None

    # Shorter beats padded. A thin theme filled out to the requested length
    # ranked BELOW the same theme returned short under blind judging: the
    # strangers do not add to the subject, they remove it.
    chosen = order_for_refrain(trim_to_theme(chosen, anchor))

    # Say what the paragraph turned out to be about. With no prompt there is
    # still a subject and the reader should be told it; with a prompt that
    # matched nothing, `theme` is null rather than the default one, so the
    # request is not reported as answered when it was only served.
    if anchor is None and not asked:
        from collections import Counter
        shared: Counter = Counter()
        for unit in chosen:
            shared.update(content_words(unit))
        anchor = next((w for w, n in shared.most_common() if n > 1), None)

    text = " ".join(u.capitalize() + "." for u in refrain(chosen))
    return {
        "mode": "letter",
        "text": text,
        "units": chosen,
        "theme": anchor,
        "prompted": bool(asked),
        "words": len(re.findall(r"[A-Za-z]+", text)),
        "letterPalindrome": is_palindrome(text),
        "wordPalindrome": is_word_palindrome(text),
        "note": "Letter-level palindrome: the letters read the same both ways. "
                "Each sentence is itself a palindrome and the sequence mirrors, "
                "so the whole is one at any length.",
    }


@router.get("/paragraph")
def paragraph(sentences: int = Query(7, ge=1, le=40),
              prompt: str = Query("", max_length=200),
              mode: str = Query("letter", pattern="^(letter|word)$")):
    """A palindromic paragraph. Letter-level by default.

    The word mode mirrors the SENTENCE SEQUENCE and not the letters — a
    different and much easier constraint, since it pays nothing per letter. It
    was the default while the letter-level paragraph was still a list of
    fragments. It is kept, and labelled, as the curiosity it is.
    """
    if mode == "letter":
        return letter_paragraph(sentences=sentences, prompt=prompt)

    import json as _json
    import time as _time
    from pathlib import Path as _Path
    nonce = _time.time_ns()
    banks = _json.loads(_Path(WORD_BANKS_PATH).read_text())
    name, bank = pick_bank(banks, prompt, nonce)
    # The word mode's own measured threshold: a length ladder judged
    # three times put its through-line at ~105 words, about 17 units.
    out = word_paragraph(bank, sentences=max(sentences, 18), nonce=nonce)
    out["bank"] = name
    return out


@router.get("/health")
def health():
    return {"ok": True, "version": 2, "vocab": _tries is not None,
            "sentences": len(_sentences), "error": _load_error}


@router.get("/generate")
async def generate(prompt: str = Query("", max_length=200),
                   budget: float = Query(DEFAULT_BUDGET)):
    budget = max(1.0, min(float(budget), MAX_BUDGET))

    async def stream():
        if _tries is None:
            yield _sse({"type": "status", "phase": "warming"})
            for _ in range(240):
                await asyncio.sleep(0.5)
                if _tries is not None or _load_error:
                    break
            if _tries is None:
                yield _sse({"type": "error",
                            "message": _load_error or "still warming up"})
                return

        yield _sse({"type": "plan", "expectLetters": 3 * LENGTH_FLOOR,
                    "version": 2})

        t0 = time.time()
        drafts: queue.Queue = queue.Queue(maxsize=64)
        task = asyncio.create_task(
            asyncio.to_thread(_search, prompt, budget, _offer(drafts),
                              time.time_ns()))
        while not task.done():
            while True:
                try:
                    yield _sse(drafts.get_nowait())
                except queue.Empty:
                    break
            yield _sse({"type": "status", "phase": "searching",
                        "elapsed": round(time.time() - t0, 1)})
            await asyncio.sleep(0.25)
        try:
            result = task.result()
        except Exception as exc:
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            return
        if result is None:
            yield _sse({"type": "error", "message": "no palindrome closed; try again"})
            return
        result["seconds"] = round(time.time() - t0, 1)
        yield _sse(result)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _offer(q: "queue.Queue"):
    def put(payload: dict) -> None:
        try:
            q.put_nowait(payload)
        except queue.Full:
            pass
    return put


# Words that carry stakes rather than scenery. A paragraph built only from
# equipment and weather was judged "pure formal reversals without thematic
# progression"; the one that held a subject contained doubt, fatigue, theories
# and dawn. These are what the through-line is made of.
ARC_WORDS = frozenset({
    "doubt", "doubted", "doubters", "fatigue", "failure", "failed", "grief",
    "hunger", "silence", "silenced", "loss", "lost", "fear", "patience",
    "certainty", "theories", "dawn", "spring", "mercy", "faith", "hope",
    "debts", "buried", "mourned", "remembered", "forgot", "believed",
    "survived", "outlived", "outlasted", "rebuilt", "returned", "healed",
})


def select_units(outer: list, sentences: int, nonce: int,
                 gaps: Optional[dict] = None) -> list:
    """Choose which units appear, keeping the arc and seating it inward.

    A uniform shuffle treats "doubt shadowed certainty" and "frost coated
    glass" alike, so a short request often returns only scenery. Stakes-bearing
    units are kept first and placed nearest the centre — the centre is the
    turn, and the turn is where the stakes belong.

    `gaps` is the measured reversal table (see llm_palindrome.reversal). Every
    unit chosen here appears twice — forward, then mirrored in the second half
    — so a unit whose mirror does not mean anything spends two seats to say one
    thing. When the bank has surplus, the least stable go first.
    """
    import random
    rng = random.Random(nonce)
    want = max(0, min(sentences, len(outer)))
    if gaps:
        outer = drop_worst(outer, gaps, want=want, fraction=0.25)
    arc = [u for u in outer if ARC_WORDS & set(u.lower().split())]
    rest = [u for u in outer if u not in arc]
    rng.shuffle(arc)
    rng.shuffle(rest)
    # Fill from the arc first, then scenery, then order scenery-outward so the
    # arc lands beside the centre.
    keep_arc = arc[:max(1, want // 2)] if arc else []
    keep_rest = rest[:want - len(keep_arc)]
    if len(keep_arc) + len(keep_rest) < want:
        keep_arc += arc[len(keep_arc):want - len(keep_rest)]
    return keep_rest + keep_arc


def pick_bank(banks: dict, prompt: str, nonce: int = 0) -> tuple:
    """Choose a themed bank from what the visitor typed.

    The letter mode centres a palindrome on the visitor's own words; the word
    mode ignored them entirely and served one bank. Scoring the prompt against
    each bank's name and vocabulary is the cheapest way to make it answer.
    """
    import random
    import re
    asked = set(re.findall(r"[a-z]+", prompt.lower()))
    best, score = None, 0
    for name, bank in banks.items():
        vocab = {w for s in bank["outer"] for w in re.findall(r"[a-z]+", s.lower())}
        hits = len(asked & vocab) + (3 if name in asked else 0)
        if hits > score:
            best, score = name, hits
    if best is None:
        best = random.Random(nonce).choice(sorted(banks))
    return best, banks[best]


def word_paragraph(bank: dict, sentences: int = 12,
                   nonce: Optional[int] = None) -> dict:
    """A word-order palindromic paragraph, assembled from authored units.

    A different constraint from the rest of this service. Letter-level
    generation is bounded by the mirror cost — 3.3 bits per free letter — and
    its paragraphs are refrain poetry assembled from canonical material. Word
    order costs nothing per letter, so the same symmetric assembly holds a
    subject across 200+ words and judges as an intentional composition.

    The response says which it served: a visitor told "palindrome" deserves to
    know whether the letters mirror or the words do.
    """
    import random
    import re
    # A fixed bank read in fixed order is a static page served over HTTP. The
    # subset and its order are the variation — chosen so the arc survives.
    if nonce is None:
        outer = list(bank["outer"])[:max(0, sentences)]
    else:
        outer = select_units(list(bank["outer"]), sentences, nonce,
                             bank.get("reversal"))
    text = word_assemble(outer, bank["center"])
    words = re.findall(r"[A-Za-z]+", text)
    return {
        "mode": "word",
        "text": text,
        "words": len(words),
        "pairs": len(outer),
        "wordPalindrome": is_word_palindrome(text),
        "letterPalindrome": is_palindrome(text),
        "note": "Word-order palindrome: the sentence sequence reads the same "
                "both ways. The letters do not mirror — that is the other mode.",
    }
