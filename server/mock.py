"""A stand-in for the generation service, so the page can be worked on alone.

The real service needs a vocabulary, a bigram table and GPT-2 in memory, and
then spends its whole budget searching. None of that is interesting when the
thing being changed is the page, and all of it makes the loop slow.

This replays a recorded session instead: the same SSE events, in the same
order, with the same field names, paced to the timings the search actually
produced. `server/fixtures/session.json` holds a real 410-letter palindrome and
the six append-only frames that built it — recorded, not fabricated, so the
append-only guarantee the page relies on genuinely holds.

    python -m server.mock                 # 127.0.0.1:8011
    PAL_API=http://127.0.0.1:8011 npm --prefix web run dev

Endpoints match the real service exactly:
  GET /health        liveness, plus which fixture is loaded
  GET /api/generate  SSE: plan, status, partial frames, result
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

FIXTURE_DIR = Path(os.environ.get(
    "PALINDROME_FIXTURES", Path(__file__).parent / "fixtures"))
DEFAULT_FIXTURE = os.environ.get("PALINDROME_FIXTURE", "session")
# 1.0 replays at the speed the search actually ran. Raise it to shorten the
# loop; lower it when the pacing itself is what is being looked at.
SPEED = float(os.environ.get("PALINDROME_MOCK_SPEED", "1.0"))

app = FastAPI(title="palindrome-mock")

_sessions: dict[str, dict] = {}
_errors: dict[str, str] = {}


def _load() -> None:
    """Load every fixture, and refuse any that is not actually a palindrome.

    A mock that serves a broken palindrome teaches the page to render one, so
    the check is worth keeping even though the recorder already ran it.
    """
    _sessions.clear()
    _errors.clear()
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            norm = data.get("normalized", "")
            if not norm or norm != norm[::-1]:
                raise ValueError("fixture is not a palindrome")
            _sessions[path.stem] = data
        except Exception as exc:
            _errors[path.stem] = f"{type(exc).__name__}: {exc}"


def _pick(name: Optional[str], prompt: str = "") -> Optional[dict]:
    """Fixture by explicit name, else by prompt, else the default.

    Matching on the prompt is what makes both of the page's centre cases
    reachable by typing rather than by restarting with a flag: a fixture
    recorded with "level" put the mirror inside a word and echoed the prompt
    there, and one recorded without a prompt put it in a gap. The real service
    varies the same way for the same reason.
    """
    if name and name in _sessions:
        return _sessions[name]
    wanted = " ".join(prompt.lower().split())
    if wanted:
        for session in _sessions.values():
            if " ".join((session.get("prompt") or "").lower().split()) == wanted:
                return session
    if DEFAULT_FIXTURE in _sessions:
        return _sessions[DEFAULT_FIXTURE]
    return next(iter(_sessions.values()), None)


_load()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/health")
def health():
    """Same shape as the real service, plus what is being served.

    `mock: true` is deliberate and load-bearing: something has to be able to
    tell that a page is talking to a fixture, or a recorded palindrome ends up
    in a screenshot captioned as a fresh one.
    """
    ok = bool(_sessions)
    return {
        "ok": ok,
        "mock": True,
        "vocab": ok,          # the real service reports these; keep the keys
        "lm": False,          # nothing is scored in a replay
        "bigrams": ok,
        "lm_error": None,
        "default": DEFAULT_FIXTURE if DEFAULT_FIXTURE in _sessions else None,
        "speed": SPEED,
        "fixtures": {
            name: {
                "recorded": s.get("recorded"),
                "letters": s.get("letters"),
                "frames": len(s.get("frames", [])),
                "prompt": s.get("prompt") or None,
                # The page draws these two cases differently, so which one a
                # fixture covers is the useful thing to see here.
                "mirrorInWord": bool(s.get("result", {}).get("center")),
                "oddLetters": bool(s.get("result", {}).get("pivotOdd")),
                "coherence": s.get("result", {}).get("coherence"),
            }
            for name, s in _sessions.items()
        },
        "errors": _errors or None,
    }


@app.get("/api/generate")
async def generate(prompt: str = Query("", max_length=200),
                   budget: float = Query(14.0),
                   fixture: Optional[str] = Query(None),
                   speed: Optional[float] = Query(None)):
    session = _pick(fixture, prompt)
    pace = speed if speed and speed > 0 else SPEED

    async def stream():
        if session is None:
            yield _sse({"type": "error",
                        "message": f"no usable fixture in {FIXTURE_DIR}: {_errors}"})
            return

        frames = session["frames"]
        result = dict(session["result"])

        # The page sizes its grid from this before the first frame and then
        # never rewraps, so it has to lead.
        yield _sse({"type": "plan", "expectLetters": session["letters"]})

        t0 = time.time()
        elapsed = 0.0
        for frame in frames:
            wait = max(0.0, frame["at"] - elapsed) / max(0.01, pace)
            # Heartbeats between frames, as the real service emits them.
            while wait > 0:
                step = min(0.25, wait)
                await asyncio.sleep(step)
                wait -= step
                yield _sse({"type": "status", "phase": "searching",
                            "elapsed": round(time.time() - t0, 1)})
            elapsed = frame["at"]
            yield _sse({"type": "partial",
                        **{k: v for k, v in frame.items() if k != "at"}})

        result["seconds"] = round(time.time() - t0, 1)
        yield _sse(result)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()

    import uvicorn
    if not _sessions:
        print(f"warning: no usable fixture in {FIXTURE_DIR}: {_errors}")
    for name, s in _sessions.items():
        mark = " (default)" if name == DEFAULT_FIXTURE else ""
        print(f"  {name}{mark}: {s['letters']} letters, "
              f"{len(s['frames'])} frames"
              + (f", mirror inside {s['result']['center']!r}"
                 if s["result"].get("center") else ", mirror in a gap"))
    print(f"http://{args.host}:{args.port}/health")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
