"""v5: long palindrome paragraphs, grown centre-out with crossing debts.

    GET /api/v5/health
    GET /api/v5/paragraph?theme=sea,ship&letters=600     JSON
    GET /api/v5/stream?theme=sea,ship&letters=600        SSE: status ..., result
    GET /                                                the page

Run locally:  python -m server.v5   (http://127.0.0.1:8045)
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from llm_palindrome import paragraph as P
from llm_palindrome.prose_lm import ProseLM, theme_bonus, theme_vectors
from llm_palindrome.spectrum import Spectrum, Tables
from llm_palindrome.validator import is_palindrome

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "server" / "v5_page.html"
EXAMPLES = ROOT / "data" / "v5_examples.json"
FEATURED = ROOT / "data" / "v5_featured.json"
MIN_LETTERS, MAX_LETTERS = 120, 1200
BEAM = int(os.environ.get("PALINDROME_V5_BEAM", "200"))
SLACK = int(os.environ.get("PALINDROME_V5_SLACK", "40"))
SITE = os.environ.get("PALINDROME_SITE", "https://palindrome.ericspencer.us")

router = APIRouter(prefix="/api/v5", tags=["v5 paragraphs"])

_models: dict = {}
_ready = threading.Event()
_lock = threading.Semaphore(int(os.environ.get("PALINDROME_V5_WORKERS", "2")))


def _warm() -> None:
    lm, tb = ProseLM.load(), Tables.load()
    _models.update(lm=lm, lex=P.Lexicon(lm, tb), sp=Spectrum(tb), vec=theme_vectors())
    _ready.set()


def _theme(raw: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z]+", raw.lower()) if len(w) > 2][:6]
    return [w for w in words if w in _models["vec"]]


def _shape(text: str) -> dict:
    """Two halves and the mirror, in printed words, for the page to draw."""
    words = text.split()
    total = sum(sum(c.isalpha() for c in w) for w in words)
    half, run = total / 2, 0
    for i, w in enumerate(words):
        n = sum(c.isalpha() for c in w)
        if run + n > half:
            if run == half:
                return {"left": words[:i], "center": "", "right": words[i:]}
            return {"left": words[:i], "center": w, "right": words[i + 1:]}
        run += n
    return {"left": words, "center": "", "right": []}


def generate(theme: str, letters: int, seed: int = 0, on_draft=None) -> dict:
    _ready.wait()
    m = _models
    letters = max(MIN_LETTERS, min(MAX_LETTERS, letters))
    themes = _theme(theme)
    t0 = time.time()
    with _lock:
        closed = P.grow(m["lm"], m["lex"], letters, beam=BEAM, spectrum=m["sp"], sent_w=2.0,
                        content_bonus=1.6, theme=theme_bonus(m["vec"], themes) if themes else None,
                        theme_w=15.0, seed=seed, start_noise=4.0, slack=SLACK,
                        on_step=(lambda st: on_draft(P.render(st.seq), st.n)) if on_draft else None)
    if not closed:
        raise RuntimeError("no paragraph closed")
    text = P.render(closed[0].seq)
    if not is_palindrome(text):  # never serve a broken palindrome
        raise RuntimeError("internal: closure failed the letter check")
    tape = re.sub("[^a-z]", "", text.lower())
    return {"text": text, "letters": len(tape), "words": len(text.split()),
            "sentences": text.count("."), "theme": themes, "seed": seed,
            "seconds": round(time.time() - t0, 1), **_shape(text)}


@router.get("/health")
def health():
    return {"ok": True, "ready": _ready.is_set(), "beam": BEAM}


@router.get("/paragraph")
async def paragraph(theme: str = Query("", max_length=120), letters: int = Query(400),
                    seed: int = Query(0, ge=0, le=10_000)):
    try:
        return await asyncio.to_thread(generate, theme, letters, seed)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@router.get("/stream")
async def stream(theme: str = Query("", max_length=120), letters: int = Query(400),
                 seed: int = Query(0, ge=0, le=10_000)):
    async def events():
        drafts: list = []
        task = asyncio.create_task(asyncio.to_thread(
            generate, theme, letters, seed, lambda text, n: drafts.append((text, n))))
        t0, sent = time.time(), 0
        while not task.done():
            phase = "searching" if _ready.is_set() else "warming"
            msg = {"type": "status", "phase": phase, "elapsed": round(time.time() - t0, 1)}
            if len(drafts) > sent:  # the best state so far, still owing letters
                sent = len(drafts)
                msg.update(type="draft", text=drafts[-1][0], letters=drafts[-1][1])
            yield f"data: {json.dumps(msg)}\n\n"
            await asyncio.sleep(0.5)
        try:
            yield f"data: {json.dumps({'type': 'result', **task.result()})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _examples() -> list[dict]:
    try:
        return json.loads(EXAMPLES.read_text())
    except FileNotFoundError:
        return []


def render_page() -> str:
    """The page with the examples rendered into the HTML, so crawlers and
    no-script readers get the text itself, not an empty shell."""
    ex = _examples()
    cards = "\n".join(
        f'<article class="ex" id="ex-{i + 1}"><h3>{html.escape(", ".join(e["theme"]).title() or "Untitled")}'
        f' <span>{e["letters"]} letters · {e["words"]} words</span></h3>'
        f'<p>{html.escape(e["text"])}</p></article>'
        for i, e in enumerate(ex))
    ld = {"@context": "https://schema.org", "@type": "WebApplication",
          "name": "Palindrome Paragraph Generator", "url": SITE,
          "applicationCategory": "EntertainmentApplication", "operatingSystem": "Any",
          "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
          "description": "Generates long palindrome paragraphs that read the same forwards and backwards, letter by letter.",
          "author": {"@type": "Person", "name": "Eric Spencer"}}
    try:
        hero = json.loads(FEATURED.read_text())
        hero_text = html.escape(hero["text"])
        hero_meta = (f'<span><b>{hero["letters"]}</b> letters</span><span><b>{hero["words"]}</b> words</span>'
                     '<span class="ok">reads the same backwards ✓</span>')
    except FileNotFoundError:
        hero_text = hero_meta = ""
    return (PAGE.read_text()
            .replace("{{HERO_TEXT}}", hero_text)
            .replace("{{HERO_META}}", hero_meta)
            .replace("{{EXAMPLES}}", cards)
            .replace("{{LD_JSON}}", json.dumps(ld))
            .replace("{{SITE}}", SITE))


app = FastAPI(title="palindrome v5", docs_url=None, redoc_url=None)
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
def index():
    return render_page()


@app.get("/robots.txt")
def robots():
    return Response(f"User-agent: *\nAllow: /\nSitemap: {SITE}/sitemap.xml\n", media_type="text/plain")


@app.get("/sitemap.xml")
def sitemap():
    return Response(f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    f"<url><loc>{SITE}/</loc><changefreq>weekly</changefreq></url></urlset>\n",
                    media_type="application/xml")


@app.get("/favicon.svg")
def favicon():
    return Response('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#5b3fd9"/>'
                    '<text x="32" y="44" font-family="Georgia,serif" font-size="34" text-anchor="middle" fill="#fff">ↄc</text></svg>',
                    media_type="image/svg+xml")


threading.Thread(target=_warm, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8045")))
