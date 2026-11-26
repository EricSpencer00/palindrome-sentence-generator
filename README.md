# Palindrome Paragraph Generator

Whole paragraphs that read the same backwards. Live at
**[palindrome.ericspencer.us](https://palindrome.ericspencer.us)**, with the full
walkthrough at [/how](https://palindrome.ericspencer.us/how/).

> Pull away. And I ask law is all at a net for ever, even till I hate me. Here are we
> not say a war of evil. All its name. Now as it is a wise man names, I was it? I saw
> one man still alive? For a way as to new era. Ere he met a hill. It never ever.
> Often, a tall, as I walk said, nay, a wall. Up.

## How it works

The paragraph grows from the middle out as a beam search over letter debts: whatever
one side writes, the other side owes in reverse. A word that overpays hands the surplus
to the opposite side, so the two halves never need matching word boundaries. Commas,
full stops, question marks and apostrophes cost no letters, so each half ends its
sentences where its own grammar wants. A trigram model trained on Brown and Gutenberg
scores every step, and a count-based verifier judges each sentence as it closes. An
empty prompt draws a random theme. The search keeps Dan Hoey's and Peter Norvig's
bookkeeping ([The Algorithm](https://norvig.com/pal-alg.html)) and demands sentences.

## Paper

**[paper/v5/main.pdf](paper/v5/main.pdf)**: *Debt Crossing: Growing Paragraph-Length
Palindromes from the Centre Out*. Source in [paper/v5/main.tex](paper/v5/main.tex);
targeting the ARR January 2027 cycle (ACL 2027).

```bash
cd paper/v5 && make          # tectonic; needs ../acl.sty
make numbers                 # regenerate numbers.tex and figdata/ from runs/*.json
```

## Run it

```bash
pip install -r requirements.txt && python -m nltk.downloader brown gutenberg
PALINDROME_ENGINE=v5 uvicorn server.app:app --port 8045     # API
cd web && npm install && PAL_API=http://127.0.0.1:8045 npm run dev
```

Tests: `pytest` and `cd web && npm test`.

## Production

| Piece | Where |
|---|---|
| Site | Cloudflare Pages project `palindrome`; `web/functions/api` proxies `/api` |
| API | Mac mini, launchd [`com.ericspencer.palindrome-v5`](deploy/com.ericspencer.palindrome-v5.plist), port 8045, behind a Cloudflare tunnel |
| Load guard | [`server/v5_serve.py`](server/v5_serve.py): 2 search workers, one search per visitor; everyone else gets a finished paragraph from a pool |
| Fallback | [`web/public/replay.json`](web/public/replay.json): 50 paragraphs the page plays when the API is down |

Deploy the site:

```bash
cd web && npm run build && cd .. && tools/build_v1_snapshot.sh
cd web && npx wrangler pages deploy dist --project-name palindrome --branch main
```

## Layout

| Path | What |
|---|---|
| `llm_palindrome/paragraph.py` | the v5 search |
| `llm_palindrome/prose_lm.py`, `spectrum.py`, `contractions.py` | language model, sentence verifier, contractions |
| `server/` | API (`app.py`, `v5.py`, `v5_serve.py`) and archive mode for v1–v4 |
| `web/` | site (React) and `/how` explainer (`web/public/how`) |
| `tools/` | paper metrics, replay and load-test scripts |
| `runs/` | measurements the paper and `/how` are built from |

v1–v4 and the experiments before v5 are recorded in [docs/HISTORY.md](docs/HISTORY.md)
and [RESULTS.md](RESULTS.md). `tools/archive.sh` replays them locally.

## Credit

The overhang search is Peter Norvig's
([World's Longest Palindrome?](https://norvig.com/palindrome.html), 2002), building on
Dan Hoey's 1984 program. Developed within the AI4FM group. MIT license.
