# Palindrome Paragraph Generator

Whole paragraphs that read the same backwards:
**[palindrome.ericspencer.us](https://palindrome.ericspencer.us)** ·
[how it works](https://palindrome.ericspencer.us/how/) ·
**[paper](paper/v5/main.pdf)**

The paragraph grows from the middle out as a beam search over letter debts, scored by a
trigram model and a sentence verifier. It keeps Hoey's and
[Norvig's](https://norvig.com/pal-alg.html) bookkeeping and demands sentences.

```bash
pip install -r requirements.txt && python -m nltk.downloader brown gutenberg
PALINDROME_ENGINE=v5 uvicorn server.app:app --port 8045
cd web && npm install && PAL_API=http://127.0.0.1:8045 npm run dev
```

Earlier versions: [docs/HISTORY.md](docs/HISTORY.md). MIT license.
