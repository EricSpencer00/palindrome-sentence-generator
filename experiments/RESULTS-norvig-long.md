# Long Norvig–Hoey search, 5 September 2026

This experiment follows the user's explicit length-first request: grow the
"A man, a plan … a canal, Panama" construction into many paragraphs using
hardcoded search. It is a separate target from coherent novel prose in
`docs/NORTH-STAR.md`.

## Method

`experiments/norvig_long.py` implements inward depth-first overhang matching,
using the existing forward/reverse tries and `consume` operation. It starts
with left `a man, a plan`, right `a canal, panama`, and right-owned debt `aca`.
A palindromic debt closes the entire text. Closed improvements are verified and
saved while the walk continues. Dead ends backtrack with exact removal of
phrase and word counts. The latest version also backs up 100 moves after
2,000 accepted moves without a length improvement; this is a bounded heuristic,
not an exhaustive proof of a maximum.

The dictionary is the previously cached [Norvig phrase inventory](https://norvig.com/npdict.txt).
The algorithm is credited to [Norvig, building on Hoey](https://norvig.com/pal-alg.html).
The new output is a searched arrangement of dictionary material, not a copy of
Norvig's finished palindrome. The dictionary includes obscure names and
abbreviations, as inspection of the resulting text confirms.

Hard constraints:

- No phrase is reused, including alternate spacings with identical letters.
- No adjacent word repeats, including across phrase boundaries and the centre.
- Each non-function word occurs at most three times. The explicit function-word
  exception list is `a an the and or of to in on at for with by`.
- The complete output is checked against its reverse after removing nonletters.

Paragraph breaks are inserted roughly every 100 words, at phrase boundaries.
They are presentation breaks in one large palindromic list, not assertions of
independent grammatical sentences or paragraph coherence.

## Reproduction

```
.venv-v3/bin/python -m experiments.norvig_long --seconds 120 --seed 4 --out runs/norvig-long-4
.venv-v3/bin/python -m pytest -q tests/test_norvig_long.py tests/test_growth_convention.py tests/test_graph_search.py tests/test_llm_palindrome.py
```

Wall-clock budgeting makes the exact stopping point machine-dependent. Saved
`phrases.json` files retain the complete construction. Runs 0–2 used ordinary
DFS without the later stagnation backjump; use `--stagnation-nodes 0` for that
variant. Search order is otherwise determined by the dictionary order and seed.

The final delivery and independent audit are stored in `artifacts/norvig-long/`.
Counts below are filled from the completed runs, not estimates or targets.

| seed | letters | words | paragraphs |
|---|---:|---:|---:|
| 0 | 43,871 | 10,219 | 102 |
| 1 | 55,679 | 12,908 | 129 |
| 2 | 56,244 | 13,068 | 131 |
| 3 | 65,365 | 15,202 | 152 |
| 4 | 70,825 | 16,552 | 166 |

Winner: seed 4, **70,825 letters, 16,552 words, 166 paragraphs**, using 13,245 distinct phrases.

Independent standard-library verification checked the reverse equality,
phrase-to-text identity, uniqueness of normalized phrases, adjacent repeats,
content-word cap, preserved opening/ending, and delivered-file SHA-256.

Validation: 27 focused tests passed. A missing NumPy dependency initially
blocked the existing graph tests; installing it in the local environment
resolved that failure. The new checksum assertion also passed.

This is the longest result found in these runs, not a proven maximum, a
world-record claim, or a result on the coherent-prose north star.

## Publication

Published at https://palindrome.ericspencer.us/panama on 5 September 2026.
Cloudflare Pages deployment: `f66df4a2.palindrome-3u1.pages.dev`.

The static reading page includes all 166 paragraphs, the original download,
a highlighted centre letter, and a browser-side letter comparison control.
`tools/publication/build_panama.py` builds the HTML from the verified artifact;
styles and interaction live in `web/public/panama/`.

The production build passed. Live desktop (1440px) and phone (390px) browser
checks confirmed 166 paragraphs, no horizontal overflow, working letter
inspection and centre navigation, and no JavaScript errors. The downloaded
production text is byte-identical to the original. The generator homepage
still responds. Screenshots were visually inspected at both viewport sizes.

The page compares the result with Hoey's 540-word version (about 30.7x), while
crediting Norvig's longer 21,012-word version. It makes no world-record claim.

### Plain HTML redesign

At the user's request, `/panama` now ships as one self-contained `index.html`
with browser-default serif typography and ordinary links. The small stylesheet
and letter inspector are inline; the exact text download is embedded as a data
URL. The former separate CSS, JS, and TXT publication files were removed.
The source artifact remains intact in `artifacts/norvig-long/`.

Deployment: `5519e97a.palindrome-3u1.pages.dev`. Verified desktop and mobile,
including direct `file://` use: zero network requests, working inspector and
centre link, and a byte-identical downloaded artifact. The build passed.
