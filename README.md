# Palindrome Sentence Generator

Generates long, multi-sentence character-level palindromes: text that reads
identically forwards and backwards once case, spaces, and punctuation are
stripped. A dictionary search enforces the palindrome constraint; a language
model chooses among the branches that satisfy it.

```
$ python -m llm_palindrome.generate --min-letters 200 --seeds 24

Xi from life with girl law one. Most egan at eyes in go certified. Is nine
popular referred role to how. Type else my sat animal ha as. Air on a lot
in it fit is. So gone last as it in if. Trap arts ally as no oh no. Say
llas trap art fin if it. Sal tsa lengo si fit in it. Ol a no ri ash ala
mina stay melse. Pyt wohot el orde referral. Up open inside if it recognise
yet. An a get some now all right. I we film or fix.

[letters=201 sentences=14 lm_score=-2.382]
```

## How it works

Word-by-word generation cannot enforce a palindrome, and neither can an LLM
generating left to right: the last character is decided by the first. So the
text is built from both ends inward.

At every step one half owes the other a run of letters — the **overhang**. A
word added on the left must match the overhang forwards; a word added on the
right must match it reversed. The palindrome closes when the overhang is
itself a palindrome, and that becomes the center.

```
left:  "rats live on"          overhang: "no"   (right owes "on" reversed)
right:            "no evil star"
                        ^ closes when the overhang reads the same both ways
```

This overhang search is **Peter Norvig's palindrome algorithm**, building on
Dan Hoey's 1984 program — see Credit below. What this repository adds is the
scoring: at each step the
search faces hundreds of letter-valid continuations, nearly all of them
gibberish. A language model ranks them, so the search follows the branches
that read as English.

Two ways to apply the model, both implemented:

- **Rerank** (default): search with word-frequency scoring, then score the
  finished palindromes with GPT-2 and keep the most fluent.
- **In-loop** (`--lm-in-loop`): GPT-2 rescores the beam *during* the search, so
  unfluent branches die before they consume the budget.

## Results

Matched budgets, 12 seeds, min 200 letters, beam 60, GPT-2 small on an M-series
Mac. `lm_score` is mean token log-probability per letter (higher is better).

| Configuration | Best score | Mean score | Letters | Time | Valid |
|---|---|---|---|---|---|
| Zipf + GPT-2 rerank | −2.382 | −2.455 | 201 | 5.5 s | 12/12 |
| Zipf + GPT-2 in-loop | **−2.297** | **−2.405** | 216 | 170 s | 12/12 |

In-loop scoring produces measurably more fluent text at roughly 31× the cost.
Every candidate in both arms is a valid palindrome — validity comes from the
search, not the model, so the model can only affect readability.

Reproduce with:

```bash
python benchmark.py --seeds 12 --min-letters 200 --beam 60
```

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m llm_palindrome.generate --min-letters 200 --seeds 24
```

| Option | Description | Default |
|---|---|---|
| `--min-letters` | Minimum palindrome length in letters | 120 |
| `--seeds` | Independent search runs to rank against each other | 24 |
| `--beam` | Beam width per search | 60 |
| `--vocab` | Size of the frequency-ranked vocabulary | 30000 |
| `--model` | Hugging Face model for scoring (`''` disables) | `gpt2` |
| `--lm-in-loop` | Prune the beam with the LM during search | off |
| `--words-per-sentence` | Sentence length in the formatted output | 7 |
| `--out` | Write the result to a file | none |

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers overhang matching, trie candidate generation, end-to-end
search validity, and the formatting invariant that punctuation and casing
never alter the normalized letters.

## Project structure

```
llm_palindrome/
├── search.py       # overhang matching, tries, beam search
├── scoring.py      # frequency-based scorer
├── lm_scoring.py   # GPT-2 fluency scoring (batched)
├── textify.py      # sentence formatting, letter-preserving
├── validator.py    # normalize / is_palindrome
└── generate.py     # CLI
benchmark.py        # configuration comparison
tests/              # pytest suite
```

## Known limits

Output is locally fluent — phrases parse, sentences scan — but not globally
coherent; it does not hold a topic across its full length. The bottleneck is
the vocabulary's letter statistics: past a few hundred letters, the overhang
tends toward runs no English word can absorb, and the search leans on rare
short words to escape. Larger scoring models and a corpus-derived phrase
inventory are the next things to try.

## Credit

The two-sided overhang search is **Peter Norvig's**, described in
[World's Longest Palindrome?](https://norvig.com/palindrome.html) and in detail
in [The Algorithm](https://norvig.com/pal-alg.html) (2002). It builds on **Dan
Hoey's** 1984 program, which produced the first long "A man, a plan, a canal"
palindrome by the same remainder-matching idea. This project contributes the
bidirectional-bigram and language-model scoring layer on top of that search.

An earlier version of this file credited John Tromp for the algorithm. That was
wrong — he has no connection to this work, and the URL it cited did not exist.

Developed within the **AI4FM group**.

## License

MIT
