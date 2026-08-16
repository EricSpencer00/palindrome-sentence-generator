# North star

**A paragraph of coherent English prose, at least 100 words, whose letters read
identically forwards and backwards, built from sentences that are not
themselves palindromes and were not written by somebody else.**

That is the whole goal. Everything below exists to stop it being quietly
replaced by something easier.

## Why it is stated this precisely

Two versions exist, and each has half of it.

**v1** — `llm_palindrome.generate`. Structurally exactly right: one search
closes one mirror across the whole text, no unit is a palindrome on its own,
nothing is quoted. It reads as gibberish.

> Xi from life with girl law one. Most egan at eyes in go certified. Is nine
> popular referred role to how.

**v2** — `/api/v2/paragraph`. Reads as English, and gets there by an
uninteresting shortcut: it nests mirror-pairs harvested from catalogued
palindromes. The mirror is real, but the material is borrowed and the units are
fragments.

> Swap god for a janitor. Go hang a salami. Lived on decaf. … Faced no devil.
> Ima lasagna hog. Rot in ajar of dog paws.

**v3 is both at once.** v1's structure with v2's readability, and neither one
purchased by giving up the other.

## Acceptance criteria

All nine. Not a scorecard to average — a conjunction. Failing one means not
done, whatever the others say.

| # | criterion | how it is checked |
|---|-----------|-------------------|
| 1 | ≥ 100 words | count |
| 2 | the whole text is a letter-level palindrome | strip to letters, compare to its reverse — not `is_palindrome`, which is ours |
| 3 | at most **one** sentence is a palindrome on its own | the centre may be; nothing else |
| 4 | no sentence repeats | set size equals list length |
| 5 | the halves share no sentence | first-half set ∩ second-half set is empty |
| 6 | every sentence is grammatical English | blind judging, salad controls |
| 7 | it has a subject a reader can name unprompted | blind judging |
| 8 | judged as coherent prose, not as "sentences that read" | blind, against consecutive prose from one document AND disparate real sentences — must beat both |
| 9 | novel | `is_novel_palindrome` on the whole text and on every unit |

## Forbidden shortcuts

Each of these was actually taken in this repository. They are listed with the
iteration that took them so the pattern is recognisable, not so it is
apologised for.

**Word-order palindromes** (iteration 49). The sentence sequence mirrors, the
letters do not. Pays nothing per letter. I flagged it as "a much easier
constraint" in the same paragraph that adopted it, and it was the endpoint's
default for about forty iterations.

**Self-palindromic units** (iteration 105–121). A mirrored sequence of
sentences that are each already palindromes. Passes `is_palindrome`; reverse it
and every sentence returns unchanged, so the mirror does no work. Criterion 3
exists for this one.

**Verbatim repetition.** Any construction whose second half is the first half
again, in any order. Criteria 4 and 5.

**Optimising a proxy and reporting it as quality** (iterations 96–97). GPT-2
score improved 0.58 under guards and was invisible to a blind judge. Cohesion
1.200 beat 1.095 on the metric and lost on the judge. Four proxies have now
disagreed with blind judging; none has ever agreed on ranking. Proxies may
filter and may propose. They may not decide.

**Borrowed material presented as generated** (current). Every readable unit in
v2 is a catalogued palindrome. The assembly is ours; the sentences are the
record's. Criterion 9.

**Preferring the version that scores better by not attempting the
constraint.** The refrain beat the pair construction under blind judging, and I
shipped the refrain. It read better *because* it wasn't doing the hard thing.
A measurement that rewards the shortcut is not a licence to take it.

## Where v2 stands against the criteria

Honestly, so the gap is the thing being worked on rather than argued about.

| # | status |
|---|--------|
| 1 | **fail** — 91 words |
| 2 | pass — 237 letters, verified independently |
| 3 | pass — one, the centre |
| 4 | pass |
| 5 | pass |
| 6 | **fail** — "A slut nixes", "Eva can is tab", "Never a foot" are fragments |
| 7 | **fail** — no subject |
| 8 | **fail** — ranks below both controls |
| 9 | **fail** — every unit is catalogued |

Four of nine. The four that pass are the structural ones, which are free once
the assembly is right; the five that fail are all about the material.

## What the constraint actually costs

Not a reason to stop — a reason not to expect a shortcut to work.

Reversing English letters and re-segmenting into the vocabulary costs **3.296
bits per free letter**, against forward English's 1.63. Every letter is placed
twice and both placements must be English, so the coherent feasible set thins
by roughly 10× every three letters.

Measured consequences, all of which any v3 approach has to survive:

- Mining 272k attested bigrams yields 3,894 mirror-pairs; **131** have both
  halves attested. At three-word halves, 27. At four-word halves — where prose
  would start — **0**, from 34,688 attested 4-grams.
- The 40–60 letter band was searched: 880k closures, 112,523 of the first
  300,000 using only dictionary words, **none** within three unattested joins
  of reading.
- Thematic selection over two-word halves is impossible: of every content-word
  pair across the 26 usable halves, **2** co-occur anywhere in 3,932 sentences.
- 29 pairs exist whose two halves both read. All 29 are catalogued.

The one thing that has moved: whole self-palindromic sentences pay the same
3.296 bits and do carry subjects. The cost forces short **units**, not short
**palindromes**. Any v3 route has to find units that are long enough to mean
something and are not already in the record.

## Why the unit is a mirror-pair

Three ways exist to pay the mirror across a paragraph, and two of them are
closed.

**Free-running.** One search, one palindrome, sentence boundaries wherever they
fall. This is v1, and it is the only structure with no repetition and no
independent units — which is why it stays the thing to beat. It is also the one
that has been measured and failed: the 40–60 letter band yielded 880k closures
and nothing within three unattested joins of reading, and a paragraph needs
about 500.

**Self-palindromic units.** Any concatenation of units that are each palindromes
reverses into *those same units in the opposite order*, so the whole is a
palindrome only when the unit sequence is itself a palindrome — unit k must be
unit n+1−k. That is not a stylistic preference for refrains; it is forced, and
it is forced whether the unit is one sentence or five. Criteria 4 and 5 close
this door, and the refrain is what stands behind it.

**Mirror-pairs.** The unit is two halves that spell each other backwards, placed
at mirrored positions. The sequence still mirrors and nothing repeats, because
what comes back at position n+1−k is the OTHER half. This is the only structure
that survives all nine criteria, which is why the work is on finding pairs
rather than on choosing between constructions.

The cost lands entirely on the material: both halves have to read, and pairs
where both halves read are the scarcest thing in this project. Where they come
from is `llm_palindrome/pairs.py` — a walk over the vocabulary, constrained so
every join inside a half is one English has been seen to make.

## Falsifying this

`tests/test_north_star.py` checks criteria 1–5 and 9 mechanically against the
shipped endpoint, and is expected to fail until v3 exists. It is a target in
the suite rather than a paragraph in a document, because a paragraph in a
document is how the last three shortcuts got taken.

Criteria 6–8 need a blinded batch with real-prose and salad controls. They
cannot be automated and must not be replaced by a proxy that can.
