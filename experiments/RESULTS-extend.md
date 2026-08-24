# Growing palindromes from the outside in

Run 23 August 2026. `experiments/extend_ops.py`, judged via
`runs/punct/make_grow_blind.py` and `runs/punct/score_grow.py`.

## The idea, which is the account holder's

Shown `non academia aimed a canon` (22 letters), they wrote by hand:

    we racecar non academia anna aimed a canon racecar ew        44 letters
    we racecared non academia anna aimed a canon de racecar ew   48 letters

Both verify. Twice the length in one step, and no search involved. Every
operation preserves the mirror by construction:

    WRAP-PAIR   `w` at the front, `reverse(w)` at the back, both words
    WRAP-SELF   the same self-palindromic word at both ends
    CENTRE      a self-palindromic word at the exact midpoint
    GLUE        a multi-word wrap whose two sides mirror in LETTERS while
                their word boundaries differ: `racecared` against `de racecar`

## Why these work where chunking did not

Placing a chunk one-sidedly advances the mirror by 1.09 letters whatever its
size, because it leaves its own length behind as debt — measured over 60,000
placements in `RESULTS-chunking`. These operations place a unit **and its
mirror in the same move**, so the debt one creates the other discharges. Net
debt zero, progress the full length on both sides.

The conservation law is therefore not a wall. It is a statement about
one-sided placement, and these operations are two-sided.

## It works, mechanically

750 verified seeds — the catalogue plus the Polaris output. With GLUE enabled
(4,656 mined mirror-pairs, 2,492 of them with different word counts on the two
sides):

**750 of 750 grew. Mean gain +36.1 letters. Longest 87.**

    87  Man are was. Am, a time on. Doc note I. Dissent a fast: never
        prevents a. Fatness, I, diet on cod, no emit a. Ma saw era. Nam.

Against the search's best of 36 letters on 32 Polaris cores, in a fraction of
the compute.

## And it makes the text worse, unanimously

Twenty pairs of a seed against its own grown version, both written out by the
same punctuator so presentation is not the variable, position balanced, two
independent annotators with no access to the key or the hypothesis, the second
shown every pair with the sides exchanged. Both were told explicitly to ignore
length, since the grown text is always longer and that is the confound most
likely to manufacture a result.

| | calibration (n=6) | grown vs seed (n=20) |
|---|---|---|
| judge 1 | 6/6 | **grown preferred 0/20** |
| judge 2 | 6/6 | **grown preferred 0/20** |
| chose the left passage | 3/6 each | 10/20 each |
| agreement on the text | 6/6 | **20/20** |
| Cohen's kappa | +1.000 | **+1.000** |

Perfect on calibration, no position bias, and complete agreement — so the
protocol has full power and the judges are reading the text rather than the
slot. In that condition they chose the **seed** on all twenty items.

Seed preferred 20 of 20, p is about 1e-6.

This is the opposite shape from the punctuation null, and worth contrasting.
There, both judges defaulted to position on every item and agreed on the text
zero times, which is what no signal looks like. Here they agree on the text
twenty times out of twenty and always in the same direction, which is what a
large real effect looks like — pointing the wrong way.

## What is actually established

The operations **produce valid longer palindromes** and do it reliably. They do
**not** produce better ones. Automated wrap selection is a length hack.

The failure is visible in the output. `A, draw delivered der. Evil edward a.`
becomes `But are not. Was am no: a draw delivered. Did der evil, edward a on,
ma saw ton. Era, tub.` — forty letters longer, and the additions are `but are
not was am no` and `on ma saw ton era tub`, which are filler at both ends. The
seed said little; the grown version says the same little with a preamble and a
trailing mutter.

## What is NOT established

**That the operations are hopeless in a human's hands.** The account holder's
own `we racecar non academia anna aimed a canon racecar ew` was chosen by
reading it. My selection rule is an edge score over Brown tag shapes, which is
the same class of proxy that has now failed five times in this repository. The
negative result is about automated selection, not about the operations.

Distinguishing those needs the hand-made examples judged against their seeds by
the same protocol, which is n=2 and therefore suggestive at best. A better test
is a batch where a person picks the wraps.

**That growth cannot help a text that has somewhere to go.** Every seed here is
a short palindrome that already says nearly nothing. Wrapping something with no
subject cannot give it one, and the north-star criterion these all fail is
about subject, not length.

## The first version of this experiment, recorded because it was wrong

Greedy first-fit reported 87 letters and mean gain +38.7, better than the final
numbers. It got there by stacking `and`/`dna` six deep:

    And, and and and and: and doc note, I dissent a. Fast, never prevents a.
    Fatness, I, diet on cod, dna, dna, dna, dna, dna, dna.

Optimising length and reporting length. Requiring each wrap to keep the edges
parsing, and forbidding repeated glue words, dropped the headline to 75 and
made the text readable — and the blind judging then said even that is worse
than not growing at all.
