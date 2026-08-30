# The whole thing in one paragraph

The version the paper's abstract is cut from, and the version to give someone
who asks what the work is.

---

Producing palindromes was solved in 1984 and producing readable ones was not,
and the gap is usually blamed on search quality. It is not the search. The
state that makes a palindrome checkable while you build it is the overhang, the
run of letters one side owes the other: it is a function of both ends rather
than of a prefix, and it is a sufficient statistic for feasibility, so two
half-built palindromes with completely different text and the same overhang
have identical sets of legal continuations. Posed over prefixes the constraint
admits no incremental check at all, which is why a left-to-right model cannot
enforce it; posed over overhangs it is an ordinary finite-state constraint and
the constrained-decoding literature applies. Over that state the space has
three regimes, and picking the wrong method for a length is the usual way to
waste a budget: below about 30 letters it is walkable end to end, and
exhaustive enumeration is the only route that returns novel readable sentences,
because best-of-everything is a different object from best-of-N over a beam's
proposals; above 30 the tree branches about as wide as the vocabulary, so a
time-budgeted walk returns a deep prefix of one corner and silently reports it
as coverage, which is why it needs an acceptance test — we use recall of the
catalogued palindromes, and it goes from 10 of 18 at 83 seeded words to 0 of 27
at 14,000; and at any length, units that pay the constraint internally nest
like brackets, which takes length out of the problem and puts unit supply in
its place. Of the three ways to nest, one is closed by algebra rather than by
taste: a concatenation of units that are each palindromes reverses into those
same units in the opposite order, so the sequence itself has to mirror and
every unit but the centre must appear twice, which is why long human
palindromic poetry is always a refrain. What actually bounds the output is the
material. Seven construction routes — beam search, exhaustive enumeration,
corpus mining, closed-form reversible chains, LLM authoring, authoring one half
and segmenting the other, and a sharded vocabulary walk — stop in the same
place, and the sharpest number is that of 34,688 attested English four-grams,
none has a mirror image that reads. The two levers everyone reaches for are
measured and both fail: best-of-N reranking is flat by N=24 and the winner sits
where order statistics say it should, so no future judge beats it on that pool;
and widening the beam quadruples the vocabulary in use while readability falls
monotonically. As for why, about half the letters of reversed English cannot be
placed inside a dictionary word at all — a model-free fact, stable across
lengths and segmentation strategies — and priced under a language model the
reversed reading costs 2.2 to 3.6 bits per free letter more than the forward
one, against the 1.4 to 1.9 bits per letter the same spans carry read normally.
That price is a range rather than a constant and it grows with model capacity,
because a bigger model finds forward English cheaper and finds no reading of
reversed English at all.
