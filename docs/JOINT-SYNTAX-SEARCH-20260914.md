# Joint typed syntax/character search

This run makes the verb/noun/adjective state space part of the exact search,
not a readability score applied after a free character search.  Every center-
out state carries:

- a live typed-plan frontier (`VT`, `NOUN`, `PERSON`, `THING`, `ADJ`, and
  function-word roles);
- the exact character residual owed by the opposite edge; and
- a content-word distinctness guard.

Brown bigrams are a soft child-ordering prior only.  They never admit a state,
and no Brown sentence is copied.  A separate normalization and admission pass
audits each closure.

## Reproducible evaluation

Command:

```text
python3 experiments/joint_syntax_palindrome_search_20260914.py \
  --out runs/joint_syntax_palindrome_search_20260914_v1.json \
  --seeds 64 --beam 800 --candidate-limit 700 --max-steps 160
```

The run emitted 47 exact typed closures, all repetitions of the existing
38-letter control.  It emitted no novel closure in the requested 39–140-letter
band and therefore produced zero mechanically eligible new items.

Rendered control (not a new result):

> An aide rips nine memos; some men inspire Diana.

Independent letter tape:

```text
anaideripsninememossomemeninspirediana
```

The tape is 38 letters and equals its reverse.  The independent audit also
passes lexical, distinct-content, catalogue, and anti-shortcut checks; the
only failed mechanical field is the requested minimum length.  It has not been
sent to readers in this run because it is the known control, not a new output.

## What failed and what changes next

The failure is search coverage, not evidence that readable long palindromes are
impossible.  The next constructive operator is typed-plan expansion at the
deepest replayed residual: add a new complete verb–argument/adjective plan,
then replay the same residual with the syntax frontier still hard-constrained.
Relaxing to free tape scoring is explicitly out of scope.

Only a novel closure of at least 39 letters that passes the independent
mechanical gate can enter the next reader-facing test: randomized blinded
intact prose versus shuffled controls, with the reproducible rater package.
Programmatic scores in this report do not certify readability.
