# Forward lexicalized CSP remote run

Run the bounded Brown inventory on the mini-agent:

```sh
python forward_lexicalized_grammar_20260920.py --brown --limit 5000 --max-nodes 20000
```

Artifacts report `SAT`, `UNSAT`, or `timeout` separately from solver node
count. Entries are deterministic DET/NOUN/VERB/PROPN rows from Brown.
