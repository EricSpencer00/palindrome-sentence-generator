from experiments.argonne_free_typed_clauses_20261009 import ClauseFrontier, connected, run_job
from experiments.argonne_typed_overhang_growth_20261009 import Clause, Grammar
from llm_palindrome.bigram import BigramModel


def test_nfa_retains_longer_clause_and_end_reset_and_rejects_repeat():
    a = Clause("a", (("x", "does", "y"),), "short")
    ab = Clause("a b", (("x", "sees", "z"),), "long")
    b = Clause("b", (("y", "does", "z"),), "other")
    frontier = ClauseFrontier(Grammar([a, ab, b]))
    states = frontier.states(("a", "b"))
    assert len(states) == 2  # long clause, or a then b
    assert not frontier.states(("a", "a", "a"))


def test_fact_graph_distinguishes_entity_reuse_from_disconnected_events():
    assert connected([("ada", "read", "memo"), ("ada", "file", "memo")])
    assert not connected([("ada", "read", "memo"), ("eve", "file", "report")])
    assert not connected([])


def test_root_free_child_closure_saved_at_final_step_and_repeats_rejected():
    clauses = [Clause("ab", (("x", "does", "y"),), "one"),
               Clause("ba", (("y", "helps", "z"),), "two")]
    model = BigramModel({}, {"ab": 1, "ba": 1})
    result = run_job(clauses, model, min_letters=4, seconds=1, beam_width=8, max_letters=4, max_steps=2)
    assert result["longest_exact"]["audit"]["exact"]
    assert result["counters"]["unique_admitted_exact_closures"] == 2
    assert result["candidates"][0]["meaning_evidence"] == "not_evaluated"
    same = [Clause("ab", (("x", "does", "y"),), "one"),
            Clause("ba", (("x", "does", "y"),), "two")]
    rejected = run_job(same, model, min_letters=4, seconds=1, beam_width=8, max_letters=4, max_steps=2)
    assert rejected["longest_exact"] is None
    assert rejected["counters"]["closure_repeated_fact"] == 2


def test_question_then_statement_is_distinct_use_not_a_meaning_claim():
    facts = (("x", "does", "y"),)
    clauses = [Clause("ab", facts, "question", "question"), Clause("ba", facts, "assertion")]
    model = BigramModel({}, {"ab": 1, "ba": 1})
    row = run_job(clauses, model, min_letters=4, seconds=1, beam_width=8, max_letters=4, max_steps=2)
    assert row["longest_exact"] is not None
    assert {mode for mode, fact in row["longest_exact"]["canonical_fact_uses"]} == {"question", "statement"}
    assert row["longest_exact"]["meaning_evidence"] == "not_evaluated"
