from experiments.argonne_typed_overhang_growth_20261009 import (
    CENTERS, Clause, ContextSupport, Grammar, exact_audit, novelty_cost,
    scene_clauses, search_job,
)
from llm_palindrome.bigram import BigramModel


def test_grammar_enforces_roles_and_agreement_with_shared_fact_identity():
    clauses = scene_clauses()
    grammar = Grammar(clauses)
    active = grammar.parse(tuple("diana reads a memo".split()))
    passive = grammar.parse(tuple("a memo is read by diana".split()))
    assert active is not None and passive is not None
    assert clauses[active[0]].facts == clauses[passive[0]].facts
    assert grammar.parse(tuple("some men read a memo".split())) is not None
    assert grammar.parse(tuple("some men reads a memo".split())) is None
    assert grammar.parse(tuple("a memo reads diana".split())) is None
    assert grammar.parse(tuple("diana reads a memo diana files a report".split())) is not None


def test_entity_reuse_is_free_but_fact_and_frame_reuse_cost():
    read = Clause("diana reads a memo", (("diana", "read", "memo"),), "active")
    file = Clause("diana files a memo", (("diana", "file", "memo"),), "active")
    assert novelty_cost(file, read.facts, ()) == 0
    assert novelty_cost(read, read.facts, ()) > 0
    assert novelty_cost(file, read.facts, ("active",)) > 0


def test_context_repayment_respects_grammar_and_unknown_horizon():
    clauses = [Clause("ab c", (("a", "does", "b"),), "x"),
               Clause("cb a", (("b", "does", "a"),), "y")]
    support = ContextSupport(Grammar(clauses), Grammar(clauses, reverse=True))
    assert support.check("R", 0, "cba", 4) == "supported"
    assert support.check("R", 0, "zz", 4) == "impossible"
    assert support.check("R", 0, "zz", 0) == "unknown"


def test_toy_grammar_cross_word_debt_closes_independently_exact():
    # Artificial terminals test construction, not English meaning or quality.
    clauses = [Clause("ab c", (("a", "does", "b"),), "x"),
               Clause("cb a", (("b", "does", "a"),), "y")]
    model = BigramModel({}, {w: 1 for w in ("ab", "c", "cb", "a", "eve")})
    row = search_job("Eve.", "typed_context_debt", clauses, model,
                     seconds=1, seed=3, min_letters=9, max_letters=15, beam_width=24, max_steps=12)
    assert row["selected"] is not None
    assert row["selected"]["audit"]["exact"]
    assert row["selected"]["complete_clause_extensions"]
    assert row["selected"]["growth_letters"] > 0
    assert row["counters"]["complete_clause_closures_seen"] > 0


def test_seed_controls_are_exact_and_empty_is_not():
    assert all(exact_audit(text)["exact"] for text in CENTERS.values())
    assert {exact_audit(text)["letters"] for text in CENTERS.values()} == {38, 54}
    assert not exact_audit("!?")["exact"]


def test_last_step_closure_is_saved_before_another_expansion():
    clauses = [Clause("a", (("a", "does", "a"),), "toy")]
    model = BigramModel({}, {"a": 1, "eve": 1})
    # Two additions suffice: a / Eve / a. The second step is the last one.
    row = search_job("Eve.", "lexical_debt", clauses, model,
                     seconds=1, seed=0, min_letters=5, max_letters=5,
                     beam_width=4, max_steps=2)
    assert row["selected"] is not None
    assert row["selected"]["letters"] == 5
    assert row["selected"]["audit"]["exact"]


def test_typed_rejects_same_fact_even_when_last_clauses_have_not_transitioned():
    clauses = [Clause("a", (("a", "does", "a"),), "toy")]
    model = BigramModel({}, {"a": 1, "eve": 1})
    row = search_job("Eve.", "typed_context_debt", clauses, model,
                     seconds=1, min_letters=5, max_letters=5, beam_width=4, max_steps=2)
    assert row["selected"] is None
    assert row["counters"]["repeated_fact_closure_rejections"] > 0


def test_reused_predicate_requires_distinct_fact_bindings_and_paraphrase_is_banned():
    from experiments.argonne_typed_overhang_growth_20261009 import predicate_reuse_audit, repeats_fact
    diana = Clause("diana reads a memo", (("diana", "read", "memo"),), "active")
    ada = Clause("ada reads a memo", (("ada", "read", "memo"),), "active")
    passive = Clause("a memo is read by diana", diana.facts, "passive")
    assert not repeats_fact(ada, diana.facts)
    assert repeats_fact(passive, diana.facts)
    audit, = predicate_reuse_audit([diana, ada])
    assert audit["predicate_span"] == "reads a memo" and audit["licensed_distinct_use"]
    audit, = predicate_reuse_audit([diana, diana])
    assert not audit["licensed_distinct_use"]


def test_inherited_seed_is_not_silently_licensed_by_good_extensions():
    from experiments.argonne_typed_overhang_growth_20261009 import State, describe
    grammar = Grammar(scene_clauses())
    clear = describe(State(), CENTERS["38-control"], grammar)
    unusual = describe(State(), CENTERS["54-np-candidate"], grammar)
    assert clear["seed_grammar_licensed"] and clear["whole_text_grammar_licensed"]
    assert unusual["complete_clause_extensions"]  # Empty extensions add no errors.
    assert not unusual["seed_grammar_licensed"]
    assert not unusual["whole_text_grammar_licensed"]
