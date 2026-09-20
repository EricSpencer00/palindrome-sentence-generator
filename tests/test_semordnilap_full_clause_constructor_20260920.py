from experiments.semordnilap_full_clause_constructor_20260920 import audit, consume, run, self_palindromic
def test_full_clause_controls_and_index():
 d=run(1000); assert d["control_count"]>=20 and d["reverse_index_keys"]>0
def test_cross_word_equation_and_rejection():
 assert consume("abcd","abc")==("d","") and consume("abcd","abx") is None and self_palindromic("level")
def test_exact_rows_have_independent_audit():
 d=run(12000)
 for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
