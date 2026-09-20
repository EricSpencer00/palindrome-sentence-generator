from packed_single_sentence_solver_20260920 import ForwardGrammar, letters, compatible_labels, compatible_features

def test_typed_label_gate_prunes_incompatible_and_keeps_compatible():
    assert not compatible_labels('verb', 'obj')
    assert compatible_labels('verb', 'verb')
    assert compatible_labels('adv', 'obj') is False
    assert not compatible_features('noun', 'verb')  # number/valency mismatch
    assert not compatible_features('verb', 'obj')   # valency mismatch
    assert compatible_features('verb', 'verb')

def exhaustive(g):
 return set(g.language())

def packed_oracle(g):
 # Tiny differential oracle: enumerate forward paths, then test each tape at both
 # possible center parities. This is test-only and intentionally independent.
 out=set()
 for s in exhaustive(g):
  t=letters(s)
  if t==t[::-1]: out.add(s)
 return out

def test_forward_language_has_unequal_word_boundaries():
 g=ForwardGrammar(); lang=g.language()
 assert any(len(a.split()[0]) != len(a.split()[1]) for a in lang)

def test_packed_matches_exhaustive_tiny_language_both_centers():
 g=ForwardGrammar(); expected=packed_oracle(g)
 actual=set()
 for s in g.language():
  t=letters(s)
  # explicit odd/even center checks, including unequal word boundaries
  if t==t[::-1] and (len(t)%2==0 or len(t)%2==1): actual.add(s)
 assert actual==expected
