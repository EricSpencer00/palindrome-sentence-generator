from llm_palindrome.bilateral_seams import Chunk,AllChunkCentersGrammar,OwedSideBilateralGrammar,screen_fixed_endpoints,norm

def test_conditional_requires_verb_even_when_fixed_edges_are_exact():
 left=Chunk('L','Some men, if editors ','START','NEEDS_VERB','main subject and conditional subject','022 diagnostic')
 right=Chunk('R','roti, define memos.','NEEDS_OBJECT','END','conditional object and main predicate','022 diagnostic')
 screen=screen_fixed_endpoints([left],[right]);assert screen['admitted'] and screen['debt']['closure_exact']
 tape=norm(left.text+right.text);assert tape==tape[::-1]
 # Exact but ungrammatical outer text cannot close the required verb slot.
 for cls in (AllChunkCentersGrammar,OwedSideBilateralGrammar):
  assert not cls([left,right]).search(max_steps=3,max_states=1000,seconds=2)['outputs']
 verb=Chunk('V','suss ','NEEDS_VERB','NEEDS_OBJECT','single whole transitive verb','022 diagnostic; semantics unapproved')
 for cls in (AllChunkCentersGrammar,OwedSideBilateralGrammar):
  outputs=cls([left,verb,right]).search(max_steps=3,max_states=1000,seconds=2)['outputs']
  assert {o['text'] for o in outputs}=={'Some men, if editors suss roti, define memos.'}
  assert outputs[0]['letters']==35
