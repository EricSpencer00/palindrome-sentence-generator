import itertools
from llm_palindrome.bilateral_seams import Chunk,screen_fixed_endpoints,debt,norm

def chunk(i,text,bindings=()):return Chunk(i,text,'A','B','fixed licensed outer phrase','test',bindings)

def test_mismatched_endpoints_cannot_be_repaired_interior():
 l=chunk('l','ab');r=chunk('r','x')
 assert not screen_fixed_endpoints([l],[r])['admitted']
 for length in range(5):
  for mid in itertools.product('abx',repeat=length):
   t=norm(l.text+''.join(mid)+r.text);assert t!=t[::-1]

def test_nonpalindromic_debt_can_be_completed():
 l=chunk('l','No, sir, a ');r=chunk('r','radar is on.')
 g=screen_fixed_endpoints([l],[r]);assert g['admitted'] and not g['closure_claim']
 assert g['debt']['letters']!=g['debt']['letters'][::-1]
 t=norm(l.text+'dark '+r.text);assert t==t[::-1]

def test_payoff_is_not_admission_gate_and_actor_binding_is():
 l=chunk('l','abc');r=[]
 # Free growth pays nothing yet is compatible and may later close.
 g=screen_fixed_endpoints([l],r,previous=debt('',''),extension_side='left',extension_text='abc')
 assert g['admitted'] and g['payoff_utility']==0
 l=chunk('l','a',(('agent','Eve'),));r=chunk('r','a',(('agent','Noel'),))
 assert screen_fixed_endpoints([l],[r])['reason']=='actor_conflict'
