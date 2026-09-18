"""Balanced relative clauses with coordinated determiner changes."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='balanced-clause-determiner-pair-20260918'
ROWS=[('the','the','chart','ledger','that guides the crew','that marks the route','beside the inlet'),('a','a','map','journal','that charts the shore','that charts the shore','near the harbor'),('the','the','plan','book','that guards the pier','that guards the pier','along the coast'),('','', 'stars','paths','that name the sky','that name the sky','by the garden wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(dl,dr,lo,ro,left,right,boundary) in enumerate(ROWS):
  text=f'At dawn, {dl} keeper marks {dl+" " if dl else ""}{lo} {left} {boundary}; {dr+" " if dr else ""}sailor reads {dr+" " if dr else ""}{ro} {right} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'determiners':{'left':dl or 'bare','right':dr or 'bare','coordinated':True},'balance':{'left_letters':len(letters(left)),'right_letters':len(letters(right)),'equal':len(letters(left))==len(letters(right))},'audit':audit(text),'provenance':{'balanced_clause_determiner_pair':True,'attachment_role':'location','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'balanced clause determiner plus agreement-aware verb pair','reason':'coordinated determiners preserve balanced attachment but do not close the tape; next couple the determiner changes to agreement-aware verbs','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
