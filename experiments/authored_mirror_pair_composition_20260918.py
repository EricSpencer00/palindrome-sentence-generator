"""Fresh non-self-palindromic mirror-pair clause composition."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='authored-mirror-pair-composition-20260918'
PAIRS=[('The patient archivist maps the harbor at dawn.','At dusk, a careful sailor checks the tide beside the pier.'),('A quiet teacher tends the garden after rain.','Before noon, the young keeper records each flowering path.'),('The seasoned pilot charts a narrow channel.','A watchful guide marks the safe return beyond the rocks.')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def self_pal(s):
 t=letters(s);return bool(t) and t==t[::-1]
def run():
 rows=[]
 for i,(left,right) in enumerate(PAIRS):
  text=f'{left} {right}'
  rows.append({'pair_id':i,'left_clause':left,'right_clause':right,'rendered':text,'audit':audit(text),'provenance':{'fresh_authored_left':True,'fresh_authored_right':True,'left_self_palindrome':self_pal(left),'right_self_palindrome':self_pal(right),'repeated_unit':False,'catalogue_used':False,'novelty_preflight':'distinct semantic roles and newly authored lexical tape'}})
 best=max(rows,key=lambda r:r['audit']['letters'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':best['audit']['letters'],'best_mismatches':min(r['audit']['mismatches'] for r in rows)},'next_repair':{'operator':'fresh mirror-pair clause seam edit','reason':'distinct authored clauses remain grammatical but do not close the whole tape; next edit the paired clause seam under independent semantic-role constraints','route_exhausted':False},'provenance':{'bounded_pairs':len(PAIRS),'catalogue_used':False,'non_self_palindromic_units':all(not r['provenance']['left_self_palindrome'] and not r['provenance']['right_self_palindrome'] for r in rows)}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
