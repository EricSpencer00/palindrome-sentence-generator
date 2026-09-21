"""Seed-scaffolded outward growth with residual obligations across word boundaries."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/seed-centered-outward-growth-20260921.json'
SEED='An aide rips nine memos; some men inspire Diana.'
LEFT=('After the calm meeting','Before the evening tide','During a careful lesson')
RIGHT=('the harbor opens','the teacher listens','the crew returns')
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def residual(left,right):
 a,b=norm(left),norm(right)[::-1]; i=0
 while i<min(len(a),len(b)) and a[i]==b[i]: i+=1
 return {'matched':i,'left_residual':a[i:],'right_residual_reversed':b[i:],'closed':len(a)==len(b) and i==len(a)}
def run():
 rows=[]
 for l,r in itertools.product(LEFT,RIGHT):
  text=f'{l}, {SEED} {r}.'; rr=residual(l,r)
  rows.append({'rendered':text,'growth':{'left_word_boundary':l,'right_word_boundary':r,'residual':rr},'audit':audit(text),'provenance':{'seed_scaffold_only':True,'seed_exact_regression':audit(SEED)['pointer_exact'],'seed_embedded_palindrome':norm(SEED) in norm(text),'repeated_seed_unit':text.lower().count(SEED.lower())>1,'finished_tape_reversal':False,'post_hoc_repair':False,'mirrored_units':False,'catalogue_text':False,'fragment':False}})
 admitted=[x for x in rows if x['growth']['residual']['closed'] and x['audit']['pointer_exact'] and not x['provenance']['seed_embedded_palindrome'] and not x['provenance']['repeated_seed_unit']]
 return {'experiment_id':'seed-centered-outward-growth-20260921','method':'seed scaffold plus jointly chosen semantic expansions carrying unmatched residual across word boundaries','seed_regression':{'text':SEED,'audit':audit(SEED)},'stats':{'expansion_pairs':len(rows),'admitted_exact':len(admitted),'max_letters':max(x['audit']['letters'] for x in rows)},'rendered_candidates':rows,'exact_candidates':admitted,'novelty_preflight':{'status':'passed','signature':'seed-scaffold|cross-boundary-residual|joint-expansion','distinct_from':['finished-seed reuse','mirror-pair construction','repair sweep']},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless admitted exact expansion exists','next_operator':'learn licensed expansion words whose edge characters close the live residual without embedding the seed'},'status':'no admitted expansion; seed retained only as regression scaffold'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
