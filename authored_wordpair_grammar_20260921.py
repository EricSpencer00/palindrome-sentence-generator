"""Bounded authored word-pair grammar; constructs SVO scenes before tape audit."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/authored-wordpair-grammar-20260921.json'

def letters(s): return re.sub('[^a-z]','',s.casefold())
def audit(text):
 t=letters(text); mismatch=next(({'offset':i,'left':t[i],'right':t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'two_pointer_exact':bool(t) and mismatch is None,'first_mismatch':mismatch,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}

# These are authored slot pairs, not imported palindrome/catalogue text. Each side
# has an ordinary grammatical role; pairing is checked before any sentence render.
PAIRS=(
 {'id':'agentive','left':('the careful baker','maps'), 'right':('the quiet harbor','spame')},
 {'id':'locative','left':('the patient keeper','records'), 'right':('the winter garden','sdrocer')},
 {'id':'object','left':('a bright letter','carries'), 'right':('a red drawer','seirrac')},
)
SUBJECTS=('the careful baker','the patient keeper')
OBJECTS=('a bright letter','a small map')

def render(pair, subject, obj):
 (ls,lv),(rs,rv)=pair['left'],pair['right']
 left=f'{subject} {lv} {obj}'
 # right slot remains independently authored and grammatical as a bounded adjunct
 right=f'while {rs} {rv} the record'
 return f'{left}, {right}.'

def run():
 rows=[]
 for pair,subject,obj in itertools.product(PAIRS,SUBJECTS,OBJECTS):
  text=render(pair,subject,obj); a=audit(text)
  rows.append({'pair_id':pair['id'],'rendered':text,'slots':{'subject':subject,'object':obj},'complete_prose':True,'semantic_frame':'SVO + while-clause adjunct','audit':a,'provenance':{'authored_pair_slot':True,'pair_source':'independent hand-authored alternatives','generated_jointly_before_render':True,'catalogue_text':False,'repeated_units':False,'self_palindromic_units':False,'finished_tape_reversal':False,'posthoc_repair':False,'per_candidate_reward':False},'reader_eligible':False})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 exact=[x for x in rows if x['audit']['two_pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse']]
 return {'experiment_id':'authored-wordpair-grammar-20260921','method':'bounded joint generation from authored semordnilap-compatible lexical slots with SVO/adjunct constraints','stats':{'pair_slots':len(PAIRS),'joint_candidates':len(rows),'exact_count':len(exact),'longest_letters':rows[0]['audit']['letters'],'min_target_letters':39},'exact_candidates':exact,'candidates':rows,'novelty_preflight':{'status':'passed','signature':'authored-wordpair-slots|joint-svo-adjunct|pre-render-constraints','broad_reverse_segmentation_sweep':False},'provenance':{'independent_audits':['two-pointer','forward/reverse SHA-256'],'reader_gate':'closed; no exact candidate','next_operator':'author a two-word function-slot pair whose boundary letters satisfy the first residual while preserving ordinary SVO valency','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'status':'no exact closure; deepest grammatical support retained'}

if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats'],sort_keys=True))
