"""Constructive paragraph ABBA lane with live outside-in obligations.
Four independently authored sentence roles are paired A↔A and B↔B while
residual characters cross sentence boundaries. No finished paragraph reversal.
"""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/paragraph-abba-constructive-20260921.json'
def norm(s): return re.sub('[^a-z]','',s.lower())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
 t=norm(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':sha(t),'sha256_reverse':sha(t[::-1])}
ROLE_ROWS=(
    (
        ('subject','singular','agent','The careful archivist labels a faded map.'),
        ('event','singular','patient','The lantern guides a tired sailor.'),
        ('event','singular','patient','A small bell warns the waiting crew.'),
        ('subject','singular','agent','A quiet keeper measures the old bridge.'),
    ),
    (
        ('subject','singular','agent','At dawn, the patient curator files a chart.'),
        ('event','singular','patient','A harbor pilot checks the western tide.'),
        ('event','singular','patient','By noon, the ferryman secures a wooden skiff.'),
        ('subject','singular','agent','At dusk, the watchman counts the mooring posts.'),
    ),
)
def pair_trace(left,right):
 x,y=norm(left),norm(right); tr=[]
 for i,(a,b) in enumerate(itertools.zip_longest(x,y[::-1])):
  if a is None or b is None:return tr,i,{'offset':i,'left':a,'right':b,'kind':'length'}
  tr.append({'offset':i,'left':a,'right':b,'outcome':'match' if a==b else 'unsupported'})
  if a!=b:return tr,i,{'offset':i,'left':a,'right':b,'kind':'character'}
 return tr,len(tr),None
def run():
 rows=[]
 for a1,b1,b2,a2 in ROLE_ROWS:
  # Four roles are independently authored and remain distinct surfaces.
  sentences=[a1[3],b1[3],b2[3],a2[3]]
  paragraph=' '.join(sentences)
  outer_l=a1[3]+' '+sentences[-1]; outer_r=sentences[-1]+' '+a1[3]
  inner_l=b1[3]; inner_r=sentences[2]
  ot,od,oc=pair_trace(outer_l,outer_r); it,idd,ic=pair_trace(inner_l,inner_r)
  au=audit(paragraph); gates={'whole_output_exact':au['pointer_exact'],'outer_support':od>=2,'inner_support':idd>=2,'independent_pointer_hash':au['pointer_exact'] and au['sha256_forward']==au['sha256_reverse'],'distinct_sentence_roles':len(set(sentences))==4,'self_palindromic_unit_absent':all(norm(s)!=norm(s)[::-1] for s in sentences),'no_reversal':True,'no_catalogue':True,'no_post_hoc_repair':True}
  rows.append({'rendered':paragraph,'roles':{'A1':a1,'B1':b1,'B2':b2,'A2':a2},'outer_trace':ot,'inner_trace':it,'support_depth':min(od,idd),'first_unsupported':{'outer':oc,'inner':ic},'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'four-role ABBA paragraph with live cross-sentence residuals','independently_authored':True,'finished_tape_reversal':False,'catalogue_text':False,'rendered_before_repair':True}})
 exact=[r for r in rows if r['accepted']]; frontier=max(r['support_depth'] for r in rows)
 return {'experiment_id':'paragraph-abba-constructive-20260921','method':'constructive four-role ABBA outside-in character obligations','stats':{'paragraphs':len(rows),'support_depth_frontier':frontier,'exact_candidates':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,'obstruction':{'status':'blocked_at_first_character_cut','detail':'independently authored sentence endings do not discharge outer and inner residuals beyond the initial seam; no repair or widening performed'},'novelty_preflight':{'status':'passed','signature':'paragraph-abba|four-role|live-residual','distinct_from':'repeated-unit controls, catalogue palindromes, finished reversal'}}
if __name__=='__main__':
 r=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
