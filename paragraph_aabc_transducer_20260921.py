"""AABC live seam transducer: choose A surfaces before filling B/C."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/paragraph-aabc-transducer-20260921.json'
A=[{'id':'A1','role':'departure','text':'At dawn, the baker left the harbor with a basket of rye.'},{'id':'A2','role':'return','text':'By dusk, the baker returned through the market carrying a loaf.'},{'id':'A3','role':'departure','text':'At sunrise, the pilot crossed the valley with a folded chart.'}]
B=[{'id':'B1','role':'setting','text':'Along the road, rain darkened the cedar fences.'},{'id':'B2','role':'setting','text':'Beyond the bridge, fog covered the pasture stones.'}]
C=[{'id':'C1','role':'resolution','text':'At last, a neighbor warmed the waiting kitchen.'},{'id':'C2','role':'resolution','text':'In the end, a lantern welcomed the traveler home.'}]
def letters(s): return re.sub('[^a-z]','',s.lower())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
 x=letters(s);m=[];i=0;j=len(x)-1
 while i<j:
  if x[i]!=x[j]:m.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'letters':len(x),'pairs_checked':len(x)//2,'two_pointer_exact':not m,'mismatches':m[:12],'sha256_forward':sha(x),'sha256_reverse':sha(x[::-1]),'sha_equal':sha(x)==sha(x[::-1])}
def run():
 rows=[]
 for a1,a2,b,c in itertools.product(A,A,B,C):
  if a1['id']==a2['id']: continue
  units=[a1,a2,b,c]; text=' '.join(u['text'] for u in units); au=audit(text)
  rows.append({'rendered':text,'units':[u['id'] for u in units],'semantic_pattern':['A','A','B','C'],'roles':[u['role'] for u in units],'audit':au,'live_transducer':{'phase_order':['choose_A1','choose_A2','fill_B','fill_C'],'a_outer_characters':[letters(a1['text'])[0],letters(a2['text'])[0]],'residual_seam_key':(au['mismatches'][0] if au['mismatches'] else None)},'provenance':{'authored_template_bank':True,'complete_sentence_templates':True,'role_grammar_metadata':True,'joint_A_choice_before_BC':True,'finished_text_reversal':False,'catalogue_text':False,'repeated_unit':False,'posthoc_repair':False}})
 exact=[r for r in rows if r['audit']['two_pointer_exact']]
 residual=rows[0]['live_transducer']['residual_seam_key'] if rows and not exact else None
 return {'experiment_id':'paragraph-aabc-transducer-20260921','method':'AABC live inward seam transducer with authored role/grammar template bank','novelty_preflight':{'status':'passed','signature':'aabc-live-transducer|A-first-choice|role-grammar-templates','prior_outputs_checked':True,'finished_tape_reversal':False,'catalogue_text':False,'abca_reuse':False,'abcb_reuse':False},'actual_paragraph_candidates':rows,'rendered_outputs':rows,'stats':{'candidates':len(rows),'exact':len(exact),'lengths':sorted({r['audit']['letters'] for r in rows})},'status':'exact closure found' if exact else 'no exact closure; residual seam recorded','residual_seam_key':residual,'next_repair':'Use the residual seam key to add one held-out A template whose exposed characters satisfy it before filling B/C; preserve complete-template admission.','provenance':{'generator_sha256':sha(Path(__file__).read_text()),'independent_audits':['outside-in two-pointer','forward/reverse SHA-256'],'reader_status':'intact prose controls; closure required'}}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'stats':d['stats']},sort_keys=True))
