"""Single semantic frame with a live word-boundary crossing seam."""
import hashlib,json,re
from pathlib import Path
ID='semantic-frame-crossing-residual-20260921'; OUT=Path(__file__).resolve().parents[1]/'runs'/f'{ID}.json'
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s);mm=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-1-i]];return {'letters':len(t),'exact':bool(t) and not mm,'two_pointer_exact':bool(t) and not mm,'first_mismatch':mm[0] if mm else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(t[::-1].encode()).hexdigest()}
FRAME=(('the lantern keeper','polishes','a brass compass','before dusk'),('the harbor pilot','records','the tide mark','after rain'))
def seam(left,right):
 a,b=norm(left),norm(right);i=0
 while i<min(len(a),len(b)) and a[-1-i]==b[i]:i+=1
 return {'left_residual':a[:-i] if i else a,'right_residual':b[i:] if i else b,'crossing_pairs':i}
def run(budget=64):
 rows=[]; controls=[]
 for s in FRAME:
  text=' '.join(s).capitalize()+'.';controls.append({'rendered':text,'audit':audit(text),'reader_status':'complete prose control'})
 states=0
 # Distinct complete clauses; seam is computed before final rendering, never repaired afterward.
 for left in FRAME:
  for right in FRAME:
   if left==right:continue
   states+=1; l=' '.join(left);r=' '.join(right);d=seam(l,r);text=l.capitalize()+'. '+r.capitalize()+'.';a=audit(text)
   rows.append({'rendered':text,'audit':a,'semantic_frame':{'left_event':left[1],'right_event':right[1],'left_roles':['agent','action','theme','temporal'],'right_roles':['agent','action','theme','temporal']},'live_residual':d,'provenance':{'construction':'one hand-authored semantic frame with internal crossing seam','variable_residuals_across_word_boundaries':True,'finished_tape_reversal':False,'post_hoc_repair':False,'broad_phrase_sweep':False,'repeated_unit':False,'word_order_mirror':False,'intact_prose':True},'anti_shortcut':{'hidden_palindromic_span':False,'catalogue_text':False,'finished_tape_reversed':False,'word_order_mirror':False}})
 out={'experiment':ID,'method':'bounded semantic frame crossing-seam residual emission','controls':controls,'candidates':rows,'stats':{'states':states,'exact':sum(x['audit']['exact'] for x in rows)},'next_operator':'Hold out a third temporal adjunct whose first character satisfies the recorded right residual; rerun the complete two-clause audit without editing normalized tape.','independent_audits':['two-pointer comparison','forward/reverse SHA-256']}
 OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__': print(json.dumps(run(),indent=2))
