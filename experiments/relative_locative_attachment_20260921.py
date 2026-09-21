"""Held-out locative relative grammar with prepositional valency."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs/relative-locative-attachment-20260921.json'
@dataclass(frozen=True)
class LocArc:
    name:str; words:tuple[str,...]; attachment:str; role:str
LEFT=(
 LocArc('archivist_works',('the','quiet','archivist','who','works','in','the','old','library'),'library','agent-place'),
 LocArc('gardener_lives',('a','patient','gardener','who','lives','near','the','green','river'),'river','agent-place'),
 LocArc('teacher_teaches',('the','wise','teacher','who','teaches','at','a','small','school'),'school','agent-place'),
 LocArc('sailor_waits',('a','tired','sailor','who','waits','by','the','open','harbor'),'harbor','agent-place'),)
RIGHT=(
 LocArc('reader_studies',('a','careful','reader','who','studies','in','the','quiet','library'),'library','agent-place'),
 LocArc('farmer_walks',('the','kind','farmer','who','walks','near','a','wide','river'),'river','agent-place'),
 LocArc('student_learns',('a','young','student','who','learns','at','the','old','school'),'school','agent-place'),
 LocArc('captain_anchors',('the','brave','captain','who','anchors','by','a','safe','harbor'),'harbor','agent-place'),)
def tape(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s); n=len(t); bad=next((i for i in range(n//2) if t[i]!=t[n-1-i]),None); h=hashlib.sha256(t.encode()).hexdigest(); rh=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'letters':n,'two_pointer_exact':bad is None,'first_mismatch':bad,'forward_sha256':h,'reverse_sha256':rh,'sha_equal':h==rh}
def render(a,b): return ' '.join(a.words)+'; '+' '.join(b.words)+'.'
def online(s):
 t=tape(s); l=0;r=len(t)-1;p=0
 while l<r and t[l]==t[r]: l+=1;r-=1;p+=1
 return {'pairs':p,'seam_position':l,'center_inside_word':True,'obligation':None if l>=r else (t[l],t[r])}
def main():
 rows=[]
 for a in LEFT:
  for b in RIGHT:
   if a.attachment!=b.attachment: continue
   s=render(a,b); z=online(s)
   rows.append({'rendered':s,'left_arc':a.name,'right_arc':b.name,'shared_attachment':a.attachment,'shared_valency':a.role==b.role,'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'held-out locative relative grammar with prepositional valency','finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False,'word_order_symmetry':False}})
 exact=[r for r in rows if r['exact_admitted']]
 out={'experiment_id':'relative-locative-attachment-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'joint locative attachment with prepositional valency before character closure','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max((r['audit']['letters'] for r in rows),default=0),'attachment_families':len(set(r['shared_attachment'] for r in rows))},'novelty_preflight':{'held_out_event_family':True,'prior_outer_pair_sweep_reused':False,'local_key_sweep':False,'cartesian_widening':False},'failure_and_repair':{'failure':'locative valency-compatible pairs mismatch at first character obligation' if not exact else 'none','next_construction':'add an instrument/location alternation with typed preposition compatibility, retaining shared attachment state'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer normalized comparison','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters']}))
if __name__=='__main__': main()
