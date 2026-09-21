"""Matrix-attached temporal adjunct follow-up to temporal5.

The relative clause keeps a fresh three-item adjunct bank while one distinct
since/after adjunct is attached to the matrix clause. Both attachments are
typed before ordinary-order emission and checked by live opposing cursors.
"""
import hashlib,json,re,sys
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.unequal_center_ditransitive_relative_complement_20260920 import Word,walk_pair
OUT=ROOT/"runs/semantic-role-matrix-adjunct-temporal6-csp-20260920.json"; REG=ROOT/"docs/experiment-novelty-registry.json"
ID="semantic-role-matrix-adjunct-temporal6-csp-20260920"; SIG="typed-seam-machine|subject-relative-future-perfect|matrix-perfect|relative-temporal-attachment|matrix-temporal-attachment|independent-pointer-sha"
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);i,j=0,len(t)-1; bad=None
 while i<j:
  if t[i]!=t[j]: bad={'left_index':i,'right_index':j,'left_char':t[i],'right_char':t[j]};break
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'normalized_tape':t,'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal_under_reversal':f==r,'independent_exact':bool(t) and bad is None and f==r}
@dataclass(frozen=True)
class C:
 frame_id:str; number:str; rel_adj:str; matrix_adj:str; words:tuple[Word,...]; roles:tuple[str,...]
 @property
 def surface(self): return ' '.join(w.surface for w in self.words)
 @property
 def letters(self): return len(n(self.surface))
SUB={'sg':('the patient pilot',), 'pl':('patient pilots',)}
REL={'sg':(('charted','the old map'),('guarded','the quiet inlet')), 'pl':(('charted','old maps'),('guarded','quiet inlets'))}
MAT={'sg':(('guided','the narrow bridge'),('observed','a calm harbor')), 'pl':(('guided','narrow bridges'),('observed','calm harbors'))}
RELADJ=(('during rain','during'),('beneath moonlight','beneath'),('as winter faded','as'))
MATADJ=(('since dawn','since'),('after first light','after'))
def ws(s,r,i): return tuple(Word(x,r,i) for x in s.split())
def paths():
 out=[]
 for num,subs in SUB.items():
  for subject in subs:
   for rv,rt in REL[num]:
    for mv,mt in MAT[num]:
     for ra,rr in RELADJ:
      for ma,mr in MATADJ:
       chunks=(ws(subject,'head',0),ws('who will have','relative_aux',1),ws(rv,'relative_participle',2),ws(rt,'relative_theme',3),ws(ra,'relative_adjunct',4),ws('will have','matrix_perfect_aux',5),ws(mv,'matrix_participle',6),ws(mt,'matrix_theme',7),ws(ma,'matrix_adjunct',8))
       out.append(C(f'{num}:{subject}:{rv}:{mv}:{rr}:{mr}',num,rr,mr,tuple(x for z in chunks for x in z),('head','relative_aux','relative_participle','relative_theme','relative_adjunct','matrix_perfect_aux','matrix_participle','matrix_theme','matrix_adjunct')))
 return tuple(out)
def render(a,b): return a.surface[:1].upper()+a.surface[1:]+'; '+b.surface+'.'
def row(a,b,w):
 text=render(a,b)
 return {'rendered':text,'letters':len(n(text)),'audit':audit(text),'seam_machine':{'status':w.status,'online_equations':w.online_equations,'matched_characters':w.matched_characters,'first_mismatch':w.first_mismatch,'seam_mode':w.seam_mode},'provenance':{'left_frame_id':a.frame_id,'right_frame_id':b.frame_id,'left_roles':list(a.roles),'right_roles':list(b.roles),'relative_adjunct_left':a.rel_adj,'matrix_adjunct_left':a.matrix_adj,'relative_adjunct_right':b.rel_adj,'matrix_adjunct_right':b.matrix_adj,'lexical_source':'fresh relative and matrix temporal attachment slots','grammar_paths_independent':True,'lexicalized_during_cursor_walk':True,'right_clause_read_inward_by_cursor':True,'finished_tape_reversal':False,'post_hoc_repair':False,'mirrored_units':False,'catalogue_text':False,'word_order_symmetry':False,'per_search_rlaif':False},'reader_facing_eligible':False,'reader_evidence':{'status':'not_run','human_raters':0}}
def novelty():
 es=json.loads(REG.read_text()).get('entries',[]); hits=[e.get('id') for e in es if e.get('id')==ID or e.get('signature')==SIG]
 related=[e.get('id') for e in es if 'matrix-perfect' in e.get('signature','') or 'temporal-subject-relative' in e.get('signature','')]
 return {'status':'passed' if not hits else 'collision','registry_inspected':True,'registry_entries_read':len(es),'exact_signature_collision':hits,'related_temporal_runs':related[-6:],'signature':SIG,'not_a_duplicate_sweep':True,'distinction':'Relocates one temporal adjunct to matrix attachment while retaining a separate relative-clause adjunct state; both are live typed obligations during emission.'}
def run():
 ps=paths(); obs=[]; exact=[]; eq=pr=0
 for i,a in enumerate(ps):
  for j,b in enumerate(ps):
   w=walk_pair(a,b);obs.append((i,j,a,b,w));eq+=w.online_equations;pr+=w.status=='mismatch_pruned'
   if w.status=='closed':
    x=row(a,b,w)
    if x['audit']['independent_exact']: exact.append(x)
 controls=[];seen=set()
 for i,j,a,b,w in sorted(obs,key=lambda q:-(q[2].letters+q[3].letters)):
  k=(a.surface,b.surface)
  if k not in seen and a.surface!=b.surface: controls.append(row(a,b,w));seen.add(k)
  if len(controls)>=24: break
 clean=[x for x in exact if x['letters']>38]
 result={'experiment_id':ID,'signature':SIG,'method':'Fresh dual-attachment temporal grammar: future-perfect relative clause with its own adjunct and matrix perfect clause with a distinct since/after adjunct; opposing cursor equations are evaluated during ordinary-order realization.','operator_added':{'name':'matrix temporal attachment relocation','relative_adjunct_bank':['during','beneath','as'],'matrix_adjunct_bank':['since','after'],'fresh_lexical_bank':True,'post_hoc_repair':False},'stats':{'heldout_clause_paths':len(ps),'paired_grammar_states':len(obs),'online_character_equations':eq,'mismatch_prunes':pr,'rendered_controls':len(controls),'mechanical_exact_candidates':len(exact),'exact_clean_above_38':len(clean),'longest_rendered_control_letters':max((x['letters'] for x in controls),default=0),'longest_exact_clean_letters':max((x['letters'] for x in clean),default=0)},'rendered_controls':controls,'exact_candidates':exact,'exact_clean_candidates':clean,'reader_facing_candidates':[],'reader_gate':{'status':'closed','reason':'No exact-clean closure; blinded reading is not claimed.','human_raters':0},'novelty_preflight':novelty(),'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'shared_cursor_source':'experiments/unequal_center_ditransitive_relative_complement_20260920.py','independent_audits':['literal outside-in pointer scan','forward/reverse SHA-256'],'ordinary_order_grammar_emission':True,'right_clause_read_inward_by_cursor':True,'catalogue_text':False,'finished_tape_reversal':False,'post_hoc_repair':False,'mirrored_or_self_palindromic_units':False,'word_order_symmetry':False,'fragment_output':False,'per_search_rlaif':False},'status':'completed_no_exact_closure' if not exact else 'mechanical_exact_requires_reader_gate','falsifier':'A closure is valid only when cursor closure and independent pointer/SHA equality agree; no near miss is promoted.','next_construction':'Change attachment topology to a center-bearing adjunct with agreement-carrying subject state, not another lexical sweep.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 x=run();print(json.dumps({'experiment_id':ID,'stats':x['stats']},sort_keys=True))
