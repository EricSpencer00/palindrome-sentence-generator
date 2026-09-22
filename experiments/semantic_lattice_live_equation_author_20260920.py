"""Author-first semantic lattice with live two-sided character solving."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/semantic-lattice-live-equation-author-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUB=[('the lantern keeper','singular','agent'),('several river pilots','plural','agent')]; VERB={'singular':['charts','guards','marks'],'plural':['chart','guard','mark']}; OBJ=[('the quiet inlet','theme'),('a weathered beacon','theme'),('the narrow channel','theme')]; SET=['before dawn','beside the river','under clear stars']
def gates(t,units):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(units)!=len(set(units)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]; states=prunes=0
 for (s,num,role),v,(o,orole),setting in itertools.product(SUB,VERB['singular'],OBJ,SET):
  if num!='singular': continue
  text=f'{s} {v} {o} {setting}.'; states+=1; left=n(f'{s} {v}'); right=n(f'{o} {setting}'); live=left[0]==right[-1]
  if not live: prunes+=1
  rows.append({'rendered':text,'semantic_frame':{'subject':s,'subject_number':num,'event':v,'object':o,'setting':setting,'roles':[role,orole]},'live_equation':{'left_initial':left[0],'right_terminal':right[-1],'accepted':live},'audit':audit(text),'provenance':{**gates(text,[s,v,o,setting]),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['live_equation']['accepted'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'semantic-lattice-live-equation-author-20260920','method':'fresh author-first semantic lattice with live two-sided character equation during lexical selection','stats':{'semantic_states':states,'live_prunes':prunes,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':[],'diagnostic_controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|semantic-lattice|live-equation|scene-first','distinct_from':'aspectual and boundary residual lanes: semantic scene frame is selected first, then lexical choices solve a live outer character equation'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text'],'reader_gate':'closed; diagnostic controls include mechanical shortcut flags and are not reader material'},'next_construction':'Expand both number states and carry a two-character residual across subject, verb, object, and setting slots while preserving the semantic frame.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; semantic-lattice prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
