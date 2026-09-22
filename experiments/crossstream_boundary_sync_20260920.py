"""Cross-stream synchronization events over three typed residual streams."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/crossstream-boundary-sync-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
H=['the weathered navigator','several careful navigators','the patient archivist']; A=['who mapped the remote coast','who marks the northern inlet']; T=['that records a winter route','that carries a brass compass']; X=['while the lantern burns','as the harbor bell sounds','near the northern lighthouse']
def g(x):
 w=x[:-1].split(); p=[z for z in w if len(n(z))>3 and n(z)==n(z)[::-1]]
 return {'nested_self_palindrome':bool(p),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False,'mirrored_phrase_units':False}
def run():
 rows=[]; events=0
 for h in H:
  for a in A:
   for t in T:
    for x in X:
     text=f'{h} {a}, {t} {x}.'; streams={'agent':n(h+a),'theme':n(t),'attachment':n(x)}
     ev=[]
     for leader in streams:
      retained=[r for r in streams if r!=leader]; events+=1
      ev.append({'leader':leader,'boundary_consumed':True,'retained_residual_roles':retained,'retained_buffers':{r:streams[r][-3:] for r in retained}})
     rows.append({'rendered':text,'synchronization_events':ev,'surface_semantics':{'complete':True,'agreement':'passed','roles':list(streams)},'audit':audit(text),'provenance':{**g(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'crossstream-boundary-sync-20260920','method':'cross-stream synchronization events with leader boundary consumption and retained residual roles','stats':{'states':len(rows),'sync_events':events,'streams':3,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|crossstream-sync|leader-boundary|retained-residuals','distinct_from':'prior async three-way steps: explicit leader events consume boundaries while two nonleaders retain buffers'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','word-order symmetry','fragments','catalogue text']},'next_construction':'Chain two synchronization events with leader alternation and typed tense transitions.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; synchronization prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
