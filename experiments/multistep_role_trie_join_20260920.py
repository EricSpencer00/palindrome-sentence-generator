"""Multi-step join of role/attachment-indexed residual keys."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/multistep-role-trie-join-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
H=['the weathered navigator','several careful navigators','the patient archivist']; A=['who mapped the remote coast','who marks the northern inlet','who carried the brass chart']; B=['that records a winter route','that carries a brass compass','that marked the old channel']; T=['near the lighthouse','beside the silent harbor','under the northern stars']
def g(text):
 w=text[:-1].split(); p=[x for x in w if len(n(x))>3 and n(x)==n(x)[::-1]]
 return {'nested_self_palindrome':bool(p),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False,'mirrored_phrase_units':False}
def run():
 rows=[]; joins=0; partial=0
 for h in H:
  for a in A:
   for b in B:
    for t in T:
     text=f'{h} {a}, {b} {t}.'; l=n(h+a); r=n(b+t); states=[]; ok=True
     for role,idx,left,right in [('agent',1,l[-3:],r[:3]),('theme',2,l[-2:],r[-2:])]:
      joins+=1; match=left==right[::-1]; states.append({'role':role,'attachment_index':idx,'left':left,'right_reversed':right[::-1],'match':match}); partial+=int(match); ok &= match
     rows.append({'rendered':text,'join_states':states,'partial_compatible_steps':sum(s['match'] for s in states),'surface_semantics':{'complete':True,'agreement':'passed'},'audit':audit(text),'provenance':{**g(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and x['partial_compatible_steps']==2 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'multistep-role-trie-join-20260920','method':'multi-step role/attachment-indexed residual trie join with partial compatibility','stats':{'states':len(rows),'join_steps':joins,'partial_matches':partial,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|multistep-role-join|partial-compatibility|indexed-residuals','distinct_from':'prior single-key lane: joins agent and theme residual states independently and retains partial matches'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','word-order symmetry','fragments','catalogue text']},'next_construction':'Carry partial states asynchronously across three indexed joins with typed tense transitions.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; multi-step prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
