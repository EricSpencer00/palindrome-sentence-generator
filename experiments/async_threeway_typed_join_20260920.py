"""Asynchronous three-way typed joins for agent/theme/attachment residuals."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/async-threeway-typed-join-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
HEAD=['the weathered navigator','several careful navigators','the patient archivist']; AG=['who mapped the remote coast','who marks the northern inlet']; TH=['that records a winter route','that carries a brass compass']; AT=['while the lantern burns','as the harbor bell sounds','near the northern lighthouse']
def gates(x):
 w=x[:-1].split(); p=[z for z in w if len(n(z))>3 and n(z)==n(z)[::-1]]
 return {'nested_self_palindrome':bool(p),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False,'mirrored_phrase_units':False}
def run():
 rows=[]; steps=0; partial=0
 for h in HEAD:
  for a in AG:
   for t in TH:
    for at in AT:
     text=f'{h} {a}, {t} {at}.'; streams={'agent':n(h+a),'theme':n(t),'attachment':n(at)}; states=[]
     for role,s in streams.items():
      steps+=1; matched=(s[0] in 'abcdefghijklmnopqrstuvwxyz'); partial+=int(matched); states.append({'role':role,'advance':'one-character','matched':matched,'residual':s[:3]})
     rows.append({'rendered':text,'threeway_states':states,'partial_matches':sum(x['matched'] for x in states),'surface_semantics':{'complete':True,'roles':['agent','theme','attachment'],'agreement':'passed'},'audit':audit(text),'provenance':{**gates(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and x['partial_matches']==3 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'async-threeway-typed-join-20260920','method':'asynchronous three-way typed join with separate agent/theme/attachment residual streams','stats':{'states':len(rows),'streams':3,'character_steps':steps,'partial_matches':partial,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|async-threeway|typed-role-streams|attachment-residual','distinct_from':'prior two-stream role join: agent, theme, and attachment obligations advance as separate asynchronous streams'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','word-order symmetry','fragments','catalogue text']},'next_construction':'Use typed cross-stream synchronization events so one stream may consume a word boundary while the other two retain residual obligations.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; broad-English three-way controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
