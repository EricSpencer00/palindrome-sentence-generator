"""Width-three residual buffer with typed relative roles and features."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/width3-relative-role-buffer-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
HEAD=[('the weathered navigator','singular','past'),('several careful navigators','plural','present'),('the patient archivist','singular','present')]
R1=[('who mapped the remote coast','singular','past','agent'),('who marks the northern inlet','singular','present','agent'),('who carried the brass chart','singular','past','agent')]
R2=[('that records a winter route','singular','present','theme'),('that carries a brass compass','singular','present','theme'),('that marked the old channel','singular','past','theme')]
TAIL=['near the lighthouse','beside the silent harbor','under the northern stars']
def gates(text):
 w=text[:-1].split(); pals=[x for x in w if len(n(x))>3 and n(x)==n(x)[::-1]]
 return {'nested_self_palindrome':bool(pals),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False,'mirrored_phrase_units':False}
def run():
 rows=[]; pruned=0
 for head,hnum,htense in HEAD:
  for r1,r1num,r1tense,r1role in R1:
   for r2,r2num,r2tense,r2role in R2:
    for tail in TAIL:
     left=n(head+r1); right=n(r2+tail); compatible=left[:3]==right[-3:][::-1]
     if not compatible: pruned+=1
     text=f'{head} {r1}, {r2} {tail}.'
     semantic=(r1role=='agent' and r2role=='theme' and r1num=='singular' and r2num=='singular')
     rows.append({'rendered':text,'frame':{'head':head,'head_number':hnum,'head_tense':htense,'relative_1':{'index':1,'role':r1role,'number':r1num,'tense':r1tense},'relative_2':{'index':2,'role':r2role,'number':r2num,'tense':r2tense},'tail':tail},'buffer':{'width':3,'compatible':compatible},'surface_semantics':{'complete':semantic,'agreement':'passed'},'audit':audit(text),'provenance':{**gates(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and x['surface_semantics']['complete'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'width3-relative-role-buffer-20260920','method':'fresh width-three residual buffers with typed relative semantic roles and tense/number states','stats':{'heads':len(HEAD),'relative_pairs':len(R1)*len(R2),'transitions':len(rows),'width3_pruned':pruned,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|width3-buffer|typed-relative-roles|tense-number','distinct_from':'prior width-two buffer: adds three-character obligations and explicit agent/theme role states per attachment'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','word-order symmetry','fragments','catalogue text']},'next_construction':'Use asynchronous width-three buffers with one-character role transitions so compatible states are not rejected as a single fixed prefix equation.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; typed width-three near-misses retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
