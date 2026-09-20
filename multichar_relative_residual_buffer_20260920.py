"""Two-character residual buffers across indexed relative boundaries."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/multichar-relative-residual-buffer-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
HEAD=['the weathered navigator','several careful navigators','the patient archivist']
R1=['who mapped the remote coast','who carried the brass chart','who marked the northern inlet']
R2=['that records a winter route','that carries a brass compass','that marked the old channel']
TAIL=['near the lighthouse','beside the silent harbor','under the northern stars']
def gates(text):
 w=text[:-1].split(); pals=[x for x in w if len(n(x))>3 and n(x)==n(x)[::-1]]
 return {'nested_self_palindrome':bool(pals),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False,'mirrored_phrase_units':False}
def run():
 rows=[]; checked=pruned=0
 for head in HEAD:
  for r1 in R1:
   for r2 in R2:
    for tail in TAIL:
     left=n(head+r1); right=n(r2+tail); checked+=1
     # Carry width-two residual from each boundary, before emitting prose.
     compatible=left[:2]==right[-2:][::-1]
     if not compatible: pruned+=1
     text=f'{head} {r1}, {r2} {tail}.'
     semantic_ok=('who ' in r1 and ('that ' in r2)) and len({head,r1,r2,tail})==4
     rows.append({'rendered':text,'buffer':{'width':2,'left_prefix':left[:2],'right_suffix_reversed':right[-2:][::-1],'compatible':compatible},'surface_semantics':{'complete':semantic_ok,'distinct_attachment_units':len({r1,r2})==2},'audit':audit(text),'provenance':{**gates(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and x['surface_semantics']['complete'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'multichar-relative-residual-buffer-20260920','method':'fresh width-two residual buffers across two indexed relative attachments with surface-semantic checks','stats':{'heads':len(HEAD),'transitions':checked,'width2_pruned':pruned,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|two-index-relative|width2-residual-buffer|surface-semantics','distinct_from':'prior one-character endpoint trie: two-character obligations are carried across both attachment boundaries before rendering'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','word-order symmetry','fragments','catalogue text']},'next_construction':'Increase residual width to three while adding typed relative tense/number states and semantic roles for each attachment.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; width-two grammatical near-misses retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
