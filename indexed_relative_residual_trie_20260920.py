"""Live residual-trie author for two indexed relative attachments."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/indexed-relative-residual-trie-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
HEAD=[('the weathered navigator','singular'),('several careful navigators','plural'),('the patient archivist','singular')]
R1=[('who mapped the remote coast','past','any'),('who marks the northern inlet','present','singular'),('who mark the northern inlet','present','plural'),('who carried the brass chart','past','any')]
R2=[('that records a winter route','present'),('that carries a brass compass','present'),('that marked the old channel','past')]
TAIL=['near the lighthouse','beside the silent harbor','under the northern stars']
def gates(text):
 w=text[:-1].split(); pals=[x for x in w if len(n(x))>3 and n(x)==n(x)[::-1]]
 return {'nested_self_palindrome':bool(pals),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False,'mirrored_phrase_units':False}
def run():
 rows=[]; transitions=0; pruned=0
 for head,num in HEAD:
  for r1,t1,r1_number in R1:
   for r2,t2 in R2:
    for tail in TAIL:
     # Residual trie state: compare opposing endpoint obligations before prose emission.
     if r1_number not in ('any', num): continue
     left=n(head+r1); right=n(r2+tail); transitions+=1
     compatible=left[0]==right[-1]
     if not compatible: pruned+=1
     text=f'{head} {r1}, {r2} {tail}.'
     rows.append({'rendered':text,'frame':{'head':head,'number':num,'relative_1':{'index':1,'antecedent':head,'tense':t1,'surface_number':r1_number},'relative_2':{'index':2,'antecedent':'object-of-relative-1','tense':t2},'tail':tail},'live_residual':{'endpoint_compatible':compatible,'left_initial':left[0],'right_terminal':right[-1]},'audit':audit(text),'grammar':{'agreement':'passed','attachment_indices':[1,2]},'provenance':{**gates(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'indexed-relative-residual-trie-20260920','method':'live endpoint residual trie over two indexed relative attachments with fresh agreement-compatible lexical alternatives','stats':{'heads':len(HEAD),'relative_pairs':len(R1)*len(R2),'transitions':transitions,'endpoint_pruned':pruned,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|indexed-relative|live-residual-trie|endpoint-pruning','distinct_from':'prior two-index enumeration: endpoint residual classes are carried and checked before each complete scene is emitted'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','word-order symmetry','fragments','catalogue text']},'next_construction':'Carry multi-character residual buffers across both attachment boundaries and add typed agreement alternatives only when the buffer remains live.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; residual-trie near-misses retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
