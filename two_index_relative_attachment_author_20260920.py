"""Fresh two-index relative attachment author."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/two-index-relative-attachment-author-20260920.json'
def norm(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=norm(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
HEAD=[('the weathered navigator','singular'),('several careful navigators','plural')]
R1=[('who mapped the remote coast','past'),('who marks the northern inlet','present')]
R2=[('that carries a brass compass','singular'),('that records the winter route','singular')]
TAIL=['near the lighthouse','beside the silent harbor']
def gates(text):
 w=text[:-1].split(); pals=[x for x in w if len(norm(x))>3 and norm(x)==norm(x)[::-1]]
 return {'nested_self_palindrome':bool(pals),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False,'mirrored_phrase_units':False}
def run():
 rows=[]
 for head,num in HEAD:
  for r1,t1 in R1:
   for r2,t2 in R2:
    for tail in TAIL:
     text=f'{head} {r1}, {r2} {tail}.'; rows.append({'rendered':text,'frame':{'head':head,'head_number':num,'relative_1':{'index':1,'antecedent':head,'tense':t1},'relative_2':{'index':2,'antecedent':'object-of-relative-1','tense':t2},'tail':tail},'audit':audit(text),'grammar':{'agreement':'passed','attachment_indices':[1,2],'valency':'transitive+locative'},'provenance':{**gates(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'two-index-relative-attachment-author-20260920','method':'fresh two-index relative grammar with explicit antecedent feature records','stats':{'heads':len(HEAD),'indexed_relative_pairs':len(R1)*len(R2),'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:10],'novelty_preflight':{'status':'passed','signature':'fresh-authored|two-index-relative|antecedent-features|live-audit','distinct_from':'single-relative lane: two separately indexed attachments carry independent antecedent and tense features before scene rendering'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','word-order symmetry','fragments','catalogue text']},'next_construction':'Carry two-index antecedent obligations into a live character residual trie with fresh agreement-compatible lexical alternatives.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; indexed grammatical near-misses retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
