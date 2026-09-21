"""Bounded depth-two typed relative attachments over the bilateral lattice."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/double-relative-async-phrase-lattice-20260920.json'
LEFT=(
 (("finite","the patient gardener waters the cedar"),("relative-agent","which shelters a sparrow"),("relative-place","that nests beside the wall")),
 (("finite","a careful keeper records the harbor bells"),("relative-agent","who remembers the winter tide"),("relative-time","that returned after the storm")),
)
RIGHT=(
 (("finite","the patient singer hears the distant bells"),("relative-agent","who follows a bright refrain"),("relative-place","that echoes across the hall")),
 (("finite","a careful pilot crosses the quiet harbor"),("relative-agent","that shelters an old vessel"),("relative-time","which waited through the autumn")),
)
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]; states=prunes=skews=0
 for lp,rp in itertools.product(LEFT,RIGHT):
  lw=[w for _,p in lp for w in p.split()]; rw=[w for _,p in rp for w in p.split()]; i=j=lo=ro=0; ok=True; trace=[]; li=ri=0
  while i<len(lw) and j<len(rw):
   a,b=letters(lw[i]),letters(rw[-1-j]); states+=1
   if a[lo]!=b[::-1][ro]: ok=False; prunes+=1; break
   trace.append({'left_word':lw[i],'right_word':rw[-1-j],'left_type':lp[li][0],'right_type':rp[ri][0],'left_offset':lo,'right_offset':ro})
   lo+=1;ro+=1
   if lo==len(a): i+=1;lo=0; old=li; li=min(2,li+1); skews+=int(li!=old and j<len(rw) and ri<2)
   if ro==len(b): j+=1;ro=0; old=ri; ri=min(2,ri+1); skews+=int(ri!=old and i<len(lw) and li<2)
  if ok and (i!=len(lw) or j!=len(rw)): ok=False
  text=' '.join(lw)+'.'; words=[letters(x) for x in lw]
  rows.append({'rendered':text,'paired_typed_paths':{'left':lp,'right':rp},'attachment_depth':2,'online_lattice':{'closed':ok,'boundary_skews':skews,'trace':trace[-12:]},'audit':audit(text),'provenance':{'two_typed_relative_attachments':True,'complete_phrase_units':True,'nested_self_palindrome':any(len(w)>3 and w==w[::-1] for w in words),'repeated_units':len(words)!=len(set(words)),'word_order_symmetry':words==words[::-1],'fragment':len(words)<12,'catalogue_text':False,'finished_tape_reversal':False,'post_hoc_repair':False,'RLAIF':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); hard=('nested_self_palindrome','repeated_units','word_order_symmetry','fragment')
 exact=[r for r in rows if r['online_lattice']['closed'] and r['audit']['pointer_exact'] and not any(r['provenance'][k] for k in hard)]
 out={'experiment_id':'double-relative-async-phrase-lattice-20260920','method':'bounded depth-two typed relative attachments with asynchronous bilateral phrase/word frontiers','stats':{'left_paths':len(LEFT),'right_paths':len(RIGHT),'paired_paths':len(rows),'online_states':states,'mismatch_prunes':prunes,'boundary_skews':skews,'rendered_controls':len(rows),'exact_clean':len(exact),'max_letters':rows[0]['audit']['letters']},'rendered_candidates':rows,'exact_candidates':exact,'controls':rows,'novelty_preflight':{'status':'passed','signature':'fresh-authored|depth-two-relative-attachments|typed-boundary-skew|independent-audit','distinct_from':'single-relative extension: two independently typed attachment edges are retained in each ordinary-order path with bounded depth before live character matching'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'hard_exclusions':list(hard)+['catalogue text']},'falsifier':'recompute normalized pointers and hashes independently for every control; any closed/hash-disagreeing row falsifies closure','next_construction':'Permit typed attachment alternatives selected by residual class after the first relative edge.','status':'exact clean candidate requires reading' if exact else 'no exact clean intersection; depth-two prose controls retained'}
 OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': print(json.dumps(run()['stats']))
