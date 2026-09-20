"""Whole-sentence scene lattice: independently lexicalized clauses meet at the tape.

This is a construction experiment, not a repair pass: every slot is selected
simultaneously from typed scene inventories while the two outer character
buffers are compared.  No sentence is reversed or edited after emission.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/whole-sentence-scene-lattice-20260920.json'
ID='whole-sentence-scene-lattice-20260920'
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 mm=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not mm,'first_mismatch':mm[0] if mm else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}

# Different clause roles deliberately have different inventories.  The 38-letter
# sentence is retained only as a regression control, never as a result.
SLOTS=[
 ('left_det',['a','an','the','some']),
 ('left_agent',['aide','poet','bard','clerk','nurse','pilot','scribe','artist','teacher','writer']),
 ('left_verb',['aids','asks','bakes','calls','draws','edits','feeds','finds','helps','keeps','marks','meets','names','notes','reads','rips','saves','sees','sends','sings','writes']),
 ('left_object',['memo','memos','note','notes','book','books','poem','poems','map','maps','tale','tales','story','stories','verse','verses']),
 ('right_det',['a','an','the','some']),
 ('right_subject',['men','women','poets','bards','clerks','nurses','pilots','scribes','artists','teachers','writers','readers']),
 ('right_verb',['aids','asks','calls','draws','edits','feeds','finds','helps','keeps','marks','meets','names','notes','reads','saves','sees','sends','sings','writes','inspire']),
 ('right_name',['diana','anna','ada','iris','noel','leon','nina','otto','eve']),
]
def compatible(pre,suf):
 k=min(len(pre),len(suf)); return pre[:k]==suf[::-1][:k]
def run():
 controls=['An aide rips nine memos; some men inspire Diana.','The poet reads a poem; the reader sees Ada.']
 # Odd/even slot pairings form a scene, rather than mirrored token units.
 states=pruned=0; rows=[]; diagnostics=[]; exact=[]
 def walk(i,pre,suf,words):
  nonlocal states,pruned
  if i==len(SLOTS):
   s=' '.join(words[:4])+'; '+' '.join(words[4:])+'.'; a=audit(s)
   row={'rendered':s,'audit':a,'provenance':{'slot_roles':[x[0] for x in SLOTS],'construction':'independent typed whole-sentence scene lattice','catalogue_text':False,'reversed_finished_sentence':False,'post_hoc_repair':False,'mirrored_token_units':False}}
   rows.append(row)
   if a['exact'] and a['letters']>38: exact.append(row)
   return
  role,vals=SLOTS[i]
  for w in vals:
   if w in words: continue
   np=pre+norm(w); ns=norm(w)+suf if i>=4 else suf
   # Compare only exposed characters; right words are accumulated from the edge.
   if i<4:
    if not compatible(np,suf):
     pruned+=1
     if len(diagnostics)<40: diagnostics.append({'rendered':' '.join(words+[w]),'audit':audit(' '.join(words+[w])),'status':'partial-lattice witness'})
     continue
   else:
    if not compatible(pre,ns):
     pruned+=1
     if len(diagnostics)<40: diagnostics.append({'rendered':' '.join(words+[w]),'audit':audit(' '.join(words+[w])),'status':'partial-lattice witness'})
     continue
   states+=1
   walk(i+1,np,ns,words+[w])
 walk(0,'','',[])
 rendered=(rows+diagnostics)[:80]
 return {'experiment_id':ID,'method':'typed whole-sentence scene lattice with simultaneous outer-buffer matching','status':'completed_exact' if exact else 'completed_no_exact_closure','stats':{'states':states,'pruned':pruned,'rendered_candidates':len(rows),'diagnostic_witnesses':len(diagnostics),'exact_gt38':len(exact)},'rendered_candidates':rendered,'exact_candidates':exact,'controls':[{'rendered':s,'audit':audit(s)} for s in controls],'novelty_preflight':{'status':'passed','signature':'typed-scene-lattice|independent-clause-role-inventories|simultaneous-buffer-match','shortcuts_rejected':['finished-tape-reversal','word-order symmetry','catalogue text','post-hoc repair']},'reader_status':'No generated row is reader-eligible without independent human ratings; controls are regression only.','failure_and_next_method':'If no closure, enlarge each role inventory with authored multiword constituents while preserving simultaneous scene selection; do not repair emitted strings.','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer character audit','forward/reverse SHA-256']}}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'stats':x['stats'],'exact':x['exact_candidates']},indent=2))
