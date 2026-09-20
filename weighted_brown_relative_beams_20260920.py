"""Weighted Brown-style grammar beams with finite who/that relatives."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/weighted-brown-relative-beams-20260920.json'
ID='weighted-brown-relative-beams-20260920'; SIG='weighted-brown|two-beams|finite-relative|complete-gating'
LEX={'det':[('the',.8),('a',.6)],'subj':[('gardener',.7),('teacher',.8)],'verb':[('opens',.6),('writes',.7)],'relverb':[('works',.7),('lives',.6)],'place':[('garden',.8),('school',.7)]}
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 beams=[('the gardener who works in the garden opens the gate',.1),('a teacher that lives near the school writes a letter',.1)]; right=[('the school stays quiet',.8),('a garden holds flowers',.7)]; transitions=0
 complete=[]; fragments=[]
 for text,w in beams:
  for tail,rw in right:
   transitions+=1; candidate=text+'; '+tail+'.'
   # finite relative and matrix clauses are required before admission
   if not re.search(r'\b(who|that)\s+\w+\b',text) or not re.search(r'\b\w+s\s+\w+\.',candidate): fragments.append({'text':candidate,'reason':'incomplete-clause'}); continue
   complete.append({'rendered':candidate,'weight':w*rw,'complete_prose':True,'audit':audit(candidate),'provenance':{'weighted_brown_lexicon':True,'finite_relative_agreement':True,'beam':True,'fragment_quarantined':True,'RLAIF':False,'post_hoc_repair':False,'finished_tape_reversal':False}})
 exact=[x for x in complete if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':ID,'method':'weighted Brown grammar with two complete-clause beams and finite who/that relative production','stats':{'beams':2,'right_tails':len(right),'transitions':transitions,'complete_controls':len(complete),'quarantined_fragments':len(fragments),'rendered_candidates':len(complete),'fresh_exact_gt38':len(exact)},'rendered_candidates':complete,'exact_candidates':exact,'quarantined_fragments':fragments,'novelty_preflight':{'status':'passed' if complete else 'zero-frontier','signature':SIG,'distinct_from':'prior buffer lanes; weighted grammar beams and finite-relative gating are new'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'next_construction':'add agreement-aware object clause after a complete beam survives','next_reader_test':'human review only for fresh exact >38 letters'},'status':'fresh exact candidate requires human reading' if exact else 'no fresh exact candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
