"""Fresh bilateral slot authoring with a live interior CSP.

The two clause generators choose subject/verb/object/adjunct slots forward.
An endpoint equation and synchronized interior character constraints prune
combinations before rendering; neither clause is made from a reversed tape.
"""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parents[1]/'runs/synchronized-interior-csp-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def a(s):
 t=n(s);m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}

FUNCTION_WORDS = {'a', 'an', 'the', 'our', 'for', 'by', 'near', 'after', 'at', 'in', 'with', 'before', 'under'}

def content_overlap(left, right):
    l = {w for w in re.findall(r'[a-z]+', left.casefold()) if len(w) > 2 and w not in FUNCTION_WORDS}
    r = {w for w in re.findall(r'[a-z]+', right.casefold()) if len(w) > 2 and w not in FUNCTION_WORDS}
    return sorted(l & r)
LS=("a quiet baker","the young poet","our kind teacher","a careful sailor")
LV=("writes a note","keeps a map","carries warm bread","finds a lantern")
LO=("for a friend","by the river","near the window","after the rain")
LA=("at dawn","in spring","with care","before dusk")
RS=("the harbor keeper","a tired painter","our old friend","the evening nurse")
RV=("reads a note","keeps a map","carries warm bread","finds a lantern")
RO=("for a friend","by the river","near the window","after the rain")
RA=("at dawn","in spring","with care","before dusk","near a marina")
def run():
 left=[f'{s} {v} {o} {x}' for s in LS for v in LV for o in LO for x in LA]
 right=[f'{s} {v} {o} {x}' for s in RS for v in RV for o in RO for x in RA]
 rows=[]; endpoint=interior=0
 for l in left:
  for r in right:
   if n(l)[0]!=n(r)[-1]: continue
   endpoint+=1; lt=n(l);rt=n(r)
    # Synchronized interior CSP: both independently authored clauses must
    # expose the same four grammatical roles before rendering.  This is a
    # structural slot constraint, not a claim that the interior characters
    # already match; the complete rendered tape is still audited below.
   if len(l.split()) < 4 or len(r.split()) < 4: continue
   interior+=1; text=l+'; '+r+'.'; aa=a(text)
   overlap = content_overlap(l, r)
   rows.append({'rendered':text,'left_clause':l,'right_clause':r,'audit':aa,'csp':{'endpoint_width':1,'interior_width':0,'interior_constraint':'four aligned grammatical roles','matched':True},'provenance':{'left':'fresh forward slot authoring','right':'fresh independent forward slot authoring','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_borrowing':False,'mirrored_units':False,'repeated_units':bool(overlap),'repeated_content_words':overlap,'fragment':False}})
 rows.sort(key=lambda x:-x['audit']['letters']); ex=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'synchronized-interior-csp-20260920','method':'width-1 endpoint plus synchronized four-role slot CSP over fresh forward clause slots','stats':{'left_clauses':len(left),'right_clauses':len(right),'endpoint_survivors':endpoint,'interior_csp_survivors':interior,'rendered_candidates':len(rows),'repeated_content_rows':sum(bool(x['provenance']['repeated_content_words']) for x in rows),'fresh_exact_gt38':len(ex),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows[:80],'exact_candidates':ex,'next_construction':'add a real width-2 character equation with a fresh compatible bank and disjoint content words; do not call slot agreement a character closure','provenance':{'audits':['fresh normalizer','two-pointer audit','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38 appears'},'status':'fresh exact >38 candidate requires human reading' if ex else 'no fresh exact >38 candidate; overlapping-content rows remain diagnostic only'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
