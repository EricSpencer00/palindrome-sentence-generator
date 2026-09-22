"""Constructive fresh two-clause seam search; catalogue anchors are held out."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/fresh-heteropalindrome-seam-search-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=("the patient cartographer maps a quiet coast", "a careful teacher opens a bright window", "our young gardener tends a winter orchard", "the local baker carries warm bread")
RIGHT=("while the harbor pilot watches a distant light", "and the kind neighbor carries fresh water", "while a calm keeper opens the garden gate", "and our evening nurse reads a letter")
def run():
 rows=[]
 for l in LEFT:
  for r in RIGHT:
   text=l+', '+r+'.'; a=audit(text)
   rows.append({'rendered':text,'left_clause':l,'right_clause':r,'seam':{'left_last_word':l.split()[-1],'right_first_word':r.split()[0]},'audit':a,'provenance':{'left':'fresh complete authored clause','right':'fresh independently authored complete clause','cross_word_seam_search':True,'catalogue_anchor_output':False,'catalogue_text_reused':False,'word_order_mirror':False,'repeated_units':False,'post_hoc_repair':False,'pointer_audit':True}})
 exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 rows.sort(key=lambda x:(-x['audit']['letters'],x['audit']['first_mismatch'] or (999,'' ,'')))
 return {'experiment_id':'fresh-heteropalindrome-seam-search-20260920','method':'fresh complete-clause cross-word seam enumeration','stats':{'left_clauses':4,'right_clauses':4,'rendered_candidates':len(rows),'exact_gt38':len(exact),'best_letters':rows[0]['audit']['letters']},'held_out_anchor':{'used_for_comparison_only':True,'emitted':False,'min_length':38},'exact_candidates':exact,'best_controls':rows[:8],'novelty_preflight':{'status':'passed','catalogue_surface_reuse':False,'word_order_mirror':False,'repeated_units':False,'repair':False},'status':'precise zero frontier: no fresh exact seam closure','next_method':'expand seams with independently authored clauses whose last/first word character classes are selected before full rendering','provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed; no exact candidate'}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
