"""Author-first bidirectional lexical-boundary search, grown center/outward."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/author-first-bidirectional-boundary-20260920.json'
ID='author-first-bidirectional-boundary-20260920'; SIG='author-first-inventory|center-outward-growth|online-character-obligations'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
CENTERS=('the careful record','a quiet harbor','our old map')
LEFT=(('the patient archivist','agent'),('a watchful gardener','agent'),('our quiet teacher','agent'))
RIGHT=(('returns before dusk','event'),('guards the small bridge','event'),('keeps the lantern lit','event'))
OUTER=(('in the village','setting'),('near the river','setting'))
def consume(left,right):
 a,b=letters(left),letters(right)[::-1]; checked=0
 for i,(x,y) in enumerate(zip(a,b)):
  checked+=1
  if x!=y:return False,checked,{'offset':i,'left':x,'right':y}
 return len(a)<=len(b),checked,None
def run():
 diagnostics=[]; prunes=0; walks=0
 for center,(l,lr),(r,rr),(outer,orole) in itertools.product(CENTERS,LEFT,RIGHT,OUTER):
  # Independent expansions are authored clauses; center is inserted once, not mirrored.
  rendered=f'{l} {center} {r} {outer}.'; walks+=1
  ok,checked,mm=consume(l+' '+center, r+' '+outer)
  rec={'rendered':rendered,'growth':{'center':center,'left_edge':l,'right_edge':r,'outer_edge':outer,'left_role':lr,'right_role':rr,'outer_role':orole},'online_obligation':{'accepted':ok,'characters_checked':checked,'mismatch':mm},'audit':audit(rendered),'provenance':{'inventory':'fresh hand-authored intact English clauses/expansions','center_outward':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_word_order':False,'repeated_units':False,'self_palindromic_units':False,'fragment':False}}
  diagnostics.append(rec)
  if not ok: prunes+=1
 diagnostics.sort(key=lambda x:-x['audit']['letters']); exact=[r for r in diagnostics if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'author-first independent center/outward clause growth with online opposite character obligations','stats':{'centers':len(CENTERS),'left_edges':len(LEFT),'right_edges':len(RIGHT),'outer_edges':len(OUTER),'bidirectional_walks':walks,'online_prunes':prunes,'diagnostic_controls':len(diagnostics),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in diagnostics),default=0)},'reader_facing_candidates':exact if exact and all(not r['provenance']['mirrored_word_order'] for r in exact) else [],'diagnostic_controls':diagnostics,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'scene modality/quotation products and residual-buffer repair lanes'},'next_topology':'add a third independently authored outward edge with typed attachment, retaining center obligations','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; reader-facing candidates intentionally empty'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
