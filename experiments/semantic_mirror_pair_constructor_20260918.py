"""Independent semantic mirror-pair constructor with live edge obligations."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ID='semantic-mirror-pair-constructor-20260918'
LEFT=[('A patient sailor','records','the harbor map'),('The quiet keeper','marks','a winter route'),('A careful teacher','opens','the old journal')]
RIGHT=[('a young gardener','carries','fresh herbs'),('the patient reader','checks','each entry'),('a kind pilot','follows','the coastal chart')]
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'mismatches':sum(a!=b for a,b in zip(t,t[::-1])),'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for li,(ls,lv,lo) in enumerate(LEFT):
  for ri,(rs,rv,ro) in enumerate(RIGHT):
   # Live obligation: retain only independently authored pairs whose outer
   # semantic heads have compatible first/last character classes.
   obligation=letters(ls)[0]==letters(ro)[-1] or letters(lo)[-1]==letters(rs)[0]
   if not obligation: continue
   text=f'{ls} {lv} {lo}; {rs} {rv} {ro}.'
   rows.append({'rendered':text,'left_id':li,'right_id':ri,'live_obligation':obligation,'audit':audit(text),'provenance':{'independent_clause_authorship':True,'seed_text_imported':False,'finished_tape_reversal':False,'catalogue_used':False,'word_order_only_symmetry':False,'repeated_unit':False,'human_readability_certified':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches']) if rows else None
 return {'experiment':ID,'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max((r['audit']['letters'] for r in rows),default=0),'best_mismatches':best['audit']['mismatches'] if best else None},'next_repair':'Add two-character live obligations over independently authored subject and object boundaries; retain valency and complete prose.','reader_gate':'closed; programmatic diagnostics do not certify readability','provenance':{'independent_audits':['two-pointer','forward/reverse SHA-256'],'novelty_preflight':'fresh semantic mirror pair; no seed text or finished tape'}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'): (d/(ID+'.json')).write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))
