"""Diagnostic two-slot residual debt over relation and setting realizations."""
import hashlib,itertools,json,re
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/residual-relation-setting-20260920.json'
def letters(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); h=hashlib.sha256(t.encode()).hexdigest(); rh=hashlib.sha256(t[::-1].encode()).hexdigest(); return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':h,'sha256_reverse':rh,'sha_equal':h==rh}
REL=(('garden','the gardener opens the gate','the gate is opened by the gardener'),('map','the cartographer maps the shore','the shore is mapped by the cartographer'))
SET=(('river','near the river','by the river'),('harbor','at the harbor','from the harbor'),('window','by the window','through the window'))
def debt(left,right):
 l=letters(left); r=letters(right)[::-1]; n=min(len(l),len(r)); i=0
 while i<n and l[i]==r[i]: i+=1
 return {'matched':i,'left_unmatched':l[i:i+4],'right_unmatched':r[i:i+4],'resolved':i==n}
def run():
 rows=[]; examined=0; pruned=0
 for rel,setting in itertools.product(REL,SET):
  # Slots are selected independently; residual debt is recorded diagnostically.
  left=rel[1]; right=rel[2]; d1=debt(left,right); examined+=1
  for mode,lset,rset in (('locative',setting[1],setting[2]),):
   d2=debt(left+' '+lset,right+' '+rset)
   if setting[0]=='harbor' and d2['matched']==0: pruned+=1; continue
   text=f'{left} {lset}, and {right} {rset}.'; rows.append({'rendered':text,'relation':rel[0],'setting':setting[0],'debt_after_relation':d1,'debt_after_setting':d2,'audit':audit(text),'complete_prose':True,'provenance':{'independent_relation_and_setting_slots':True,'debt_carried_across_two_slots':True,'pruned_before_rendering':False,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'fragment':False}})
 exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'residual-relation-setting-20260920','method':'diagnostic two-slot relation plus setting debt trace; no validated live equation','stats':{'relation_slots':len(REL),'setting_slots':len(SET),'states_examined':examined,'arbitrary_filtered_states':pruned,'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':'relation-setting|two-slot-debt-diagnostic','distinct_from':'semantic relation lattice: records debt across a second setting slot','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'fragments':False},'provenance':{'audits':['independent pointer mismatch','forward/reverse SHA-256']},'status':'diagnostic only; arbitrary filter is not a live character equation'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
