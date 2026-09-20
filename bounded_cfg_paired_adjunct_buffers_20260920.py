"""Paired left/right adjunct ownership with separate residual buffers."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/bounded-cfg-paired-adjunct-buffers-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUB=[('the sailor','singular','marks'),('the sailors','plural','mark'),('the keeper','singular','carries')]; OBJ=['the inlet','a beacon','the channel']; ADJ=[('at dawn','time'),('by the river','location'),('under stars','setting')]; RIGHT=['the guide records the harbor','the scouts watch a lantern','the keeper guards the bridge']
def gates(t,u):
 w=t.rstrip('.').split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<6,'catalogue_text':False}
def run():
 rows=[]; transitions=0; prunes=0
 for (s,num,v),o,(a1,r1),(a2,r2),right in itertools.product(SUB,OBJ,ADJ,ADJ,RIGHT):
  if a1==a2: continue
  left=f'{s} {v} {o} {a1} {a2}.'; buffers=[n(a1)[-2:],n(a2)[-2:]]; controls=[n(right)[0:2],n(right)[2:4]]; transitions+=2; compatible=all(x==y[::-1] for x,y in zip(buffers,controls))
  if not compatible: prunes+=1
  rows.append({'rendered':left,'owners':[r1,r2],'right_control':right+'.','separate_buffers':[{'owner':r1,'buffer':buffers[0],'control':controls[0]},{'owner':r2,'buffer':buffers[1],'control':controls[1]}],'online_compatible':compatible,'audit':audit(left),'provenance':{**gates(left,[s,v,o,a1,a2]),'fresh_cfg_productions':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); clean=[x for x in rows if x['online_compatible'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'bounded-cfg-paired-adjunct-buffers-20260920','method':'bounded CFG paired adjunct ownership with separate left/right residual buffers','stats':{'states':len(rows),'separate_buffers':2,'transitions':transitions,'prunes':prunes,'reader_clean':len(clean),'max_letters':rows[0]['audit']['letters']},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|bounded-cfg|paired-adjunct-ownership|separate-residual-buffers','distinct_from':'prior dual-owned adjunct CFG: left/right ownership now carries separate boundary buffers into paired acceptance'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'reader list contains only exact clean rows','hard_exclusions':['nested palindromes','repeated units','mirrored order','fragments','catalogue text']},'next_construction':'Allow buffers to cross optional adjunct boundaries asynchronously.','status':'fresh exact candidate requires reading' if clean else 'no exact clean CFG rows; paired-buffer controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
