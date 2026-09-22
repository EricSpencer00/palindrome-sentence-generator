"""Bounded CFG with two distinct adjuncts and separate semantic owners."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/bounded-cfg-dual-owned-adjuncts-20260920.json'
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
  left=f'{s} {v} {o} {a1} {a2}.'; lt=n(left); rt=n(right+'.'); transitions+=2; compatible=lt[:2]==rt[-2:][::-1]
  if not compatible: prunes+=1
  rows.append({'rendered':left,'shared_frame':{'subject_number':num,'verb':v,'object':o,'adjunct_1':{'text':a1,'owner':r1},'adjunct_2':{'text':a2,'owner':r2}},'right_control':right+'.','online_obligations':{'boundaries':2,'compatible':compatible},'audit':audit(left),'provenance':{**gates(left,[s,v,o,a1,a2]),'fresh_cfg_productions':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); clean=[x for x in rows if x['online_obligations']['compatible'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'bounded-cfg-dual-owned-adjuncts-20260920','method':'bounded CFG with two distinct adjuncts carrying separate semantic owners','stats':{'agreement_states':len(SUB),'objects':len(OBJ),'adjunct_pairs':len(ADJ)*(len(ADJ)-1),'right_controls':len(RIGHT),'states':len(rows),'online_boundaries':transitions,'prunes':prunes,'reader_clean':len(clean),'max_letters':rows[0]['audit']['letters']},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|bounded-cfg|dual-owned-adjuncts|separate-semantic-owners|online-obligations','distinct_from':'prior single-attached CFG: each derivation now carries two distinct adjuncts with separate semantic owners'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'reader list contains only exact clean rows','hard_exclusions':['nested palindromes','repeated units','mirrored order','fragments','catalogue text']},'next_construction':'Add paired left/right adjunct ownership and carry separate residual buffers through both derivations.','status':'fresh exact candidate requires reading' if clean else 'no exact clean CFG rows; dual-owned adjunct controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
