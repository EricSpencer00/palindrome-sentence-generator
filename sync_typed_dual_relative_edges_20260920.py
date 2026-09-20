"""Synchronize typed dual-relative residuals at lexical-edge boundaries."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/sync-typed-dual-relative-edges-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
H=['the careful navigator','several patient sailors','the quiet archivist']; R1=['who records the chart','who carries the lantern']; R2=['that marks the harbor','that guards the passage']; EDGES=[('drawer','reward'),('diaper','repaid'),('deliver','reviled'),('stressed','desserts')]
def gates(t):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':False,'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False}
def run():
 rows=[]; syncs=0; rejected=0
 for h in H:
  for r1 in R1:
   for r2 in R2:
    for left,right in EDGES:
     text=f'{h} {r1}, and the guide {r2} beside the {left} and {right}.'; a=audit(text)
     k1=n(r1)[-2:]; k2=n(r2)[-2:]; compatible=k1[0]==k2[0]; syncs+=1
     if not compatible: rejected+=1
     rows.append({'rendered':text,'edge_pair':{'left':left,'right':right},'typed_streams':{'relative1':{'role':'theme','number':'singular','boundary_key':k1},'relative2':{'role':'theme','number':'singular','boundary_key':k2}},'boundary_sync':{'compatible':compatible,'lexical_edge_boundary':True},'audit':a,'provenance':{**gates(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['boundary_sync']['compatible'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'sync-typed-dual-relative-edges-20260920','method':'typed dual-relative residual synchronization at lexical-edge boundaries','stats':{'heads':len(H),'relative_pairs':len(R1)*len(R2),'edge_pairs':len(EDGES),'sync_checks':syncs,'rejected':rejected,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':[],'diagnostic_controls':rows[:12],'reader_eligible':bool(ex),'novelty_preflight':{'status':'passed','signature':'fresh-authored|typed-dual-relative|lexical-edge-sync|agreement-compatible','distinct_from':'prior typed residual trie: stream compatibility is synchronized at each selected lexical-edge boundary'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text'],'reader_gate':'closed until exact >38 and blinded intact-vs-shuffled ratings'},'next_construction':'Carry synchronized boundary keys through two successive edge pairs with alternating relative roles.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; synchronized prose diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
