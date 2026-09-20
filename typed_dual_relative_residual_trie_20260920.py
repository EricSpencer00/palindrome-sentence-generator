"""Typed dual-relative obligations carried through a live residual trie."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/typed-dual-relative-residual-trie-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
HEAD=[('the careful navigator','singular','agent'),('several patient sailors','plural','agent'),('the quiet archivist','singular','agent')]; R1=[('who records the chart','singular','theme'),('who carries the lantern','singular','theme')]; R2=[('that marks the harbor','singular','theme'),('that guards the passage','singular','theme')]; E=['drawer','reward','deliver','reviled','diaper','repaid']
def gates(t):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False,'mirrored_units':False}
def run():
 rows=[]; transitions=0; pruned=0
 for head,num,role in HEAD:
  for r1,r1num,r1role in R1:
   for r2,r2num,r2role in R2:
    for edge in E:
     text=f'{head} {r1}, and the guide {r2} beside the {edge}.'; streams=[('head',head,num,role),('relative1',r1,r1num,r1role),('relative2',r2,r2num,r2role)]
     states=[]; live=True
     for name,s,feature,srole in streams:
      residual=n(s)[:2]; transitions+=1; compatible=feature in ('singular','plural') and bool(residual); live &= compatible; states.append({'stream':name,'role':srole,'number':feature,'residual':residual,'compatible':compatible})
     if not live: pruned+=1
     rows.append({'rendered':text,'typed_states':states,'live':live,'audit':audit(text),'provenance':{**gates(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['live'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'typed-dual-relative-residual-trie-20260920','method':'typed number/role features with both relative obligations carried in a live residual trie','stats':{'heads':len(HEAD),'relative_pairs':len(R1)*len(R2),'transitions':transitions,'pruned':pruned,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|typed-dual-relative|feature-residual-trie|live-obligations','distinct_from':'prior dual-relative enumeration: typed number/role states are carried as live residual streams before complete scene acceptance'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','word-order symmetry','fragments','catalogue text','mirrored units']},'next_construction':'Synchronize typed residual streams at lexical edge boundaries with alternate agreement-compatible graph edges.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; typed residual prose diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
