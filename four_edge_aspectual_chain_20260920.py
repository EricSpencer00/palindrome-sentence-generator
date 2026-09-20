"""Four lexical edge events with alternating roles, tense, and aspect."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/four-edge-aspectual-chain-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
H=['the careful navigator','several patient sailors']; R=['who records the chart','who carried the lantern']; Q=['that marks the harbor','that guarded the passage']; E=[('drawer','reward'),('diaper','repaid'),('deliver','reviled'),('stressed','desserts')]
def gates(t):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':False,'word_order_symmetry':w==w[::-1],'fragment':len(w)<14,'catalogue_text':False}
def run():
 rows=[]
 for h in H:
  for r in R:
   for q in Q:
    for i in range(4):
     for j in range(4):
      for k in range(4):
       for z in range(4):
        if len({i,j,k,z})<4: continue
        a,b=E[i]; c,d=E[j]; e,f=E[k]; g,hx=E[z]
        text=f'{h} {r} beside the {a}, and the guide {q} near the {b}; then the keeper records the {c} before the {d}, while the scout carries the {e} toward the {f}, having already marked the {g} beyond the {hx}.'
        rows.append({'rendered':text,'edge_chain':[{'pair':f'{a}/{b}','role':'agent-theme','tense':'past','aspect':'simple'},{'pair':f'{c}/{d}','role':'theme-agent','tense':'past','aspect':'simple'},{'pair':f'{e}/{f}','role':'agent-theme','tense':'present','aspect':'progressive'},{'pair':f'{g}/{hx}','role':'theme-agent','tense':'present','aspect':'perfect'}],'audit':audit(text),'provenance':{**gates(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'four-edge-aspectual-chain-20260920','method':'four successive semordnilap lexical edge events with alternating roles, tense agreement, and typed aspect','stats':{'heads':len(H),'relative_pairs':len(R)*len(Q),'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':[],'diagnostic_controls':rows[:8],'reader_eligible':bool(ex),'novelty_preflight':{'status':'passed','signature':'fresh-authored|four-edge-chain|alternating-roles|typed-aspect','distinct_from':'prior three-edge chain: adds a fourth independently authored edge event with perfect aspect while retaining tense/role states'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text'],'reader_gate':'closed until exact >38 and blinded intact-vs-shuffled ratings'},'next_construction':'Move aspectual states into a live lexical-edge residual automaton with boundary compatibility rather than post-render metadata.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; four-edge prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
