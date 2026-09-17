"""Seam-conditioned semantic substitution over agreement-carrying role packages."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs'/'grammar-relation-semantic-substitution-20260917.json'
ID='grammar-relation-semantic-substitution-20260917'
# Fresh semantic scene: caretakers act on tangible objects. Number and valency
# are carried by each package; alternatives are deliberately small and authored.
SUBJECTS=(('the','nurse','checks'),('the','guard','opens'),('the','artist','mixes'),('the','child','finds'))
OBJECTS=(('a','map'),('a','gate'),('the','paint'),('a','note'))
PATTERN=('SUBJ','OBJ','ADV','SUBJ','OBJ'); ADVS=('again','often','still')
FUNCTION=frozenset('the a again often still'.split())
def letters(s): return ''.join(re.findall('[a-z]',''.join(s).lower()))
def audit(s):
    t=letters(s); mm=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]]
    return {'letters':len(t),'exact':bool(t) and not mm,'mismatches':mm[:8],
            'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),
            'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def flags(ws):
    c=[x for x in ws if x not in FUNCTION]
    return {'repeated_content':len(c)!=len(set(c)),
            'self_palindromic_words':[x for x in ws if len(x)>1 and x==x[::-1]],
            'word_order_mirror':list(ws)==[x[::-1] for x in reversed(ws)]}
def solve(budget=100000):
    choices={'SUBJ':tuple(' '.join(x) for x in SUBJECTS),'OBJ':tuple(' '.join(x) for x in OBJECTS),'ADV':ADVS}
    st=[(0,4,None,0,None,0,(None,)*5,frozenset(),0)]; seen=set(); states=bad=0; exact=[]; rejected=[]; best=None
    def record(a,reason,matched):
        if any(x is None for x in a): return
        text=' '.join(a); au=audit(text); fs=flags(tuple(text.split()))
        row={'rendered':text,'audit':au,'shortcut_flags':fs,'provenance':{'experiment':ID,'construction':'seam-conditioned semantic role substitution','reason':reason,'matched_character_pairs':matched,'pattern':PATTERN}}
        if au['exact'] and not any(fs.values()): exact.append(row)
        elif au['exact']: rejected.append(row)
    while st and states<budget:
        li,ri,lw,lp,rw,rp,a,u,matched=st.pop(); states+=1
        if lw is not None and lp==len(lw): st.append((li+1,ri,None,0,rw,rp,a,u,matched)); continue
        if rw is not None and rp==0: st.append((li,ri-1,lw,lp,None,0,a,u,matched)); continue
        key=(li,ri,lw,lp,rw,rp,a,u)
        if key in seen: continue
        seen.add(key)
        if best is None or matched>best['matched_character_pairs']:
            best={'matched_character_pairs':matched,'assignment':a,'rendered_partial':' '.join(x for x in a if x),'pattern':PATTERN}
        if li>ri: record(a,'complete semantic relation',matched); continue
        if li==ri and lw is None and rw is None: continue
        if li==ri and lw is None and rw is not None:
            rem=rw[:rp]
            if rem and rem==rem[::-1]: q=list(a);q[li]=rw;record(tuple(q),'right residual closure',matched)
            continue
        if li==ri and rw is None and lw is not None:
            rem=lw[lp:]
            if rem and rem==rem[::-1]: q=list(a);q[ri]=lw;record(tuple(q),'left residual closure',matched)
            continue
        if lw is None:
            for w in choices[PATTERN[li]]:
                content=set(w.split())-FUNCTION
                if not content&u: q=list(a);q[li]=w;st.append((li,ri,w,0,rw,rp,tuple(q),u|content,matched))
            continue
        if rw is None:
            for w in choices[PATTERN[ri]]:
                content=set(w.split())-FUNCTION
                if not content&u: q=list(a);q[ri]=w;st.append((li,ri,lw,lp,w,len(w),tuple(q),u|content,matched))
            continue
        if lp<len(lw) and rp>0:
            if lw[lp]!=rw[rp-1]: bad+=1; continue
            st.append((li,ri,lw,lp+1,rw,rp-1,a,u,matched+1))
    return {'states':states,'mismatch_edges':bad,'candidates':exact,'rejected_exact':rejected,'best_partial':best,'budget_exhausted':states>=budget}
def main():
    r=solve(); probe='The nurse checks a map often; the guard opens a gate.'
    r['near_miss_probes']=[{'rendered':probe,'audit':audit(probe),'shortcut_flags':flags(tuple(probe.lower().replace('.','').split())),'provenance':'fresh authored intact semantic scene'}]
    r['next_repair']='permit controlled determiner and clitic substitutions at the seam while preserving semantic valency'
    OUT.write_text(json.dumps({'experiment_id':ID,'status':'completed','independent_validator':'audit','novelty_preflight':{'catalogue_family_imported':False,'construction_signature':ID},'patterns':[r]},indent=2)+'\n')
    print(json.dumps({'states':r['states'],'mismatch_edges':r['mismatch_edges'],'exact':len(r['candidates']),'best':r['best_partial']},indent=2))
if __name__=='__main__':main()

