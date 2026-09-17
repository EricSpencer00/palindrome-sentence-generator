"""Agreement-pair relation search: number is carried with grammar state."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs'/'grammar-relation-agreement-pair-repair-20260917.json'
ID='grammar-relation-agreement-pair-repair-20260917'
PAIRS={'S':(('the','teacher','reads'),('the','writer','writes'),('the','artist','helps')),'P':(('the','teachers','read'),('the','writers','write'),('the','artists','help'))}
OBJECTS=(('a','letter'),('a','story'),('the','garden'),('a','message')); ADV=('again','often','still')
FUNCTION=frozenset('the a again often still'.split()); PATTERN=('SUBJ','OBJ','ADV','SUBJ','OBJ')
def letters(s): return ''.join(re.findall('[a-z]',''.join(s).lower()))
def audit(s):
    t=letters(s); m=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]]
    return {'letters':len(t),'exact':bool(t) and not m,'mismatches':m[:8],'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def shortcuts(ws):
    c=[w for w in ws if w not in FUNCTION]
    return {'repeated_content':len(c)!=len(set(c)),'self_palindromic_words':[w for w in ws if len(w)>1 and w==w[::-1]],'word_order_mirror':list(ws)==[w[::-1] for w in reversed(ws)]}
def solve(budget=100000):
    choices={'SUBJ':tuple(' '.join(x) for x in PAIRS['S']+PAIRS['P']),'OBJ':tuple(' '.join(x) for x in OBJECTS),'ADV':ADV}
    stack=[(0,len(PATTERN)-1,None,0,None,0,(None,)*len(PATTERN),frozenset(),0)]; seen=set(); states=bad=0; exact=[]; reject=[]; best=None
    def record(a,reason,matched):
        if any(x is None for x in a): return
        text=' '.join(a); au=audit(text); sc=shortcuts(tuple(text.split()))
        row={'rendered':text,'audit':au,'shortcuts':sc,'provenance':{'experiment':ID,'construction':'typed singular/plural agreement pair','reason':reason,'matched_character_pairs':matched,'pattern':PATTERN}}
        if au['exact'] and not any(sc.values()): exact.append(row)
        elif au['exact']: reject.append(row)
    while stack and states<budget:
        li,ri,lw,lp,rw,rp,a,u,matched=stack.pop(); states+=1
        if lw is not None and lp==len(lw): stack.append((li+1,ri,None,0,rw,rp,a,u,matched)); continue
        if rw is not None and rp==0: stack.append((li,ri-1,lw,lp,None,0,a,u,matched)); continue
        k=(li,ri,lw,lp,rw,rp,a,u)
        if k in seen: continue
        seen.add(k)
        if best is None or matched>best['matched_character_pairs']: best={'matched_character_pairs':matched,'assignment':a,'rendered_partial':' '.join(x for x in a if x),'pattern':PATTERN}
        if li>ri: record(a,'agreement relation closure',matched); continue
        if li==ri and lw is None and rw is None: continue
        if li==ri and lw is None and rw is not None:
            rem=rw[:rp]
            if rem and rem==rem[::-1]: q=list(a); q[li]=rw; record(tuple(q),'right residual',matched)
            continue
        if li==ri and rw is None and lw is not None:
            rem=lw[lp:]
            if rem and rem==rem[::-1]: q=list(a); q[ri]=lw; record(tuple(q),'left residual',matched)
            continue
        if lw is None:
            for w in choices[PATTERN[li]]:
                content=set(w.split())-FUNCTION
                if not content&u: q=list(a); q[li]=w; stack.append((li,ri,w,0,rw,rp,tuple(q),u|content,matched))
            continue
        if rw is None:
            for w in choices[PATTERN[ri]]:
                content=set(w.split())-FUNCTION
                if not content&u: q=list(a); q[ri]=w; stack.append((li,ri,lw,lp,w,len(w),tuple(q),u|content,matched))
            continue
        if lp<len(lw) and rp>0:
            if lw[lp]!=rw[rp-1]: bad+=1; continue
            stack.append((li,ri,lw,lp+1,rw,rp-1,a,u,matched+1))
    return {'states':states,'mismatch_edges':bad,'candidates':exact,'rejected_exact':reject,'best_partial':best,'budget_exhausted':states>=budget}
def main():
    r=solve(); probe='The teachers read a letter often; the writer writes a story.'
    r['near_miss_probes']=[{'rendered':probe,'audit':audit(probe),'provenance':'fresh intact authored agreement control'}]
    r['next_repair']='add typed transitive-valency pairs at the seam without reusing content words'
    OUT.write_text(json.dumps({'experiment_id':ID,'status':'completed','independent_validator':'audit','novelty_preflight':{'catalogue_family_imported':False,'construction_signature':ID},'patterns':[r]},indent=2)+'\\n')
    print(json.dumps({'states':r['states'],'mismatch_edges':r['mismatch_edges'],'exact':len(r['candidates']),'best':r['best_partial']},indent=2))
if __name__=='__main__': main()

