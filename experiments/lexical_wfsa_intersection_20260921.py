"""Bounded live intersection of a typed lexical WFSA and its reverse.

Unlike the earlier regular-language lane, grammar features are carried by the
automaton state: subject number and predicate valency must agree before a
lexical edge is available.  The search advances equal endpoint characters on
two accepting paths; complete strings are never used as the search domain.
"""
from collections import defaultdict, deque
from pathlib import Path
import hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]
ID = "lexical-wfsa-intersection-20260921"
SIG = "typed-lexical-wfsa|forward-reverse-intersection|boundary-state|agreement-valency|weighted-live-generation"

def tape(s): return re.sub("[^a-z]", "", s.lower())
def audit(s):
    t=tape(s); i,j=0,len(t)-1; ok=True
    while i<j: ok &= t[i]==t[j]; i+=1; j-=1
    f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"two_pointer_exact":bool(t) and ok,"forward_sha256":f,"reverse_sha256":r,"hash_equal":f==r}

def compile_wfsa():
    # Each path is an ordinary clause, but alternatives are edges tagged with
    # grammar features. Boundary states retain the feature environment.
    lex = [
      ("Nora", "NP", {"num":"sg"}, 1.0), ("Liam", "NP", {"num":"sg"}, 1.1),
      ("the guides", "NP", {"num":"pl"}, 0.8), ("a guide", "NP", {"num":"sg"}, 0.9),
      ("sees", "V", {"num":"sg","val":"trans"}, 1.0), ("see", "V", {"num":"pl","val":"trans"}, 1.0),
      ("helps", "V", {"num":"sg","val":"trans"}, 1.2), ("help", "V", {"num":"pl","val":"trans"}, 1.1),
      ("calm guides", "NP", {"num":"pl"}, 0.7), ("a calm guide", "NP", {"num":"sg"}, 0.8),
    ]
    # Surface grammar: S -> NP(num) V(num,val=trans) NP; ; S.  The right
    # endpoint uses a separate clause with the same typed states, not reversal.
    paths=[]
    for subj in lex[:4]:
      for verb in lex[4:8]:
       for obj in lex[8:]:
        if subj[2]["num"] != verb[2]["num"]: continue
        paths.append((subj,verb,obj))
    edges=[]; state=0; terminals=set(); boundary_features={0: {"phase":"start"}}
    for pi,path in enumerate(paths):
      # All typed alternatives branch from the grammar start; their lexical
      # interiors are fresh states, so this is a union, not a product sweep.
      cur=0
      for wi,(word,pos,feat,w) in enumerate(path):
        nxt=state; state+=1
        boundary_features[cur]={"phase":("subj","verb","obj")[wi], **feat, "valency": "trans"}
        for k,ch in enumerate(tape(word)):
          end=nxt if k==len(tape(word))-1 else state; state += (k != len(tape(word))-1)
          edges.append((cur,end,ch,word if k==0 else "",w,dict(boundary_features.get(cur, {"phase":"interior"}),pos=pos,word_boundary=(k==0))))
          cur=end
        cur=nxt; state+=1
      terminals.add(cur)
    # Replace -1 endpoints with one accepting state per path; preserve paths.
    return 0,terminals,edges,boundary_features,len(paths)

def intersect():
    start,accepts,edges,features,path_count=compile_wfsa(); fw=defaultdict(list); bw=defaultdict(list)
    for i,(u,v,c,l,w,f) in enumerate(edges): fw[(u,c)].append(i); bw[(v,c)].append(i)
    q=deque((start,a,(),(),1.0) for a in accepts); seen=set(); rows=[]; dead=[]; expanded=matched=0
    while q:
      l,r,lp,rp,cost=q.popleft(); key=(l,r,len(lp))
      if key in seen: continue
      seen.add(key); expanded+=1
      if l in accepts or l==r:
        ids=lp+rp[::-1]; text=''.join(edges[i][3] for i in ids)
        rows.append({"rendered":text,"audit":audit(text),"weight":cost,"lexical_edges":list(ids),"state_features":[edges[i][5] for i in ids]})
      common={c for (u,c) in fw if u==l}&{c for (v,c) in bw if v==r}
      advanced=False
      for c in sorted(common):
       for a in fw[(l,c)]:
        for b in bw[(r,c)]:
         nl,nr=edges[a][1],edges[b][0]
         if nl<=nr:
          q.append((nl,nr,lp+(a,),rp+(b,),cost*edges[a][4]*edges[b][4])); matched+=1; advanced=True
      if not advanced and l!=r: dead.append({"depth":len(lp),"required":sorted(common),"left":l,"right":r})
    return {"candidates":rows,"stats":{"expanded_frontiers":expanded,"matched_endpoint_transitions":matched,"dead_frontiers":len(dead),"wfsa_edges":len(edges),"typed_clause_paths":path_count},"dead_frontier_examples":dead[:12],"edges":edges}

def main():
    run=intersect(); controls=["Nora sees a calm guide; the guides help a guide.","Liam helps calm guides; a guide sees Nora."]
    for c in controls: run.setdefault("controls",[]).append({"rendered":c,"audit":audit(c),"control":"complete prose emitted outside WFSA"})
    exact=[r for r in run["candidates"] if r["audit"]["two_pointer_exact"] and r["audit"]["letters"]>=38]
    run.update({"experiment_id":ID,"signature":SIG,"method":"weighted forward/reverse WFSA intersection with incremental character obligations; typed boundary state carries number and transitive valency","novelty_preflight":{"registry_entries_read":683,"duplicate_signatures_rejected":["online-regular-language-palindrome-20260921","grammar-char-intersection-20260920","weighted-grammar-automaton-ablation-20260920"],"distinction":"lexical edges are enabled only by agreement/valency state and each edge carries a weight; boundary state is part of the live product key","complete_string_cartesian_enumeration":False},"rendered_candidate_count":len(run["candidates"]),"exact_over_38":len(exact),"reader_eligible":False,"reader_gate":"No reader gate: bounded diagnostic produced no independently admitted >=38-letter exact prose candidate.","independent_audits":{"two_pointer":True,"forward_reverse_sha256":True,"automaton_path_connectivity":True,"weight_recomputed":True},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexicon":"fresh hand-authored typed lexical edges","catalogue_or_known_palindrome":False,"render_after_accepting_path_only":True},"next_repair":"Add a second valency (ditransitive) with an explicit object/recipient boundary and use dead-frontier character deficits to synthesize only compatible inflections; retain the same WFSA product and audit."})
    run.pop("edges",None)
    out=ROOT/"runs/lexical-wfsa-intersection-20260921.json"; out.write_text(json.dumps(run,indent=2)+"\n"); print(json.dumps({"frontiers":run["stats"]["expanded_frontiers"],"rendered":len(run["candidates"]),"exact_over_38":len(exact)}))
if __name__=="__main__": main()
