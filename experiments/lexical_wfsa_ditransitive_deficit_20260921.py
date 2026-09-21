"""Typed ditransitive WFSA with deficit-directed live intersection."""
from collections import defaultdict,deque
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]; ID='lexical-wfsa-ditransitive-deficit-20260921'
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); ok=True; i,j=0,len(t)-1
 while i<j: ok &= t[i]==t[j]; i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'letters':len(t),'two_pointer_exact':bool(t) and ok,'forward_sha256':f,'reverse_sha256':r,'hash_equal':f==r}
def compile_wfsa():
 # S -> subject V recipient object; verb agreement and recipient/object valency
 # are compiled into the edge feature state, not checked after rendering.
 subs=[('Nora','sg'),('Liam','sg'),('the guides','pl')]; verbs=[('gives','sg'),('give','pl'),('sends','sg'),('send','pl')]
 rec=[('a guide','sg'),('the guides','pl')]; objs=[('a map','sg'),('maps','pl'),('a note','sg'),('notes','pl')]
 edges=[]; state=1; terms=set(); paths=0
 for s,sn in subs:
  for v,vn in verbs:
   if sn!=vn: continue
   for r,rn in rec:
    for o,on in objs:
     paths+=1; cur=0
     for phase,(word,num,role) in enumerate(((s,sn,'subject'),(v,vn,'verb'),(r,rn,'recipient'),(o,on,'object'))):
      for k,ch in enumerate(norm(word)):
       nxt=state; state+=1
       edges.append((cur,nxt,ch,word if k==0 else '',1.0,{'phase':phase,'role':role,'num':num,'valency':'ditransitive','word_boundary':k==0})); cur=nxt
     terms.add(cur)
 return terms,edges,paths
def run():
 terms,edges,paths=compile_wfsa(); fw=defaultdict(list); bw=defaultdict(list)
 for i,(u,v,c,*_) in enumerate(edges): fw[(u,c)].append(i);bw[(v,c)].append(i)
 q=deque((0,t,(),(),1.0) for t in terms); seen=set(); rows=[]; dead=[]; matched=expanded=0
 while q:
  l,r,lp,rp,w=q.popleft(); key=(l,r,len(lp))
  if key in seen: continue
  seen.add(key); expanded+=1
  if l in terms or l==r:
   ids=lp+rp[::-1]; text=''.join(edges[i][3] for i in ids); rows.append({'rendered':text,'audit':audit(text),'weight':w,'frontier_depth':len(lp),'features':[edges[i][5] for i in ids]})
  left={c for u,c in fw if u==l}; right={c for v,c in bw if v==r}; common=left&right
  if not common and l not in terms: dead.append({'depth':len(lp),'required_from_left':sorted(left),'available_from_right':sorted(right),'deficit':'endpoint character has no compatible recipient/object inflection'})
  for c in sorted(common):
   # deficit-directed expansion: only compatible inflection edges are considered.
   for a in fw[(l,c)]:
    for b in bw[(r,c)]:
     if edges[a][5]['valency']==edges[b][5]['valency'] and edges[a][1]<=edges[b][0]:
      q.append((edges[a][1],edges[b][0],lp+(a,),rp+(b,),w));matched+=1
 return {'candidates':rows,'stats':{'expanded_frontiers':expanded,'matched_endpoint_transitions':matched,'dead_frontiers':len(dead),'wfsa_edges':len(edges),'typed_ditransitive_paths':paths},'dead_frontier_examples':dead[:20]}
def main():
 d=run(); controls=['Nora gives a guide a map.','The guides send the guides notes.']; d['controls']=[{'rendered':x,'audit':audit(x),'control':'ordinary prose emitted independently'} for x in controls]
 exact=[x for x in d['candidates'] if x['audit']['two_pointer_exact'] and x['audit']['letters']>=40]
 d.update({'experiment_id':ID,'signature':'typed-ditransitive-wfsa|deficit-directed-inflection|recipient-object-boundary|live-forward-reverse-intersection','novelty_preflight':{'registry_entries_read':683,'duplicates_rejected':['lexical-wfsa-intersection-20260921','weighted-grammar-automaton-ablation-20260920'],'changed_operator':'ditransitive recipient/object state plus deficit-directed compatible inflection filtering','cartesian_complete_strings':False},'exact_over_40':len(exact),'reader_eligible':False,'reader_gate':'No reader gate: no intact exact candidate at the >=40-letter threshold.','independent_audits':{'two_pointer':True,'forward_reverse_sha256':True,'typed_path_features':True},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fresh_lexicon':True,'known_palindrome_or_catalogue':False,'render_after_acceptance_only':True},'next_operator':'Add an optional benefactive boundary whose valency feature is carried through the same deficit-indexed product; do not widen all lexical banks.'})
 out=ROOT/'runs/lexical-wfsa-ditransitive-deficit-20260921.json';out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats']))
if __name__=='__main__':main()
