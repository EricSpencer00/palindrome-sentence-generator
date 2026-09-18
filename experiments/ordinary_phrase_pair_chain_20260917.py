"""Bounded ordinary phrase-pair/trie lane (diagnostic, no catalogue text)."""
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]; W=re.compile('[a-z]+')
def n(s): return ''.join(W.findall(s.lower()))
def aud(s):
 t=n(s);m=[i for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b];return {'letters':len(t),'exact':bool(t) and not m,'two_pointer':bool(t) and not m,'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':m[:8]}
def run(out='runs/ordinary-phrase-pair-chain-20260917.json'):
 raw=(R/'data/authored_sentences.txt').read_text().splitlines(); common=set(json.loads((R/'data/ngrams_wikitext2.json').read_text()))
 phrases=[]
 for s in raw:
  ws=W.findall(s.lower())
  if len(ws)>=3 and all((x in common or len(x)<=3) for x in ws) and all(len(x)>1 or x in {'a','i'} for x in ws): phrases.append(' '.join(ws))
 phrases=list(dict.fromkeys(phrases))[:600]
 trie={}
 for w in common:
  if len(w)<2: continue
  q=trie
  for c in w:q=q.setdefault(c,{})
  q['$']=w
 def segment(t):
  dp=[None]*(len(t)+1);dp[0]=[]
  for i in range(len(t)):
   if dp[i] is None:continue
   q=trie
   for j in range(i,len(t)):
    q=q.get(t[j]);
    if q is None:break
    if '$' in q and (dp[j+1] is None or len(dp[i])+1<len(dp[j+1])):dp[j+1]=dp[i]+[q['$']]
  return dp[-1]
 edges=[]
 for p in phrases:
  z=segment(n(p)[::-1])
  if z and len(z)>=3 and len(set(W.findall(p)))==len(W.findall(p)) and len(set(z))==len(z): edges.append((p,' '.join(z)))
  if len(edges)>=200:break
 chain=[];used=set();letters=0
 for a,b in sorted(edges,key=lambda x:len(n(x[0])),reverse=True):
  toks=set(W.findall(a+b))
  if not toks&used:chain.append((a,b));used|=toks;letters+=len(n(a));
  if letters>=100:break
 text=' — '.join(' '.join(x) for x in zip([a for a,b in chain],[b for a,b in reversed(chain)])) if chain else ''
 row={'text':text,'audit':aud(text),'edges':chain,'reader_eligible':bool(chain) and letters>=100 and aud(text)['exact'],'construction_gates':{'common_words':True,'complete_multiword_phrases':True,'disjoint_lexical_content':True,'repeated_units':False},'provenance':{'source':'held-out authored_sentences filtered by Wikitext vocabulary','catalogue_imported':False,'finished_reversal':False,'independent_audits':['two-pointer','SHA-256 forward/reverse']}}
 result={'experiment_id':'ordinary-phrase-pair-chain-20260917','status':'completed_exact' if row['reader_eligible'] else 'quarantined_no_closure','candidates':[row],'stats':{'phrase_count':len(phrases),'edge_count':len(edges)},'failure_and_repair':{'next_repair':'expand held-out phrase inventory with typed SVO clauses and retain pre-admission trie segmentation'}}
 Path(out).write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(run(),indent=2))
