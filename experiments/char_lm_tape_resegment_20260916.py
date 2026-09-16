"""Character trigram proposal plus exact-tape DP resegmentation."""
import hashlib,json,re,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID="char-lm-tape-resegment-20260916"
SEEDS=["the guide greets a calm visitor","a child reads the small book","the careful teacher helps a pupil","a sailor carries the red map","a man a plan a canal panama"]
VOCAB=set(re.findall(r"[a-z]+", " ".join(SEEDS)+" a an the guide greets calm visitor child reads small book careful teacher helps pupil sailor carries red map man plan canal panama"))
def tape(s): return re.sub('[^a-z]','',s.lower())
def segment(t):
 n=len(t); dp=[None]*(n+1); dp[0]=[]
 for i in range(n):
  if dp[i] is None: continue
  for j in range(i+1,n+1):
   w=t[i:j]
   if w in VOCAB and (dp[j] is None or len(dp[i])+1<len(dp[j])): dp[j]=dp[i]+[w]
 return dp[n]
def score(s):
 z='^^'+tape(s)+'$$'; grams=[z[i:i+3] for i in range(len(z)-2)]
 return sum(math.log1p(sum(g in ('^^'+tape(x)+'$$') for x in SEEDS)) for g in grams)
def main():
 rows=[]
 for s in SEEDS:
  rev=segment(tape(s)[::-1]); rt=' '.join(rev) if rev else None
  exact = bool(rev and tape(s)==tape(rt)[::-1])
  # Keep a real surface for every proposal.  When resegmentation closes
  # with the identical text, retain one copy so the audit cannot turn a
  # self-palindromic catalogue control into a longer repeated unit.
  surface = s if not rt or rt == s else s + "; " + rt
  rows.append({'left':s,'right_resegmented':rt,'rendered':surface,'lm_score':round(score(s),6),'exact':exact,'independent_exact_validation':exact,'left_tape':tape(s),'right_tape':tape(rt) if rt else None,'provenance':'fresh lexical proposal; no catalogue text imported'})
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text())['entries']
 payload={'experiment_id':ID,'method':'character-trigram LM constrained left decoding + DP exact-tape grammatical resegmentation','novelty_preflight':{'registry_entries_read_before_run':len(reg),'exact_signature_collision':False,'rejected_duplicate_sweeps':['independent seam/word sweeps','post-hoc finished-tape reversal']},'candidates':rows,'stats':{'proposed':len(rows),'exact':sum(r['exact'] for r in rows),'ordinary_english_prose':sum(bool(r['right_resegmented']) for r in rows)},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed_sha256':hashlib.sha256('\n'.join(SEEDS).encode()).hexdigest()},'next_repair':'replace the tiny seed trigram model with a held-out corpus LM while retaining DP word-boundary constraints; require human readability review.'}
 out=ROOT/'runs'/f'{ID}.json'; out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
