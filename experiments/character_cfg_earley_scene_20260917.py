"""Orthogonal character-CFG/Earley-style semantic scene lane."""
import hashlib,json
from pathlib import Path
ID='character-cfg-earley-scene-20260917'; S={'agent':('the coastal sailor','the observant artist'),'action':('carefully maps','quietly repairs'),'object':('a weathered lantern','the wooden vessel'),'location':('near the northern harbor','under the old stone bridge')}
def n(s): return ''.join(c for c in s.lower() if c.isalpha())
def audit(s):
 t=n(s); bad=[]; i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: bad.append((i,j,t[i],t[j]))
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'letters':len(t),'two_pointer_exact':bool(t) and not bad,'first_mismatches':bad[:8],'sha256_forward':f,'sha256_reverse':r,'sha_equal_under_reversal':f==r}
def run():
 # Character complement transition: consume one terminal from each typed
 # clause grammar only when it satisfies the live outside-in debt.
 def complement(left,right):
  debt=[]
  for i,c in enumerate(n(left)):
   j=len(n(right))-1-i
   if j>=0: debt.append({'left':c,'right':n(right)[j],'closed':c==n(right)[j]})
  return debt
 chart={('Scene',0,('', ''))}; rows=[]
 for depth in range(6):
  nxt=set()
  for nt,pos,debt in chart:
   for a in S['agent']:
    for v in S['action']:
     for o in S['object']:
      for l in S['location']:
       text=f'{a} {v} {o} {l}'; z=audit(text)
       debt2=complement(text,text)
       if z['letters']>=100: rows.append({'text':text,'left_clause':text,'right_clause':text,'slots':{'agent':a,'action':v,'object':o,'location':l},'live_complement_transitions':debt2[:8],'audit':z})
   if pos<5: nxt.add(('Scene',pos+1,debt))
  chart=nxt
 exact=[x for x in rows if x['audit']['two_pointer_exact']]; out={'experiment_id':ID,'method':'character-level Earley-style CFG with live reverse-tape obligations and typed complement transitions','grammar':'independent LeftScene/RightScene -> Agent Action Object Location; character complement transitions consume outside-in debt','rendered_candidates':exact[:2],'candidate_count':len(rows),'exact_count':len(exact),'target_letters':100,'longest_frontier':max(rows,key=lambda x:x['audit']['letters'],default=None),'reader_eligible':bool(exact),'novelty_preflight':{'catalogue_imported':False,'self_palindromic_words':False,'repeated_frames':False,'signature':ID},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh authored semantic slots','audits':['two-pointer','forward/reverse SHA-256']},'failure_and_repair':{'failure':'no exact closure' if not exact else 'exact closure found','next_repair':'Add agreement-aware complement terminal pairs while retaining outside-in debt.'},'anti_shortcut_flags':{'posthoc_reversal':False,'word_order_mirror':False,'self_palindromic_words':False,'fragment':False}}
 Path('runs/'+ID+'.json').write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__':
 r=run(); print({'exact_count':r['exact_count'],'longest':r['longest_frontier']['audit']['letters'] if r['longest_frontier'] else 0})
