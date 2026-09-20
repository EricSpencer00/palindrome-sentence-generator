"""Bounded recipient/theme and locative-attachment endpoint product."""
import argparse,hashlib,itertools,json,re,socket
from pathlib import Path
FRAMES=[
 ("ditransitive","sg",("a","the"),("bard","pilot","keeper"),("gives","sends","shows"),("the","a"),("child","sailor","visitor"),("the","a"),("map","song","key")),
 ("ditransitive","pl",("the",),("bards","pilots","keepers"),("give","send","show"),("the",),("children","sailors","visitors"),("the",),("maps","songs","keys")),
 ("locative","sg",("a","the"),("traveler","singer","gardener"),("rests","sings","works"),("by","near","under"),("the",),("harbor","river","willow")),
 ("locative","pl",("the",),("travelers","singers","gardeners"),("rest","sing","work"),("by","near","under"),("the",),("harbors","rivers","willows")),
]
def tape(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=tape(s);r=t[::-1]
 return {'letters':len(t),'two_pointer_exact':bool(t) and all(t[i]==t[-1-i] for i in range(len(t))), 'pointer_mismatches':sum(a!=b for a,b in zip(t,r))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest()}
def yields():
 for kind,num,*parts in FRAMES:
  for vals in itertools.product(*parts):
   if kind=='ditransitive':
    d,s,v,rd,rec,td,theme=vals; words=[d,s,v,rd,rec,td,theme]; features={'recipient':rec,'theme':theme,'attachment':'recipient-theme'}
   else:
    d,s,v,prep,td,loc=vals; words=[d,s,v,prep,td,loc]; features={'recipient':None,'theme':None,'attachment':'locative','prep':prep}
   yield {'words':words,'frame':{'kind':kind,'number':num,'agreement':True,'valency':kind,'features':features}}
def key(x):
 t=tape(' '.join(x['words']));f=x['frame'];return (t[0],t[-1],len(t)%5,f['valency'],f['features']['attachment'])
def run(min_letters,limit):
 items=list(yields());b={}
 for x in items:b.setdefault(key(x),[]).append(x)
 rows=[];probes=pruned=0
 for l in items:
  lt=tape(' '.join(l['words']))
  for k in [(lt[-1],lt[0],len(lt)%5,l['frame']['valency'],l['frame']['features']['attachment'])]:
   probes+=1
   for r in b.get(k,[]):
    if r['words']==l['words'] or r['frame']['number']!=l['frame']['number']:continue
    t=tape(' '.join(l['words']+r['words']))
    if len(t)<min_letters:continue
    if any(t[i]!=t[-1-i] for i in range(len(t)//2)):pruned+=1;continue
    rows.append({'rendered':' '.join(l['words']+r['words']),'audit':audit(' '.join(l['words']+r['words'])),'reader_worthy':False,'provenance':{'construction':'recipient/theme and locative attachment endpoint product','left_frame':l,'right_frame':r,'live_character_invariant':True,'agreement_checked':True,'valency_checked':True,'attachment_feature_keyed':True,'no_finished_tape_reversal':True,'no_posthoc_repair':True,'catalogue_text':False}})
    if len(rows)>=limit:break
   if len(rows)>=limit:break
  if len(rows)>=limit:break
 return {'experiment':'role-attachment-endpoint-product-20260926','host':socket.gethostname(),'parameters':{'min_letters':min_letters,'limit':limit},'endpoint_buckets':len(b),'probes':probes,'pruned':pruned,'candidates':rows,'closures':len(rows),'reader_worthy':0,'controls':[{'rendered':'The pilot sends a song to the sailor.','audit':audit('The pilot sends a song to the sailor.')}],'next_construction':'Add semantic recipient/theme compatibility and locative scene roles inside the same live bucket state before any optional adjunct.'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--min-letters',type=int,default=40);ap.add_argument('--limit',type=int,default=10);ap.add_argument('--out',required=True);a=ap.parse_args();p=run(a.min_letters,a.limit);Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('endpoint_buckets','probes','pruned','closures')}))
if __name__=='__main__':main()
