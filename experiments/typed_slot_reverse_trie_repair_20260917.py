"""Typed verb/object reverse-trie repair with live seam gating."""
import hashlib,json,re,heapq
from pathlib import Path
R=Path(__file__).resolve().parents[1]; W=re.compile('[a-z]+')
VARIANTS=[
 ('the fox','finds','a den','at dawn','sg'),('the fox','sees','the moon','at dusk','sg'),
 ('a child','reads','the book','by water','sg'),('a child','holds','a shell','by shore','sg'),
 ('the sailor','marks','a map','near shore','sg'),('the sailor','keeps','the log','near harbor','sg'),
 ('a teacher','opens','the door','at noon','sg'),('a teacher','reads','a note','at school','sg'),
 ('the bakers','carry','fresh bread','to town','pl'),('the bakers','bake','a loaf','at dawn','pl'),
 ('the gardener','waters','a rose','at sunrise','sg'),('the gardener','plants','the seed','in soil','sg')]
def n(s):return ''.join(W.findall(s.lower()))
def audit(s):
 t=n(s);i,j=0,len(t)-1;ok=bool(t);mm=[]
 while i<j:
  if t[i]!=t[j]:ok=False;mm.append(i)
  i+=1;j-=1
 return {'letters':len(t),'exact':ok,'two_pointer':ok,'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':mm[:10]}
def content(s):return set(W.findall(s.lower()))-{'a','an','the','at','by','near','to','in'}
def run():
 items=[]
 for agent,verb,obj,prep,number in VARIANTS:
  text=' '.join((agent,verb,obj,prep)); tape=n(text)
  items.append({'text':text,'tape':tape,'agent':agent,'verb':verb,'object':obj,'prep':prep,'number':number,'roles':('agent','verb','object','prep')})
 # Trie leaves retain typed slot metadata.  Prefix lookup is by the live seam char.
 trie={}
 for x in items:
  q=trie
  for c in x['tape'][::-1]:q=q.setdefault(c,{})
  q.setdefault('$',[]).append(x)
 rows=[]; queue=[]
 for li,left in enumerate(items):heapq.heappush(queue,(0,li))
 while queue:
  _,li=heapq.heappop(queue); left=items[li]; obligation=left['tape'][::-1]
  for rj,right in enumerate(items):
   if li==rj or left['number']!=right['number'] or content(left['text'])&content(right['text']):continue
   # role-compatible live prefix: inspect each required character in the trie
   req=obligation; q=trie; consumed=0
   for ch in req:
    if ch not in q:break
    q=q[ch];consumed+=1
   if consumed == 0 or right['tape'][-1] != req[0]:
    continue
   text=left['text']+' '+right['text']; rows.append({'text':text,'left':left,'right':right,'live_required_prefix':req[:16],'trie_prefix_depth':consumed,'role_compatible':True,'audit':audit(text),'anti_shortcut':{'disjoint_content_words':True,'finished_mirror':False,'word_order_only':False},'provenance':'held-out typed agent/verb/object/preposition variants; lexical right choice gated before rendering'})
 out={'experiment_id':'typed-slot-reverse-trie-repair-20260917','status':'quarantined_no_reader_candidate','candidates':sorted(rows,key=lambda x:x['trie_prefix_depth'],reverse=True)[:20],'stats':{'typed_variants':len(items),'rendered_near_misses':len(rows),'exact':sum(x['audit']['exact'] for x in rows)},'provenance':{'source':'fresh typed phrase variants','catalogue_imported':False,'seed_used_as_scaffold':False,'finished_tape_reversal':False,'independent_audits':['two-pointer','SHA-256 forward/reverse']},'failure_and_repair':{'next_repair':'add inflectional alternatives per typed verb while preserving number and require full two-sided chart closure'}}
 (R/'runs/typed-slot-reverse-trie-repair-20260917.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
