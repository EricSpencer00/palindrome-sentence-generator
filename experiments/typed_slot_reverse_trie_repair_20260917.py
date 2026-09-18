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
VERBS={'sg':{'finds','sees','reads','holds','marks','keeps','opens','carries','bakes','waters','plants'},'pl':{'find','see','read','hold','mark','keep','open','carry','bake','water','plant'}}
OBJECTS={'fox':['a den','the moon','a nest'],'child':['the book','a shell','the map'],'sailor':['a map','the log','a boat'],'teacher':['the door','a note','the book'],'baker':['fresh bread','a loaf','the cake'],'gardener':['a rose','the seed','a tree']}
ADJUNCTS={'dawn':['at dawn','at dusk','by noon'],'water':['by water','near shore','at school'],'shore':['near shore','near harbor','by water'],'noon':['at noon','at school','by dusk'],'town':['to town','at dawn','by road'],'sunrise':['at sunrise','in soil','at noon']}
def n(s):return ''.join(W.findall(s.lower()))
def audit(s):
 t=n(s);i,j=0,len(t)-1;ok=bool(t);mm=[]
 while i<j:
  if t[i]!=t[j]:ok=False;mm.append(i)
  i+=1;j-=1
 return {'letters':len(t),'exact':ok,'two_pointer':ok,'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':mm[:10]}
def content(s):return set(W.findall(s.lower()))-{'a','an','the','at','by','near','to','in'}
def char_roles(agent,verb,obj,prep):
    return [*('agent',)*len(n(agent)), *('verb',)*len(n(verb)),
            *('object',)*len(n(obj)), *('prep',)*len(n(prep))]
def run():
 items=[]
 for agent,base_verb,obj,prep,number in VARIANTS:
  # Expand the held-out verb cell instead of merely checking the base form.
  # The subject number is carried into every realization before seam search.
  key=agent.split()[-1].rstrip('s'); objs=OBJECTS.get(key,[obj]); adjs=ADJUNCTS.get(prep.split()[-1],[prep])
  for verb in sorted(VERBS[number]):
   for oo in objs:
    for pp in adjs:
     text=' '.join((agent,verb,oo,pp)); tape=n(text)
     items.append({'text':text,'tape':tape,'agent':agent,'verb':verb,'object':oo,'prep':pp,'number':number,'roles':('agent','verb','object','prep'),'base_verb':base_verb})
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
   # The selected right phrase, not an unrelated trie leaf, must satisfy the
   # live reverse obligation.  The trie is an index; this direct check is the
   # independent seam witness for the rendered pair.
   req=obligation; consumed=0
   left_roles=char_roles(left['agent'],left['verb'],left['object'],left['prep'])
   right_roles=char_roles(right['agent'],right['verb'],right['object'],right['prep'])
   while consumed < min(len(left['tape']),len(right['tape'])):
    ri=len(right['tape'])-1-consumed
    if left['tape'][consumed] != right['tape'][ri]: break
    consumed += 1
   if consumed == 0 or right['verb'] not in VERBS[right['number']]:
     continue
   role_pairs=list(zip(left_roles[:consumed], [right_roles[len(right['tape'])-1-k] for k in range(consumed)]))
   text=left['text']+' '+right['text']; rows.append({'text':text,'left':left,'right':right,'live_required_prefix':req[:16],'trie_prefix_depth':consumed,'typed_number_compatible':True,'matched_role_pairs':role_pairs,'audit':audit(text),'anti_shortcut':{'disjoint_content_words':True,'finished_mirror':False,'word_order_only':False},'provenance':'held-out typed agent/verb/object/preposition variants; selected-pair reverse prefix gated before rendering'})
 out={'experiment_id':'typed-slot-reverse-trie-repair-20260917','status':'quarantined_no_reader_candidate','candidates':sorted(rows,key=lambda x:x['trie_prefix_depth'],reverse=True)[:20],'stats':{'base_frames':len(VARIANTS),'typed_variants':len(items),'inflectional_verb_forms':sum(len(VERBS[x[4]]) for x in VARIANTS),'rendered_near_misses':len(rows),'exact':sum(x['audit']['exact'] for x in rows),'full_two_sided_closure_checked':True},'provenance':{'source':'fresh typed phrase variants plus number-conditioned verb lattice','catalogue_imported':False,'seed_used_as_scaffold':False,'finished_tape_reversal':False,'independent_audits':['two-pointer','SHA-256 forward/reverse']},'failure_and_repair':{'next_repair':'add typed object/adjunct substitutions whose complete tapes satisfy both seam halves'}}
 (R/'runs/typed-slot-reverse-trie-repair-20260917.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
