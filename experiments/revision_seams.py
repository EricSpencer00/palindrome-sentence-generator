"""Frozen prospective model calibration and fixed-material seam-order experiment.
Run build before judge. Raw replies are retained; no failed judge is hidden.
"""
import argparse, collections, datetime, hashlib, itertools, json, pathlib, random, re, statistics, sys, urllib.request
sys.path.insert(0,'.')
from server.v3 import harvest_pair
from llm_palindrome.bigram import BigramModel
ROOT=pathlib.Path('runs/revision-2026-09-07')
def norm(s): return ''.join(re.findall('[a-z]',s.lower()))
def write(name,data): (ROOT/name).write_text(json.dumps(data,indent=2)+'\n')
def nest(pairs): return ' '.join(w for a,b in pairs for w in a)+' '+' '.join(w for a,b in reversed(pairs) for w in b)
def build():
 ROOT.mkdir(exist_ok=True)
 if (ROOT/'protocol.json').exists(): raise SystemExit('Frozen protocol exists; refusing overwrite')
 prose=['The nurse checked the label before giving the patient his medicine.', 'After the rain stopped, we carried the chairs back into the garden.', 'I missed the last train, so my brother drove me home.', 'The baker left the bread on a rack to cool before slicing it.', 'She saved the receipt because the new lamp did not work.', 'When the dog heard the doorbell, it ran toward the front door.', 'We planted the seeds in spring and picked the tomatoes in summer.', 'The teacher drew a map to show the class where the river began.', 'He turned down the music so he could hear the person on the phone.', 'The bridge was closed for repairs, so the bus took a different road.', 'My sister borrowed my coat because she had forgotten to bring hers.', 'They put the leftovers in the refrigerator after everyone finished eating.']
 rng=random.Random(20260907);cal=[]
 for i,p in enumerate(prose):
  words=re.findall('[a-z]+',p.lower()); rng.shuffle(words)
  opts=[p,' '.join(words)]; target=i%2
  if target:opts.reverse()
  cal.append(dict(id=f'cal{i:02}',a=opts[0],b=opts[1],target='AB'[target]))
 bank=json.load(open('data/v3_bank.json'));pairs=[];seen=set()
 for row in bank:
  p=harvest_pair(row['text'].split())
  if p and norm(' '.join(p[0])) not in seen:
   seen.add(norm(' '.join(p[0])));pairs.append(p)
 lengths=sorted(len(norm(nest([p]))) for p in pairs)
 support={str(k):[sum(lengths[:k]),sum(lengths[-k:])] for k in [2,4,8]}
 bm=BigramModel.from_file('data/count_2w.txt',vocab={w for p in pairs for half in p for w in half})
 items=[];groups=[]
 for g in range(12):
  ps=rng.sample(pairs,8)
  # Both joins forced by an outer i followed by inner j, normalized unigram gain.
  edge=[[bm.forward_order_gain(ps[i][0][-1],ps[j][0][0])+bm.forward_order_gain(ps[j][1][-1],ps[i][1][0]) for j in range(8)] for i in range(8)]
  def score(order):return sum(edge[a][b] for a,b in zip(order,order[1:]))
  order=max(itertools.permutations(range(8)),key=score)
  base=nest(ps); opt=nest([ps[i] for i in order]);assert norm(base)==norm(base)[::-1] and norm(opt)==norm(opt)[::-1]
  assert collections.Counter(base.split())==collections.Counter(opt.split())
  groups.append(dict(group=g,pairs=ps,order=order,baseline_score=score(range(8)),optimized_score=score(order),letters=len(norm(base))))
  for arm,t in [('random',base),('optimized',opt),('mid_single',nest([ps[0]]))]:items.append(dict(id=f'g{g:02}_{arm}',group=g,arm=arm,text=t))
 controls=[x[1] for x in json.load(open('runs/punct/punct_after_items.json'))['items'] if x[0]=='hand'][:12]
 for i,t in enumerate(controls):items.append(dict(id=f'best{i:02}',group=i,arm='catalogue',text=t))
 rng.shuffle(items)
 protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=['gpt-oss:20b','mistral:latest','deepseek-r1:8b','gemma3:4b','llama3.1:8b'],gate='12/12 correct on twelve held-out prose versus shuffle pairs; ties/malformed/missing fail; fixed before experimental ratings',rubric='0: word salad with no recoverable connected meaning; 1: local meaningful phrases but no coherent whole; 2: a recoverable connected meaning despite strained English; 3: clear natural connected English.',analysis='Show all model gates. Grade all experimental items for each passing model only. Report paired optimized minus random, catalogue versus mid as scale check, all arm means and floor rates. Model results are not human judgments. Ordering optimizes a bigram proxy, not the outcome. No causal seam-count claim: 2 and 8 have disjoint length support. No independence claim across model families.',length_support=support,pairs=len(pairs),calibration=cal,items=items,groups=groups)
 write('protocol.json',protocol);(ROOT/'protocol.sha256').write_text(hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()+'\n')
 write('human-blind-items.json',[dict(id=x['id'],text=x['text'],score=None) for x in items]);print(support)
def call(model,prompt):
 body=dict(model=model,messages=[dict(role='user',content=prompt)],stream=False,options=dict(temperature=0,num_predict=512),think=False,keep_alive='2m')
 req=urllib.request.Request('http://127.0.0.1:11435/api/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as f:return json.load(f)
def judge():
 p=json.loads((ROOT/'protocol.json').read_text());results=[]
 for model in p['models']:
  row=dict(model=model,calibration=[],ratings=[])
  for c in p['calibration']:
   try:
    raw=call(model,'Which text has more coherent ordinary English meaning? Answer exactly A or B.\nA: '+c['a']+'\nB: '+c['b']);reply=raw.get('message',{}).get('content','').strip();pick=reply if reply in ['A','B'] else None
    row['calibration'].append(dict(id=c['id'],correct=pick==c['target'],pick=pick,raw=raw))
   except Exception as e:row['calibration'].append(dict(id=c['id'],correct=False,error=str(e)));break
   write('judges.partial.json',results+[row])
  row['hits']=sum(x['correct'] for x in row['calibration']);row['pass']=row['hits']==12 and len(row['calibration'])==12
  if row['pass']:
   for item in p['items']:
    try:
     raw=call(model,'Rate the entire text for connected meaning. '+p['rubric']+' Judge the whole, not just recognizable words. Answer exactly one digit 0, 1, 2, or 3.\nTEXT: '+item['text']);reply=raw.get('message',{}).get('content','').strip();score=int(reply) if reply in ['0','1','2','3'] else None
     row['ratings'].append(dict(id=item['id'],arm=item['arm'],group=item['group'],score=score,raw=raw))
    except Exception as e:row['ratings'].append(dict(id=item['id'],arm=item['arm'],group=item['group'],score=None,error=str(e)))
    write('judges.partial.json',results+[row])
  results.append(row);write('judges.json',results);print(model,row['hits'],len(row['ratings']),flush=True)
if __name__=='__main__':
 {'build':build,'judge':judge}[sys.argv[1]]()
