"""Schema-aware discovery; never assumes one file equals one independent experiment."""
import hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/block-seams'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def run():
 output=OUT/'historical-schema-inventory-001.json'
 if output.exists():raise RuntimeError('immutable inventory exists')
 discovery=json.loads((OUT/'other-output-source-discovery-001.json').read_text());rows=[];errors=[]
 outputkeys={'outputs','candidates','results','hits','closures','terminals','accepted','promoted','reader_worthy','exact_near_misses'}
 for entry in discovery['sources']:
  p=ROOT/entry['path'];raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
  try:
   d=[json.loads(line) for line in raw.splitlines() if line.strip()] if p.suffix=='.jsonl' else json.loads(raw)
  except Exception as e:errors.append({'path':entry['path'],'error':str(e)[:200]});continue
  obj=d if isinstance(d,dict) else {};keys=sorted(obj)
  method=str(obj.get('method',obj.get('experiment',obj.get('version',obj.get('algorithm','')))))[:300]
  stem=re.sub(r'-(local|remote|summary|audit|analysis|results|provenance)$','',p.stem)
  lineage=str(obj.get('experiment',obj.get('run_id',stem)))
  lists=[]
  for k in sorted(outputkeys & obj.keys()):
   if isinstance(obj[k],list):
    values=obj[k];textrows=[]
    for i,v in enumerate(values):
     if isinstance(v,str):textrows.append({'index':i,'text':v})
     elif isinstance(v,dict):
      for tk in ('text','rendered_text','rendered','raw_text','sentence','palindrome'):
       if isinstance(v.get(tk),str):textrows.append({'index':i,'text':v[tk],'record':v});break
    lists.append({'key':k,'items':len(values),'direct_text_count':len(textrows),'payload_sha256':digest(values),'text_records':textrows})
  rows.append({'path':entry['path'],'sha256':sha,'schema':keys if isinstance(d,dict) else ['JSONL' if p.suffix=='.jsonl' else 'ARRAY'],'method':method,'lineage':lineage,'top_level_lists':lists,'declared_attempts':{k:obj[k] for k in ('attempts','trials','proposed','evaluated','completed_runs','total_runs') if isinstance(obj.get(k),(int,float))},'provenance_present':any(k in obj for k in ('provenance','parameters','config','manifest_sha256','source','source_sha256')),'role': 'summary_or_review' if any(x in p.stem for x in ('analysis','audit','review','summary','feedback','ratings','plan','manifest')) else 'generation_or_unresolved'})
 exact=defaultdict(list);derived=defaultdict(list);schemas=Counter()
 for r in rows:
  exact[r['sha256']].append(r['path']);schemas[tuple(r['schema'])]+=1
  for li in r['top_level_lists']:
   if li['items']:derived[(r['lineage'],li['key'],li['payload_sha256'])].append(r['path'])
 groups=defaultdict(list)
 for r in rows:groups[r['method'] or r['lineage']].append(r)
 families=[]
 for name,rs in groups.items():
  valid=[r for r in rs if r['role']=='generation_or_unresolved' and r['provenance_present'] and any(li['direct_text_count'] for li in r['top_level_lists'])]
  families.append({'family':name,'files':len(rs),'candidate_representatives':[r['path'] for r in valid[:3]],'status':'needs stage/attempt-denominator verification before sampling' if valid else 'no direct provenance-backed output list identified','listed_direct_text_occurrences':sum(li['direct_text_count'] for r in rs for li in r['top_level_lists'])})
 record={'source_count':len(rows),'errors':errors,'unique_byte_payloads':len(exact),'exact_copy_groups':[v for v in exact.values() if len(v)>1],'same_lineage_nonempty_payload_copy_groups':[{'lineage':k[0],'field':k[1],'sha256':k[2],'files':v} for k,v in derived.items() if len(v)>1],'schemas':[{'keys':list(k),'files':n} for k,n in schemas.most_common()],'families':sorted(families,key=lambda x:(not bool(x['candidate_representatives']),-x['listed_direct_text_occurrences'],x['family'])),'files':rows,'deduplication_policy':'Exact byte copies and same explicit lineage plus identical nonempty output payloads are copies. Shared controls, empty arrays, similar filenames or identical output text across independent runs do not establish run duplication.','coverage':'Discovery includes all 2663 listed historical sources. Output/attempt stages unresolved are not eligible for denominator inflation or quality claims; controls/examples explicitly excluded from generation lists.'}
 output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] if k not in ('families','schemas') else record[k][:12] for k in ('source_count','unique_byte_payloads','errors','schemas','families')},indent=2))
if __name__=='__main__':run()
