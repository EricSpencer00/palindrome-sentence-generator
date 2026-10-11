"""Compare exact text against surviving local structured/text evidence."""
from pathlib import Path
import gzip,json,re,hashlib,time
from llm_palindrome.bilateral_seams import norm
ROOT=Path.cwd();D=ROOT/'research/block-seams/constructive-019'
target=norm('Do geese on a cedar trade canoe see God?')
records=[];matches=[];start=time.monotonic();unsupported=0
def comparable(value):
 global unsupported
 try:return norm(value)
 except ValueError:unsupported+=1;return None
# Search every surviving structured evidence file in data/research/experiments.
# JSON strings are normalized whole, never inferred from substring matches.
selected=[ROOT/'data/v3_bank.json']
for p in (ROOT/'research/block-seams').rglob('*'):
 if not p.is_file() or D in p.parents or 'imports' in p.parts:continue
 if any(x in p.name for x in ('outputs','complete-pool','bank.json','raw-results.json','results.json')) and any(p.name.endswith(x) for x in ('.json','.jsonl','.json.gz','.jsonl.gz')):
  if any(x in p.name for x in ('controlled-seam-ablation','boundary-root-diagnostic','mined-results','comparison-run','depth-diagnostic')):continue
  selected.append(p)
for p in sorted(set(selected)):
  opener=gzip.open if p.suffix=='.gz' else open
  strings=0;hits=[]
  with opener(p,'rt',encoding='utf-8',errors='replace') as f:
   for line_number,line in enumerate(f,1):
    if 'canoe' not in line.lower():continue
    if p.name.endswith('.txt'):
     if comparable(line.strip())==target:hits.append(line_number)
    else:
     for m in re.finditer(r'"(?:[^"\\]|\\.)*"',line):
      value=json.loads(m.group());strings+=1
      if comparable(value)==target:hits.append(line_number)
  records.append(dict(path=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,strings_checked=strings,matching_lines=hits))
  if hits:matches.append(records[-1])
result=dict(target_tape=target,files_checked=len(records),unsupported_strings=unsupported,matches=matches,records=records,elapsed_seconds=time.monotonic()-start,scope='Canonical surviving v3 bank and block-seams output/result/pool files, excluding import snapshots and failure/state logs; ASCII canoe byte-prefilter before whole-string normalization; standard surviving renderings and normalized tapes. No missing cloud records or worldwide originality claim.')
(D/'corpus-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
