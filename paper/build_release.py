"""Build source/evidence archives with a content hash manifest; no network calls."""
from pathlib import Path
import hashlib, json, subprocess, zipfile
ROOT=Path(__file__).resolve().parents[1]
def run(*args): subprocess.run(args,cwd=ROOT,check=True)
run('python3','paper/build_revision_tables.py')
run('python3','experiments/verify_revision.py')
source=[ROOT/'paper/naacl2027.tex', ROOT/'paper/refs.bib', *sorted((ROOT/'paper').glob('revision-*.tex'))]
evidence=[]
for pattern in ['artifacts/norvig-v3/*','runs/revision-2026-09-07/*.json','runs/revision-2026-09-07/*.sha256','runs/revision-2026-09-07/*.csv','runs/revision-2026-09-07/*.txt','runs/punct/after_*.json','experiments/revision_*.py','experiments/verify_revision.py','experiments/audit_norvig_result.py','experiments/sentence_intersection-results.json','experiments/RESULTS-*.md','runs/polaris/scale_20260823/summaries.jsonl','runs/polaris/sentence_plan_20260904_204815/aggregate.json','runs/polaris/sentence_quality_20260905_011235/aggregate.json','runs/sentence_quality*120b.json','paper/SOURCE-AUDIT.md','paper/build_revision_tables.py','paper/build_release.py']:
 evidence.extend(p for p in ROOT.glob(pattern) if p.is_file())
evidence=sorted(set(evidence))
inputs={}
for pattern in ['runs/norvig/npdict.txt','runs/norvig/pal21txt.html','runs/norvig/pal3.py','data/v3_bank.json','data/*2w*']:
 for p in ROOT.glob(pattern):
  if p.is_file(): inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
for name,files,flat in [('overleaf-revision-2026-09-07.zip',source,True),('palindrome-evidence-2026-09-07.zip',evidence,False)]:
 with zipfile.ZipFile(ROOT/'paper'/name,'w',zipfile.ZIP_DEFLATED) as z:
  hashes={}
  for p in files:
   key=p.name if flat else str(p.relative_to(ROOT));data=p.read_bytes();z.writestr(key,data);hashes[key]=hashlib.sha256(data).hexdigest()
  z.writestr('MANIFEST-SHA256.json',json.dumps(hashes,indent=2)+'\n')
  z.writestr('EXTERNAL-INPUT-SHA256.json',json.dumps(inputs,indent=2)+'\n')
  z.writestr('README.txt','Main document: naacl2027.tex. Compile with XeLaTeX/BibTeX or Tectonic.\nEvidence archive retains repository-relative paths. Run python3 experiments/verify_revision.py from its root.\nSee paper/SOURCE-AUDIT.md for provenance and missing evidence. External corpora are not bundled.\nThe Overleaf archive contains generated tables and needs no Python.\n')
 print(name,len(files),'files')
