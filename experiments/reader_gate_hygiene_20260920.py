"""Reproducible cleanup for historical reader-facing candidate gates."""
from __future__ import annotations
import json,glob
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/reader-gate-hygiene-20260920.json'

def cleanable(row):
 a=row.get('audit',{}); p=row.get('provenance',{})
 if p.get('catalogue_replay') or p.get('borrowed_catalogue') or p.get('borrowed_text'):
  return None # explicitly protected
 return bool(a.get('two_pointer_exact')) and a.get('letters',0)>38 and not any(p.get(k) for k in ('mirrored_token_units','repeated_units','posthoc_repair','finished_tape_reversal'))

def run():
 paths=[]
 for p in sorted(glob.glob(str(ROOT/'runs/*.json'))):
  try:d=json.load(open(p))
  except Exception:continue
  if isinstance(d,dict) and isinstance(d.get('reader_facing_candidates'),list): paths.append(Path(p))
 targets=paths[:20]; records=[]
 for p in targets:
  d=json.load(open(p)); old=d.get('reader_facing_candidates',[]); keep=[]; moved=[]; protected=[]
  for row in old:
   ok=cleanable(row)
   if ok is True: keep.append(row)
   elif ok is None: protected.append(row); keep.append(row)
   else:moved.append(row)
  if moved:d.setdefault('diagnostic_controls',[]).extend(moved)
  d['reader_facing_candidates']=keep
  p.write_text(json.dumps(d,indent=2)+'\n')
  records.append({'artifact':str(p.relative_to(ROOT)),'before':len(old),'after':len(keep),'moved_to_diagnostic_controls':len(moved),'protected_borrowed_rows':len(protected)})
 result={'targets':len(targets),'records':records,'rule':'retain only exact >38 anti-shortcut-clean rows; protect borrowed/catalogue rows; move all other rows to diagnostic_controls'}
 OUT.write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(run(),indent=2))
