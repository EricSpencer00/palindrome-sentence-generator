"""Bounded character-ledger authoring probe (no catalogue/Brown material)."""
import glob, json, hashlib, re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
ROOT=Path(__file__).parents[1]
LEFTS=[
 "Careful makers restore old radios.",
 "Bright students solve hard puzzles.",
 "Patient nurses record each dosage.",
 "Quiet artists frame winter scenes.",
]
RIGHT_GUESSES=["So does the team.","Then the work ends.","The notes remain.","Before dawn."]
def fingerprint():
 s=set()
 for f in glob.glob(str(ROOT/'data'/'*.json'))+glob.glob(str(ROOT/'runs'/'**'/'*.json*'),recursive=True):
  try:x=json.loads(Path(f).read_text())
  except:continue
  def w(v):
   if isinstance(v,str):
    try:s.add(normalize_letters(v))
    except ValueError:pass
   elif isinstance(v,dict):
    for z in v.values():w(z)
   elif isinstance(v,list):
    for z in v:w(z)
  w(x)
 return s
def main():
 known=fingerprint(); rows=[]
 for left in LEFTS:
  tape=normalize_letters(left)
  target=tape[::-1]
  for right in RIGHT_GUESSES:
   text=left.rstrip('.')+' '+right
   t=normalize_letters(text); c=mechanical_admission_checks(text,min_letters=39,max_letters=180)
   rows.append({'left_clause':left,'reversed_tape_constraint':target,'right_clause':right,'rendered':text,'tape':t,'exact':t==t[::-1],'known_tape':t in known,'checks':c,'admitted':all(c.values())})
 out={'status':'complete_character_ledger_promptbank','state_space_signature':hashlib.sha256(b'char-ledger-v2|4 natural left clauses|16 independently authored right guesses|reverse-tape constraint').hexdigest(),'repository_tapes':len(known),'proposals':rows,'next_operator':'Use local gpt-oss constrained decoding to author right clauses while exposing only reverse-tape-compatible prefixes.'}
 p=ROOT/'runs'/'luna_character_ledger_promptbank_20260915.json';p.write_text(json.dumps(out,indent=2)+'\n');print({'proposals':len(rows),'exact':sum(x['exact'] for x in rows),'admitted':sum(x['admitted'] for x in rows)})
if __name__=='__main__':main()
