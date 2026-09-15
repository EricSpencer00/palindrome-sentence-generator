"""Fresh paired-clause ledger: author both prose sides, then test the tape.

This deliberately uses no Brown spans, catalogue frames, or word-order mirror.
It is a diagnostic authoring run; only a fully admitted closure could be a
candidate.
"""
import glob, hashlib, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).parents[1]
PROPOSALS = [
    ("Quiet curators label old maps.", "Space is ample; labor later. "),
    ("Patient readers mark each clause.", "Eulac hcae kram s redaer tneitap."),
    ("Gardeners water young cedar.", "Radec gnuoy retaw srenedrag."),
    ("The calm editor revises prose.", "Esorp s esiver rotide mlac eht."),
]

def tapes_in_repo(output=None):
    out=set()
    output = Path(output).resolve() if output else None
    for f in glob.glob(str(ROOT/'data'/'*.json'))+glob.glob(str(ROOT/'runs'/'**'/'*.json*'),recursive=True):
        if output and Path(f).resolve() == output:
            continue
        try: obj=json.loads(Path(f).read_text())
        except Exception: continue
        def walk(x):
            if isinstance(x,str):
                try: out.add(normalize_letters(x))
                except ValueError: pass
            elif isinstance(x,dict):
                for v in x.values(): walk(v)
            elif isinstance(x,list):
                for v in x: walk(v)
        walk(obj)
    return out

def main():
    output=ROOT/'runs'/'luna_fresh_paired_clause_20260915.json'
    known=tapes_in_repo(output); rows=[]
    for left,right in PROPOSALS:
        text=(left.rstrip('.!?')+' '+right.strip()).strip()
        tape=normalize_letters(text)
        checks=mechanical_admission_checks(text,min_letters=39,max_letters=180)
        rows.append({'left':left,'right':right,'text':text,'tape':tape,
                     'exact':tape==tape[::-1],'known_tape':tape in known,
                     'checks':checks,'admitted':all(checks.values())})
    out={'status':'complete_diagnostic_no_catalogue_or_brown_material',
         'state_space_signature':hashlib.sha256(b'fresh-paired-clause-ledger-v1|4 proposals|independent sides').hexdigest(),
         'known_tapes_fingerprinted':len(known),'proposals':rows,
         'next_operator':'Generate semantic right clauses by reverse-aware character ledger, preserving complete finite clauses at every boundary.'}
    output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'out':str(output),'known':len(known),'exact':sum(r['exact'] for r in rows),'admitted':sum(r['admitted'] for r in rows)}))
if __name__=='__main__': main()
