"""Small fresh boundary-ending replacement operator.

Each row starts from an authored grammatical opening and replaces only the
ending constituent.  The operator checks live outside-in character agreement
before any ranking, then records how far the continuation survives.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; ID="fresh-boundary-ending-replacement-20260919"
OUT=ROOT/"runs"/(ID+".json")

# Openings and endings are fresh, ordinary constituents.  The first pair is a
# deliberately strong 4-character boundary witness ("I saw" / "was I").
CASES=[
 ("I saw", "the careful keeper mark the ledger", ["was I", "left early", "the harbor"]),
 ("We sew", "a blue banner beside the window", ["so we", "at noon", "the cloth"]),
 ("At dawn", "the patient baker carries warm loaves", ["near the inn", "at dusk", "home"]),
 ("A calm", "young teacher gathers paper models", ["for class", "after rain", "today"]),
]

def tape(s): return ''.join(c.lower() for c in s if c.isascii() and c.isalpha())
def outer(s):
 t=tape(s); reach=0; first=None
 for i in range(len(t)//2):
  j=len(t)-1-i
  if t[i]!=t[j]: first={"offset":i,"left":t[i],"right":t[j]}; break
  reach+=1
 return {"letters":len(t),"outer_match_chars":reach,"first_mismatch":first,
         "exact":bool(t) and first is None,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
         "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def flags(s):
 ws=[tape(w) for w in re.findall('[A-Za-z]+',s)]; c=[w for w in ws if len(w)>2]
 return {"repeated_content":len(c)!=len(set(c)),"self_palindromic_content_words":[w for w in c if w==w[::-1]],
         "word_order_mirror":ws==[w[::-1] for w in ws],"borrowed_catalogue_text":False,"finished_tape_reversed":False}

def run():
 rows=[]
 for opening,middle,replacements in CASES:
  for ending in replacements:
   s=f"{opening} {middle}, {ending}."; a=outer(s)
   rows.append({"rendered":s,"opening":opening,"middle":middle,"replacement_ending":ending,
    "audit":a,"reach_4_8_12_16":{str(k):a["outer_match_chars"]>=k for k in (4,8,12,16)},"shortcut_flags":flags(s),
    "provenance":{"generator":ID,"phrase_author":"fresh local constituent inventory","replacement_operator":"ending constituent only","catalogue_imported":False,"posthoc_reverse":False}})
 exact=[r for r in rows if r['audit']['exact'] and not any(r['shortcut_flags'].values())]
 best=max(rows,key=lambda r:(r['audit']['outer_match_chars'],r['audit']['letters']))
 return {"experiment_id":ID,"status":"exact_candidate" if exact else "completed_no_exact_closure","cases":len(CASES),"rows":rows,"exact_candidates":exact,"best":best,"independent_validation":"two-pointer outside-in audit plus SHA-256 forward/reverse tapes","reader_gate":"closed unless exact novel survivor","next_repair":"Replace only the first mismatching ending constituent in the best row, preserving its opening and middle, and test the next continuation against the same 4/8/12/16 gates.","generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 OUT.parent.mkdir(exist_ok=True); d=run(); OUT.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps({'status':d['status'],'rows':len(d['rows']),'best':d['best']['rendered'],'reach':d['best']['audit']['outer_match_chars']}))
