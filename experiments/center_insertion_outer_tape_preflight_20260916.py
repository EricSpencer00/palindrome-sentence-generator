#!/usr/bin/env python3
"""Two-request preflight: both dimensions are already retained."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/center-insertion-outer-tape-preflight-20260916.json'
def run():
 return {'experiment_id':'center-insertion-outer-tape-preflight-20260916','status':'preflight_blocked','routes':{'A1':{'signature':'center-phrase-insertion|semantic-cfg|bilateral-character-obligations','overlaps':['gpt2-center-letter-bridge','reversible-grammar-insertion-20260916'],'reason':'Center insertion and stacked grammar wrappers are retained.'},'A2':{'signature':'outer-tape-preserving-lexical-repair|inner-material-change|exact-reaudit','overlaps':['semantic-insertion-repair','seed-local-mirrored-character-edit'],'reason':'Outer-tape-preserving insertion and lexical repair are retained.'}},'reason':'Neither requested route is disjoint; running them would replay prior center insertion or local outer-tape repair.','pivot':'Construct a fresh whole-scene operator with nonlocal semantic reassignment and no fixed outer tape.','rendered_candidates':[],'reader_eligible':False}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps({'status':'preflight_blocked','routes':2,'rendered':0}))
