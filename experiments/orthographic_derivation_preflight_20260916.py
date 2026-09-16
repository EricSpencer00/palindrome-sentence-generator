#!/usr/bin/env python3
"""Preflight orthographic derivation route; blocked because morphology is saturated."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/orthographic-derivation-preflight-20260916.json'
def run():
 return {'experiment_id':'orthographic-derivation-preflight-20260916','status':'preflight_blocked','requested_signature':'orthographic-derivational-composition|allomorphic-morpheme-choice|semantic-role-live-state|joint-character-obligations|complete-prose-repair','overlaps':['morphology-semantic-template-csp','morphological-derivational-seam','morphology-first-dependency-lattice-20260916','inflectional-fst-clitic-tape'],'reason':'The registry already covers derivational/inflectional morphology, allomorph/FST choices, and live character obligations in complete semantic clauses. A new orthographic morphology route would replay that state dimension.','pivot':'Move to a non-morphological construction: discourse-source/evidential planning with fresh scene roles and independently authored lexicalization.','rendered_candidates':[],'reader_eligible':False}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps({'status':'preflight_blocked','rendered':0}))
