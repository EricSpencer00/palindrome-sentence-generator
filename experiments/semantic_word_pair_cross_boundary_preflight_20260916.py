#!/usr/bin/env python3
"""Preflight requested semordnilap word-pair grammar; blocked on overlap."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/semantic-word-pair-cross-boundary-preflight-20260916.json'
def run():
 return {'experiment_id':'semantic-word-pair-cross-boundary-preflight-20260916','status':'preflight_blocked','requested_signature':'attested-semordnilap-word-pairs|cross-boundary-segmentation|distinct-syntactic-roles|complete-scene-search|length-gated-character-constraints','overlaps':['typed-semordnilap','semordnilap-template-inventory','whole-sentence-semordnilap-clauses-20260916','brown-attested-sentence-pairs'],'reason':'Existing retained routes already use attested/reversible word pairs, typed roles, cross-boundary equations, and complete-clause repair. Rendering this route would replay the same construction dimension.','pivot':'Use semantic tense/aspect alternation with discourse-connective state expansion and no reversible-word inventory.','rendered_candidates':[],'reader_eligible':False}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps({'status':'preflight_blocked','rendered':0}))
