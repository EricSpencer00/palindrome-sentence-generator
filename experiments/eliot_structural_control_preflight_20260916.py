#!/usr/bin/env python3
"""Preflight Eliot-control alternative segmentation; blocked as catalogue-derived."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/eliot-structural-control-preflight-20260916.json'
def run():
 return {'experiment_id':'eliot-structural-control-preflight-20260916','status':'preflight_blocked','requested_signature':'eliot-control-structure-learning|alternative-cross-boundary-segmentation|independent-role-substitution|online-character-equation|catalogue-span-rejection','reason':'Learning a slot pattern from the known 85-letter Eliot palindrome is catalogue-derived structural imitation; existing registry already records abstract-role analogy and catalogue-gated cross-boundary routes. Substitution would not establish independent generation.','overlaps':['abstract-role-shape-analogy','corpus-semantic-frame-planning','model-authored-clause-bank'],'pivot':'Use the Eliot string only as a blinded non-generated control in reader packaging; generate candidates from a fresh semantic grammar without importing its spans or slot pattern.','rendered_candidates':[],'reader_eligible':False}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps({'status':'preflight_blocked','rendered':0}))
