#!/usr/bin/env python3
"""Preflight coordination/ellipsis topology; blocked on retained attachment route."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/coordination-ellipsis-preflight-20260916.json'
def run():
 return {'experiment_id':'coordination-ellipsis-preflight-20260916','status':'preflight_blocked','requested_signature':'coordination-ellipsis-topology|typed-conjunct-growth|appositive-attachment|live-character-obligations|complete-prose-repair','overlaps':['coordination-attachment-constructor-20260916','brown-two-svo-attachments','typed-proper-name-caption-records','semantic-scene-graph'],'reason':'Registry already retains coordination attachment, SVO attachments, appositive incident reports, and coordinated semantic graphs. A conjunct-growth grammar would replay that topology.','pivot':'Use a new non-attachment topology: modal/evidential scope nesting with fresh semantic operators and independent lexicalization.','rendered_candidates':[],'reader_eligible':False}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps({'status':'preflight_blocked','rendered':0}))
