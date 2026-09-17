import json
from pathlib import Path
RUN=Path(__file__).parents[1]/"runs/seam-conditioned-lexical-trie-20260917.json"
def test_trie_run_has_audited_prose_rows():
 d=json.loads(RUN.read_text()); assert d["frontier_size"]>0; assert d["candidate_count"]==len(d["candidates"])
 for r in d["candidates"]:
  assert r["audit"]["exact"]==r["audit"]["independent_two_pointer"]
  assert len(r["audit"]["sha256"])==64 and r["anti_shortcut"]["intact_prose"]
def test_repeated_units_are_flagged():
 d=json.loads(RUN.read_text())
 for r in d["candidates"]:
  assert r["anti_shortcut"]["repeated_unit"]==(r["left_bundle"]==r["right_bundle"] and r["left_setting"]==r["right_setting"])
