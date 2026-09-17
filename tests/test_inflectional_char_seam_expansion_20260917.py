import json
from pathlib import Path
RUN=Path(__file__).parents[1]/"runs/inflectional-char-seam-expansion-20260917.json"
def test_character_expansion_rows_are_audited():
 d=json.loads(RUN.read_text()); assert d["candidate_count"]==len(d["candidates"])>0
 for r in d["candidates"]:
  assert r["character_checks"]>0 and r["audit"]["exact"]==r["audit"]["independent_two_pointer"]
  assert len(r["audit"]["sha256"])==64 and r["novelty_preflight"]["signature"]
def test_ordinary_prose_and_shortcut_flags():
 d=json.loads(RUN.read_text())
 for r in d["candidates"]:
  assert r["anti_shortcut"]["intact_prose"]
  assert r["anti_shortcut"]["repeated_unit"]==(r["words"][:4]==r["words"][4:])
