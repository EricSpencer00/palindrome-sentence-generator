import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.semantic_valency_attachment_solver_20260917 import *

def test_typed_frames_and_rendered_audit():
 e=Event(VERBS[0],"i","a rat","at noon")
 assert graph(e)["arguments"]["patient"]=="a rat"
 a=audit(e,render(e,True)); assert a["typed_valency"] and a["attachment_valid"]
 assert a["rendered_prose"] and not a["exact_letter_palindrome"]

def test_residual_is_exact_and_independent():
 assert residual("was it a rat i saw","was it a rat i saw")["exact"]
 assert not residual("i saw a rat","i read a book")["exact"]

def test_bounded_solver_and_heldout_repair():
 x=solve(10); assert x["states"]==10 and len(x["held_out_repair"])==len(HELD_OUT)
 assert all(r["source"]=="held_out_adjunct" for r in x["held_out_repair"])

def test_run_artifact_has_provenance_and_next_repair():
 p=run(); q=json.loads(Path("runs/semantic-valency-attachment-solver-20260917.json").read_text())
 assert q["provenance"]["source"].startswith("human-authored")
 assert q["next_repair"] and q["anti_shortcut_checks"]
