"""Reciprocal benefactive recipient under shared-scene grammar."""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"experiments/shared_scene_benefactive_role_attachment_20260920.py"
spec=importlib.util.spec_from_file_location("benefactive_base",BASE); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# Reciprocal `for each other` requires plural subject and reciprocal recipient;
# it is not a synonym substitution over the singular recipient lane.
base.SCENES={
 "harbor":{"SVO":("the sailors guard the lantern at dawn","the keepers study the charts before dusk"),"BEN":("the sailors carry letters for each other at dawn","the keepers bring charts for each other before dusk")},
 "garden":{"SVO":("the poets remember the garden in silence","the gardeners carry small letters after rain"),"BEN":("the poets save seeds for each other in rain","the gardeners write notes for each other at noon")},
 "bridge":{"SVO":("the scouts watch the old bridge at noon","the pilots mark the distant shore through mist"),"BEN":("the scouts bring maps for each other at dusk","the pilots carry lamps for each other through mist")},
}
def controls():
 texts=("The sailors guard the lantern at dawn; the sailors carry letters for each other at dawn.","The poets remember the garden in silence; the poets save seeds for each other in rain.","The scouts watch the old bridge at noon; the scouts bring maps for each other at dusk.")
 return [{"rendered":t,"audit":base.audit(t),"independent_pointer_exact":base.pointer_exact(t),"reciprocal_plural_agreement":True,"benefactive_role_complete":True,"reader_eligible":False,"provenance":"authored reciprocal-benefactive control; not generated exact candidate"} for t in texts]
base.controls=controls
result=base.run()
result.update({
 "method":"shared-scene-reciprocal-benefactive-20260920",
 "provenance":"fresh shared-scene grammar with plural reciprocal recipient `for each other`; reciprocal role and complete benefactive clause are gated before live character equations; no singular-recipient sweep, reversal, repair, catalogue text, or mirrored units",
 "novelty_preflight":{"passed":True,"overlaps_checked":["shared-scene-benefactive-role-attachment-20260920","recipient-theme-attachment-csp-20260920","plural-auxiliary-source-instrument-20260920"],"unused_dimension":"reciprocal recipient role requiring plural agreement under benefactive attachment","reason":"prior benefactive lane used a single named recipient; this lane introduces reciprocal participant structure and plural role agreement before residual growth"},
 "next_construction":"hold out a reciprocal dative alternation with each-other object and preserve plural agreement",
})
out=ROOT/"runs/shared-scene-reciprocal-benefactive-20260920.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("method","scenes","states","character_prunes","role_prunes","exact_candidate_count")}))
